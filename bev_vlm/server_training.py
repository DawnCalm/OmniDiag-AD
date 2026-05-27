from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_model_input_text


def build_prompt(question, task_type):
    return f"Task: {task_type}\nQuestion: {question}\nAnswer:"


def _pad_sequences(sequences, pad_value):
    max_len = max(seq.numel() for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.numel()] = seq
    return padded


def _load_images(image_paths, max_images):
    images = []
    for image_path in image_paths[:max_images]:
        if image_path and Path(image_path).exists():
            images.append(Image.open(image_path).convert("RGB"))
    return images


def _resize_chw_tensor(tensor, size):
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=tuple(size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _stack_chw_tensors(tensors):
    target_h = max(int(tensor.shape[-2]) for tensor in tensors)
    target_w = max(int(tensor.shape[-1]) for tensor in tensors)
    resized = [_resize_chw_tensor(tensor, (target_h, target_w)) for tensor in tensors]
    return torch.stack(resized, dim=0)


def collate_server_batch(
    batch,
    tokenizer,
    image_processor,
    max_text_length=512,
    max_images=5,
    prompt_mode="question_only",
):
    text_inputs = []
    text_masks = []
    label_inputs = []
    all_image_tensors = []
    batch_has_images = False
    bev_tensors = {"camera": [], "lidar": [], "fused": []}

    for record in batch:
        prompt_question = (
            build_model_input_text(record)
            if prompt_mode == "structured"
            else record["question"]
        )
        prompt = build_prompt(prompt_question, record.get("task_type", "scene"))
        full_text = prompt + " " + record["answer"]

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )["input_ids"][0]
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = min(prompt_ids.numel(), labels.numel())
        labels[:prompt_len] = -100

        text_inputs.append(input_ids)
        text_masks.append(attention_mask)
        label_inputs.append(labels)

        for modality in bev_tensors:
            bev_tensors[modality].append(
                torch.load(record["bev_features"][modality], map_location="cpu").float()
            )

        if image_processor is not None:
            images = _load_images(record.get("images", []), max_images=max_images)
            if images:
                batch_has_images = True
                encoded = image_processor(images=images, return_tensors="pt")
                all_image_tensors.append(encoded["pixel_values"])
            else:
                all_image_tensors.append(None)

    input_ids = _pad_sequences(text_inputs, tokenizer.pad_token_id)
    attention_mask = _pad_sequences(text_masks, 0)
    labels = _pad_sequences(label_inputs, -100)

    stacked_bev = {
        modality: _stack_chw_tensors(bev_list)
        for modality, bev_list in bev_tensors.items()
    }

    pixel_values = None
    if image_processor is not None and batch_has_images:
        template = next(tensor for tensor in all_image_tensors if tensor is not None)
        max_images_in_batch = min(
            max(tensor.size(0) if tensor is not None else 0 for tensor in all_image_tensors),
            max_images,
        )
        padded_images = []
        for tensor in all_image_tensors:
            if tensor is None:
                tensor = torch.zeros(
                    max_images_in_batch,
                    *template.shape[1:],
                    dtype=template.dtype,
                )
            elif tensor.size(0) < max_images_in_batch:
                pad = torch.zeros(
                    max_images_in_batch - tensor.size(0),
                    *tensor.shape[1:],
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, pad], dim=0)
            else:
                tensor = tensor[:max_images_in_batch]
            padded_images.append(tensor)
        pixel_values = torch.stack(padded_images, dim=0)

    return {
        "records": batch,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "bev_tensors": stacked_bev,
        "pixel_values": pixel_values,
    }


def move_server_batch_to_device(batch, device):
    moved = {
        "records": batch["records"],
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "bev_tensors": {
            key: value.to(device) for key, value in batch["bev_tensors"].items()
        },
        "pixel_values": None,
    }
    if batch["pixel_values"] is not None:
        moved["pixel_values"] = batch["pixel_values"].to(device)
    return moved


def train_server_epoch(
    model,
    data_loader,
    optimizer,
    device,
    max_steps=None,
    gradient_accumulation_steps=1,
    max_grad_norm=None,
    use_autocast=False,
    autocast_dtype=torch.float16,
):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()
    for step_idx, batch in enumerate(data_loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        batch = move_server_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(
            enabled=use_autocast and str(device).startswith("cuda"),
            dtype=autocast_dtype,
        ):
            outputs = model(
                bev_tensors=batch["bev_tensors"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
            )
            raw_loss = outputs.loss
            loss = raw_loss / max(gradient_accumulation_steps, 1)
        loss.backward()

        should_step = (step_idx + 1) % max(gradient_accumulation_steps, 1) == 0
        if should_step:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += float(raw_loss.item())
        steps += 1
    if steps > 0 and steps % max(gradient_accumulation_steps, 1) != 0:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_server_loss(
    model,
    data_loader,
    device,
    max_steps=None,
    use_autocast=False,
    autocast_dtype=torch.float16,
):
    model.eval()
    total_loss = 0.0
    steps = 0
    for step_idx, batch in enumerate(data_loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        batch = move_server_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(
            enabled=use_autocast and str(device).startswith("cuda"),
            dtype=autocast_dtype,
        ):
            outputs = model(
                bev_tensors=batch["bev_tensors"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
            )
        total_loss += float(outputs.loss.item())
        steps += 1
    return total_loss / max(steps, 1)
