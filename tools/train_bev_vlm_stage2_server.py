import argparse
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from bev_vlm.connectors import count_trainable_parameters
from bev_vlm.data import MultiModalRecordDataset, load_records, save_json, split_records
from bev_vlm.server_models import (
    ServerQFormerLLM,
    build_image_processor,
    build_server_tokenizer,
)
from bev_vlm.server_training import collate_server_batch, evaluate_server_loss, train_server_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Server Stage 2: freeze the BEV backbone and text backbone, then train the Q-Former connector against multimodal BEV + image inputs."
    )
    parser.add_argument("data", help="flat json/jsonl dataset")
    parser.add_argument("--llm-model", required=True, help="text LLM used for Stage 2-server")
    parser.add_argument("--vision-model", required=True, help="frozen vision encoder model")
    parser.add_argument("--output-dir", default="outputs/bev_vlm/stage2_server")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-query-tokens", type=int, default=32)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-dropout", type=float, default=0.1)
    parser.add_argument("--token-grid", nargs=2, type=int, default=[4, 4])
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        choices=["question_only", "structured"],
        default="question_only",
        help="question_only keeps the server path closer to the final multimodal setup; structured can help with smoke tests",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    records = load_records(args.data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No training samples were found for Stage 2-server.")

    tokenizer = build_server_tokenizer(args.llm_model)
    image_processor = build_image_processor(args.vision_model)

    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=42)
    train_loader = DataLoader(
        MultiModalRecordDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_server_batch,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_text_length=args.max_text_length,
            max_images=args.max_images,
            prompt_mode=args.prompt_mode,
        ),
    )
    val_source = val_records if val_records else train_records[:1]
    val_loader = DataLoader(
        MultiModalRecordDataset(val_source),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_server_batch,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_text_length=args.max_text_length,
            max_images=args.max_images,
            prompt_mode=args.prompt_mode,
        ),
    )

    model = ServerQFormerLLM(
        llm_model_name=args.llm_model,
        vision_model_name=args.vision_model,
        qformer_token_grid=tuple(args.token_grid),
        num_query_tokens=args.num_query_tokens,
        qformer_layers=args.qformer_layers,
        qformer_heads=args.qformer_heads,
        qformer_dropout=args.qformer_dropout,
        freeze_llm=True,
        freeze_vision=True,
        use_lora=False,
        use_qlora=False,
        torch_dtype=args.torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        runtime_device=args.device,
    ).to(args.device)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    if image_processor is not None:
        image_processor.save_pretrained(output_dir / "image_processor")

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_server_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            max_steps=args.smoke_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_autocast=args.torch_dtype in {"float16", "bfloat16"},
            autocast_dtype=torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16,
        )
        val_loss = evaluate_server_loss(
            model,
            val_loader,
            args.device,
            max_steps=args.smoke_steps,
            use_autocast=args.torch_dtype in {"float16", "bfloat16"},
            autocast_dtype=torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "stage2_server_best.pt")
            torch.save(model.connector.state_dict(), output_dir / "qformer_connector.pt")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"trainable_connector_params={count_trainable_parameters(model.connector)}"
        )

    save_json(output_dir / "history.json", history)
    save_json(
        output_dir / "stage2_server_config.json",
        {
            "data": str(Path(args.data).resolve()),
            "llm_model": args.llm_model,
            "vision_model": args.vision_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_query_tokens": args.num_query_tokens,
            "token_grid": args.token_grid,
            "torch_dtype": args.torch_dtype,
            "prompt_mode": args.prompt_mode,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "trainable_connector_params": count_trainable_parameters(model.connector),
        },
    )
    torch.save(model.state_dict(), output_dir / "stage2_server_last.pt")
    print(f"Saved Stage 2-server artifacts to {output_dir}")


if __name__ == "__main__":
    main()
