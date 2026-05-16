import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


TASK_TO_ID = {
    "scene": 0,
    "miss_summary": 1,
    "attribution": 2,
    "attribution_object": 2,
    "trust": 3,
}


def task_type_to_id(task_type):
    return TASK_TO_ID.get(task_type, 0)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value):
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return f"{parsed:.2f}"


def describe_anchor_brief(anchor):
    if not anchor:
        return None
    direction = anchor.get("direction")
    label = anchor.get("label_name") or anchor.get("label")
    if direction and label:
        return f"{direction}的 {label}"
    if label:
        return str(label)
    return None


def build_local_context_prompt(record):
    metadata = record.get("metadata", {}) or {}
    task_type = record.get("task_type", "")
    context_bits = []

    if task_type == "miss_summary":
        missed_gt_count = metadata.get("missed_gt_count")
        if missed_gt_count is not None:
            context_bits.append(f"漏检目标数量={int(missed_gt_count)}")
        missed_gt_objects = metadata.get("missed_gt_objects") or []
        briefs = [describe_anchor_brief(anchor) for anchor in missed_gt_objects]
        briefs = [brief for brief in briefs if brief]
        if briefs:
            context_bits.append("漏检目标=" + "、".join(briefs[:8]))
    elif task_type == "trust":
        scene_uncertainty = _format_float(metadata.get("scene_uncertainty"))
        if scene_uncertainty is not None:
            context_bits.append(f"场景不确定性={scene_uncertainty}")
        primary_anchor = metadata.get("primary_anchor_object") or metadata.get("anchor_object")
        primary_anchor_brief = describe_anchor_brief(primary_anchor)
        if primary_anchor_brief:
            context_bits.append(f"关键目标={primary_anchor_brief}")
        local_stats = metadata.get("primary_anchor_local_stats") or metadata.get("anchor_local_stats") or {}
        response_strengths = local_stats.get("response_strengths") or {}
        camera_strength = _format_float(response_strengths.get("camera"))
        lidar_strength = _format_float(response_strengths.get("lidar"))
        fused_strength = _format_float(response_strengths.get("fused"))
        if camera_strength and lidar_strength and fused_strength:
            context_bits.append(
                "局部响应="
                f"camera:{camera_strength},lidar:{lidar_strength},fused:{fused_strength}"
            )
        edl_stats = local_stats.get("edl_stats") or {}
        local_uncertainty = _format_float(edl_stats.get("local_uncertainty_mean"))
        if local_uncertainty is not None:
            context_bits.append(f"局部EDL不确定性={local_uncertainty}")
    elif task_type in {"attribution", "attribution_object"}:
        anchor = metadata.get("anchor_object") or metadata.get("primary_anchor_object")
        anchor_brief = describe_anchor_brief(anchor)
        if anchor_brief:
            context_bits.append(f"目标={anchor_brief}")
        nearby_prediction = (anchor or {}).get("nearby_prediction") or {}
        nearby_label = nearby_prediction.get("label_name")
        nearby_score = _format_float(nearby_prediction.get("score"))
        if nearby_label is not None and nearby_score is not None:
            context_bits.append(f"最近预测={nearby_label}:{nearby_score}")
        local_stats = metadata.get("anchor_local_stats") or metadata.get("primary_anchor_local_stats") or {}
        response_strengths = local_stats.get("response_strengths") or {}
        camera_strength = _format_float(response_strengths.get("camera"))
        lidar_strength = _format_float(response_strengths.get("lidar"))
        fused_strength = _format_float(response_strengths.get("fused"))
        if camera_strength and lidar_strength and fused_strength:
            context_bits.append(
                "局部响应="
                f"camera:{camera_strength},lidar:{lidar_strength},fused:{fused_strength}"
            )
        edl_stats = local_stats.get("edl_stats") or {}
        local_uncertainty = _format_float(edl_stats.get("local_uncertainty_mean"))
        if local_uncertainty is not None:
            context_bits.append(f"局部EDL不确定性={local_uncertainty}")

    if not context_bits:
        return record["question"]
    return "线索：" + "；".join(context_bits) + "\n问题：" + record["question"]


def build_model_input_text(record):
    return build_local_context_prompt(record)


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_records(records, val_ratio=0.1, seed=42):
    records = list(records)
    random.Random(seed).shuffle(records)
    val_count = int(round(len(records) * val_ratio))
    if val_count <= 0:
        return records, []
    return records[val_count:], records[:val_count]


class FlatBEVQADataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        max_question_length=256,
        max_answer_length=256,
        modalities=("camera", "lidar", "fused"),
    ):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.modalities = tuple(modalities)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        model_input_text = build_model_input_text(record)
        question_ids = self.tokenizer.encode(
            model_input_text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_question_length,
        )
        answer_ids = self.tokenizer.encode(
            record["answer"],
            add_bos=True,
            add_eos=True,
            max_length=self.max_answer_length,
        )
        bev_tensors = {}
        for modality in self.modalities:
            bev_tensors[modality] = torch.load(
                record["bev_features"][modality], map_location="cpu"
            ).float()
        return {
            "id": record["id"],
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "task_id": torch.tensor(task_type_to_id(record["task_type"]), dtype=torch.long),
            "bev_tensors": bev_tensors,
            "model_input_text": model_input_text,
            "record": record,
        }


def _pad_sequence(sequences, pad_value):
    max_len = max(seq.numel() for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.numel()] = seq
    return padded


def collate_flat_bev_qa(batch, pad_id=0):
    question_ids = _pad_sequence([item["question_ids"] for item in batch], pad_id)
    answer_ids = _pad_sequence([item["answer_ids"] for item in batch], pad_id)
    task_ids = torch.stack([item["task_id"] for item in batch], dim=0)

    modality_names = list(batch[0]["bev_tensors"].keys())
    bev_tensors = {}
    for modality in modality_names:
        bev_tensors[modality] = torch.stack(
            [item["bev_tensors"][modality] for item in batch],
            dim=0,
        )

    return {
        "ids": [item["id"] for item in batch],
        "question_ids": question_ids,
        "answer_ids": answer_ids,
        "task_ids": task_ids,
        "bev_tensors": bev_tensors,
        "model_input_texts": [item["model_input_text"] for item in batch],
        "records": [item["record"] for item in batch],
    }


class MultiModalRecordDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
