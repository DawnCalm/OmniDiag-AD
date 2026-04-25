import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


TASK_TO_ID = {
    "scene": 0,
    "attribution": 1,
    "trust": 2,
}


def task_type_to_id(task_type):
    return TASK_TO_ID.get(task_type, 0)


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
        question_ids = self.tokenizer.encode(
            record["question"],
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
        "records": [item["record"] for item in batch],
    }


class MultiModalRecordDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
