import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_MANIFEST_FIELDS = [
    "sample_token",
    "camera_bev_path",
    "lidar_bev_path",
    "fused_bev_path",
    "pred_path",
    "camera_bev_render_path",
    "lidar_bev_render_path",
    "fused_bev_render_path",
    "edl_evidence_path",
    "edl_render_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate exported BEV-VLM artifacts before moving them to a server-side LLM environment."
    )
    parser.add_argument(
        "--root",
        default="outputs/bev_vlm",
        help="artifact root directory containing manifest/data outputs",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="override enriched manifest path; defaults to <root>/bev_vlm_sharegpt_v2_manifest.jsonl",
    )
    parser.add_argument(
        "--flat",
        default=None,
        help="override flat jsonl path; defaults to <root>/bev_vlm_sharegpt_v2_flat.jsonl",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_manifest_rows(rows):
    missing_fields = Counter()
    missing_files = []
    for row in rows:
        sample_token = row.get("sample_token")
        for field in REQUIRED_MANIFEST_FIELDS:
            value = row.get(field)
            if not value:
                missing_fields[field] += 1
                continue
            if field == "sample_token":
                continue
            if not Path(value).exists():
                missing_files.append((sample_token, field, value))
        anchor_crop_path = row.get("anchor_crop_path")
        if anchor_crop_path and not Path(anchor_crop_path).exists():
            missing_files.append((sample_token, "anchor_crop_path", anchor_crop_path))
    return missing_fields, missing_files


def validate_flat_rows(rows):
    missing_files = []
    task_counter = Counter()
    for row in rows:
        task_counter[row.get("task_type", "unknown")] += 1
        bev_features = row.get("bev_features", {})
        for field in ("camera", "lidar", "fused"):
            value = bev_features.get(field)
            if not value or not Path(value).exists():
                missing_files.append((row.get("id"), f"bev_features.{field}", value))
        for image_path in row.get("images", []):
            if image_path and not Path(image_path).exists():
                missing_files.append((row.get("id"), "images", image_path))
    return task_counter, missing_files


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else root / "bev_vlm_sharegpt_v2_manifest.jsonl"
    flat_path = Path(args.flat).resolve() if args.flat else root / "bev_vlm_sharegpt_v2_flat.jsonl"

    manifest_rows = load_jsonl(manifest_path)
    flat_rows = load_jsonl(flat_path)

    missing_fields, manifest_missing_files = validate_manifest_rows(manifest_rows)
    task_counter, flat_missing_files = validate_flat_rows(flat_rows)

    print(f"artifact_root={root}")
    print(f"manifest_rows={len(manifest_rows)}")
    print(f"flat_rows={len(flat_rows)}")
    print(f"task_counts={dict(task_counter)}")
    print(f"manifest_missing_fields={dict(missing_fields)}")
    print(f"manifest_missing_files={len(manifest_missing_files)}")
    print(f"flat_missing_files={len(flat_missing_files)}")

    if manifest_missing_files:
        print("manifest file issues:")
        for item in manifest_missing_files[:20]:
            print(item)
    if flat_missing_files:
        print("flat file issues:")
        for item in flat_missing_files[:20]:
            print(item)

    if missing_fields or manifest_missing_files or flat_missing_files:
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
