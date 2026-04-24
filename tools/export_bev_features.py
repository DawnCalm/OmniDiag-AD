import argparse
import copy
import json
from pathlib import Path

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export BEVFusion camera/lidar/fused BEV features and predictions"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument(
        "--output-dir",
        default="outputs/bev_vlm",
        help="directory that stores exported features and manifests",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="dataset split to export",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional cap on exported samples",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="dtype used when persisting BEV tensors",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options",
    )
    return parser.parse_args()


def load_config(config_path, cfg_options=None):
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    return cfg


def resolve_dataset_cfg(cfg, split):
    split_cfg = copy.deepcopy(cfg.data[split])
    if split == "train" and split_cfg.get("type") == "CBGSDataset":
        split_cfg = split_cfg["dataset"]
        split_cfg["pipeline"] = copy.deepcopy(cfg.data.val.pipeline)
    split_cfg.test_mode = True
    return split_cfg


def iter_meta_list(batch_metas):
    metas = batch_metas.data[0]
    if isinstance(metas, dict):
        return [metas]
    return metas


def json_ready_prediction(result, class_names):
    boxes = result["boxes_3d"].tensor.cpu().tolist()
    scores = result["scores_3d"].cpu().tolist()
    labels = result["labels_3d"].cpu().tolist()
    uncertainties = None
    if "uncertainty_3d" in result:
        uncertainties = result["uncertainty_3d"].cpu().tolist()

    objects = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        obj = {
            "label": int(label),
            "label_name": class_names[int(label)],
            "score": float(score),
            "center": [float(v) for v in box[:3]],
            "size": [float(v) for v in box[3:6]],
            "yaw": float(box[6]),
        }
        if len(box) >= 9:
            obj["velocity"] = [float(v) for v in box[7:9]]
        if uncertainties is not None and idx < len(uncertainties):
            obj["uncertainty"] = float(uncertainties[idx])
        objects.append(obj)

    return {"num_objects": len(objects), "objects": objects}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("`tools/export_bev_features.py` currently requires a CUDA device.")

    cfg = load_config(args.config, args.cfg_options)
    dataset_cfg = resolve_dataset_cfg(cfg, args.split)
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=max(1, cfg.data.get("workers_per_gpu", 1)),
        dist=False,
        shuffle=False,
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    model.module.configure_feature_export(
        output_dir=args.output_dir,
        split=args.split,
        save_dtype=args.save_dtype,
    )

    output_root = Path(args.output_dir)
    split_dir = output_root / args.split
    manifest_rows = []
    exported = 0

    with torch.no_grad():
        for batch in data_loader:
            results = model(return_loss=False, rescale=True, **batch)
            metas = iter_meta_list(batch["metas"])
            for result, meta in zip(results, metas):
                sample_token = str(meta.get("token") or meta.get("sample_idx"))
                sanitized = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sample_token
                )
                pred_path = split_dir / f"{sanitized}_pred.json"
                prediction_payload = json_ready_prediction(result, dataset.CLASSES)
                write_json(pred_path, prediction_payload)

                feature_paths = result.get("bev_feature_paths", {})
                row = {
                    "id": sample_token,
                    "sample_token": sample_token,
                    "split": args.split,
                    "image_paths": feature_paths.get("image_paths", meta.get("filename", [])),
                    "camera_bev_path": (
                        str(Path(feature_paths["camera_bev_path"]).resolve())
                        if feature_paths.get("camera_bev_path")
                        else None
                    ),
                    "lidar_bev_path": (
                        str(Path(feature_paths["lidar_bev_path"]).resolve())
                        if feature_paths.get("lidar_bev_path")
                        else None
                    ),
                    "fused_bev_path": (
                        str(Path(feature_paths["fused_bev_path"]).resolve())
                        if feature_paths.get("fused_bev_path")
                        else None
                    ),
                    "pred_path": str(pred_path.resolve()),
                    "gt_ref": {
                        "ann_file": str(Path(dataset.ann_file).resolve()),
                        "sample_token": sample_token,
                    },
                    "metadata": {
                        "timestamp": meta.get("timestamp"),
                        "lidar_path": meta.get("lidar_path"),
                        "config": args.config,
                        "checkpoint": args.checkpoint,
                    },
                }
                manifest_rows.append(row)
                exported += 1
                if args.max_samples is not None and exported >= args.max_samples:
                    break
            if args.max_samples is not None and exported >= args.max_samples:
                break

    manifest_path = output_root / f"manifest_{args.split}.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    write_json(
        output_root / f"summary_{args.split}.json",
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "num_samples": len(manifest_rows),
            "manifest_path": str(manifest_path),
        },
    )
    print(
        f"Exported {len(manifest_rows)} samples to {output_root}. "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
