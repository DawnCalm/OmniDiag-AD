import argparse
import json
from collections import Counter
from pathlib import Path

import mmcv
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ShareGPT/LLaVA-style BEV-VLM QA data from exported BEV features"
    )
    parser.add_argument("manifest", help="path to manifest jsonl")
    parser.add_argument(
        "--output",
        default="outputs/bev_vlm/bev_vlm_sharegpt.json",
        help="output json path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional cap on generated QA samples",
    )
    parser.add_argument(
        "--point-cloud-range",
        nargs=4,
        type=float,
        default=[-54.0, -54.0, 54.0, 54.0],
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="XY bounds used to map lidar coordinates into the saved BEV tensor grid",
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


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_path(path_value):
    return str(Path(path_value).expanduser().resolve())


def build_gt_lookup(manifest_rows):
    ann_files = sorted({row["gt_ref"]["ann_file"] for row in manifest_rows})
    gt_lookup = {}
    for ann_file in ann_files:
        ann_data = mmcv.load(resolve_path(ann_file))
        for info in ann_data["infos"]:
            gt_lookup[info["token"]] = info
    return gt_lookup


def direction_from_xy(x, y):
    front = "前方" if x >= 0 else "后方"
    if abs(y) < 2.5:
        return front
    lateral = "左侧" if y >= 0 else "右侧"
    if front == "前方":
        return f"前{lateral}"
    return f"后{lateral}"


def filter_gt_objects(info):
    names = info.get("gt_names", [])
    boxes = info.get("gt_boxes", [])
    num_lidar_pts = info.get("num_lidar_pts")
    objects = []
    for idx, (name, box) in enumerate(zip(names, boxes)):
        if num_lidar_pts is not None and num_lidar_pts[idx] <= 0:
            continue
        objects.append(
            {
                "label_name": str(name),
                "center": [float(v) for v in box[:3]],
                "size": [float(v) for v in box[3:6]],
                "yaw": float(box[6]),
                "direction": direction_from_xy(float(box[0]), float(box[1])),
            }
        )
    return objects


def load_prediction_objects(pred_path):
    with Path(resolve_path(pred_path)).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objects", [])


def xy_to_grid(x, y, height, width, point_cloud_range):
    x_min, y_min, x_max, y_max = point_cloud_range
    x_norm = (x - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (y - y_min) / max(y_max - y_min, 1e-6)
    grid_x = int(round(x_norm * (width - 1)))
    grid_y = int(round(y_norm * (height - 1)))
    return max(0, min(width - 1, grid_x)), max(0, min(height - 1, grid_y))


def region_feature_strength(feature_path, center, size, point_cloud_range):
    tensor = torch.load(resolve_path(feature_path), map_location="cpu").float()
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor in {feature_path}, got shape {tuple(tensor.shape)}")

    _, height, width = tensor.shape
    x, y = center[:2]
    length = max(size[0], 1.0)
    width_m = max(size[1], 1.0)
    cx, cy = xy_to_grid(x, y, height, width, point_cloud_range)
    x_radius = max(1, int(round(length / (point_cloud_range[2] - point_cloud_range[0]) * width * 0.5)))
    y_radius = max(1, int(round(width_m / (point_cloud_range[3] - point_cloud_range[1]) * height * 0.5)))
    x0, x1 = max(0, cx - x_radius), min(width, cx + x_radius + 1)
    y0, y1 = max(0, cy - y_radius), min(height, cy + y_radius + 1)
    patch = tensor[:, y0:y1, x0:x1]
    if patch.numel() == 0:
        return 0.0
    return float(patch.abs().mean().item())


def describe_scene(gt_objects):
    counts = Counter(obj["label_name"] for obj in gt_objects)
    front_objects = [obj for obj in gt_objects if obj["center"][0] >= 0]
    front_counts = Counter(obj["label_name"] for obj in front_objects)
    car_count = counts.get("car", 0)
    pedestrian_count = counts.get("pedestrian", 0)
    scene_bits = []
    if car_count:
        scene_bits.append(f"车辆以 car 为主，共 {car_count} 辆")
    if pedestrian_count:
        scene_bits.append(f"pedestrian 共 {pedestrian_count} 人")
    if not scene_bits:
        most_common = counts.most_common(3)
        if most_common:
            scene_bits.append(
                "主要目标包括 "
                + "、".join(f"{name} {count} 个" for name, count in most_common)
            )
        else:
            scene_bits.append("当前场景中的有效目标较少")

    if front_counts:
        scene_bits.append(
            "前方区域最显著的目标有 "
            + "、".join(f"{name} {count} 个" for name, count in front_counts.most_common(3))
        )
    else:
        scene_bits.append("前方区域暂无显著目标")
    return "；".join(scene_bits) + "。"


def choose_anchor_object(pred_objects):
    if not pred_objects:
        return None
    front_objects = [obj for obj in pred_objects if obj["center"][0] >= 0]
    candidate_pool = front_objects if front_objects else pred_objects
    return sorted(candidate_pool, key=lambda obj: obj.get("score", 0.0))[0]


def explain_anchor(record, anchor, point_cloud_range):
    if anchor is None:
        return (
            "当前帧没有稳定的预测目标，因此无法给出对象级归因。"
            "这通常意味着场景较空或检测输出整体偏弱。"
        )

    camera_strength = region_feature_strength(
        record["camera_bev_path"], anchor["center"], anchor["size"], point_cloud_range
    )
    lidar_strength = region_feature_strength(
        record["lidar_bev_path"], anchor["center"], anchor["size"], point_cloud_range
    )
    fused_strength = region_feature_strength(
        record["fused_bev_path"], anchor["center"], anchor["size"], point_cloud_range
    )

    score = float(anchor.get("score", 0.0))
    direction = direction_from_xy(anchor["center"][0], anchor["center"][1])
    reasons = [f"{direction}的 {anchor['label_name']} 预测分数为 {score:.2f}"]
    if camera_strength < lidar_strength * 0.8:
        reasons.append("视觉 BEV 响应偏弱，说明该目标更可能受到遮挡、裁剪或纹理不足影响")
    elif lidar_strength < camera_strength * 0.8:
        reasons.append("LiDAR BEV 响应偏弱，更像是距离偏远、点云稀疏或几何回波不足")
    else:
        reasons.append("相机与 LiDAR 的局部响应接近，低分更像是类别边界模糊而不是单模态缺失")

    if fused_strength < max(camera_strength, lidar_strength):
        reasons.append("融合特征没有明显放大局部证据，说明跨模态互补有限")
    else:
        reasons.append("融合特征仍保留了该区域的证据，说明目标并非完全不可见")
    return "；".join(reasons) + "。"


def build_confidence_answer(pred_objects, anchor, record, point_cloud_range):
    if not pred_objects:
        return (
            "当前场景整体感知可信度约为 0.15。"
            "建议自车保持低速并等待更多观测，因为检测结果几乎无法支撑稳定决策。"
        )

    top_scores = sorted((float(obj.get("score", 0.0)) for obj in pred_objects), reverse=True)[:5]
    mean_score = sum(top_scores) / max(len(top_scores), 1)
    if anchor is not None:
        camera_strength = region_feature_strength(
            record["camera_bev_path"], anchor["center"], anchor["size"], point_cloud_range
        )
        lidar_strength = region_feature_strength(
            record["lidar_bev_path"], anchor["center"], anchor["size"], point_cloud_range
        )
        agreement = 1.0 - abs(camera_strength - lidar_strength) / max(
            camera_strength + lidar_strength, 1e-6
        )
    else:
        agreement = 0.2

    confidence = max(0.0, min(1.0, 0.7 * mean_score + 0.3 * agreement))
    if confidence >= 0.7:
        advice = "建议维持当前行驶策略，但继续监测前方动态目标。"
    elif confidence >= 0.45:
        advice = "建议保守通过，优先关注前方或侧前方的低分目标。"
    else:
        advice = "建议减速并准备让行，因为当前感知证据不足以支持激进操作。"

    return f"当前场景整体感知可信度约为 {confidence:.2f}。{advice}"


def build_sample(record, info, point_cloud_range):
    gt_objects = filter_gt_objects(info)
    pred_objects = load_prediction_objects(record["pred_path"])
    anchor = choose_anchor_object(pred_objects)

    scene_question = "请概括这个场景的主要交通参与者，尤其说明前方区域有多少车辆和行人。"
    explanation_question = "为什么当前最不稳定的前向目标置信度偏低？请结合相机、LiDAR 与融合 BEV 的局部证据解释。"
    trust_question = "当前系统整体感知可信度大约是多少？自车更适合采取什么保守操作？"

    scene_answer = describe_scene(gt_objects)
    explanation_answer = explain_anchor(record, anchor, point_cloud_range)
    trust_answer = build_confidence_answer(pred_objects, anchor, record, point_cloud_range)

    return {
        "id": record["sample_token"],
        "images": record["image_paths"],
        "bev_features": {
            "camera": resolve_path(record["camera_bev_path"]),
            "lidar": resolve_path(record["lidar_bev_path"]),
            "fused": resolve_path(record["fused_bev_path"]),
        },
        "conversations": [
            {"from": "human", "value": scene_question},
            {"from": "gpt", "value": scene_answer},
            {"from": "human", "value": explanation_question},
            {"from": "gpt", "value": explanation_answer},
            {"from": "human", "value": trust_question},
            {"from": "gpt", "value": trust_answer},
        ],
        "metadata": {
            "sample_token": record["sample_token"],
            "split": record["split"],
            "pred_path": record["pred_path"],
            "gt_ref": record["gt_ref"],
        },
    }


def main():
    args = parse_args()
    manifest_rows = load_jsonl(args.manifest)
    if args.max_samples is not None:
        manifest_rows = manifest_rows[: args.max_samples]
    gt_lookup = build_gt_lookup(manifest_rows)

    samples = []
    for record in manifest_rows:
        info = gt_lookup.get(record["sample_token"])
        if info is None:
            continue
        samples.append(build_sample(record, info, args.point_cloud_range))

    save_json(args.output, samples)
    print(f"Saved {len(samples)} BEV-VLM QA samples to {args.output}")


if __name__ == "__main__":
    main()
