import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from bev_vlm.data import save_json, save_jsonl
from bev_vlm.visualization import save_feature_render

CAMERA_YAW_DEGREES = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 60.0,
    "CAM_BACK_LEFT": 120.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_RIGHT": -120.0,
    "CAM_FRONT_RIGHT": -60.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create BEV-VLM QA samples, enriched manifests, and flat training jsonl."
    )
    parser.add_argument("manifest", help="path to the base manifest jsonl")
    parser.add_argument(
        "--output",
        default="outputs/bev_vlm/bev_vlm_sharegpt.json",
        help="output ShareGPT-style json path",
    )
    parser.add_argument(
        "--flat-output",
        default=None,
        help="output flat jsonl path used by the local Stage 2 training script",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="output enriched manifest jsonl path",
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
    parser.add_argument(
        "--image-mode",
        choices=["bev_only", "anchor_crop"],
        default="anchor_crop",
        help="whether to attach the anchor crop into the final image list",
    )
    parser.add_argument(
        "--crop-dir",
        default=None,
        help="directory used to save target-centric crops",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=128,
        help="minimum crop side length in pixels",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.15,
        help="relative padding applied to the selected 2D crop",
    )
    parser.add_argument(
        "--match-distance-thresh",
        type=float,
        default=2.5,
        help="minimum XY distance threshold used when deciding whether a GT object was missed",
    )
    parser.add_argument(
        "--local-patch-dir",
        default=None,
        help="directory used to save target-centric local BEV/EDL patches",
    )
    parser.add_argument(
        "--local-patch-scale",
        type=float,
        default=4.0,
        help="context multiplier applied when cropping local BEV/EDL patches around an anchor",
    )
    parser.add_argument(
        "--local-patch-grid",
        nargs=2,
        type=int,
        default=[32, 32],
        metavar=("HEIGHT", "WIDTH"),
        help="fixed H W used when saving local BEV/EDL patch tensors",
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


def resolve_path(path_value):
    return str(Path(path_value).expanduser().resolve())


def normalize_label_name(name):
    return str(name).strip().lower().replace(" ", "_")


def build_gt_lookup(manifest_rows):
    ann_files = sorted({row["gt_ref"]["ann_file"] for row in manifest_rows})
    gt_lookup = {}
    for ann_file in ann_files:
        ann_data = mmcv.load(resolve_path(ann_file))
        for info in ann_data["infos"]:
            gt_lookup[info["token"]] = info
    return gt_lookup


def angle_from_xy_deg(x, y):
    return float(np.degrees(np.arctan2(float(y), float(x))))


def angular_distance_deg(a, b):
    delta = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(delta)


def direction_from_xy(x, y):
    angle = angle_from_xy_deg(x, y)
    if -22.5 <= angle < 22.5:
        return "前方"
    if 22.5 <= angle < 67.5:
        return "前左侧"
    if 67.5 <= angle < 112.5:
        return "左侧"
    if 112.5 <= angle < 157.5:
        return "后左侧"
    if angle >= 157.5 or angle < -157.5:
        return "后方"
    if -157.5 <= angle < -112.5:
        return "后右侧"
    if -112.5 <= angle < -67.5:
        return "右侧"
    return "前右侧"


def rank_cameras_for_anchor(center):
    target_angle = angle_from_xy_deg(center[0], center[1])
    ranked = sorted(
        CAMERA_YAW_DEGREES.items(),
        key=lambda item: angular_distance_deg(target_angle, item[1]),
    )
    return [(name, angular_distance_deg(target_angle, yaw_deg)) for name, yaw_deg in ranked]


def camera_alignment_weight(center, camera_name):
    if camera_name not in CAMERA_YAW_DEGREES:
        return 1.0, None
    target_angle = angle_from_xy_deg(center[0], center[1])
    angle_diff = angular_distance_deg(target_angle, CAMERA_YAW_DEGREES[camera_name])
    weight = float(np.exp(-((angle_diff / 55.0) ** 2)))
    return weight, angle_diff


def filter_gt_objects(info):
    names = info.get("gt_names", [])
    boxes = info.get("gt_boxes", [])
    num_lidar_pts = info.get("num_lidar_pts")
    objects = []
    for idx, (name, box) in enumerate(zip(names, boxes)):
        if num_lidar_pts is not None and idx < len(num_lidar_pts) and num_lidar_pts[idx] <= 0:
            continue
        objects.append(
            {
                "label_name": str(name),
                "center": [float(v) for v in box[:3]],
                "size": [float(v) for v in box[3:6]],
                "yaw": float(box[6]),
                "direction": direction_from_xy(float(box[0]), float(box[1])),
                "source": "gt",
            }
        )
    return objects


def load_prediction_payload(pred_path):
    with Path(resolve_path(pred_path)).open("r", encoding="utf-8") as f:
        return json.load(f)


def lidar_to_camera(points_lidar, cam_info):
    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    rotation = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float32)
    translation = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float32)
    lidar2camera_r = np.linalg.inv(rotation)
    lidar2camera_t = translation @ lidar2camera_r.T
    return points_lidar @ lidar2camera_r.T - lidar2camera_t


def project_camera_points(points_camera, cam_intrinsic):
    cam_intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)
    depth = points_camera[:, 2:3]
    valid = depth[:, 0] > 1e-3
    projected = np.zeros((points_camera.shape[0], 2), dtype=np.float32)
    if valid.any():
        proj = points_camera[valid] @ cam_intrinsic.T
        projected[valid] = proj[:, :2] / proj[:, 2:3]
    return projected, valid


def box_corners_from_anchor(center, size, yaw):
    length, width, height = [float(v) for v in size[:3]]
    x_corners = np.array(
        [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2],
        dtype=np.float32,
    )
    y_corners = np.array(
        [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
        dtype=np.float32,
    )
    z_corners = np.array([0, 0, 0, 0, height, height, height, height], dtype=np.float32)
    corners = np.stack([x_corners, y_corners, z_corners], axis=1)
    cos_yaw = np.cos(float(yaw))
    sin_yaw = np.sin(float(yaw))
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center = np.asarray(center[:3], dtype=np.float32)
    return corners @ rot.T + center


def select_anchor_camera_crop(
    anchor,
    info,
    crop_dir,
    min_crop_size,
    crop_padding,
    crop_name_suffix=None,
):
    if anchor is None:
        return None

    corners_lidar = box_corners_from_anchor(anchor["center"], anchor["size"], anchor["yaw"])
    center_lidar = np.asarray(anchor["center"][:3], dtype=np.float32)[None, :]
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    ranked_cameras = rank_cameras_for_anchor(anchor["center"])
    preferred_camera_names = [
        camera_name
        for camera_name, angle_diff in ranked_cameras
        if camera_name in info.get("cams", {}) and angle_diff <= 100.0
    ]

    def find_best_candidate(camera_names):
        best_candidate = None
        for camera_name in camera_names:
            cam_info = info.get("cams", {}).get(camera_name)
            if cam_info is None:
                continue
            image_path = resolve_path(cam_info["data_path"])
            image = mmcv.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]

            center_camera = lidar_to_camera(center_lidar, cam_info)
            if center_camera[0, 2] <= 1e-3:
                continue

            corners_camera = lidar_to_camera(corners_lidar, cam_info)
            projected, valid = project_camera_points(corners_camera, cam_info["cam_intrinsic"])
            if not valid.any():
                continue
            visible = projected[valid]
            min_xy = visible.min(axis=0)
            max_xy = visible.max(axis=0)

            x1 = max(0.0, float(min_xy[0]))
            y1 = max(0.0, float(min_xy[1]))
            x2 = min(float(width), float(max_xy[0]))
            y2 = min(float(height), float(max_xy[1]))
            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = box_w * crop_padding
            pad_h = box_h * crop_padding
            side = max(min_crop_size, int(round(max(box_w + 2 * pad_w, box_h + 2 * pad_h))))

            center_proj, center_valid = project_camera_points(center_camera, cam_info["cam_intrinsic"])
            if not center_valid[0]:
                continue
            cx, cy = center_proj[0]
            crop_x1 = int(round(max(0.0, cx - side / 2)))
            crop_y1 = int(round(max(0.0, cy - side / 2)))
            crop_x2 = int(round(min(float(width), crop_x1 + side)))
            crop_y2 = int(round(min(float(height), crop_y1 + side)))
            crop_x1 = max(0, crop_x2 - side)
            crop_y1 = max(0, crop_y2 - side)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            alignment_weight, angle_diff = camera_alignment_weight(anchor["center"], camera_name)
            raw_score = float((box_w * box_h) / max(center_camera[0, 2], 1e-3))
            candidate = {
                "camera_name": camera_name,
                "image_path": image_path,
                "image": image,
                "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                "score": raw_score * alignment_weight,
                "raw_score": raw_score,
                "camera_angle_diff_deg": angle_diff,
            }
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate
        return best_candidate

    best_candidate = find_best_candidate(preferred_camera_names)
    if best_candidate is None:
        best_candidate = find_best_candidate(list(info.get("cams", {}).keys()))

    if best_candidate is None:
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = best_candidate["bbox"]
    crop = best_candidate["image"][crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    sample_token = info["token"]
    camera_name = best_candidate["camera_name"]
    suffix = "anchor" if crop_name_suffix is None else f"{crop_name_suffix}_anchor"
    crop_path = crop_dir / f"{sample_token}_{camera_name}_{suffix}.jpg"
    mmcv.imwrite(crop, str(crop_path))
    return {
        "camera_name": camera_name,
        "source_image_path": best_candidate["image_path"],
        "crop_path": str(crop_path.resolve()),
        "crop_box_xyxy": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
        "camera_angle_diff_deg": best_candidate.get("camera_angle_diff_deg"),
    }


def xy_to_grid(x, y, height, width, point_cloud_range):
    x_min, y_min, x_max, y_max = point_cloud_range
    x_norm = (x - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (y - y_min) / max(y_max - y_min, 1e-6)
    grid_x = int(round(x_norm * (width - 1)))
    grid_y = int(round(y_norm * (height - 1)))
    return max(0, min(width - 1, grid_x)), max(0, min(height - 1, grid_y))


def compute_patch_bounds(center, size, tensor_height, tensor_width, point_cloud_range, context_scale=1.0):
    x, y = center[:2]
    length = max(size[0], 1.0) * context_scale
    width_m = max(size[1], 1.0) * context_scale
    cx, cy = xy_to_grid(x, y, tensor_height, tensor_width, point_cloud_range)
    x_radius = max(1, int(round(length / (point_cloud_range[2] - point_cloud_range[0]) * tensor_width * 0.5)))
    y_radius = max(1, int(round(width_m / (point_cloud_range[3] - point_cloud_range[1]) * tensor_height * 0.5)))
    x0, x1 = max(0, cx - x_radius), min(tensor_width, cx + x_radius + 1)
    y0, y1 = max(0, cy - y_radius), min(tensor_height, cy + y_radius + 1)
    return x0, y0, x1, y1, cx, cy


def extract_feature_patch(feature_path, center, size, point_cloud_range, context_scale=1.0):
    if not feature_path:
        return None
    tensor = torch.load(resolve_path(feature_path), map_location="cpu").float()
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor in {feature_path}, got shape {tuple(tensor.shape)}")

    _, height, width = tensor.shape
    x0, y0, x1, y1, _, _ = compute_patch_bounds(
        center,
        size,
        height,
        width,
        point_cloud_range,
        context_scale=context_scale,
    )
    patch = tensor[:, y0:y1, x0:x1]
    if patch.numel() == 0:
        return None
    return patch


def resize_feature_patch(patch, output_grid):
    if patch is None:
        return None
    output_grid = tuple(int(v) for v in output_grid)
    if tuple(patch.shape[-2:]) == output_grid:
        return patch.contiguous()
    resized = F.interpolate(
        patch.unsqueeze(0),
        size=output_grid,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.contiguous()


def save_tensor_patch(path, patch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(patch.detach().cpu().half(), str(path))
    return str(path.resolve())


def build_local_patch_assets(
    record,
    anchor,
    patch_dir,
    sample_id,
    point_cloud_range,
    local_patch_scale,
    local_patch_grid,
):
    if anchor is None:
        return None
    patch_dir = Path(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    feature_sources = {
        "camera": record.get("camera_bev_path"),
        "lidar": record.get("lidar_bev_path"),
        "fused": record.get("fused_bev_path"),
        "edl_evidence": record.get("edl_evidence_path"),
    }
    tensor_paths = {}
    render_paths = {}
    patch_shapes = {}
    for name, feature_path in feature_sources.items():
        if not feature_path:
            tensor_paths[name] = None
            render_paths[name] = None
            patch_shapes[name] = None
            continue
        patch = extract_feature_patch(
            feature_path,
            anchor["center"],
            anchor["size"],
            point_cloud_range,
            context_scale=local_patch_scale,
        )
        patch = resize_feature_patch(patch, local_patch_grid)
        if patch is None:
            tensor_paths[name] = None
            render_paths[name] = None
            patch_shapes[name] = None
            continue
        tensor_path = save_tensor_patch(patch_dir / f"{sample_id}_{name}_local.pt", patch)
        render_path = save_feature_render(patch, patch_dir / f"{sample_id}_{name}_local.png")
        tensor_paths[name] = tensor_path
        render_paths[name] = render_path
        patch_shapes[name] = [int(v) for v in patch.shape]
    return {
        "bev_features": tensor_paths,
        "renders": render_paths,
        "patch_shapes": patch_shapes,
        "local_patch_scale": float(local_patch_scale),
        "local_patch_grid": [int(v) for v in local_patch_grid],
    }


def region_feature_strength(feature_path, center, size, point_cloud_range):
    patch = extract_feature_patch(feature_path, center, size, point_cloud_range)
    if patch is None:
        return 0.0
    return float(patch.abs().mean().item())


def region_feature_stats(feature_path, center, size, point_cloud_range):
    patch = extract_feature_patch(feature_path, center, size, point_cloud_range)
    if patch is None:
        return {
            "mean_abs_response": 0.0,
            "peak_abs_response": 0.0,
            "patch_shape": None,
        }
    abs_patch = patch.abs()
    return {
        "mean_abs_response": float(abs_patch.mean().item()),
        "peak_abs_response": float(abs_patch.max().item()),
        "patch_shape": [int(v) for v in patch.shape],
    }


def region_edl_stats(feature_path, center, size, point_cloud_range):
    patch = extract_feature_patch(feature_path, center, size, point_cloud_range)
    if patch is None:
        return {
            "local_evidence_mean": 0.0,
            "local_evidence_peak": 0.0,
            "local_uncertainty_mean": 1.0,
            "local_uncertainty_min": 1.0,
            "patch_shape": None,
        }

    evidence = patch.clamp_min(0.0)
    num_classes = max(int(evidence.shape[0]), 1)
    local_strength = evidence.sum(dim=0) + float(num_classes)
    local_uncertainty = float(num_classes) / local_strength.clamp_min(1e-6)
    return {
        "local_evidence_mean": float(evidence.mean().item()),
        "local_evidence_peak": float(evidence.max().item()),
        "local_uncertainty_mean": float(local_uncertainty.mean().item()),
        "local_uncertainty_min": float(local_uncertainty.min().item()),
        "patch_shape": [int(v) for v in patch.shape],
    }


def xy_distance(a, b):
    a_xy = np.asarray(a[:2], dtype=np.float32)
    b_xy = np.asarray(b[:2], dtype=np.float32)
    return float(np.linalg.norm(a_xy - b_xy))


def find_nearest_prediction(anchor, pred_objects):
    if not pred_objects:
        return None
    ranked = sorted(pred_objects, key=lambda obj: xy_distance(anchor["center"], obj["center"]))
    nearest = copy.deepcopy(ranked[0])
    nearest["distance_xy"] = xy_distance(anchor["center"], nearest["center"])
    return nearest


def summarize_anchor_local_stats(record, anchor, point_cloud_range):
    camera_stats = region_feature_stats(
        record.get("camera_bev_path"), anchor["center"], anchor["size"], point_cloud_range
    )
    lidar_stats = region_feature_stats(
        record.get("lidar_bev_path"), anchor["center"], anchor["size"], point_cloud_range
    )
    fused_stats = region_feature_stats(
        record.get("fused_bev_path"), anchor["center"], anchor["size"], point_cloud_range
    )
    edl_stats = region_edl_stats(
        record.get("edl_evidence_path"), anchor["center"], anchor["size"], point_cloud_range
    )

    response_strengths = {
        "camera": camera_stats["mean_abs_response"],
        "lidar": lidar_stats["mean_abs_response"],
        "fused": fused_stats["mean_abs_response"],
    }
    total_strength = sum(response_strengths.values())
    if total_strength > 1e-6:
        response_shares = {
            name: float(value / total_strength) for name, value in response_strengths.items()
        }
    else:
        response_shares = {name: 0.0 for name in response_strengths}

    dominant_modality = max(response_strengths, key=response_strengths.get)
    return {
        "response_strengths": response_strengths,
        "response_shares": response_shares,
        "dominant_modality": dominant_modality,
        "camera_stats": camera_stats,
        "lidar_stats": lidar_stats,
        "fused_stats": fused_stats,
        "edl_stats": edl_stats,
    }


def choose_missed_gt_anchors(gt_objects, pred_objects, match_distance_thresh):
    missed = []
    for gt_obj in gt_objects:
        gt_label = normalize_label_name(gt_obj["label_name"])
        best_distance = None
        matched = False
        for pred_obj in pred_objects:
            if normalize_label_name(pred_obj["label_name"]) != gt_label:
                continue
            dist_xy = xy_distance(gt_obj["center"], pred_obj["center"])
            dynamic_thresh = max(
                match_distance_thresh,
                0.5
                * max(
                    gt_obj["size"][0],
                    gt_obj["size"][1],
                    pred_obj["size"][0],
                    pred_obj["size"][1],
                ),
            )
            if dist_xy <= dynamic_thresh:
                matched = True
                break
            if best_distance is None or dist_xy < best_distance:
                best_distance = dist_xy
        if not matched:
            candidate = copy.deepcopy(gt_obj)
            candidate["anchor_type"] = "missed_gt"
            candidate["score"] = 0.0
            candidate["uncertainty"] = 1.0
            candidate["nearby_prediction"] = find_nearest_prediction(gt_obj, pred_objects)
            candidate["miss_distance_hint"] = best_distance
            missed.append(candidate)

    if not missed:
        return []
    missed = sorted(
        missed,
        key=lambda obj: (
            0 if obj["center"][0] >= 0 else 1,
            float(np.linalg.norm(np.asarray(obj["center"][:2], dtype=np.float32))),
        ),
    )
    return missed


def choose_low_conf_prediction(pred_objects):
    if not pred_objects:
        return None
    front_objects = [obj for obj in pred_objects if obj["center"][0] >= 0]
    candidate_pool = front_objects if front_objects else pred_objects
    anchor = copy.deepcopy(
        sorted(candidate_pool, key=lambda obj: (float(obj.get("score", 0.0)), -obj["center"][0]))[0]
    )
    anchor["anchor_type"] = "low_conf_pred"
    anchor["source"] = "prediction"
    return anchor


def choose_anchor(gt_objects, pred_objects, match_distance_thresh):
    missed_anchors = choose_missed_gt_anchors(gt_objects, pred_objects, match_distance_thresh)
    if missed_anchors:
        return missed_anchors[0]
    return choose_low_conf_prediction(pred_objects)


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


def explain_anchor(record, anchor, point_cloud_range, local_stats=None):
    if anchor is None:
        return (
            "当前帧没有可用的漏检锚点或低置信目标，说明该场景中的目标较少，"
            "也可能是当前检测结果整体较稳定。"
        )

    if local_stats is None:
        local_stats = summarize_anchor_local_stats(record, anchor, point_cloud_range)
    camera_strength = float(local_stats["response_strengths"]["camera"])
    lidar_strength = float(local_stats["response_strengths"]["lidar"])
    fused_strength = float(local_stats["response_strengths"]["fused"])
    camera_share = float(local_stats["response_shares"]["camera"])
    lidar_share = float(local_stats["response_shares"]["lidar"])
    fused_share = float(local_stats["response_shares"]["fused"])
    edl_uncertainty = float(local_stats["edl_stats"]["local_uncertainty_mean"])
    edl_evidence = float(local_stats["edl_stats"]["local_evidence_mean"])

    direction = direction_from_xy(anchor["center"][0], anchor["center"][1])
    reasons = []
    if anchor["anchor_type"] == "missed_gt":
        reasons.append(f"{direction}存在一个 GT {anchor['label_name']}，但当前检测结果没有稳定命中该目标")
        nearby_prediction = anchor.get("nearby_prediction")
        if nearby_prediction is not None:
            reasons.append(
                f"最近的预测是 {nearby_prediction['label_name']}，分数为 {float(nearby_prediction.get('score', 0.0)):.2f}"
            )
    else:
        reasons.append(
            f"{direction}的 {anchor['label_name']} 预测分数为 {float(anchor.get('score', 0.0)):.2f}"
        )
    reasons.append(
        "以目标中心附近的局部 patch 统计来看，"
        f"camera/lidar/fused 响应约为 {camera_strength:.2f}/{lidar_strength:.2f}/{fused_strength:.2f}，"
        f"对应局部响应占比约为 {camera_share:.0%}/{lidar_share:.0%}/{fused_share:.0%}"
    )
    reasons.append(
        f"该区域的 EDL 局部平均不确定性约为 {edl_uncertainty:.2f}，平均证据强度约为 {edl_evidence:.2f}"
    )
    if camera_strength < lidar_strength * 0.8:
        reasons.append("视觉 BEV 响应偏弱，更像是遮挡、逆光、裁剪或纹理不足导致的感知退化")
    elif lidar_strength < camera_strength * 0.8:
        reasons.append("LiDAR BEV 响应偏弱，更像是距离偏远、点云稀疏或几何回波不足")
    else:
        reasons.append("相机与 LiDAR 的局部响应接近，更像是类别边界模糊或定位偏移，而不是单模态完全失效")

    if fused_strength < 0.8 * max(camera_strength, lidar_strength):
        reasons.append("融合 BEV 响应低于较强单模态响应，可能说明融合阶段未充分保留该区域证据")
    elif fused_strength > 1.2 * max(camera_strength, lidar_strength):
        reasons.append("融合 BEV 响应高于单模态响应，说明跨模态信息在该区域形成了更强证据")
    else:
        reasons.append("融合 BEV 响应与较强单模态接近，说明融合特征基本保留了该区域证据")
    return "；".join(reasons) + "。"


def build_confidence_answer(record, pred_payload, pred_objects, anchor, point_cloud_range):
    scene_uncertainty = pred_payload.get("scene_uncertainty")
    if not pred_objects:
        confidence = 0.15 if scene_uncertainty is None else max(0.05, 1.0 - float(scene_uncertainty))
        return (
            f"当前场景整体感知可信度约为 {confidence:.2f}。"
            "建议自车保持低速并等待更多观测，因为检测结果几乎无法支撑稳定决策。"
        )

    top_scores = sorted((float(obj.get("score", 0.0)) for obj in pred_objects), reverse=True)[:5]
    mean_score = sum(top_scores) / max(len(top_scores), 1)
    if anchor is not None:
        camera_strength = region_feature_strength(
            record.get("camera_bev_path"), anchor["center"], anchor["size"], point_cloud_range
        )
        lidar_strength = region_feature_strength(
            record.get("lidar_bev_path"), anchor["center"], anchor["size"], point_cloud_range
        )
        agreement = 1.0 - abs(camera_strength - lidar_strength) / max(
            camera_strength + lidar_strength, 1e-6
        )
    else:
        agreement = 0.2

    if scene_uncertainty is not None:
        edl_confidence = 1.0 - float(scene_uncertainty)
        confidence = 0.45 * mean_score + 0.25 * agreement + 0.30 * edl_confidence
    else:
        confidence = 0.70 * mean_score + 0.30 * agreement
    if anchor is not None and anchor["anchor_type"] == "missed_gt":
        confidence -= 0.15
    confidence = max(0.0, min(1.0, confidence))
    if confidence >= 0.7:
        advice = "建议维持当前行驶策略，但继续监测前方动态目标。"
    elif confidence >= 0.45:
        advice = "建议保守通过，优先关注漏检锚点或侧前方的低分目标。"
    else:
        advice = "建议减速并准备让行，因为当前感知证据不足以支持激进操作。"

    if scene_uncertainty is not None:
        return (
            f"当前场景整体感知可信度约为 {confidence:.2f}，EDL 场景不确定性约为 {float(scene_uncertainty):.2f}。"
            f"{advice}"
        )
    return f"当前场景整体感知可信度约为 {confidence:.2f}。{advice}"


def build_image_list(record, anchor_crop, image_mode, task_type=None, local_patch_assets=None):
    image_paths = []
    if task_type == "scene":
        value = record.get("fused_bev_render_path")
        return [resolve_path(value)] if value else []
    if task_type in {"miss_summary", "trust", "frame"}:
        for key in ("fused_bev_render_path", "edl_render_path"):
            value = record.get(key)
            if value:
                image_paths.append(resolve_path(value))
        return image_paths
    if task_type == "attribution_object" and local_patch_assets is not None:
        if image_mode == "anchor_crop" and anchor_crop is not None:
            image_paths.append(anchor_crop["crop_path"])
        for key in ("camera", "lidar", "fused", "edl_evidence"):
            value = local_patch_assets.get("renders", {}).get(key)
            if value:
                image_paths.append(resolve_path(value))
        global_fused = record.get("fused_bev_render_path")
        if global_fused:
            image_paths.append(resolve_path(global_fused))
        return image_paths
    if image_mode == "anchor_crop" and anchor_crop is not None:
        image_paths.append(anchor_crop["crop_path"])
    for key in (
        "camera_bev_render_path",
        "lidar_bev_render_path",
        "fused_bev_render_path",
        "edl_render_path",
    ):
        value = record.get(key)
        if value:
            image_paths.append(resolve_path(value))
    return image_paths


def sanitize_anchor_object(anchor):
    if anchor is None:
        return None
    payload = {}
    for key, value in anchor.items():
        if isinstance(value, dict):
            nested = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (float, int, str, bool)) or nested_value is None:
                    nested[nested_key] = nested_value
                elif isinstance(nested_value, list):
                    nested[nested_key] = [float(v) if isinstance(v, (int, float)) else v for v in nested_value]
            payload[key] = nested
        elif isinstance(value, list):
            payload[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
        elif isinstance(value, (float, int, str, bool)) or value is None:
            payload[key] = value
    return payload


def build_scene_question():
    return "请概括当前场景的主要交通参与者，并优先说明前方区域的车辆和行人分布。"


def build_trust_question():
    return "请综合 EDL 证据和多模态 BEV 响应，评估当前系统的整体感知可信度，并给出自车的保守建议。"


def build_miss_summary_question():
    return "请总结这一帧中总共有哪几个目标发生了漏检，并简要说明它们的大致方位和类别。"


def build_attribution_question(anchor):
    if anchor is not None and anchor["anchor_type"] == "missed_gt":
        direction = direction_from_xy(anchor["center"][0], anchor["center"][1])
        return (
            f"请结合该目标的原图 crop、三路 BEV 响应和 EDL 证据，"
            f"解释为什么位于{direction}的 {anchor['label_name']} 会被漏检。"
        )
    return "请结合目标的原图 crop、三路 BEV 响应和 EDL 证据，解释为什么当前最不稳定的目标置信度偏低。"


def build_feature_dict(record):
    return {
        "camera": resolve_path(record["camera_bev_path"]) if record.get("camera_bev_path") else None,
        "lidar": resolve_path(record["lidar_bev_path"]) if record.get("lidar_bev_path") else None,
        "fused": resolve_path(record["fused_bev_path"]) if record.get("fused_bev_path") else None,
        "edl_evidence": resolve_path(record["edl_evidence_path"]) if record.get("edl_evidence_path") else None,
    }


def build_attribution_feature_dict(record, local_patch_assets):
    if local_patch_assets is None:
        return build_feature_dict(record)
    local_features = local_patch_assets.get("bev_features", {})
    fallback = build_feature_dict(record)
    return {
        "camera": local_features.get("camera") or fallback["camera"],
        "lidar": local_features.get("lidar") or fallback["lidar"],
        "fused": local_features.get("fused") or fallback["fused"],
        "edl_evidence": local_features.get("edl_evidence") or fallback["edl_evidence"],
    }


def build_miss_summary_answer(missed_gt_anchors):
    if not missed_gt_anchors:
        return "当前帧中没有发现明确的 GT 漏检目标，主要风险更可能来自低置信预测或场景本身较为空。"

    summary_bits = []
    for anchor in missed_gt_anchors:
        direction = direction_from_xy(anchor["center"][0], anchor["center"][1])
        summary_bits.append(f"{direction}的 {anchor['label_name']}")

    if len(summary_bits) == 1:
        return f"当前帧共发现 1 个明确漏检目标，为 {summary_bits[0]}。"
    return (
        f"当前帧共发现 {len(summary_bits)} 个明确漏检目标，分别为 "
        + "、".join(summary_bits)
        + "。"
    )


def build_sample(
    record,
    info,
    point_cloud_range,
    image_mode,
    crop_dir,
    local_patch_dir,
    min_crop_size,
    crop_padding,
    match_distance_thresh,
    local_patch_scale,
    local_patch_grid,
):
    gt_objects = filter_gt_objects(info)
    pred_payload = load_prediction_payload(record["pred_path"])
    pred_objects = pred_payload.get("objects", [])
    missed_gt_anchors = choose_missed_gt_anchors(
        gt_objects,
        pred_objects,
        match_distance_thresh,
    )
    fallback_anchor = choose_low_conf_prediction(pred_objects) if not missed_gt_anchors else None
    attribution_targets = (
        missed_gt_anchors
        if missed_gt_anchors
        else ([fallback_anchor] if fallback_anchor is not None else [])
    )
    primary_anchor = attribution_targets[0] if attribution_targets else None

    frame_image_paths = build_image_list(record, None, "bev_only", task_type="frame")
    scene_question = build_scene_question()
    miss_summary_question = build_miss_summary_question()
    trust_question = build_trust_question()
    scene_answer = describe_scene(gt_objects)
    miss_summary_answer = build_miss_summary_answer(missed_gt_anchors)
    trust_answer = build_confidence_answer(
        record,
        pred_payload,
        pred_objects,
        primary_anchor,
        point_cloud_range,
    )

    base_metadata = {
        "sample_token": record["sample_token"],
        "split": record["split"],
        "pred_path": record["pred_path"],
        "gt_ref": record["gt_ref"],
        "image_mode": image_mode,
        "source_image_paths": record.get("image_paths", []),
        "scene_uncertainty": pred_payload.get("scene_uncertainty"),
        "camera_bev_render_path": record.get("camera_bev_render_path"),
        "lidar_bev_render_path": record.get("lidar_bev_render_path"),
        "fused_bev_render_path": record.get("fused_bev_render_path"),
        "edl_render_path": record.get("edl_render_path"),
        "missed_gt_count": len(missed_gt_anchors),
        "missed_gt_objects": [sanitize_anchor_object(anchor) for anchor in missed_gt_anchors],
        "attribution_target_count": len(attribution_targets),
        "primary_anchor_type": primary_anchor["anchor_type"] if primary_anchor is not None else None,
        "primary_anchor_object": sanitize_anchor_object(primary_anchor),
        "primary_anchor_local_stats": (
            summarize_anchor_local_stats(record, primary_anchor, point_cloud_range)
            if primary_anchor is not None
            else None
        ),
    }

    bev_features = build_feature_dict(record)
    sharegpt_samples = []
    flat_samples = []
    object_crop_metas = []
    object_sample_ids = []

    frame_metadata = copy.deepcopy(base_metadata)
    frame_metadata.update(
        {
            "sample_kind": "frame",
            "anchor_type": frame_metadata["primary_anchor_type"],
            "anchor_object": frame_metadata["primary_anchor_object"],
            "anchor_crop": None,
        }
    )
    frame_sharegpt_sample = {
        "id": f"{record['sample_token']}::frame",
        "sample_token": record["sample_token"],
        "sample_kind": "frame",
        "images": frame_image_paths,
        "bev_features": copy.deepcopy(bev_features),
        "conversations": [
            {"from": "human", "value": scene_question},
            {"from": "gpt", "value": scene_answer},
            {"from": "human", "value": miss_summary_question},
            {"from": "gpt", "value": miss_summary_answer},
            {"from": "human", "value": trust_question},
            {"from": "gpt", "value": trust_answer},
        ],
        "metadata": frame_metadata,
    }
    sharegpt_samples.append(frame_sharegpt_sample)

    for task_type, question, answer in (
        ("scene", scene_question, scene_answer),
        ("miss_summary", miss_summary_question, miss_summary_answer),
        ("trust", trust_question, trust_answer),
    ):
        task_metadata = copy.deepcopy(frame_metadata)
        task_metadata["task_type"] = task_type
        task_images = build_image_list(record, None, "bev_only", task_type=task_type)
        flat_samples.append(
            {
                "id": f"{record['sample_token']}::{task_type}",
                "sample_token": record["sample_token"],
                "task_type": task_type,
                "question": question,
                "answer": answer,
                "images": task_images,
                "bev_features": copy.deepcopy(bev_features),
                "metadata": task_metadata,
            }
        )

    for target_idx, anchor in enumerate(attribution_targets):
        anchor_local_stats = (
            summarize_anchor_local_stats(record, anchor, point_cloud_range)
            if anchor is not None
            else None
        )
        anchor_crop = (
            select_anchor_camera_crop(
                anchor,
                info,
                crop_dir,
                min_crop_size,
                crop_padding,
                crop_name_suffix=f"attr_{target_idx}",
            )
            if anchor is not None and image_mode == "anchor_crop"
            else None
        )
        attribution_question = build_attribution_question(anchor)
        attribution_answer = explain_anchor(
            record,
            anchor,
            point_cloud_range,
            local_stats=anchor_local_stats,
        )
        attr_sample_id = f"{record['sample_token']}::attr::{target_idx}"
        local_patch_assets = build_local_patch_assets(
            record,
            anchor,
            local_patch_dir,
            attr_sample_id.replace("::", "_"),
            point_cloud_range,
            local_patch_scale,
            local_patch_grid,
        )
        attr_images = build_image_list(
            record,
            anchor_crop,
            image_mode,
            task_type="attribution_object",
            local_patch_assets=local_patch_assets,
        )
        attr_bev_features = build_attribution_feature_dict(record, local_patch_assets)
        object_sample_ids.append(attr_sample_id)
        object_crop_metas.append(anchor_crop)

        object_metadata = copy.deepcopy(base_metadata)
        object_metadata.update(
            {
                "sample_kind": "object",
                "task_type": "attribution_object",
                "target_index": target_idx,
                "anchor_type": anchor["anchor_type"] if anchor is not None else None,
                "anchor_object": sanitize_anchor_object(anchor),
                "anchor_crop": anchor_crop,
                "anchor_local_stats": anchor_local_stats,
                "local_patch_assets": local_patch_assets,
            }
        )

        sharegpt_samples.append(
            {
                "id": attr_sample_id,
                "sample_token": record["sample_token"],
                "sample_kind": "object",
                "images": attr_images,
                "bev_features": copy.deepcopy(attr_bev_features),
                "conversations": [
                    {"from": "human", "value": attribution_question},
                    {"from": "gpt", "value": attribution_answer},
                ],
                "metadata": object_metadata,
            }
        )

        flat_samples.append(
            {
                "id": attr_sample_id,
                "sample_token": record["sample_token"],
                "task_type": "attribution_object",
                "question": attribution_question,
                "answer": attribution_answer,
                "images": attr_images,
                "bev_features": copy.deepcopy(attr_bev_features),
                "metadata": object_metadata,
            }
        )

    enriched_manifest_row = copy.deepcopy(record)
    enriched_manifest_row.update(
        {
            "anchor_type": primary_anchor["anchor_type"] if primary_anchor is not None else None,
            "anchor_object": sanitize_anchor_object(primary_anchor),
            "anchor_crop_path": object_crop_metas[0]["crop_path"] if object_crop_metas and object_crop_metas[0] is not None else None,
            "anchor_crop_meta": object_crop_metas[0] if object_crop_metas else None,
            "missed_gt_count": len(missed_gt_anchors),
            "missed_gt_objects": [sanitize_anchor_object(anchor) for anchor in missed_gt_anchors],
            "attribution_target_count": len(attribution_targets),
            "attribution_targets": [sanitize_anchor_object(anchor) for anchor in attribution_targets],
            "attribution_sample_ids": object_sample_ids,
            "anchor_crops": [meta for meta in object_crop_metas if meta is not None],
        }
    )
    return sharegpt_samples, flat_samples, enriched_manifest_row


def main():
    args = parse_args()
    manifest_rows = load_jsonl(args.manifest)
    if args.max_samples is not None:
        manifest_rows = manifest_rows[: args.max_samples]

    output_path = Path(args.output).resolve()
    flat_output = (
        Path(args.flat_output).resolve()
        if args.flat_output is not None
        else output_path.with_name(output_path.stem + "_flat.jsonl")
    )
    manifest_output = (
        Path(args.manifest_output).resolve()
        if args.manifest_output is not None
        else output_path.with_name(output_path.stem + "_manifest.jsonl")
    )
    gt_lookup = build_gt_lookup(manifest_rows)
    crop_dir = args.crop_dir
    if crop_dir is None:
        crop_dir = str(output_path.parent / "anchor_crops")
    local_patch_dir = args.local_patch_dir
    if local_patch_dir is None:
        local_patch_dir = str(output_path.parent / "local_patches")

    samples = []
    flat_samples = []
    enriched_manifest = []
    for record in manifest_rows:
        info = gt_lookup.get(record["sample_token"])
        if info is None:
            continue
        sample_sharegpt_rows, sample_flat_rows, manifest_row = build_sample(
            record,
            info,
            args.point_cloud_range,
            args.image_mode,
            crop_dir,
            local_patch_dir,
            args.min_crop_size,
            args.crop_padding,
            args.match_distance_thresh,
            args.local_patch_scale,
            args.local_patch_grid,
        )
        samples.extend(sample_sharegpt_rows)
        flat_samples.extend(sample_flat_rows)
        enriched_manifest.append(manifest_row)

    save_json(output_path, samples)
    save_jsonl(flat_output, flat_samples)
    save_jsonl(manifest_output, enriched_manifest)
    print(
        f"Saved {len(samples)} ShareGPT samples, "
        f"{len(flat_samples)} flat QA rows, "
        f"and {len(enriched_manifest)} enriched manifest rows."
    )


if __name__ == "__main__":
    main()
