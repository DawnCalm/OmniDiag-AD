from pathlib import Path

import mmcv
import numpy as np
import torch


def _to_numpy_map(feature):
    if isinstance(feature, (str, Path)):
        feature = torch.load(str(feature), map_location="cpu")
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().float().numpy()
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 3:
        feature = np.abs(feature).mean(axis=0)
    elif feature.ndim != 2:
        raise ValueError(f"Expected a 2D map or CHW tensor, got shape {feature.shape}")
    return feature


def _normalize_map(feature_map, lower_percentile=1.0, upper_percentile=99.0):
    finite_mask = np.isfinite(feature_map)
    if not finite_mask.any():
        return np.zeros_like(feature_map, dtype=np.float32)
    valid = feature_map[finite_mask]
    lo = np.percentile(valid, lower_percentile)
    hi = np.percentile(valid, upper_percentile)
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((feature_map - lo) / (hi - lo), 0.0, 1.0)
    normalized[~finite_mask] = 0.0
    return normalized


def _simple_colormap(normalized_map):
    x = normalized_map.astype(np.float32)
    red = np.clip(1.5 * x - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * x, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def render_feature_tensor(feature, lower_percentile=1.0, upper_percentile=99.0):
    feature_map = _to_numpy_map(feature)
    normalized = _normalize_map(
        feature_map,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    rgb = (_simple_colormap(normalized) * 255.0).round().astype(np.uint8)
    return rgb


def save_feature_render(feature, path, lower_percentile=1.0, upper_percentile=99.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = render_feature_tensor(
        feature,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    mmcv.imwrite(rgb[..., ::-1], str(path))
    return str(path.resolve())
