# src/gfm4mpm/infer/infer_maps.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import rasterio

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _mc_predict_single_region(
    encoder,
    mlp,
    stack,
    window_size: int,
    stride: int,
    passes: int,
    device: str,
    show_progress: bool,
    region_name: str = "",
    save_prediction: bool = False,
    save_path: str = None,
):
    encoder.eval().to(device)
    mlp.eval().to(device)
    for module in mlp.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    H, W = stack.height, stack.width
    mean_map = np.zeros((H, W), dtype=np.float32)
    var_map  = np.zeros((H, W), dtype=np.float32)
    counts   = np.zeros((H, W), dtype=np.int32)
    centers = stack.grid_centers(stride)
    if save_prediction and not hasattr(centers, "__len__"):
        centers = list(centers)
    iterator = centers
    prediction_buffer = None
    next_row = 0
    if save_prediction:
        prediction_buffer = np.empty((len(centers), 4), dtype=np.float32)

    if show_progress and tqdm is not None:
        desc = "MC inference"
        if region_name:
            desc = f"{desc} [{region_name}]"
        iterator = tqdm(centers, desc=desc)
    elif show_progress and tqdm is None:
        print("[warn] tqdm not installed; progress bar disabled.")
    for r, c in iterator:
        x = stack.read_patch(r, c, window_size)
        x = torch.from_numpy(x[None]).to(device)
        zs = encoder.encode(x)
        preds = []
        for _ in range(passes):
            p = mlp(zs)
            preds.append(p.item())

        mu = float(np.mean(preds)); sig2 = float(np.var(preds))

        # Update entire patch
        # r0, c0 = r - window_size//2, c - window_size//2
        # r1, c1 = r0 + window_size, c0 + window_size
        # r0, c0 = max(r0,0), max(c0,0)
        # r1, c1 = min(r1,H), min(c1,W)
        # mean_map[r0:r1, c0:c1] += mu
        # var_map[r0:r1, c0:c1]  += sig2
        # counts[r0:r1, c0:c1]   += 1

        # Update only the center pixel
        mean_map[r, c] += mu
        var_map[r, c]  += sig2
        counts[r, c]   += 1
        if save_prediction:
            prediction_buffer[next_row] = (r, c, mu, sig2)
            next_row += 1

    if save_prediction:
        predictions = prediction_buffer[:next_row]
        target_path = save_path
        if target_path is None:
            base_name = region_name if region_name else "prediction"
            target_path = f"{base_name}_predictions.npy"
        elif os.path.isdir(target_path):
            base_name = region_name if region_name else "prediction"
            target_path = os.path.join(target_path, f"{base_name}_predictions.npy")
        else:
            root, ext = os.path.splitext(target_path)
            if not ext:
                os.makedirs(target_path, exist_ok=True)
                base_name = region_name if region_name else "prediction"
                target_path = os.path.join(target_path, f"{base_name}_predictions.npy")
            else:
                parent_dir = os.path.dirname(target_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        np.save(target_path, predictions)

    mean_map /= np.maximum(counts, 1)
    var_map  /= np.maximum(counts, 1)
    return mean_map, np.sqrt(var_map)


@torch.no_grad()
def mc_predict_map(encoder, mlp, stack, window_size=32, stride=16, passes=30, device=None, show_progress=False, save_prediction=False, save_path=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hasattr(stack, "resolve_region_stack") and hasattr(stack, "iter_region_stacks"):
        region_results = {}
        for region_name, region_stack in stack.iter_region_stacks():
            region_mean, region_std = _mc_predict_single_region(
                encoder,
                mlp,
                region_stack,
                window_size=window_size,
                stride=stride,
                passes=passes,
                device=device,
                show_progress=show_progress,
                region_name=str(region_name),
                save_prediction=save_prediction,
                save_path=save_path
            )
            region_results[str(region_name)] = {
                "mean": region_mean,
                "std": region_std,
                "stack": region_stack,
            }
        return region_results

    mean_map, std_map = _mc_predict_single_region(
        encoder,
        mlp,
        stack,
        window_size=window_size,
        stride=stride,
        passes=passes,
        device=device,
        show_progress=show_progress,
        save_prediction=save_prediction,
        save_path=save_path
    )
    return mean_map, std_map


def _compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    interior = mask.copy()
    interior[:-1, :] &= mask[1:, :]
    interior[1:, :] &= mask[:-1, :]
    interior[:, :-1] &= mask[:, 1:]
    interior[:, 1:] &= mask[:, :-1]
    boundary &= ~interior
    return boundary


def _extract_valid_boundary_masks(stack: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not getattr(stack, "srcs", None):
        return None, None
    try:
        mask = stack.srcs[0].dataset_mask()
    except Exception:
        mask = None
    if mask is None:
        return None, None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[0]
    valid_mask = mask_arr != 0
    if not valid_mask.any():
        return None, None
    boundary_mask = _compute_boundary_mask(valid_mask)
    return valid_mask, boundary_mask


def _normalize_prediction_result(result: Any, default_stack: Any) -> Dict[str, Tuple[np.ndarray, np.ndarray, Any]]:
    normalized: Dict[str, Tuple[np.ndarray, np.ndarray, Any]] = {}

    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, dict):
                mean_map = value.get("mean")
                std_map = value.get("std")
                region_stack = value.get("stack", default_stack)
            else:
                mean_map, std_map = value
                region_stack = default_stack
            if mean_map is None or std_map is None:
                continue
            normalized[str(key)] = (mean_map, std_map, region_stack)
    else:
        mean_map, std_map = result
        normalized["GLOBAL"] = (mean_map, std_map, default_stack)

    return normalized


def group_positive_coords(
    coords: Sequence[Tuple[str, int, int]],
    stack: Any,
) -> Dict[str, List[Tuple[int, int]]]:
    grouped: Dict[str, List[Tuple[int, int]]] = {}
    has_regions = hasattr(stack, "resolve_region_stack")
    base_height = getattr(stack, "height", None)
    base_width = getattr(stack, "width", None)

    for region, row, col in coords:
        region_key = str(region)
        region_stack = None
        if has_regions:
            try:
                region_stack = stack.resolve_region_stack(region_key)
            except Exception:
                region_stack = None
        if region_stack is None:
            region_stack = stack
            region_key = "GLOBAL"
        h, w = region_stack.height, region_stack.width
        r, c = int(row), int(col)
        if 0 <= r < h and 0 <= c < w:
            grouped.setdefault(region_key, []).append((r, c))

    if "GLOBAL" not in grouped and base_height is not None and base_width is not None:
        global_coords: List[Tuple[int, int]] = []
        for _, row, col in coords:
            r, c = int(row), int(col)
            if 0 <= r < base_height and 0 <= c < base_width:
                global_coords.append((r, c))
        if global_coords:
            grouped["GLOBAL"] = global_coords

    return grouped


def write_prediction_outputs(
    result: Any,
    default_stack: Any,
    out_dir: Path,
    pos_coords_by_region: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> None:
    normalized = _normalize_prediction_result(result, default_stack)
    if not normalized:
        raise RuntimeError("No prediction data available to export.")

    out_dir.mkdir(parents=True, exist_ok=True)
    multi_region = len(normalized) > 1 or next(iter(normalized.keys())) != "GLOBAL"

    for region_name, (mean_map, std_map, region_stack) in normalized.items():
        if mean_map is None or std_map is None:
            continue

        ref_stack = region_stack if region_stack is not None else default_stack
        ref_src = ref_stack.srcs[0] if getattr(ref_stack, "srcs", None) else default_stack.srcs[0]
        valid_mask, boundary_mask = _extract_valid_boundary_masks(ref_stack)
        masked_mean = mean_map
        masked_std = std_map
        if valid_mask is not None and valid_mask.shape == mean_map.shape:
            masked_mean = np.where(valid_mask, mean_map, np.nan).astype(np.float32, copy=False)
            masked_std = np.where(valid_mask, std_map, np.nan).astype(np.float32, copy=False)

        region_positions: Optional[List[Tuple[int, int]]] = None
        if pos_coords_by_region:
            region_positions = pos_coords_by_region.get(region_name) or pos_coords_by_region.get("GLOBAL")
            if region_positions:
                filtered_positions: List[Tuple[int, int]] = []
                h, w = masked_mean.shape
                for r, c in region_positions:
                    if 0 <= r < h and 0 <= c < w:
                        if valid_mask is None or valid_mask[r, c]:
                            filtered_positions.append((r, c))
                region_positions = filtered_positions or None

        if multi_region or region_name != "GLOBAL":
            resolved_name = region_name if region_name else "GLOBAL"
            mean_path = out_dir / f"Pospectivity_region_{resolved_name}_mean.tif"
            std_path = out_dir / f"Pospectivity_region_{resolved_name}_std.tif"
        else:
            mean_path = out_dir / 'Pospectivity_mean.tif'
            std_path = out_dir / 'Pospectivity_std.tif'

        save_geotiff(str(mean_path), masked_mean, ref_src, valid_mask=valid_mask)
        save_geotiff(str(std_path), masked_std, ref_src, valid_mask=valid_mask)
        summary_png = mean_path.with_suffix('.png')
        save_png(
            summary_png,
            masked_mean,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            std_array=masked_std,
            pos_coords=region_positions,
        )
        std_png = std_path.with_suffix('.png')
        save_png(
            std_png,
            masked_std,
            cmap="gray",
            vmin=0.0,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            pos_coords=region_positions,
        )
        print(f"Wrote {mean_path}, {std_path}, summary PNG {summary_png}, and std PNG {std_png}")


def save_geotiff(path, array, ref_src, valid_mask: np.ndarray = None, nodata: float = np.nan):
    profile = ref_src.profile.copy()
    profile.update(count=1, dtype='float32')
    if nodata is not None:
        profile['nodata'] = nodata
    arr = np.asarray(array, dtype=np.float32)
    if valid_mask is not None and valid_mask.shape == arr.shape:
        arr = np.where(valid_mask, arr, nodata if np.isfinite(nodata) else np.float32(np.nan)).astype(np.float32, copy=False)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)

def save_png(
    path,
    array,
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    valid_mask: np.ndarray = None,
    boundary_mask: np.ndarray = None,
    std_array: np.ndarray = None,
    pos_coords: Sequence[Tuple[int, int]] = None,
):
    data = np.asarray(array, dtype=np.float32)
    display = data.copy()
    if valid_mask is not None and valid_mask.shape == data.shape:
        display = np.where(valid_mask, display, np.nan)
    finite_mask = np.isfinite(display)
    if not finite_mask.any():
        display = np.zeros_like(display, dtype=np.float32)
        finite_mask = np.ones_like(display, dtype=bool)
    elif not finite_mask.all():
        fill_value = float(np.nanmean(display[finite_mask])) if finite_mask.any() else 0.0
        display = np.where(finite_mask, display, fill_value).astype(np.float32, copy=False)

    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    height, width = display.shape
    dpi = 100
    fig_w = max(width / dpi, 1e-2)
    fig_h = max(height / dpi, 1e-2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    if std_array is not None:
        std_data = np.asarray(std_array, dtype=np.float32)
        std_display = std_data.copy()
        if valid_mask is not None and valid_mask.shape == std_display.shape:
            std_display = np.where(valid_mask, std_display, np.nan)
        std_finite = np.isfinite(std_display)
        if not std_finite.any():
            std_display = np.zeros_like(std_display, dtype=np.float32)
            std_finite = np.ones_like(std_display, dtype=bool)
        elif not std_finite.all():
            std_fill = float(np.nanmean(std_display[std_finite])) if std_finite.any() else 0.0
            std_display = np.where(std_finite, std_display, std_fill).astype(np.float32, copy=False)

        positive_vals = display[finite_mask & (display > 0)]
        base_vals = display[finite_mask]
        if positive_vals.size:
            hv_vmin = float(np.nanpercentile(positive_vals, 25))
        elif base_vals.size:
            hv_vmin = float(np.nanpercentile(base_vals, 25))
        else:
            hv_vmin = 0.0
        hv_vmax_data = float(np.nanmax(base_vals)) if base_vals.size else hv_vmin + 1e-3
        hv_vmax = 1.0 if hv_vmax_data >= 1.0 else hv_vmax_data
        if not np.isfinite(hv_vmax):
            hv_vmax = 1.0
        if hv_vmax <= hv_vmin:
            hv_vmin = hv_vmax - 1e-3
            if hv_vmin < 0.0:
                hv_vmin = 0.0
        base_image = np.clip(display, hv_vmin, hv_vmax)
        ax.imshow(base_image, cmap="inferno", vmin=hv_vmin, vmax=hv_vmax)

        std_vals = std_display[std_finite]
        if std_vals.size:
            std_q1 = float(np.nanpercentile(std_vals, 25))
        else:
            std_q1 = 0.0
        std_vmax = std_q1 if std_q1 > 0 else (float(np.nanmax(std_vals)) if std_vals.size else 1.0)
        if not np.isfinite(std_vmax) or std_vmax <= 0:
            std_vmax = 1.0
        std_clipped = np.clip(std_display, 0.0, std_vmax)
        ax.imshow(std_clipped, cmap="gray", vmin=0.0, vmax=std_vmax, alpha=0.35)
    else:
        ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)

    if pos_coords:
        coords_arr = np.asarray(pos_coords, dtype=np.float32)
        if coords_arr.ndim == 2 and coords_arr.size:
            ys = np.clip(coords_arr[:, 0] + 0.5, 0.0, height - 0.5)
            xs = np.clip(coords_arr[:, 1] + 0.5, 0.0, width - 0.5)
            ax.scatter(
                xs,
                ys,
                s=18,
                c="cyan",
                edgecolors="black",
                linewidths=0.4,
                marker="o",
                zorder=5,
            )

    if boundary_mask is not None and boundary_mask.shape == display.shape and boundary_mask.any():
        ax.contour(boundary_mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.8)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
