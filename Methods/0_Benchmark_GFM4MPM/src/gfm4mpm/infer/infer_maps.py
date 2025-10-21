# src/gfm4mpm/infer/infer_maps.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

import json
import numpy as np
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
import rasterio
from rasterio.transform import array_bounds, xy
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling

try:
    from shapely.geometry import MultiPolygon, Polygon, mapping
    from shapely.ops import transform as shapely_transform, unary_union
except Exception:  # pragma: no cover - optional dependency
    MultiPolygon = Polygon = mapping = shapely_transform = unary_union = None  # type: ignore[assignment]

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Transformer = None  # type: ignore[assignment]


def _require_torch():
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for this operation; install torch to enable inference utilities.")
    return torch


def _regular_grid_idw(sampled: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear IDW interpolation across a regular grid."""
    height, width = shape
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)

    # Row indices
    row_idx_low = np.searchsorted(row_coords, rows, side="right") - 1
    row_idx_low = np.clip(row_idx_low, 0, row_coords.size - 1)
    row_idx_high = np.clip(row_idx_low + 1, 0, row_coords.size - 1)
    row_low = row_coords[row_idx_low].astype(np.float32)
    row_high = row_coords[row_idx_high].astype(np.float32)
    row_denom = row_high - row_low
    row_denom[row_denom == 0] = 1.0
    row_alpha = (rows - row_low) / row_denom
    row_alpha[row_idx_high == row_idx_low] = 0.0

    # Column indices
    col_idx_low = np.searchsorted(col_coords, cols, side="right") - 1
    col_idx_low = np.clip(col_idx_low, 0, col_coords.size - 1)
    col_idx_high = np.clip(col_idx_low + 1, 0, col_coords.size - 1)
    col_low = col_coords[col_idx_low].astype(np.float32)
    col_high = col_coords[col_idx_high].astype(np.float32)
    col_denom = col_high - col_low
    col_denom[col_denom == 0] = 1.0
    col_alpha = (cols - col_low) / col_denom
    col_alpha[col_idx_high == col_idx_low] = 0.0

    row_alpha = row_alpha[:, None]
    col_alpha = col_alpha[None, :]

    v00 = sampled[row_idx_low[:, None], col_idx_low[None, :]]
    v01 = sampled[row_idx_low[:, None], col_idx_high[None, :]]
    v10 = sampled[row_idx_high[:, None], col_idx_low[None, :]]
    v11 = sampled[row_idx_high[:, None], col_idx_high[None, :]]

    w00 = (1.0 - row_alpha) * (1.0 - col_alpha)
    w01 = (1.0 - row_alpha) * col_alpha
    w10 = row_alpha * (1.0 - col_alpha)
    w11 = row_alpha * col_alpha

    interpolated = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    return interpolated.astype(np.float32, copy=False)


def _interpolate_prediction_grid(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Upsample sparse prediction grid using bilinear IDW interpolation."""
    valid_mask = counts > 0
    if valid_mask.all():
        return values
    valid_indices = np.argwhere(valid_mask)
    if valid_indices.size == 0:
        return values
    row_coords = np.unique(valid_indices[:, 0])
    col_coords = np.unique(valid_indices[:, 1])
    if row_coords.size == 0 or col_coords.size == 0:
        return values
    sampled = values[np.ix_(row_coords, col_coords)]
    # Prefer smooth spline interpolation if SciPy is available
    try:
        from scipy.interpolate import RectBivariateSpline  # type: ignore

        kx = min(3, max(1, row_coords.size - 1))
        ky = min(3, max(1, col_coords.size - 1))
        spline = RectBivariateSpline(row_coords, col_coords, sampled, kx=kx, ky=ky)
        grid_rows = np.arange(values.shape[0], dtype=np.float64)
        grid_cols = np.arange(values.shape[1], dtype=np.float64)
        filled = spline(grid_rows, grid_cols)
        return filled.astype(np.float32, copy=False)
    except Exception:
        pass
    filled = _regular_grid_idw(sampled, row_coords, col_coords, values.shape)
    return filled


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
    torch_mod = _require_torch()
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
        x = torch_mod.from_numpy(x[None]).to(device)
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

    with np.errstate(invalid="ignore"):
        mean_map = np.divide(mean_map, counts, where=counts > 0)
        var_map = np.divide(var_map, counts, where=counts > 0)
    mean_map[counts == 0] = np.nan
    var_map[counts == 0] = np.nan
    if stride > 1:
        mean_map = _interpolate_prediction_grid(mean_map, counts)
        var_map = _interpolate_prediction_grid(var_map, counts)
    return mean_map, np.sqrt(var_map)


def mc_predict_map(encoder, mlp, stack, window_size=32, stride=16, passes=30, device=None, show_progress=False, save_prediction=False, save_path=None):
    torch_mod = _require_torch()
    if device is None:
        device = 'cuda' if torch_mod.cuda.is_available() else 'cpu'

    with torch_mod.no_grad():
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
    combined: Optional[np.ndarray] = None

    boundary_mask = getattr(stack, "boundary_mask", None)
    if boundary_mask is not None:
        boundary_arr = np.asarray(boundary_mask).astype(bool, copy=False)
        if boundary_arr.any():
            combined = boundary_arr

    if getattr(stack, "srcs", None):
        try:
            dataset_mask = stack.srcs[0].dataset_mask()
        except Exception:
            dataset_mask = None
        if dataset_mask is not None:
            mask_arr = np.asarray(dataset_mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            dataset_bool = mask_arr != 0
            if dataset_bool.any():
                combined = dataset_bool if combined is None else (combined & dataset_bool)

    if combined is None or not combined.any():
        return None, None

    combined = combined.astype(bool, copy=False)
    boundary = _compute_boundary_mask(combined)
    return combined, boundary


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
        # if valid_mask is not None and valid_mask.shape == mean_map.shape:
        #     masked_mean = np.where(valid_mask, mean_map, np.nan).astype(np.float32, copy=False)
        #     masked_std = np.where(valid_mask, std_map, np.nan).astype(np.float32, copy=False)

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
            mean_std_path = out_dir / f"Pospectivity_region_{resolved_name}_mean_std.tif"
        else:
            mean_path = out_dir / 'Pospectivity_mean.tif'
            std_path = out_dir / 'Pospectivity_std.tif'
            mean_std_path = out_dir /'Pospectivity_mean_std.tif'

        save_geotiff(str(mean_path), masked_mean, ref_src, valid_mask=valid_mask)
        save_geotiff(str(std_path), masked_std, ref_src, valid_mask=valid_mask)
        
        from matplotlib.colors import LinearSegmentedColormap
        # Blue → yellow gradient for cmap1
        blue_yellow = LinearSegmentedColormap.from_list(
            "blue_yellow",
            ["#08306B", "#4EB3D3", "#FFFF00"],
        )

        # White → black gradient for cmap2
        white_black = LinearSegmentedColormap.from_list(
            "white_black",
            ["#FFFFFF", "#000000"],
        )

        summary_png = mean_path.with_suffix('.png')
        save_png(
            summary_png,
            masked_mean,
            cmap=blue_yellow,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            std_array=masked_std,
            pos_coords=region_positions,
        )
        std_png = std_path.with_suffix('.png')
        save_png(
            std_png,
            masked_std,
            cmap=white_black,
            vmin=0.0,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            pos_coords=region_positions,
        )
        mean_std_png = mean_std_path.with_suffix('.png')
        save_png2(
            mean_std_png, 
            masked_mean,
            masked_std,
            cmap1=blue_yellow,
            cmap2=white_black,
            criteria=0.25,
            valid_mask=valid_mask,
            boundary_mask=boundary_mask,
            pos_coords=region_positions,
        )
        print(f"Wrote {mean_path}, {std_path}, {mean_std_path}, mean PNG {summary_png}, std PNG {std_png}, and mean-std PNG {mean_std_png}.")


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
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.colors import Colormap
    import numpy as np

    data = np.asarray(array, dtype=np.float32)
    display = data.copy()

    # Apply valid mask (set invalid pixels to NaN)
    if valid_mask is not None and valid_mask.shape == data.shape:
        display = np.where(valid_mask, display, np.nan)

    # Replace fully NaN map with zeros
    if not np.isfinite(display).any():
        display = np.zeros_like(display)

    height, width = display.shape
    dpi = 100
    fig_w = max(width / dpi, 1e-2)
    fig_h = max(height / dpi, 1e-2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")


    # Resolve colormap properly
    if isinstance(cmap, str):
        cmap_obj: Colormap = plt.get_cmap(cmap).copy()
    else:
        cmap_obj: Colormap = cmap.copy()

    # Make invalid (NaN) pixels white instead of black
    cmap_obj.set_bad(color="white")

    im = ax.imshow(
        np.ma.masked_invalid(display),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        origin="upper",
    )

    # Overlay std array (if provided)
    if std_array is not None:
        std_data = np.asarray(std_array, dtype=np.float32)
        if valid_mask is not None and valid_mask.shape == std_data.shape:
            std_data = np.where(valid_mask, std_data, np.nan)
        std_vals = std_data[np.isfinite(std_data)]
        if std_vals.size:
            std_vmax = np.nanpercentile(std_vals, 95)
            std_clipped = np.clip(std_data, 0, std_vmax)
            ax.imshow(
                std_clipped,
                cmap="gray",
                vmin=0,
                vmax=std_vmax,
                alpha=0.25,
                interpolation="bilinear",  # 🔹 smooth overlay
            )

    # Cyan deposit markers
        if pos_coords:
            coords_arr = np.asarray(pos_coords, np.float32)
            if coords_arr.ndim == 2 and coords_arr.size:
                ys = np.clip(coords_arr[:, 0] + 0.5, 0, height - 0.5)
                xs = np.clip(coords_arr[:, 1] + 0.5, 0, width - 0.5)
                ax.scatter(
                    xs, ys, s=18, c="cyan",
                    edgecolors="black", linewidths=0.4,
                    marker="o", zorder=5
                )

    # Boundary outline
    if boundary_mask is not None and boundary_mask.shape == display.shape and boundary_mask.any():
        ax.contour(
            boundary_mask.astype(np.uint8),
            levels=[0.5],
            colors="black",
            linewidths=2.0,
        )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")


    # Optional: add colorbar for quick checks
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

def _resolve_cmap(name: Optional[str], fallback: str):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if name is None:
        return plt.get_cmap(fallback)

    try:
        return plt.get_cmap(name)
    except ValueError:
        alias: Dict[str, LinearSegmentedColormap] = {
            "blue_to_yellow": LinearSegmentedColormap.from_list(
                "blue_to_yellow",
                ["#08306B", "#2879B9", "#4EB3D3", "#FFFFD9"],
            ),
            "blue-yellow": LinearSegmentedColormap.from_list(
                "blue-yellow",
                ["#08306B", "#2879B9", "#4EB3D3", "#FFFFD9"],
            ),
        }
        cmap = alias.get(name.lower())
        if cmap is not None:
            return cmap
        try:
            return plt.get_cmap(fallback)
        except ValueError:
            return plt.get_cmap("viridis")

def save_png2(
    mean_std_path: Path,
    mean_array: np.ndarray,
    std_array: np.ndarray,
    cmap1: str = "plasma",
    cmap2: str = "gray",
    criteria: float = 0.25,
    valid_mask: np.ndarray = None,
    boundary_mask: np.ndarray = None,
    pos_coords: Sequence[Tuple[int, int]] = None,
):
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    mean = np.asarray(mean_array, np.float32)
    std = np.asarray(std_array, np.float32)

    # Mask invalid regions
    if valid_mask is not None and valid_mask.shape == mean.shape:
        mean = np.where(valid_mask, mean, np.nan)
        std = np.where(valid_mask, std, np.nan)

    if isinstance(cmap1, str):
        cmap_mean = plt.get_cmap(cmap1)
    else:
        cmap_mean = cmap1
    if isinstance(cmap2, str):
        cmap_std = plt.get_cmap(cmap2)
    else:
        cmap_std = cmap2

    combined_rgb = np.ones((*mean.shape, 3), dtype=np.float32)

    finite_mean = np.isfinite(mean)
    finite_std = np.isfinite(std)
    mean_mask = finite_mean & (mean > criteria)
    std_mask = finite_std & ~mean_mask

    norm_mean = None
    if mean_mask.any():
        vmin_mean = float(np.nanmin(mean[mean_mask]))
        vmax_mean = float(np.nanmax(mean[mean_mask]))
        if not np.isfinite(vmin_mean):
            vmin_mean = 0.0
        if not np.isfinite(vmax_mean) or vmax_mean == vmin_mean:
            vmax_mean = vmin_mean + 1.0
        norm_mean = mcolors.Normalize(vmin=vmin_mean, vmax=vmax_mean, clip=True)
        mean_rgba = cmap_mean(norm_mean(mean))
        combined_rgb[mean_mask] = mean_rgba[..., :3][mean_mask]

    norm_std = None
    if std_mask.any():
        vmin_std = float(np.nanmin(std[std_mask]))
        vmax_std = float(np.nanmax(std[std_mask]))
        if not np.isfinite(vmin_std):
            vmin_std = 0.0
        if not np.isfinite(vmax_std) or vmax_std == vmin_std:
            vmax_std = vmin_std + 1.0
        norm_std = mcolors.Normalize(vmin=vmin_std, vmax=vmax_std, clip=True)
        std_rgba = cmap_std(norm_std(std))
        combined_rgb[std_mask] = std_rgba[..., :3][std_mask]

    h, w, _ = combined_rgb.shape
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(combined_rgb, origin="upper", interpolation="none")
    ax.axis("off")

    # Boundary contour
    if boundary_mask is not None and boundary_mask.shape == mean.shape and boundary_mask.any():
        ax.contour(
            boundary_mask.astype(np.uint8),
            levels=[0.5],
            colors="black",
            linewidths=2.0,
        )

    # Positive positions
        if pos_coords:
            coords = np.asarray(pos_coords)
            try:
                xs, ys = xy(
                    transform,
                    coords[:, 0],
                    coords[:, 1],
                    offset="center",
                )
                ax.scatter(xs, ys, s=18, c="cyan", edgecolors="black", linewidths=0.4, zorder=5)
            except Exception:
                ys = np.clip(coords[:, 0] + 0.5, 0, h - 0.5)
                xs = np.clip(coords[:, 1] + 0.5, 0, w - 0.5)
                ax.scatter(xs, ys, s=18, c="cyan", edgecolors="black", linewidths=0.4, zorder=5)

    # --- Add colorbars ---
    # Likelihood bar
    cax1 = inset_axes(ax, width="2%", height="35%", loc="upper right", borderpad=0.6)
    if norm_mean is not None:
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm_mean, cmap=cmap_mean),
            cax=cax1,
            orientation="vertical",
            fraction=0.05,
        )
        cb1.set_label("Likelihood", fontsize=8)
        cb1.ax.tick_params(labelsize=7)
    else:
        cax1.set_visible(False)

    # Uncertainty bar (below)
    cax2 = inset_axes(ax, width="2%", height="35%", loc="lower right", borderpad=0.6)
    if norm_std is not None:
        cb2 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm_std, cmap=cmap_std),
            cax=cax2,
            orientation="vertical",
            fraction=0.05,
        )
        cb2.set_label("Uncertainty", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
    else:
        cax2.set_visible(False)

    fig.savefig(mean_std_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _load_boundary_mask(
    summary_boundaries: Dict[str, List[Dict]],
    region_name: str,
    dataset_root: Path,
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    region_upper = region_name.upper()
    for records in summary_boundaries.values():
        for record in records:
            if not isinstance(record, dict):
                continue
            record_region = str(record.get("region") or "GLOBAL").upper()
            if record_region != region_upper:
                continue
            path_value = record.get("path_resolved") or record.get("path")
            if not path_value:
                continue
            boundary_path = Path(path_value)
            if not boundary_path.is_absolute():
                boundary_path = (dataset_root / boundary_path).resolve()
            if not boundary_path.exists():
                continue
            try:
                with rasterio.open(boundary_path) as boundary_ds:
                    mask_data = boundary_ds.read(1, masked=True)
                    if hasattr(mask_data, "filled"):
                        mask_array = mask_data.filled(0).astype(np.uint8, copy=False)
                    else:
                        mask_array = np.asarray(mask_data, dtype=np.uint8)
                    boundary_mask = mask_array != 0
                    if boundary_mask.shape == target_shape:
                        return boundary_mask
            except Exception:
                continue
    return None


def generate_bridge_visualizations(
    bridge_mappings: Sequence[Dict[str, List[str]]],
    dataset_lookup: Dict[str, Any],
    output_dir: Path,
    label_geojson_map: Optional[Dict[str, Dict[str, List[str]]]] = None,
    overlap_geoms: Optional[Dict[str, Any]] = None,
    dataset_crs_map: Optional[Dict[str, Any]] = None,
    target_crs: Optional[Any] = None,
    overall_overlap_geom: Optional[Any] = None,
    overlap_mask_path: Optional[Path] = None,
) -> None:
    """Materialise TIFF/PNG previews for bridge features and labels (with positive masks)."""
    if not bridge_mappings:
        return
    try:
        import matplotlib.pyplot as plt
        import shutil
    except Exception as exc:
        print(f"[warn] Skipping bridge visualisations (dependencies missing): {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_mask_data = None
    overlap_mask_transform = None
    overlap_mask_crs = None
    overlap_mask_bounds = None
    if overlap_mask_path:
        try:
            with rasterio.open(overlap_mask_path) as mask_ds:
                overlap_mask_data = mask_ds.read(1).astype(np.uint8)
                overlap_mask_transform = mask_ds.transform
                overlap_mask_crs = mask_ds.crs
                overlap_mask_bounds = array_bounds(
                    overlap_mask_data.shape[0],
                    overlap_mask_data.shape[1],
                    overlap_mask_transform,
                )
        except Exception as exc:
            print(f"[warn] Failed to load overlap mask {overlap_mask_path}: {exc}")
            overlap_mask_data = None
            overlap_mask_transform = None
            overlap_mask_crs = None
            overlap_mask_bounds = None

    def _project_mask_to_grid(shape: Tuple[int, int], transform, crs) -> Optional[np.ndarray]:
        if (
            overlap_mask_data is None
            or overlap_mask_transform is None
            or overlap_mask_crs is None
            or transform is None
            or crs is None
        ):
            return None
        target = np.zeros(shape, dtype=np.uint8)
        try:
            reproject(
                overlap_mask_data,
                target,
                src_transform=overlap_mask_transform,
                src_crs=overlap_mask_crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
            return target > 0
        except Exception:
            return None

    def _project_to_overlap_grid(array: np.ndarray, transform, crs, bucket: str) -> Optional[np.ndarray]:
        if (
            overlap_mask_data is None
            or overlap_mask_transform is None
            or overlap_mask_crs is None
            or transform is None
            or crs is None
        ):
            return None
        dest = np.full(overlap_mask_data.shape, np.nan, dtype=np.float32)
        resampling_method = Resampling.nearest if bucket == "labels" else Resampling.bilinear
        try:
            reproject(
                array,
                dest,
                src_transform=transform,
                src_crs=crs,
                dst_transform=overlap_mask_transform,
                dst_crs=overlap_mask_crs,
                resampling=resampling_method,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            mask_bool = overlap_mask_data > 0
            dest = np.where(mask_bool, dest, np.nan)
            return dest
        except Exception:
            return None

    for idx, mapping in enumerate(bridge_mappings, start=1):
        bridge_dir = output_dir / f"bridge_{idx:02d}"
        bridge_overlap_assets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for dataset_id, feature_names in mapping.items():
            if not feature_names:
                continue
            summary = dataset_lookup.get(dataset_id)
            if summary is None:
                print(f"[warn] Dataset '{dataset_id}' referenced in bridge {idx} not found; skipping.")
                continue

            dataset_root = Path(getattr(summary, "root", Path.cwd()))
            regions = getattr(summary, "regions", {})
            geojson_lookup: Dict[str, List[Path]] = {}
            if label_geojson_map and dataset_id in label_geojson_map:
                geojson_lookup = {
                    region: [Path(p) for p in paths]
                    for region, paths in label_geojson_map[dataset_id].items()
                }
            overlap_geom_dataset = overlap_geoms.get(dataset_id) if overlap_geoms else None
            dataset_crs = None
            if dataset_crs_map and dataset_id in dataset_crs_map:
                dataset_crs = dataset_crs_map.get(dataset_id)
            point_transformer = None
            if dataset_crs is not None and target_crs is not None and Transformer is not None:
                try:
                    point_transformer = Transformer.from_crs(dataset_crs, target_crs, always_xy=True)
                except Exception:
                    point_transformer = None
            overlap_geom_native = None
            if overlap_geom_dataset is not None:
                overlap_geom_native = overlap_geom_dataset
                if (
                    shapely_transform is not None
                    and dataset_crs is not None
                    and target_crs is not None
                    and Transformer is not None
                    and overlap_geom_dataset is not None
                ):
                    try:
                        to_native = Transformer.from_crs(target_crs, dataset_crs, always_xy=True)
                        overlap_geom_native = shapely_transform(to_native.transform, overlap_geom_dataset)
                    except Exception:
                        overlap_geom_native = overlap_geom_dataset

            for feature_name in feature_names:
                located = False
                for region_name, region_info in regions.items():
                    region_safe = region_name.replace("/", "_")
                    region_base_dir = bridge_dir / dataset_id / region_safe
                    region_base_dir.mkdir(parents=True, exist_ok=True)

                    geojson_paths = geojson_lookup.get(region_name) or geojson_lookup.get("GLOBAL", [])
                    geojson_dest_dir = region_base_dir / "label_geojson"
                    for geo_path in geojson_paths:
                        try:
                            geojson_dest_dir.mkdir(parents=True, exist_ok=True)
                            copy_target = geojson_dest_dir / geo_path.name
                            if not copy_target.exists():
                                shutil.copy2(geo_path, copy_target)
                        except Exception as exc:
                            print(f"[warn] Failed to copy label GeoJSON {geo_path}: {exc}")

                    for bucket in ("features", "labels"):
                        entry = region_info.get(bucket, {}).get(feature_name)
                        if not isinstance(entry, dict):
                            continue
                        tif_records = entry.get("tifs", [])
                        if not tif_records:
                            continue
                        record = next(
                            (
                                rec
                                for rec in tif_records
                                if isinstance(rec, dict) and (rec.get("path_resolved") or rec.get("path"))
                            ),
                            None,
                        )
                        if record is None:
                            continue
                        path_value = record.get("path_resolved") or record.get("path")
                        if not path_value:
                            continue
                        src_path = Path(path_value)
                        if not src_path.is_absolute():
                            src_path = (dataset_root / src_path).resolve()
                        if not src_path.exists():
                            print(f"[warn] Bridge asset missing on disk: {src_path}")
                            continue

                        dest_dir = region_base_dir / bucket
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_tif = dest_dir / f"{feature_name}.tif"
                        try:
                            if not dest_tif.exists():
                                shutil.copy2(src_path, dest_tif)
                        except Exception as exc:
                            print(f"[warn] Failed to copy {src_path} -> {dest_tif}: {exc}")

                        dest_png = dest_dir / f"{feature_name}.png"
                        try:
                            label_points: List[Tuple[float, float]] = []
                            clipped_result = None
                            clipped_meta = None
                            clipped_transform = None
                            with rasterio.open(src_path) as src_ds:
                                data = src_ds.read(1, masked=True)
                                if hasattr(data, "filled"):
                                    arr_native = data.filled(np.nan).astype(np.float32, copy=False)
                                else:
                                    arr_native = np.asarray(data, dtype=np.float32)
                                native_transform = src_ds.transform
                                src_crs = src_ds.crs

                                arr = arr_native
                                transform = native_transform
                                dest_crs = src_crs
                                if target_crs is not None and src_crs is not None:
                                    try:
                                        transform, width, height = calculate_default_transform(
                                            src_crs,
                                            target_crs,
                                            src_ds.width,
                                            src_ds.height,
                                            *src_ds.bounds,
                                        )
                                        dest = np.empty((height, width), dtype=np.float32)
                                        reproject(
                                            arr_native,
                                            dest,
                                            src_transform=native_transform,
                                            src_crs=src_crs,
                                            dst_transform=transform,
                                            dst_crs=target_crs,
                                            resampling=Resampling.bilinear,
                                        )
                                        arr = dest
                                        dest_crs = target_crs
                                    except Exception:
                                        arr = arr_native
                                        transform = native_transform
                                        dest_crs = src_crs
                                else:
                                    transform = native_transform
                                    dest_crs = src_crs

                                overlap_clip_geom = None
                                if overlap_geom_dataset is not None and mapping is not None:
                                    overlap_clip_geom = overlap_geom_native or overlap_geom_dataset
                                if overlap_clip_geom is not None and mapping is not None:
                                    try:
                                        shapes_to_mask = [mapping(overlap_clip_geom)]
                                        clipped_arr, clipped_transform = mask(
                                            src_ds,
                                            shapes_to_mask,
                                            crop=True,
                                            filled=True,
                                            all_touched=True,
                                        )
                                        if target_crs is not None and src_crs is not None and clipped_arr.size:
                                            clip_bounds = array_bounds(
                                                clipped_arr.shape[1],
                                                clipped_arr.shape[2],
                                                clipped_transform,
                                            )
                                            c_transform, c_width, c_height = calculate_default_transform(
                                                src_crs,
                                                target_crs,
                                                clipped_arr.shape[2],
                                                clipped_arr.shape[1],
                                                *clip_bounds,
                                            )
                                            clipped_dest = np.empty((1, c_height, c_width), dtype=clipped_arr.dtype)
                                            reproject(
                                                clipped_arr,
                                                clipped_dest,
                                                src_transform=clipped_transform,
                                                src_crs=src_crs,
                                                dst_transform=c_transform,
                                                dst_crs=target_crs,
                                                resampling=Resampling.bilinear,
                                            )
                                            clipped_meta = src_ds.meta.copy()
                                            clipped_meta.update(
                                                height=c_height,
                                                width=c_width,
                                                transform=c_transform,
                                                crs=target_crs,
                                            )
                                            clipped_result = clipped_dest
                                            clipped_transform = c_transform
                                        elif clipped_arr.size:
                                            clipped_meta = src_ds.meta.copy()
                                            clipped_meta.update(
                                                height=clipped_arr.shape[1],
                                                width=clipped_arr.shape[2],
                                                transform=clipped_transform,
                                            )
                                            clipped_result = clipped_arr
                                    except Exception:
                                        clipped_result = None

                                boundary_mask = None
                                try:
                                    mask_native = src_ds.dataset_mask()
                                    if mask_native is not None:
                                        mask_native = mask_native != 0
                                        if mask_native.any():
                                            if target_crs is not None and src_crs is not None and mask_native.shape == arr_native.shape:
                                                mask_dest = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
                                                reproject(
                                                    mask_native.astype(np.uint8),
                                                    mask_dest,
                                                    src_transform=native_transform,
                                                    src_crs=src_crs,
                                                    dst_transform=transform,
                                                    dst_crs=dest_crs,
                                                    resampling=Resampling.nearest,
                                                )
                                                boundary_mask = mask_dest.astype(bool)
                                            else:
                                                boundary_mask = mask_native.astype(bool)
                                except Exception:
                                    boundary_mask = None

                            if boundary_mask is not None:
                                arr = np.where(boundary_mask, arr, np.nan)
                            arr_local = arr.astype(np.float32, copy=False)
                            transform_local = transform
                            crs_local = dest_crs

                            mask_local = _project_mask_to_grid(arr_local.shape, transform_local, crs_local)
                            if mask_local is not None:
                                arr_masked_local = np.where(mask_local, arr_local, np.nan)
                            else:
                                arr_masked_local = arr_local

                            global_arr = _project_to_overlap_grid(arr_local, transform_local, crs_local, bucket)

                            height, width = arr.shape

                            fig, ax = plt.subplots(figsize=(6, 5))
                            ax.set_title(f"{feature_name}\n{dataset_id} / {region_name} [{bucket[:-1]}]")
                            extent = None
                            try:
                                left, bottom, right, top = array_bounds(height, width, transform)
                                extent = (left, right, bottom, top)
                            except Exception:
                                extent = None

                            if bucket == "labels":
                                positives = np.where(arr > 0, 1.0, np.nan)
                                img = ax.imshow(
                                    positives,
                                    origin="upper",
                                    cmap="Reds",
                                    vmin=0.0,
                                    vmax=1.0,
                                    extent=extent,
                                )
                                pos_rows, pos_cols = np.where(arr > 0)
                                if pos_rows.size and extent is not None:
                                    xs, ys = xy(transform, pos_rows, pos_cols, offset="center")
                                    ax.scatter(xs, ys, s=8, c="cyan", edgecolors="black", linewidths=0.3)
                                elif pos_rows.size:
                                    ax.scatter(pos_cols + 0.5, pos_rows + 0.5, s=8, c="cyan", edgecolors="black", linewidths=0.3)
                                for geo_path in geojson_paths:
                                    try:
                                        with open(geo_path, "r", encoding="utf-8") as fh:
                                            geo_doc = json.load(fh)
                                    except Exception:
                                        continue
                                    features = geo_doc.get("features", [])
                                    for feat in features:
                                        geom = feat.get("geometry") if isinstance(feat, dict) else None
                                        if not isinstance(geom, dict):
                                            continue
                                        gtype = geom.get("type")
                                        coords = geom.get("coordinates")
                                        if coords is None:
                                            continue
                                        try:
                                            if gtype == "Point":
                                                x_val, y_val = coords
                                            elif gtype in {"MultiPoint"} and coords:
                                                x_val, y_val = coords[0]
                                            elif gtype in {"LineString"} and coords:
                                                xs_line, ys_line = zip(*coords)
                                                x_val = float(np.mean(xs_line))
                                                y_val = float(np.mean(ys_line))
                                            elif gtype in {"MultiLineString", "Polygon", "MultiPolygon"}:
                                                flat: List[Tuple[float, float]] = []
                                                def _accumulate(nodes):
                                                    if not isinstance(nodes, (list, tuple)):
                                                        return
                                                    if len(nodes) >= 2 and not isinstance(nodes[0], (list, tuple)):
                                                        try:
                                                            flat.append((float(nodes[0]), float(nodes[1])))
                                                        except Exception:
                                                            pass
                                                        return
                                                    for node in nodes:
                                                        _accumulate(node)
                                                _accumulate(coords)
                                                if not flat:
                                                    continue
                                                xs_poly, ys_poly = zip(*flat)
                                                x_val = float(np.mean(xs_poly))
                                                y_val = float(np.mean(ys_poly))
                                            else:
                                                continue
                                            label_points.append((x_val, y_val))
                                        except Exception:
                                            continue
                                if label_points:
                                    lp_x, lp_y = zip(*label_points)
                                    ax.scatter(lp_x, lp_y, s=24, c="yellow", marker="*", edgecolors="black", linewidths=0.4, zorder=7)
                            else:
                                finite = np.isfinite(arr)
                                if finite.any():
                                    valid = arr[finite]
                                    vmin = float(np.nanpercentile(valid, 2))
                                    vmax = float(np.nanpercentile(valid, 98))
                                    if vmin == vmax:
                                        vmax = vmin + 1.0
                                else:
                                    vmin, vmax = 0.0, 1.0
                                img = ax.imshow(
                                    arr,
                                    origin="upper",
                                    cmap="viridis",
                                    vmin=vmin,
                                    vmax=vmax,
                                    extent=extent,
                                )
                            if overlap_geom_dataset is not None and mapping is not None:
                                def _plot_geom(geom):
                                    if geom.is_empty:
                                        return
                                    if hasattr(geom, "geoms"):
                                        for sub in geom.geoms:
                                            _plot_geom(sub)
                                        return
                                    exterior = geom.exterior.coords if geom.exterior else []
                                    if exterior:
                                        xs, ys = zip(*exterior)
                                        ax.plot(xs, ys, color="lime", linewidth=1.2, alpha=0.9)
                                    for interior in geom.interiors:
                                        xs, ys = zip(*interior.coords)
                                        ax.plot(xs, ys, color="lime", linewidth=0.8, alpha=0.6, linestyle="--")

                                _plot_geom(overlap_geom_dataset)
                            if boundary_mask is not None:
                                try:
                                    contour_mask = _compute_boundary_mask(boundary_mask)
                                    if extent is not None:
                                        x_coords = np.linspace(extent[0], extent[1], contour_mask.shape[1])
                                        y_coords = np.linspace(extent[3], extent[2], contour_mask.shape[0])
                                        ax.contour(
                                            x_coords,
                                            y_coords,
                                            contour_mask.astype(np.uint8),
                                            levels=[0.5],
                                            colors="black",
                                            linewidths=1.0,
                                        )
                                    else:
                                        ax.contour(
                                            contour_mask.astype(np.uint8),
                                            levels=[0.5],
                                            colors="black",
                                            linewidths=1.0,
                                        )
                                except Exception:
                                    pass

                            ax.set_xlabel("X (Equal Earth m)")
                            ax.set_ylabel("Y (Equal Earth m)")

                            fig.tight_layout()
                            fig.savefig(dest_png, dpi=200)
                            plt.close(fig)

                            try:
                                array_store = arr_local
                                transform_store = transform_local
                                crs_store = crs_local
                                array_global = _project_to_overlap_grid(array_store, transform_store, crs_store, bucket)
                                bridge_overlap_assets[dataset_id].append(
                                    {
                                        "feature": feature_name,
                                        "bucket": bucket,
                                        "array": array_store,
                                        "transform": transform_store,
                                        "crs": crs_store,
                                        "masked_array": arr_masked_local,
                                        "global_array": array_global,
                                    }
                                )
                            except Exception as exc:
                                print(f"[warn] Failed to cache overlap data for {feature_name}: {exc}")
                        except Exception as exc:
                            print(f"[warn] Failed to render {src_path}: {exc}")
                        located = True
                        break
                    if located:
                        break
                if not located:
                    print(f"[warn] Feature '{feature_name}' not found for dataset '{dataset_id}' in bridge {idx}.")

        overlap_base = bridge_dir / "Overlap"
        combo_dataset_ids = [ds_id for ds_id in mapping.keys() if bridge_overlap_assets.get(ds_id)]
        if len(combo_dataset_ids) >= 2:
            combined_name = "-".join(combo_dataset_ids)
            combined_dir = overlap_base / combined_name
            combined_dir.mkdir(parents=True, exist_ok=True)

            plot_entries: List[Dict[str, Any]] = []
            for ds_id in combo_dataset_ids:
                assets = bridge_overlap_assets.get(ds_id, [])
                order = mapping.get(ds_id, [])
                for feature_name in order:
                    match = next((rec for rec in assets if rec.get("feature") == feature_name), None)
                    if match:
                        entry = dict(match)
                        entry["dataset"] = ds_id
                        plot_entries.append(entry)

            if plot_entries:
                boundary_geom = None
                if overall_overlap_geom is not None:
                    boundary_geom = overall_overlap_geom
                elif overlap_geoms:
                    geoms = [overlap_geoms.get(ds) for ds in combo_dataset_ids if overlap_geoms.get(ds) is not None]
                    if geoms:
                        boundary_geom = geoms[0]
                        for geom in geoms[1:]:
                            if boundary_geom is None:
                                boundary_geom = geom
                            else:
                                try:
                                    boundary_geom = boundary_geom.intersection(geom)
                                except Exception:
                                    boundary_geom = geom

                try:
                    import matplotlib.pyplot as plt
                except Exception as exc:
                    print(f"[warn] Failed to import matplotlib for combined overlap plot: {exc}")
                    continue

                n_plots = len(plot_entries)
                ncols = 2 if n_plots > 1 else 1
                nrows = int(np.ceil(n_plots / ncols))
                fig_height = max(4, nrows * 4)
                fig_width = max(6, ncols * 4)
                fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
                if isinstance(axes, np.ndarray):
                    axes = axes.flatten()
                else:
                    axes = [axes]

                def _plot_boundary(ax_obj, geom, target_crs_obj, clip_crs_obj):
                    if geom is None or geom.is_empty:
                        return
                    geom_to_plot = geom
                    if (
                        shapely_transform is not None
                        and Transformer is not None
                        and target_crs_obj is not None
                        and clip_crs_obj is not None
                    ):
                        try:
                            to_clip = Transformer.from_crs(target_crs_obj, clip_crs_obj, always_xy=True)
                            geom_to_plot = shapely_transform(to_clip.transform, geom)
                        except Exception:
                            geom_to_plot = geom
                    if geom_to_plot is None or geom_to_plot.is_empty:
                        return
                    if hasattr(geom_to_plot, "geoms"):
                        for sub_geom in geom_to_plot.geoms:
                            _plot_boundary(ax_obj, sub_geom, None, None)
                        return
                    exterior = geom_to_plot.exterior.coords if geom_to_plot.exterior else []
                    if exterior:
                        xs_b, ys_b = zip(*exterior)
                        ax_obj.plot(xs_b, ys_b, color="black", linewidth=2.0, alpha=0.95)
                    for interior in geom_to_plot.interiors:
                        xs_i, ys_i = zip(*interior.coords)
                        ax_obj.plot(xs_i, ys_i, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

                global_clipped_entries: List[Tuple[str, np.ndarray, Optional[Tuple[float, float, float, float]], str, Optional[Any]]] = []
                global_clip_bounds = list(overlap_mask_bounds) if overlap_mask_bounds is not None else None

                for ax, entry in zip(axes, plot_entries):
                    ds_id = entry.get("dataset")
                    feature_name = entry.get("feature")
                    bucket = entry.get("bucket")
                    arr_clip = entry.get("array")
                    clip_transform = entry.get("transform")
                    clip_crs = entry.get("crs")
                    arr_masked_local = entry.get("masked_array")
                    arr_global = entry.get("global_array")

                    if arr_clip is None or clip_transform is None:
                        continue

                    clip_height, clip_width = arr_clip.shape
                    extent = None
                    try:
                        left_c, bottom_c, right_c, top_c = array_bounds(clip_height, clip_width, clip_transform)
                        extent = (left_c, right_c, bottom_c, top_c)
                    except Exception:
                        extent = None

                    ax.set_title(f"{feature_name} ({ds_id})")
                    if bucket == "labels":
                        positives_clip = np.where(arr_clip > 0, 1.0, np.nan)
                        ax.imshow(
                            positives_clip,
                            origin="upper",
                            cmap="Reds",
                            vmin=0.0,
                            vmax=1.0,
                            extent=extent,
                        )
                    else:
                        finite_clip = np.isfinite(arr_clip)
                        if finite_clip.any():
                            valid_clip = arr_clip[finite_clip]
                            vmin_clip = float(np.nanpercentile(valid_clip, 2))
                            vmax_clip = float(np.nanpercentile(valid_clip, 98))
                            if vmin_clip == vmax_clip:
                                vmax_clip = vmin_clip + 1.0
                        else:
                            vmin_clip, vmax_clip = 0.0, 1.0
                        ax.imshow(
                            arr_clip,
                            origin="upper",
                            cmap="viridis",
                            vmin=vmin_clip,
                            vmax=vmax_clip,
                            extent=extent,
                        )

                    _plot_boundary(ax, boundary_geom, target_crs, clip_crs)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")

                    masked_local = arr_masked_local if isinstance(arr_masked_local, np.ndarray) else arr_clip
                    if global_clip_bounds is None and extent is not None:
                        global_clip_bounds = list(extent)
                    global_clipped_entries.append(
                        (
                            ds_id,
                            arr_global if isinstance(arr_global, np.ndarray) else masked_local,
                            extent,
                            bucket,
                            overlap_mask_crs if isinstance(arr_global, np.ndarray) else clip_crs,
                        )
                    )

                # Hide unused axes if any
                for leftover_ax in axes[len(plot_entries):]:
                    leftover_ax.axis("off")

                fig.tight_layout()
                combined_png = combined_dir / f"bridge_{idx:02d}_combined.png"
                fig.savefig(combined_png, dpi=200)
                plt.close(fig)

                if global_clipped_entries:
                    fig_clip, axes_clip = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
                    if isinstance(axes_clip, np.ndarray):
                        axes_clip = axes_clip.flatten()
                    else:
                        axes_clip = [axes_clip]

                    clip_extent_global = (
                        tuple(overlap_mask_bounds)
                        if overlap_mask_bounds is not None
                        else (tuple(global_clip_bounds) if global_clip_bounds is not None else None)
                    )

                    for ax_clip, (ds_id, arr_clip, extent_clip, bucket_clip, clip_crs_inner) in zip(
                        axes_clip, global_clipped_entries
                    ):
                        ax_clip.set_title(f"{bucket_clip} ({ds_id}) - clipped")
                        if bucket_clip == "labels":
                            positives_clip = np.where(arr_clip > 0, 1.0, np.nan)
                            ax_clip.imshow(
                                positives_clip,
                                origin="upper",
                                cmap="Reds",
                                vmin=0.0,
                                vmax=1.0,
                                extent=clip_extent_global,
                            )
                        else:
                            finite_clip = np.isfinite(arr_clip)
                            if finite_clip.any():
                                valid_clip = arr_clip[finite_clip]
                                vmin_clip = float(np.nanpercentile(valid_clip, 2))
                                vmax_clip = float(np.nanpercentile(valid_clip, 98))
                                if vmin_clip == vmax_clip:
                                    vmax_clip = vmin_clip + 1.0
                            else:
                                vmin_clip, vmax_clip = 0.0, 1.0
                            ax_clip.imshow(
                                arr_clip,
                                origin="upper",
                                cmap="viridis",
                                vmin=vmin_clip,
                                vmax=vmax_clip,
                                extent=clip_extent_global,
                            )
                        _plot_boundary(ax_clip, boundary_geom, target_crs, clip_crs_inner)
                        ax_clip.set_xlabel("X")
                        ax_clip.set_ylabel("Y")

                    for leftover_ax in axes_clip[len(global_clipped_entries):]:
                        leftover_ax.axis("off")

                    fig_clip.tight_layout()
                    combined_clip_png = combined_dir / f"bridge_{idx:02d}_combined_clipped.png"
                    fig_clip.savefig(combined_clip_png, dpi=200)
                    plt.close(fig_clip)
