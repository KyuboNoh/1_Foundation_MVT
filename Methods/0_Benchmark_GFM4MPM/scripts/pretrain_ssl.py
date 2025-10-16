# scripts/pretrain_ssl.py
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ensure repo-root execution can resolve the local src package
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.data_utils import clamp_coords_to_window, load_stac_rasters, MultiRegionStack
from src.gfm4mpm.data.geo_stack import GeoStack
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.training.train_ssl import _collect_preview_samples, _plot_samples, train_ssl


# TODO: How to deal with catergorical data? 1) Only use it for input, not for reconstruction 2) Use it for reconstruction, but use cross-entropy loss (how to deal with MSE?)

# TODO: later check it for AWS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class SSLDataset(Dataset):
    def __init__(
        self,
        stack,
        window: int = 32,
        n_samples: int = 200000,
        seed: int = 1337,
        skip_nan: bool = True,
        max_resample_attempts: int = 32,
    ):
        self.stack, self.window = stack, window
        self.rng = np.random.default_rng(seed)
        self.n = n_samples
        self.skip_nan = bool(skip_nan)
        self.max_resample_attempts = max(1, int(max_resample_attempts))

    def __len__(self):
        return self.n

    @staticmethod
    def _ensure_mask(mask: Optional[np.ndarray], patch: np.ndarray) -> np.ndarray:
        if mask is None:
            valid = np.isfinite(patch)
            if patch.ndim == 3:
                valid = valid.any(axis=0)
            return ~valid.astype(bool, copy=False)
        if mask.ndim == 3:
            return (~mask.any(axis=0)).astype(bool, copy=False)
        return mask.astype(bool, copy=False)

    def _pack_sample(self, patch: np.ndarray, mask_no_feature: np.ndarray) -> Dict[str, torch.Tensor]:
        mask_bool = mask_no_feature.astype(bool, copy=False)
        sample = {
            "image": torch.from_numpy(patch.astype(np.float32)),
            "mask_pixel_no_feature": torch.from_numpy(mask_bool),
        }
        return sample

    def __getitem__(self, idx):
        if hasattr(self.stack, "sample_patch"):
            attempts = self.max_resample_attempts if self.skip_nan else 1
            last_patch = None
            last_mask = None
            for _ in range(attempts):
                sample = self.stack.sample_patch(
                    self.window,
                    self.rng,
                    skip_nan=self.skip_nan,
                    max_attempts=self.max_resample_attempts,
                )
                if isinstance(sample, tuple) and len(sample) == 2:
                    patch, mask = sample
                else:
                    patch, mask = sample, None
                mask_no_feature = self._ensure_mask(mask, patch)
                if mask_no_feature.all():
                    last_patch, last_mask = patch, mask_no_feature
                    continue
                last_patch, last_mask = patch, mask_no_feature
                return self._pack_sample(patch, mask_no_feature)

            if last_patch is not None and last_mask is not None:
                return self._pack_sample(last_patch, last_mask)
            raise RuntimeError("sample_patch failed to provide a usable window; verify raster coverage")

        attempts = self.max_resample_attempts if self.skip_nan else 1
        last_patch = None
        last_mask = None
        for _ in range(attempts):
            if hasattr(self.stack, "random_coord"):
                r, c = self.stack.random_coord(self.window, self.rng)
            else:
                half = max(1, self.window // 2)
                max_row = max(half + 1, self.stack.height - half)
                max_col = max(half + 1, self.stack.width - half)
                r = int(self.rng.integers(half, max_row))
                c = int(self.rng.integers(half, max_col))
                (clamped_coord, _) = clamp_coords_to_window([(r, c)], self.stack, self.window)
                r, c = clamped_coord[0]
            if hasattr(self.stack, "read_patch_with_mask"):
                x, mask = self.stack.read_patch_with_mask(r, c, self.window)
            else:
                x = self.stack.read_patch(r, c, self.window)
                mask = None

            mask_no_feature = self._ensure_mask(mask, x)
            if mask_no_feature.all():
                continue
            if self.skip_nan and not np.isfinite(x).all():
                continue
            last_patch = x
            last_mask = mask_no_feature
            return self._pack_sample(x, mask_no_feature)

        if self.skip_nan and last_patch is not None and last_mask is not None:
            raise RuntimeError(
                f"Failed to sample a finite patch after {self.max_resample_attempts} attempts; "
                "consider disabling --skip-nan or checking raster nodata handling."
            )
        if last_patch is None:
            raise RuntimeError(
                "Unable to sample a patch from raster stack; verify raster coverage and metadata."
            )
        return self._pack_sample(last_patch, last_mask if last_mask is not None else np.zeros(last_patch.shape[-2:], dtype=bool))


def _generate_preview_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    preview_samples: int,
    preview_dir: Path,
    feature_names: Optional[Iterable[str]],
    feature_metadata: Optional[Iterable[Optional[Dict[str, Optional[str]]]]],
    mask_scope: str,
) -> None:
    if preview_samples <= 0:
        print('[info] button-inference requested but preview_samples=0; skipping preview generation.')
        return
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu')
    model.eval()
    sample_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch.get('image')
                if x is None:
                    raise ValueError("Batch dictionary missing 'image' key")
                mask_no_feature = batch.get('mask_pixel_no_feature')
            else:
                x = batch
                mask_no_feature = None

            x = x.to(device)

            mask_no_feature_tensor: Optional[torch.Tensor] = None
            if mask_no_feature is not None:
                mask_no_feature_tensor = mask_no_feature.to(device=device)
                if mask_no_feature_tensor.dtype != torch.bool:
                    mask_no_feature_tensor = mask_no_feature_tensor.bool()
                if mask_no_feature_tensor.dim() == 2:
                    mask_no_feature_tensor = mask_no_feature_tensor.unsqueeze(0).unsqueeze(0)
                elif mask_no_feature_tensor.dim() == 3:
                    mask_no_feature_tensor = mask_no_feature_tensor.unsqueeze(1)
                elif mask_no_feature_tensor.dim() == 4 and mask_no_feature_tensor.size(1) != 1:
                    mask_no_feature_tensor = mask_no_feature_tensor.any(dim=1, keepdim=True)

            expected_size = getattr(model, 'image_size', None)
            if expected_size is not None:
                if isinstance(expected_size, int):
                    expected_size = (expected_size, expected_size)
                if x.dim() == 4 and (x.shape[-2], x.shape[-1]) != expected_size:
                    x = torch.nn.functional.interpolate(x, size=expected_size, mode='nearest')
                    if mask_no_feature_tensor is not None:
                        mask_no_feature_tensor = torch.nn.functional.interpolate(
                            mask_no_feature_tensor.float(), size=expected_size, mode='nearest'
                        ) >= 0.5
            elif x.dim() == 4 and x.shape[-2] == 1 and x.shape[-1] == 1:
                target_hw = max(16, getattr(model, 'patch_size', 16))
                x = torch.nn.functional.interpolate(x, size=(target_hw, target_hw), mode='nearest')
                if mask_no_feature_tensor is not None:
                    mask_no_feature_tensor = torch.nn.functional.interpolate(
                        mask_no_feature_tensor.float(), size=(target_hw, target_hw), mode='nearest'
                    ) >= 0.5

            if mask_no_feature_tensor is not None:
                mask_no_feature_tensor = mask_no_feature_tensor.bool()

            pred, mae_mask_tokens = model(x)

            patch_size = getattr(model, 'patch_size', None)
            pixel_mae_mask: Optional[torch.Tensor] = None
            if (
                isinstance(patch_size, int)
                and pred.dim() == 4
                and mae_mask_tokens is not None
            ):
                H, W = pred.shape[-2], pred.shape[-1]
                if H % patch_size == 0 and W % patch_size == 0:
                    grid_h, grid_w = H // patch_size, W // patch_size
                    if mae_mask_tokens.numel() == pred.size(0) * grid_h * grid_w:
                        mask_grid = mae_mask_tokens.view(pred.size(0), grid_h, grid_w)
                        pixel_mae_mask = mask_grid.unsqueeze(1)
                        pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_size, dim=2)
                        pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_size, dim=3)
                        pixel_mae_mask = pixel_mae_mask.to(device=pred.device, dtype=pred.dtype)

            if pixel_mae_mask is None:
                pixel_mae_mask = torch.ones(
                    (pred.size(0), 1, pred.shape[-2], pred.shape[-1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )

            pixel_invalid_mask: Optional[torch.Tensor] = None
            if mask_no_feature_tensor is not None:
                pixel_invalid_mask = mask_no_feature_tensor.to(device=pred.device)

            new_samples = _collect_preview_samples(
                x,
                pred,
                pixel_mae_mask,
                pixel_invalid_mask,
                mask_scope=mask_scope,
                mask_ratio=float(getattr(model, 'mask_ratio', 0.0)),
            )
            for entry in new_samples:
                sample_cache.append(entry)
                if len(sample_cache) >= preview_samples:
                    break
            if len(sample_cache) >= preview_samples:
                break

    if not sample_cache:
        print('[warn] Unable to collect preview samples during button-inference run.')
        return

    limited = sample_cache[:preview_samples]
    stacked = [torch.stack(list(seq), dim=0) for seq in zip(*limited)]
    original_t, masked_t, recon_t, mae_masks, invalid_masks = stacked

    _plot_samples(
        original_t,
        masked_t,
        recon_t,
        feature_names,
        feature_metadata,
        preview_dir,
        prefix='button_inference',
        mae_masks=mae_masks,
        invalid_masks=invalid_masks,
    )
    print(f"[info] Generated {len(limited)} preview sample(s) in {preview_dir}")


def _shared_minmax_np(arrays: Iterable[np.ndarray]) -> Tuple[float, float]:
    vmin = None
    vmax = None
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        mn = float(finite.min())
        mx = float(finite.max())
        vmin = mn if vmin is None else min(vmin, mn)
        vmax = mx if vmax is None else max(vmax, mx)
    if vmin is None or vmax is None:
        return 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        eps = max(abs(vmin), abs(vmax), 1.0) * 0.01 + 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def _write_region_maps(
    region_name: str,
    feature_names: Sequence[str],
    original_map: np.ndarray,
    masked_map: np.ndarray,
    overlay_map: np.ndarray,
    ssl_mask_map: np.ndarray,
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        print(f"[warn] Skipping stitched map plotting for {region_name}; matplotlib unavailable: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    C = original_map.shape[0]
    fig, axes = plt.subplots(C, 3, figsize=(9, 3 * C), constrained_layout=True)
    if C == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(C):
        name = feature_names[idx] if idx < len(feature_names) else f"ch_{idx}"
        originals = original_map[idx]
        masked = masked_map[idx]
        overlay = overlay_map[idx]
        vmin, vmax = _shared_minmax_np([originals, masked, overlay])
        row_axes = []
        for ax, data, title in zip(
            axes[idx],
            (originals, masked, overlay),
            ("original", "masked", "recon+orig"),
        ):
            cmap = plt.colormaps['viridis']
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            if title == "masked":
                ssl_mask = ssl_mask_map.astype(bool)
                overlay_mask = np.where(ssl_mask, 1.0, np.nan)
                ax.imshow(
                    overlay_mask,
                    cmap=ListedColormap(["black"]),
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.6,
                )
            ax.set_title(f"{region_name}: {name} - {title}")
            ax.axis('off')
            row_axes.append(ax)
        cbar = fig.colorbar(im, ax=row_axes, fraction=0.046, pad=0.08)
        cbar.ax.tick_params(length=2)

    safe_region = re.sub(r"[^A-Za-z0-9_.-]+", "_", region_name)
    out_path = out_dir / f"{safe_region}_stitched.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[info] Wrote stitched map for {region_name} to {out_path}")


def _compute_patch_bounds(height: int, width: int, row: int, col: int, window: int) -> Tuple[int, int, int, int]:
    half = window // 2
    r0 = max(row - half, 0)
    c0 = max(col - half, 0)
    r1 = min(r0 + window, height)
    c1 = min(c0 + window, width)
    return r0, c0, r1, c1


def _stitch_region_maps(
    region_name: str,
    stack: GeoStack,
    model: torch.nn.Module,
    window_size: int,
    mask_scope: str,
    mask_ratio: float,
    batch_size: int,
    feature_names: Optional[Sequence[str]],
    out_dir: Path,
) -> None:
    device = next(model.parameters()).device
    coords = stack.grid_centers(window_size)
    if not coords:
        print(f"[warn] No grid centers available for region {region_name}; skipping stitched map.")
        return

    height, width = stack.height, stack.width
    channels = stack.count
    original_sum = np.zeros((channels, height, width), dtype=np.float32)
    overlay_sum = np.zeros_like(original_sum)
    weight = np.zeros((height, width), dtype=np.float32)
    mask_hits = np.zeros((height, width), dtype=np.float32)

    batch_patches: List[np.ndarray] = []
    batch_masks: List[np.ndarray] = []
    batch_bounds: List[Tuple[int, int, int, int]] = []

    def _process_batch() -> None:
        if not batch_patches:
            return
        patch_tensor = torch.from_numpy(np.stack(batch_patches)).to(device)
        invalid_tensor = torch.from_numpy(
            np.stack(batch_masks)[:, None, :, :]
        ).to(device=device)
        with torch.no_grad():
            preds, mae_mask = model(patch_tensor)
        pixel_mae_mask: Optional[torch.Tensor] = None
        patch_sz = getattr(model, 'patch_size', None)
        if (
            isinstance(patch_sz, int)
            and preds.dim() == 4
            and mae_mask is not None
        ):
            H, W = preds.shape[-2], preds.shape[-1]
            if H % patch_sz == 0 and W % patch_sz == 0:
                grid_h, grid_w = H // patch_sz, W // patch_sz
                if mae_mask.numel() == preds.size(0) * grid_h * grid_w:
                    mask_grid = mae_mask.view(preds.size(0), grid_h, grid_w)
                    pixel_mae_mask = mask_grid.unsqueeze(1)
                    pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_sz, dim=2)
                    pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_sz, dim=3)
                    pixel_mae_mask = pixel_mae_mask.to(device=preds.device, dtype=preds.dtype)
        if pixel_mae_mask is None:
            pixel_mae_mask = torch.ones(
                (preds.size(0), 1, preds.shape[-2], preds.shape[-1]),
                device=preds.device,
                dtype=preds.dtype,
            )

        samples = _collect_preview_samples(
            patch_tensor,
            preds,
            pixel_mae_mask,
            invalid_tensor,
            mask_scope=mask_scope,
            mask_ratio=mask_ratio,
        )

        for sample, bounds in zip(samples, batch_bounds):
            r0, c0, r1, c1 = bounds
            h = r1 - r0
            w = c1 - c0
            orig_np = sample[0].numpy()[:, :h, :w]
            recon_np = sample[2].numpy()[:, :h, :w]
            mae_np = sample[3].numpy()[:h, :w]
            invalid_np = sample[4].numpy()[:h, :w]
            overlay_np = recon_np.copy()
            overlay_np[:, ~mae_np.astype(bool)] = orig_np[:, ~mae_np.astype(bool)]
            valid = (~invalid_np).astype(np.float32)
            valid_expand = valid[np.newaxis, :, :]
            original_sum[:, r0:r1, c0:c1] += orig_np * valid_expand
            overlay_sum[:, r0:r1, c0:c1] += overlay_np * valid_expand
            weight[r0:r1, c0:c1] += valid
            mask_hits[r0:r1, c0:c1] += mae_np.astype(np.float32) * valid

        batch_patches.clear()
        batch_masks.clear()
        batch_bounds.clear()

    for row, col in coords:
        patch, mask_no_feature = stack.read_patch_with_mask(row, col, window_size)
        batch_patches.append(patch)
        batch_masks.append(mask_no_feature)
        batch_bounds.append(_compute_patch_bounds(height, width, row, col, window_size))
        if len(batch_patches) >= batch_size:
            _process_batch()

    _process_batch()

    nonzero = weight > 0
    if not nonzero.any():
        print(f"[warn] No valid pixels accumulated for region {region_name}; skipping stitched map.")
        return

    original_map = np.zeros_like(original_sum)
    overlay_map = np.zeros_like(overlay_sum)
    for ch in range(channels):
        np.divide(original_sum[ch], weight, out=original_map[ch], where=nonzero)
        np.divide(overlay_sum[ch], weight, out=overlay_map[ch], where=nonzero)
    original_map[:, ~nonzero] = np.nan
    overlay_map[:, ~nonzero] = np.nan
    ssl_mask_map = mask_hits > 0
    masked_map = original_map.copy()
    masked_map[:, ssl_mask_map] = np.nan

    feature_labels = (
        list(feature_names)
        if feature_names is not None and len(feature_names) >= channels
        else [f"ch_{i}" for i in range(channels)]
    )
    ssl_mask_map = mask_hits > 0
    _write_region_maps(region_name, feature_labels, original_map, masked_map, overlay_map, ssl_mask_map, out_dir)


def _generate_stitched_maps(
    model: torch.nn.Module,
    stack,
    window_size: int,
    mask_scope: str,
    mask_ratio: float,
    batch_size: int,
    preview_dir: Path,
) -> None:
    if mask_scope not in {"patch", "pixel"}:
        print('[warn] Unsupported mask_scope for stitched maps; skipping.')
        return

    regions: List[Tuple[str, GeoStack]] = []
    if isinstance(stack, MultiRegionStack):
        for region_name, region_stack in stack.iter_region_stacks():
            if getattr(region_stack, 'kind', None) == 'raster':
                regions.append((region_name, region_stack))
    elif getattr(stack, 'kind', None) == 'raster':
        regions.append(('region', stack))

    if not regions:
        print('[warn] No raster regions available for stitched map generation.')
        return

    selected = regions[:2]
    stitched_dir = preview_dir / 'stitched'
    for region_name, region_stack in selected:
        names = getattr(region_stack, 'feature_columns', None)
        _stitch_region_maps(
            region_name,
            region_stack,
            model,
            window_size,
            mask_scope,
            mask_ratio,
            batch_size,
            names,
            stitched_dir,
        )

def _persist_training_args_to_metadata(
    stac_root: Optional[Path],
    args_snapshot: Dict[str, Any],
    output_dir: Path,
    used_features: Optional[Sequence[str]] = None,
) -> None:
    if stac_root is None:
        return

    candidates = [
        stac_root / "training_metadata.json",
        stac_root / "assetization" / "training_metadata.json",
    ]
    metadata_path = None
    for candidate in candidates:
        if candidate.exists():
            metadata_path = candidate
            break
    if metadata_path is None:
        metadata_path = stac_root / "training_metadata.json"

    try:
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            data = {}
    except Exception as exc:
        print(f"[warn] Unable to load training metadata at {metadata_path}: {exc}")
        data = {}

    pretraining_entry = data.get("pretraining")
    if not isinstance(pretraining_entry, dict):
        pretraining_entry = {}
        data["pretraining"] = pretraining_entry

    pretraining_entry["args"] = args_snapshot
    pretraining_entry["output_dir"] = str(output_dir.resolve())
    pretraining_entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    features_to_record: Optional[List[str]] = None
    if used_features is not None:
        features_to_record = list(dict.fromkeys(str(feat) for feat in used_features))
    else:
        arg_features = args_snapshot.get("features")
        if isinstance(arg_features, list):
            features_to_record = list(dict.fromkeys(str(feat) for feat in arg_features))
    if features_to_record is not None:
        pretraining_entry["features"] = features_to_record

    feature_section = data.get("features")
    if not isinstance(feature_section, dict):
        feature_section = {"entries": {}, "total_features": 0}
        data["features"] = feature_section
    feature_entries = feature_section.setdefault("entries", {})
    arg_features = args_snapshot.get("features")
    if isinstance(arg_features, list):
        for feat in arg_features:
            feature_entries.setdefault(feat, {"num_tifs": 0, "tifs": []})
    feature_section["total_features"] = len(feature_entries)

    label_section = data.get("labels")
    if not isinstance(label_section, dict):
        label_section = {"entries": {}, "total_labels": 0}
        data["labels"] = label_section
    label_entries = label_section.setdefault("entries", {})
    label_column = args_snapshot.get("label_column")
    if isinstance(label_column, str) and label_column and label_column not in label_entries:
        label_entries[label_column] = {"num_tifs": 0, "tifs": []}
    label_section["total_labels"] = len(label_entries)

    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] Failed to persist training args to {metadata_path}: {exc}")


def _maybe_check_image_preproc(
    stack,
    feature_name: str,
    window: int,
    out_dir: Path,
    rng_seed: int = 0,
    *,
    variant_tag: Optional[str] = None,
) -> None:
    """Render a diagnostic plot comparing the full feature grid vs. sampled patches."""

    grid: Optional[np.ndarray] = None
    lat_axis: Optional[np.ndarray] = None
    lon_axis: Optional[np.ndarray] = None
    candidate_mask: Optional[np.ndarray] = None

    if getattr(stack, "is_table", False):
        latitudes = getattr(stack, "latitudes", None)
        longitudes = getattr(stack, "longitudes", None)
        if latitudes is None or longitudes is None:
            print("[check] Latitude/longitude columns unavailable; skipping image preprocessing diagnostic plot.")
            return

        try:
            raw_vals = stack.raw_column(feature_name)
        except KeyError:
            cols = list(getattr(stack, "feature_columns", []))
            fallback_idx: Optional[int] = None
            for idx, col in enumerate(cols):
                if col.lower() == feature_name.lower():
                    fallback_idx = idx
                    break
            if fallback_idx is None:
                print(
                    f"[check] Feature '{feature_name}' not found in STAC table; available columns include: {', '.join(cols[:10])}"
                )
                return
            raw_vals = stack.features[:, fallback_idx]

        raw_array = np.asarray(raw_vals)
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)

        def _safe_float(arr):
            result = np.empty(len(arr), dtype=np.float32)
            for idx, val in enumerate(arr):
                try:
                    result[idx] = float(val)
                except (TypeError, ValueError):
                    result[idx] = np.nan
            return result

        feature_vals = _safe_float(raw_array)
        lat_vals = _safe_float(latitudes)
        lon_vals = _safe_float(longitudes)

        valid_mask = np.isfinite(feature_vals) & np.isfinite(lat_vals) & np.isfinite(lon_vals)
        if valid_mask.sum() < window:
            print("[check] Not enough valid samples to render diagnostic plots.")
            return

        feat_valid = feature_vals[valid_mask]
        lat_valid = lat_vals[valid_mask]
        lon_valid = lon_vals[valid_mask]

        lat_keys, lat_inverse = np.unique(np.round(lat_valid, 6), return_inverse=True)
        lon_keys, lon_inverse = np.unique(np.round(lon_valid, 6), return_inverse=True)

        grid_shape = (lat_keys.size, lon_keys.size)
        grid_sum = np.zeros(grid_shape, dtype=np.float64)
        grid_count = np.zeros(grid_shape, dtype=np.int32)
        lat_sum = np.zeros(lat_keys.size, dtype=np.float64)
        lon_sum = np.zeros(lon_keys.size, dtype=np.float64)
        lat_count = np.zeros(lat_keys.size, dtype=np.int32)
        lon_count = np.zeros(lon_keys.size, dtype=np.int32)

        for idx, value in enumerate(feat_valid):
            r = lat_inverse[idx]
            c = lon_inverse[idx]
            grid_sum[r, c] += value
            grid_count[r, c] += 1
            lat_sum[r] += lat_valid[idx]
            lon_sum[c] += lon_valid[idx]
            lat_count[r] += 1
            lon_count[c] += 1

        grid = np.divide(
            grid_sum,
            grid_count,
            out=np.full_like(grid_sum, np.nan, dtype=np.float32),
            where=grid_count > 0,
        ).astype(np.float32, copy=False)

        lat_axis = np.divide(lat_sum, lat_count, out=lat_keys.astype(np.float64), where=lat_count > 0)
        lon_axis = np.divide(lon_sum, lon_count, out=lon_keys.astype(np.float64), where=lon_count > 0)
        candidate_mask = grid_count > 0
    else:
        srcs = list(getattr(stack, "srcs", []))
        if not srcs:
            print("[check] No raster sources available; skipping image preprocessing diagnostic plot.")
            return

        band_idx = 0
        feature_key = (feature_name or "").strip()
        band_paths = list(getattr(stack, "band_paths", []))
        if feature_key:
            try:
                idx = int(feature_key)
                if -len(srcs) <= idx < len(srcs):
                    band_idx = idx % len(srcs)
            except ValueError:
                lowered = feature_key.lower()
                match = None
                for idx, path in enumerate(band_paths):
                    stem = Path(path).stem.lower()
                    if lowered in stem:
                        match = idx
                        break
                if match is None:
                    for idx, path in enumerate(band_paths):
                        name = Path(path).name.lower()
                        if lowered in name:
                            match = idx
                            break
                if match is not None:
                    band_idx = match
                else:
                    print(f"[check] Feature '{feature_key}' not matched to raster band; defaulting to band 0.")

        try:
            reader = srcs[band_idx]
        except IndexError:
            print(f"[check] Raster band index {band_idx} is out of range; skipping diagnostics.")
            return

        try:
            band = reader.read(1, masked=True)
        except Exception as exc:
            print(f"[check] Unable to read raster band {band_idx}: {exc}")
            return

        if np.ma.isMaskedArray(band):
            candidate_mask = np.asarray(~band.mask, dtype=bool)
            grid = band.filled(np.nan).astype(np.float32, copy=False)
        else:
            grid = np.asarray(band, dtype=np.float32)
            candidate_mask = np.ones_like(grid, dtype=bool)

        nodata = getattr(reader, "nodata", None)
        if nodata is not None:
            invalid = np.isclose(grid, nodata, equal_nan=True)
            grid[invalid] = np.nan
            candidate_mask &= ~invalid

        candidate_mask &= np.isfinite(grid)

        transform = getattr(stack, "transform", None)
        if transform is not None:
            try:
                from rasterio.transform import xy

                rows = np.arange(grid.shape[0], dtype=np.int64)
                cols = np.arange(grid.shape[1], dtype=np.int64)
                lon_axis, _ = xy(transform, np.zeros_like(cols), cols, offset="center")
                _, lat_axis = xy(transform, rows, np.zeros_like(rows), offset="center")
                lon_axis = np.asarray(lon_axis, dtype=np.float64)
                lat_axis = np.asarray(lat_axis, dtype=np.float64)
            except Exception:
                lon_axis = np.arange(grid.shape[1], dtype=np.float64)
                lat_axis = np.arange(grid.shape[0], dtype=np.float64)
        else:
            lon_axis = np.arange(grid.shape[1], dtype=np.float64)
            lat_axis = np.arange(grid.shape[0], dtype=np.float64)

    if grid is None or candidate_mask is None or lat_axis is None or lon_axis is None:
        return

    finite_vals = grid[np.isfinite(grid)]
    if finite_vals.size == 0:
        print("[check] All grid cells are NaN for feature diagnostics; skipping plot.")
        return

    vmin = float(np.nanpercentile(finite_vals, 2))
    vmax = float(np.nanpercentile(finite_vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))

    total_valid = int(candidate_mask.sum())
    if total_valid == 0:
        print("[check] No populated grid cells available for patch sampling; skipping plot.")
        return

    rng = np.random.default_rng(rng_seed)
    max_samples = min(6, total_valid)

    if total_valid <= 1_000_000:
        candidate_centers = np.argwhere(candidate_mask)
        sampled_idx = rng.choice(candidate_centers.shape[0], size=max_samples, replace=False)
        selected = candidate_centers[sampled_idx]
    else:
        sampled: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        rows, cols = candidate_mask.shape
        max_attempts = max(1000, max_samples * 200)
        while len(sampled) < max_samples and attempts < max_attempts:
            attempts += 1
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            key = (r, c)
            if not candidate_mask[r, c] or key in seen:
                continue
            sampled.append(key)
            seen.add(key)
        if len(sampled) < max_samples:
            fallback = np.argwhere(candidate_mask)
            if fallback.size > 0:
                remaining = max_samples - len(sampled)
                fallback_idx = rng.choice(fallback.shape[0], size=min(remaining, fallback.shape[0]), replace=False)
                for idx in np.atleast_1d(fallback_idx):
                    r, c = fallback[int(idx)]
                    key = (int(r), int(c))
                    if key in seen:
                        continue
                    sampled.append(key)
                    if len(sampled) == max_samples:
                        break
        selected = np.asarray(sampled, dtype=np.int64)

    num_patches = selected.shape[0]
    if num_patches == 0:
        print("[check] Unable to select valid patches for diagnostics; skipping plot.")
        return

    patches = []
    lat_ranges = []
    lon_ranges = []
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    centers: list[tuple[int, int]] = []
    half = window // 2
    n_lat, n_lon = grid.shape
    for r, c in selected:
        row_idx = np.clip(np.arange(r - half, r - half + window), 0, n_lat - 1)
        col_idx = np.clip(np.arange(c - half, c - half + window), 0, n_lon - 1)
        patch = grid[np.ix_(row_idx, col_idx)]
        patches.append(patch)
        lat_slice = lat_axis[row_idx]
        lon_slice = lon_axis[col_idx]
        lat_ranges.append((float(np.nanmin(lat_slice)), float(np.nanmax(lat_slice))))
        lon_ranges.append((float(np.nanmin(lon_slice)), float(np.nanmax(lon_slice))))
        row_indices.append(row_idx)
        col_indices.append(col_idx)
        centers.append((int(r), int(c)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch

    fig_cols = 3
    fig_rows = int(np.ceil(num_patches / fig_cols))
    height = 4 + fig_rows * 3
    width = 6 + fig_cols * 3
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(fig_rows + 1, fig_cols, height_ratios=[4] + [1] * fig_rows)
    ax_main = fig.add_subplot(gs[0, :])

    extent = [float(np.nanmin(lon_axis)), float(np.nanmax(lon_axis)), float(np.nanmin(lat_axis)), float(np.nanmax(lat_axis))]
    base_cmap = plt.get_cmap("viridis")
    masked_cmap = base_cmap.with_extremes(bad="black")
    main_masked = grid
    if candidate_mask is not None:
        main_masked = np.ma.array(grid, mask=~candidate_mask)
    else:
        main_masked = np.ma.masked_invalid(grid)
    im = ax_main.imshow(main_masked, origin="lower", cmap=masked_cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    ax_main.set_title(f"Original '{feature_name}' values")
    ax_main.set_xlabel("Longitude" if getattr(stack, "is_table", False) else "X coordinate")
    ax_main.set_ylabel("Latitude" if getattr(stack, "is_table", False) else "Y coordinate")
    fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label=feature_name)

    cmap = plt.get_cmap("tab10", max(num_patches, 1))
    rect_handles = []

    pixel_w = (extent[1] - extent[0]) / n_lon
    pixel_h = (extent[3] - extent[2]) / n_lat

    for idx, (center_rc) in enumerate(centers):
        r_center, c_center = center_rc
        if not np.isfinite(r_center) or not np.isfinite(c_center):
            rect_handles.append(None)
            continue
        width = window * pixel_w
        height = window * pixel_h
        lon_min = extent[0] + (c_center - half) * pixel_w
        lat_min = extent[2] + (r_center - half) * pixel_h

        lon_min = min(max(lon_min, extent[0]), extent[1] - width)
        lat_min = min(max(lat_min, extent[2]), extent[3] - height)
        color = cmap(idx % cmap.N)
        rect = Rectangle(
            (lon_min, lat_min),
            width,
            height,
            linewidth=1.8,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
        )
        ax_main.add_patch(rect)
        rect_handles.append(rect)

    patch_axes: list[plt.Axes] = []

    for idx, patch in enumerate(patches):
        row = idx // fig_cols
        col = idx % fig_cols
        ax = fig.add_subplot(gs[row + 1, col])
        row_idx = row_indices[idx]
        col_idx = col_indices[idx]
        if candidate_mask is not None:
            patch_valid = candidate_mask[np.ix_(row_idx, col_idx)]
            masked_patch = np.ma.array(patch, mask=~patch_valid)
        else:
            masked_patch = np.ma.masked_invalid(patch)
        lon_min, lon_max = lon_ranges[idx]
        lat_min, lat_max = lat_ranges[idx]
        extent_patch = [lon_min, lon_max, lat_min, lat_max]
        ax.imshow(masked_patch, origin="lower", cmap=masked_cmap, vmin=vmin, vmax=vmax, extent=extent_patch, aspect="auto")
        ax.set_title(f"Window {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        color = cmap(idx % cmap.N)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        patch_axes.append(ax)

    for empty_idx in range(num_patches, fig_rows * fig_cols):
        row = empty_idx // fig_cols
        col = empty_idx % fig_cols
        fig.add_subplot(gs[row + 1, col]).axis("off")

    for idx, ax in enumerate(patch_axes):
        if idx >= len(rect_handles):
            continue
        rect = rect_handles[idx]
        if rect is None:
            continue
        color = cmap(idx % cmap.N)
        rect_x, rect_y = rect.get_xy()
        patch_center = (rect_x + rect.get_width() * 0.5, rect_y + rect.get_height() * 0.5)
        conn = ConnectionPatch(
            xyA=(0.5, 1.0),
            coordsA=ax.transAxes,
            xyB=patch_center,
            coordsB=ax_main.transData,
            color=color,
            linewidth=1.2,
            alpha=0.7,
        )
        fig.add_artist(conn)

    out_dir.mkdir(parents=True, exist_ok=True)
    def _suffix() -> str:
        base = feature_name.replace(' ', '_') if feature_name else "feature"
        if variant_tag:
            safe_variant = variant_tag.replace(' ', '_')
            return f"{base}_{safe_variant}"
        return base

    out_path = out_dir / f"preproc_check_{_suffix()}.png"
    print("out_path: ", out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[check] Saved preprocessing diagnostic plot to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', type=str, help='glob pattern to bands (e.g., /data/*.tif)')
    ap.add_argument('--stac-root', type=str, help='Path to STAC collection root (table assets)')
    ap.add_argument('--stac-table', type=str, help='Direct path to a STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC tables')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC tables')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC tables')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--patch', type=int, default=12, help='Patch size for MAE patch embedding (must divide window)')
    ap.add_argument('--window', type=int, default=144, help='Square crop size (pixels) for SSL inputs')
    ap.add_argument('--mask-ratio', type=float, default=0.75, help='Fraction of patches masked during MAE pretraining')
    ap.add_argument('--encoder-dim', type=int, default=512, help='Embedding dimension for MAE encoder tokens')
    ap.add_argument('--decoder-dim', type=int, default=384, help='Embedding dimension for MAE decoder tokens')
    ap.add_argument('--encoder-num-heads', type=int, default=8, help='Number of attention heads in encoders of MAE transformer blocks')
    ap.add_argument('--decoder-num-heads', type=int, default=8, help='Number of attention heads in decoders of MAE transformer blocks')
    
    ap.add_argument('--encoder-depth', type=int, default=6, help='Number of transformer blocks in the encoder')
    ap.add_argument('--decoder-depth', type=int, default=3, help='Number of transformer blocks in the decoder')

    ap.add_argument('--mlp-ratio', type=float, default=4.0, help='Expansion ratio for MLP layers in encoder')
    ap.add_argument('--mlp-ratio-decoder', type=float, default=4.0, help='Expansion ratio for MLP layers in decoder')

    ap.add_argument('--mask-scope', choices=['pixel', 'patch'], default='patch', help='Choose masking granularity for debug previews (pixel zeros individual pixels, patch zeros whole patches)')
    ap.add_argument('--no-norm-per-patch', dest='norm_per_patch', action='store_false',
                    help='Disable per-patch normalization when computing the reconstruction loss')
    ap.add_argument('--norm-per-patch', dest='norm_per_patch', action='store_true',
                    help='Enable per-patch normalization when computing the reconstruction loss')
    ap.add_argument('--ssim', dest='use_ssim', action='store_true',
                    help='Enable SSIM metric computation during training (disabled by default)')
    ap.add_argument('--no-ssim', dest='use_ssim', action='store_false',
                    help='Disable SSIM metric computation during training (default)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--optimizer', choices=['adamw', 'adam'], default='adamw')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--preview-samples', type=int, default=0, help='If >0, create reconstruction previews for this many samples')
    ap.add_argument('--check-image-preproc', action='store_true', help='Render diagnostic plots for patch windowing')
    ap.add_argument('--check-feature', type=str,
                    default='Gravity_Bouguer_HGM_Worms_Proximity', help='Feature name to visualize when running image preprocessing diagnostics',
                    )
    ap.add_argument('--skip-nan', dest='skip_nan', action='store_true',
                    help='Resample until patches contain only finite values (default behaviour)')
    ap.add_argument('--allow-nan', dest='skip_nan', action='store_false',
                    help='Allow NaNs in sampled patches (disables resampling)')
    ap.set_defaults(skip_nan=True)
    ap.set_defaults(norm_per_patch=True, use_ssim=False)
    ap.add_argument('--skip-nan-attempts', type=int, default=64,
                    help='Maximum resampling attempts when skipping NaN-containing patches')
    ap.add_argument('--button-inference', action='store_true',
                    help='Skip training and generate previews from existing checkpoints')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide exactly one of --bands, --stac-root, or --stac-table')

    metadata: Optional[Dict[str, Any]] = None
    feature_entries: Optional[Dict[str, Any]] = None
    stac_root_path: Optional[Path] = None
    used_feature_list: List[str] = []

    if args.stac_table:
        stac_path = Path(args.stac_table)
        stack = StacTableStack(
            stac_path,
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        window_size = args.window
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")

        if getattr(stack, "is_table", False):
            feature_meta = getattr(stack, "feature_metadata", None)
            if feature_meta:
                numeric_idx: List[int] = []
                dropped_names: List[str] = []
                for idx, meta in enumerate(feature_meta):
                    category = meta.get("category") if isinstance(meta, dict) else None
                    if category:
                        dropped_names.append(
                            stack.feature_columns[idx] if idx < len(stack.feature_columns) else f"feature_{idx}"
                        )
                    else:
                        numeric_idx.append(idx)

                if dropped_names:
                    if not numeric_idx:
                        raise RuntimeError(
                            "All STAC features resolved to categorical indicators; cannot proceed with numeric-only training."
                        )

                    stack.feature_columns = [stack.feature_columns[i] for i in numeric_idx]
                    stack.feature_metadata = [feature_meta[i] for i in numeric_idx]
                    if hasattr(stack, "features"):
                        stack.features = stack.features[:, numeric_idx]
                    if hasattr(stack, "_normalized"):
                        stack._normalized = stack._normalized[:, numeric_idx]
                    if hasattr(stack, "feature_mean"):
                        stack.feature_mean = stack.feature_mean[numeric_idx]
                    if hasattr(stack, "feature_std"):
                        stack.feature_std = stack.feature_std[numeric_idx]
                    stack.count = len(numeric_idx)
                    print(
                        f"[IMPORTANT] Using NUMERIC features only TEMPORARILY; dropped {len(dropped_names)} categorical channel(s)."
                    )
                    print("[info] Dropped categorical features:", ", ".join(dropped_names))
        used_feature_list = list(getattr(stack, "feature_columns", []) or [])
    elif args.stac_root:
        stac_root_path = Path(args.stac_root).resolve()

        def _filter_numeric_stack(stack_obj: GeoStack) -> GeoStack:
            feature_meta = getattr(stack_obj, "feature_metadata", None)
            feature_cols = getattr(stack_obj, "feature_columns", None)
            if not feature_meta or not feature_cols:
                return stack_obj
            numeric_idx: List[int] = []
            dropped_names: List[str] = []
            for idx, meta in enumerate(feature_meta):
                category = meta.get("category") if isinstance(meta, dict) else None
                if category:
                    dropped_names.append(feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}")
                else:
                    numeric_idx.append(idx)

            if dropped_names:
                if not numeric_idx:
                    raise RuntimeError(
                        "All features resolved to categorical indicators; cannot proceed with numeric-only training."
                    )

                stack_obj.feature_columns = [feature_cols[i] for i in numeric_idx]
                stack_obj.feature_metadata = [feature_meta[i] for i in numeric_idx]
                if hasattr(stack_obj, "features"):
                    stack_obj.features = stack_obj.features[:, numeric_idx]
                if hasattr(stack_obj, "_normalized"):
                    stack_obj._normalized = stack_obj._normalized[:, numeric_idx]
                if hasattr(stack_obj, "feature_mean"):
                    stack_obj.feature_mean = stack_obj.feature_mean[numeric_idx]
                if hasattr(stack_obj, "feature_std"):
                    stack_obj.feature_std = stack_obj.feature_std[numeric_idx]
                stack_obj.count = len(numeric_idx)
                print(
                    f"[IMPORTANT] Using NUMERIC features only TEMPORARILY; dropped {len(dropped_names)} categorical channel(s)."
                )
                print("[info] Dropped categorical features:", ", ".join(dropped_names))

            return stack_obj

        stack, _, feature_names_resolved, metadata, feature_entries, _ = load_stac_rasters(
            stac_root_path,
            args.features,
            numeric_filter=_filter_numeric_stack,
        )
        window_size = args.window
        used_feature_list = list(feature_names_resolved or [])
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        window_size = args.window
        if getattr(stack, "band_paths", None):
            used_feature_list = [Path(p).stem for p in stack.band_paths]
        else:
            used_feature_list = []

    if not used_feature_list:
        feature_candidates = getattr(stack, "feature_columns", None)
        if feature_candidates:
            used_feature_list = list(feature_candidates)
        elif getattr(stack, "band_paths", None):
            used_feature_list = [Path(p).stem for p in stack.band_paths]
        else:
            used_feature_list = []
    used_feature_list = list(dict.fromkeys(str(name) for name in used_feature_list))

    requested_out = Path(args.out)
    if stac_root_path is not None:
        output_base = (stac_root_path / "work").resolve()
        relative_part = requested_out
        if requested_out.is_absolute():
            relative_part = Path(requested_out.name)
        relative_str = str(relative_part).strip()
        if not relative_str or relative_str in {".", "./"}:
            output_dir = output_base
        else:
            output_dir = (output_base / relative_part).resolve()
        print(f"[info] Output artifacts will be stored under {output_dir}")
    else:
        output_dir = requested_out.resolve()

    if args.window <= 0:
        ap.error('--window must be positive')
    if args.patch <= 0:
        ap.error('--patch must be positive')
    if args.window % args.patch != 0:
        ap.error('--window must be divisible by --patch to form an integer patch grid')

    preview_dir = output_dir / 'previews'

    if args.check_image_preproc:
        if metadata and feature_entries:
            if stac_root_path is None:
                raise RuntimeError("Internal error: stac_root_path is undefined for metadata-driven preprocessing checks.")
            for feature_name, entry in feature_entries.items():
                tif_records = entry.get("tifs", []) or []
                for record in tif_records:
                    rel_path = record.get("path")
                    if not rel_path:
                        continue
                    region = str(record.get("region") or "GLOBAL").upper()
                    tif_path = (stac_root_path / rel_path).resolve()
                    tmp_stack = GeoStack([str(tif_path)])
                    tmp_stack.feature_columns = [feature_name]
                    tmp_stack.feature_metadata = [{"regions": [region]}]
                    _maybe_check_image_preproc(
                        tmp_stack,
                        feature_name,
                        window_size,
                        preview_dir,
                        variant_tag=region,
                    )
        elif isinstance(stack, MultiRegionStack):
            for region_name, region_stack in stack.iter_region_stacks():
                _maybe_check_image_preproc(
                    region_stack,
                    args.check_feature,
                    window_size,
                    preview_dir,
                    variant_tag=region_name,
                )
        else:
            _maybe_check_image_preproc(stack, args.check_feature, window_size, preview_dir)

    ds = SSLDataset(
        stack,
        window=window_size,
        skip_nan=args.skip_nan,
        max_resample_attempts=args.skip_nan_attempts,
    )

    # TODO: Make it generalized for future development... How so?
    worker_count = 8
    if getattr(stack, "kind", None) == "raster":
        worker_count = 0
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=not args.button_inference,
        num_workers=worker_count,
        pin_memory=True,
    )

    model = MAEViT(
        in_chans=stack.count,
        mask_ratio=args.mask_ratio,
        image_size=window_size,
        patch_size=args.patch,
        depth=args.encoder_depth,
        dec_depth=args.decoder_depth,
        embed_dim=args.encoder_dim,
        dec_dim=args.decoder_dim,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        mlp_ratio_dec=args.mlp_ratio_decoder,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / 'ssl_history.json'
    checkpoint_path = output_dir / 'mae_encoder.pth'

    if not args.button_inference:
        args_snapshot = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                args_snapshot[key] = str(value)
            elif isinstance(value, (list, tuple)):
                args_snapshot[key] = [str(item) if isinstance(item, Path) else item for item in value]
            else:
                args_snapshot[key] = value
        args_snapshot["features"] = list(used_feature_list)
        args_path = output_dir / 'training_args_1_pretrain.json'
        args_path.write_text(json.dumps(args_snapshot, indent=2), encoding='utf-8')
        if stac_root_path is not None:
            _persist_training_args_to_metadata(stac_root_path, args_snapshot, output_dir, used_feature_list)

        checkpoint_percentages = (0.25, 0.5, 0.75, 1.0)
        checkpoint_epochs = sorted(
            {
                min(args.epochs, max(1, math.ceil(args.epochs * pct)))
                for pct in checkpoint_percentages
            }
        )

        def _save_checkpoint(epoch: int, current_model: torch.nn.Module, history_entries: List[Dict[str, float]]):
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            torch.save(current_model.state_dict(), checkpoint_path)
            if history_path.exists():
                history_path.unlink()
            history_path.write_text(json.dumps(history_entries, indent=2), encoding='utf-8')
            print(
                f"[info] Saved checkpoint artifacts at epoch {epoch} to {checkpoint_path.parent}"
            )

        model, history = train_ssl(
            model,
            dl,
            epochs=args.epochs,
            lr=args.lr,
            optimizer=args.optimizer,
            preview_samples=args.preview_samples,
            preview_dir=preview_dir if args.preview_samples > 0 else None,
            feature_names=getattr(stack, 'feature_columns', None),
            feature_metadata=getattr(stack, 'feature_metadata', None),
            mask_scope=args.mask_scope,
            norm_per_patch=args.norm_per_patch,
            use_ssim=args.use_ssim,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_callback=_save_checkpoint,
        )
        if args.epochs not in checkpoint_epochs:
            _save_checkpoint(args.epochs, model, history)
        print(f"Training history available at {history_path}")
    else:
        state_source: Optional[Path] = None
        if checkpoint_path.exists():
            state_source = checkpoint_path
        elif Path(args.encoder).exists():
            state_source = Path(args.encoder)
        if state_source is None or not state_source.exists():
            raise FileNotFoundError(
                "button-inference requested but no checkpoint found at output directory "
                f"({checkpoint_path}) or encoder path ({args.encoder})."
            )
        model.load_state_dict(torch.load(state_source, map_location='cpu'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        _generate_preview_inference(
            model,
            dl,
            args.preview_samples,
            preview_dir,
            getattr(stack, 'feature_columns', None),
            getattr(stack, 'feature_metadata', None),
            args.mask_scope,
        )
        stitch_mask_ratio = float(getattr(model, 'mask_ratio', args.mask_ratio))
        _generate_stitched_maps(
            model,
            stack,
            window_size,
            args.mask_scope,
            stitch_mask_ratio,
            max(1, args.batch),
            preview_dir,
        )
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding='utf-8'))
                if history:
                    print(
                        f"[info] Loaded existing training history with {len(history)} entries from {history_path}"
                    )
            except Exception as exc:
                print(f"[warn] Failed to read existing history at {history_path}: {exc}")
