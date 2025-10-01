# scripts/pretrain_ssl.py
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.gfm4mpm.data.geo_stack import GeoStack
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.training.train_ssl import train_ssl

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


class MultiRegionStack:
    """Round-robin sampler that stitches multiple regional GeoStacks."""

    def __init__(self, feature_names: Sequence[str], region_to_paths: Dict[str, Sequence[str]]):
        if not region_to_paths:
            raise ValueError("region_to_paths must not be empty")

        self.feature_columns: List[str] = list(feature_names)
        self._regions: List[Tuple[str, GeoStack]] = []
        for region, paths in region_to_paths.items():
            path_list = list(paths)
            if len(path_list) != len(self.feature_columns):
                raise ValueError(
                    f"Region '{region}' expected {len(self.feature_columns)} feature rasters, "
                    f"found {len(path_list)}."
                )
            stack = GeoStack(path_list)
            self._regions.append((region, stack))
            print(f"[info] Region '{region}' prepared with {len(path_list)} feature map(s).")

        if not self._regions:
            raise ValueError("No usable regional stacks were created.")

        self.count = self._regions[0][1].count
        self.kind = "raster"
        self.is_table = False
        self.feature_metadata = [
            {"regions": [region for region, _ in self._regions]}
            for _ in self.feature_columns
        ]
        self._cycle_index = int(np.random.default_rng().integers(0, len(self._regions)))

    @property
    def regions(self) -> List[str]:
        return [region for region, _ in self._regions]

    def iter_region_stacks(self) -> Iterable[Tuple[str, GeoStack]]:
        return tuple(self._regions)

    def sample_patch(
        self,
        window: int,
        rng: np.random.Generator,
        *,
        skip_nan: bool = True,
        max_attempts: int = 32,
    ) -> np.ndarray:
        attempts = max(1, int(max_attempts))
        last_patch: Optional[np.ndarray] = None
        last_mask: Optional[np.ndarray] = None
        for _ in range(attempts):
            region_idx = self._cycle_index
            self._cycle_index = (self._cycle_index + 1) % len(self._regions)
            region_name, region_stack = self._regions[region_idx]
            r, c = region_stack.random_coord(window, rng)
            if hasattr(region_stack, "read_patch_with_mask"):
                patch, mask = region_stack.read_patch_with_mask(r, c, window)
                mask_no_feature = mask.astype(bool, copy=False)
            else:
                patch = region_stack.read_patch(r, c, window)
                mask_valid = np.isfinite(patch)
                if mask_valid.ndim == 3:
                    mask_no_feature = ~mask_valid.any(axis=0)
                else:
                    mask_no_feature = ~mask_valid.astype(bool)

            if mask_no_feature.all() or np.allclose(patch, 0.0):
                continue
            if skip_nan and not np.isfinite(patch).all():
                continue
            last_patch = patch
            last_mask = mask_no_feature
            return patch, mask_no_feature

        if skip_nan:
            raise RuntimeError(
                "Failed to sample a finite patch from any regional raster; "
                "consider disabling --skip-nan or inspecting raster coverage."
            )

        if last_patch is not None:
            fallback_mask = last_mask if last_mask is not None else np.zeros(last_patch.shape[-2:], dtype=bool)
            return last_patch, fallback_mask

        raise RuntimeError(
            "Unable to sample a patch from regional rasters; verify raster coverage and metadata."
        )

        raise RuntimeError(
            "sample_patch could not retrieve any data; check raster inputs and metadata."
        )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    return slug or "feature"


def _load_training_metadata(stac_root: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        stac_root / "training_metadata.json",
        stac_root / "assetization" / "training_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fh:
                try:
                    return json.load(fh)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse training metadata JSON at {candidate}: {exc}") from exc
    return None


def _find_feature_entry(entries: Dict[str, Any], feature_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if feature_name in entries:
        return feature_name, entries[feature_name]
    lowered = feature_name.lower()
    for key, value in entries.items():
        if key.lower() == lowered:
            return key, value
    return None


def _collect_stac_raster_paths(
    stac_root: Path,
    features: Optional[Sequence[str]],
) -> Tuple[List[str], Dict[str, List[str]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    metadata = _load_training_metadata(stac_root)
    if metadata:
        features_section = metadata.get("features", {})
        entries: Dict[str, Any] = features_section.get("entries", {})
        total_features = features_section.get("total_features", len(entries))
        print(f"[info] training_metadata.json reports {total_features} feature(s).")

        if entries:
            selected_pairs: List[Tuple[str, Dict[str, Any]]] = []
            if features:
                for feature in features:
                    lookup = _find_feature_entry(entries, feature)
                    if lookup is None:
                        raise FileNotFoundError(
                            f"Feature '{feature}' not present in training_metadata.json; available keys: {', '.join(entries.keys())}"
                        )
                    selected_pairs.append(lookup)
            else:
                selected_pairs = sorted(entries.items(), key=lambda kv: kv[0].lower())
                summary_names = ", ".join(name for name, _ in selected_pairs)
                print(
                    f"[info] No explicit --features provided; using all {len(selected_pairs)} features"
                    f" from metadata: {summary_names}"
                )

            feature_names = [name for name, _ in selected_pairs]
            region_to_feature_paths: Dict[str, Dict[str, str]] = defaultdict(dict)

            for feature_name, entry in selected_pairs:
                tif_records = entry.get("tifs", [])
                if not tif_records:
                    raise FileNotFoundError(
                        f"No TIFF records listed for feature '{feature_name}' in training_metadata.json"
                    )

                print(f"[info] Feature '{feature_name}' has {len(tif_records)} map(s):")
                for record in tif_records:
                    rel_path = record.get("path")
                    if not rel_path:
                        continue
                    region = str(record.get("region") or "GLOBAL").upper()
                    tif_path = (stac_root / rel_path).resolve()
                    region_to_feature_paths[region][feature_name] = str(tif_path)
                    print(f"        - {tif_path} (region={region})")

            usable_regions: Dict[str, List[str]] = {}
            for region, feature_path_map in region_to_feature_paths.items():
                missing = [name for name in feature_names if name not in feature_path_map]
                if missing:
                    print(
                        f"[warn] Region '{region}' is missing {len(missing)} feature(s) ({', '.join(missing)}); skipping this region."
                    )
                    continue
                ordered_paths = [feature_path_map[name] for name in feature_names]
                usable_regions[region] = ordered_paths

            if not usable_regions:
                raise FileNotFoundError(
                    "No regions contained complete raster coverage for the requested features."
                )

            selected_entries = {name: entry for name, entry in selected_pairs}
            return feature_names, usable_regions, metadata, selected_entries

        print("[warn] training_metadata.json contained no feature entries; falling back to filesystem scan.")
        metadata = None

    # Fallback: file system scan
    assets_dir = (stac_root / "assets" / "rasters").resolve()
    if not assets_dir.exists():
        raise FileNotFoundError(f"No raster assets directory found at {assets_dir}")

    candidates = [
        path for path in assets_dir.rglob("*_cog.tif") if "thumbnails" not in {p.lower() for p in path.parts}
    ]
    if not candidates:
        candidates = [
            path for path in assets_dir.rglob("*.tif") if "thumbnails" not in {p.lower() for p in path.parts}
        ]

    if not candidates:
        raise FileNotFoundError(f"No raster GeoTIFFs found under {assets_dir}")

    if features:
        ordered: list[Path] = []
        remaining = candidates.copy()
        for feature in features:
            slug = _slugify(feature)
            match = None
            for path in remaining:
                stem_lower = path.stem.lower()
                stem_lower = stem_lower[:-4] if stem_lower.endswith("_cog") else stem_lower
                if slug in stem_lower:
                    match = path
                    break
            if match is None:
                raise FileNotFoundError(f"No raster asset matching feature '{feature}' (slug '{slug}') under {assets_dir}")
            ordered.append(match)
            remaining.remove(match)
        candidates = ordered
        feature_names = list(features)
    else:
        candidates = sorted(candidates)
        feature_names = [Path(p).stem for p in candidates]

    region_paths = {"GLOBAL": [str(path.resolve()) for path in candidates]}
    return feature_names, region_paths, metadata, None


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
    ap.add_argument('--patch', type=int, default=224, help='Patch size for MAE patch embedding (must divide window)')
    ap.add_argument('--window', type=int, default=16, help='Square crop size (pixels) for SSL inputs')
    ap.add_argument('--mask-ratio', type=float, default=0.75, help='Fraction of patches masked during MAE pretraining')
    ap.add_argument('--encoder-dim', type=int, default=768, help='Embedding dimension for MAE encoder tokens')
    ap.add_argument('--decoder-dim', type=int, default=512, help='Embedding dimension for MAE decoder tokens')
    ap.add_argument('--encoder-num-heads', type=int, default=12, help='Number of attention heads in encoders of MAE transformer blocks')
    ap.add_argument('--decoder-num-heads', type=int, default=16, help='Number of attention heads in decoders of MAE transformer blocks')
    
    ap.add_argument('--encoder-depth', type=int, default=12, help='Number of transformer blocks in the encoder')
    ap.add_argument('--decoder-depth', type=int, default=8, help='Number of transformer blocks in the decoder')

    ap.add_argument('--mlp-ratio', type=float, default=4.0, help='Expansion ratio for MLP layers in encoder')
    ap.add_argument('--mlp-ratio-decoder', type=float, default=2.0, help='Expansion ratio for MLP layers in decoder')

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
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide exactly one of --bands, --stac-root, or --stac-table')

    metadata: Optional[Dict[str, Any]] = None
    feature_entries: Optional[Dict[str, Any]] = None
    stac_root_path: Optional[Path] = None

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

        # Temporarily restrict to numeric-only features to avoid categorical indicators
        if getattr(stack, "is_table", False):
            feature_meta = getattr(stack, "feature_metadata", None)
            if feature_meta:
                numeric_idx: list[int] = []
                dropped_names: list[str] = []
                for idx, meta in enumerate(feature_meta):
                    category = None
                    if isinstance(meta, dict):
                        category = meta.get("category")
                    if category:
                        dropped_names.append(stack.feature_columns[idx] if idx < len(stack.feature_columns) else f"feature_{idx}")
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
    elif args.stac_root:
        stac_root_path = Path(args.stac_root).resolve()
        (
            feature_names,
            region_paths,
            metadata,
            feature_entries,
        ) = _collect_stac_raster_paths(stac_root_path, args.features)

        def _filter_numeric_stack(stack_obj):
            feature_meta = getattr(stack_obj, "feature_metadata", None)
            feature_cols = getattr(stack_obj, "feature_columns", None)
            if not feature_meta or not feature_cols:
                return stack_obj
            numeric_idx: List[int] = []
            dropped_names: List[str] = []
            for idx, meta in enumerate(feature_meta):
                category = None
                if isinstance(meta, dict):
                    category = meta.get("category")
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

        if len(region_paths) == 1:
            region_name, paths = next(iter(region_paths.items()))
            print(
                f"[info] Using {len(paths)} raster asset(s) from region '{region_name}' under {stac_root_path}"
            )
            stack = GeoStack(paths)
            # Attach metadata so downstream preview uses canonical feature names.
            stack.feature_columns = list(feature_names)
            stack.feature_metadata = [
                {"regions": [region_name]}
                for _ in feature_names
            ]
            stack = _filter_numeric_stack(stack)
        else:
            print(
                f"[info] Using {sum(len(v) for v in region_paths.values())} raster asset(s) "
                f"across {len(region_paths)} regions under {stac_root_path}"
            )
            stack = MultiRegionStack(feature_names, region_paths)
            for _, region_stack in stack.iter_region_stacks():
                _filter_numeric_stack(region_stack)
        window_size = args.window
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        window_size = args.window

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


    # why new GeoStack is created here?
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

    # print('[dev] stack:', stack)
    # if hasattr(stack, '__len__'):
    #     print('[dev] stack length:', len(stack))
    # print('[dev] dataset:', ds)
    # try:
    #     first_item = ds[0]
    #     print('[dev] first item type:', type(first_item))
    #     if isinstance(first_item, dict):
    #         for key, value in first_item.items():
    #             value_info = getattr(value, 'shape', None) or getattr(value, 'size', None)
    #             print(f"[dev] ds[0]['{key}'] ->", value_info if value_info is not None else type(value))
    #     elif hasattr(first_item, 'shape'):
    #         print('[dev] first item shape:', first_item.shape, first_item)
    # except Exception as exc:
    #     print('[dev] failed to inspect first item:', exc)
    # exit()


    # TODO: Make it generalized for future development... How so?
    worker_count = 8
    if getattr(stack, "kind", None) == "raster":
        worker_count = 0
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=worker_count, pin_memory=True)

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
