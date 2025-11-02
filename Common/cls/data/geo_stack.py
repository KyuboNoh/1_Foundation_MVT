# src/gfm4mpm/data/geo_stack.py
from __future__ import annotations
import os, json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from shapely.geometry import shape

@dataclass
class GeoStack:
    band_paths: List[str]

    def __post_init__(self):
        assert len(self.band_paths) > 0, "No band paths provided"
        self.srcs = [rasterio.open(p) for p in self.band_paths]
        ref = self.srcs[0]
        self.height, self.width = ref.height, ref.width
        self.transform = ref.transform
        self.crs = ref.crs
        self.count = len(self.srcs)
        self.is_table = False
        self.kind = "raster"
        self.band_mean, self.band_std = self._compute_band_stats()
        self._valid_rows: Optional[np.ndarray] = None
        self._valid_cols: Optional[np.ndarray] = None
        self.boundary_mask: Optional[np.ndarray] = None
        self._base_valid_mask: Optional[np.ndarray] = None
        self._combined_valid_mask: Optional[np.ndarray] = None
        self._valid_indices: Optional[np.ndarray] = None
        try:
            mask = ref.dataset_mask()
            if mask is not None:
                mask_arr = np.asarray(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                mask_bool = mask_arr != 0
                if mask_bool.any():
                    self._base_valid_mask = mask_bool.astype(bool, copy=False)
                    self._update_combined_mask(None)
                else:
                    self._set_valid_mask(None)
            else:
                self._set_valid_mask(None)
        except Exception:
            self._set_valid_mask(None)

    # ------------------------------------------------------------------
    def _compute_band_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for src in self.srcs:
            mean, std = self._estimate_band_stats(src)
            means.append(mean)
            stds.append(std)
        mean_arr = np.asarray(means, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.asarray(stds, dtype=np.float32).reshape(-1, 1, 1)
        return mean_arr, std_arr

    def _estimate_band_stats(self, src) -> Tuple[float, float]:
        stats = None
        try:
            if hasattr(src, "stats"):
                result = src.stats(1, approx=True)
                if isinstance(result, dict):
                    stats = result
                elif isinstance(result, (list, tuple)) and result:
                    stats = result[0]
            elif hasattr(src, "statistics"):
                stats = src.statistics(1, approx=True)
        except Exception:
            stats = None

        if stats is not None:
            count = stats.get("count") if isinstance(stats, dict) else getattr(stats, "count", 0)
            mean_val = stats.get("mean") if isinstance(stats, dict) else getattr(stats, "mean", None)
            std_val = stats.get("std") if isinstance(stats, dict) else getattr(stats, "std", None)
            if count and np.isfinite(mean_val) and np.isfinite(std_val):
                mean = float(mean_val)
                std = float(std_val)
                if std < 1e-6:
                    std = 1.0
                return mean, std

        sum_vals = 0.0
        sum_sq = 0.0
        total = 0
        max_samples = 1_000_000
        for _, window in src.block_windows(1):
            band = src.read(1, window=window, masked=True)
            if np.ma.isMaskedArray(band):
                band = band.astype(np.float64, copy=False)
                arr = band.filled(np.nan)
            else:
                arr = np.asarray(band, dtype=np.float64)
            mask = np.isfinite(arr)
            if not mask.any():
                continue
            vals = arr[mask]
            sum_vals += float(vals.sum())
            sum_sq += float(np.square(vals).sum())
            total += vals.size
            if total >= max_samples:
                break

        if total == 0:
            return 0.0, 1.0

        mean = sum_vals / total
        variance = max(sum_sq / total - mean * mean, 0.0)
        std = float(math.sqrt(variance))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        return float(mean), std

    def _read_patch_arrays(
        self,
        row: int,
        col: int,
        size: int,
        nodata_val: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        half = size // 2
        r0, c0 = max(row-half, 0), max(col-half, 0)
        r1, c1 = min(r0+size, self.height), min(c0+size, self.width)
        window = Window(c0, r0, c1-c0, r1-r0)
        band_arrays = []
        band_masks = []
        for src in self.srcs:
            band = src.read(1, window=window, masked=True)
            if np.ma.isMaskedArray(band):
                band = band.astype(np.float32, copy=False)
                arr = band.filled(np.nan)
                raw_mask = band.mask
                if raw_mask is np.ma.nomask:
                    mask = np.ones_like(arr, dtype=bool)
                else:
                    mask = np.asarray(raw_mask, dtype=bool)
                    if mask.shape != arr.shape:
                        mask = np.broadcast_to(mask, arr.shape)
                    mask = ~mask
            else:
                arr = np.asarray(band, dtype=np.float32)
                nodata = src.nodata
                if nodata is not None:
                    mask = ~np.isclose(arr, nodata, equal_nan=True)
                    arr = np.where(mask, arr, np.nan)
                else:
                    mask = np.ones_like(arr, dtype=bool)
            band_arrays.append(arr)
            band_masks.append(mask)

        patch = np.stack(band_arrays, axis=0)
        mask = np.stack(band_masks, axis=0).astype(bool, copy=False)

        # pad if near edges
        pad_h, pad_w = size - patch.shape[1], size - patch.shape[2]
        if pad_h or pad_w:
            pad_config = ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w)))
            patch = np.pad(patch, pad_config, mode='constant', constant_values=np.nan)
            mask = np.pad(mask, pad_config, mode='constant', constant_values=False)

        if self._combined_valid_mask is not None:
            combined_slice = self._combined_valid_mask[r0:r1, c0:c1]
            if pad_h or pad_w:
                combined_slice = np.pad(
                    combined_slice,
                    ((0, max(0, pad_h)), (0, max(0, pad_w))),
                    mode='constant',
                    constant_values=False,
                )
            mask &= combined_slice[np.newaxis, ...]

        if nodata_val is not None:
            invalid = np.isclose(patch[0], nodata_val, equal_nan=True)
            mask[:, invalid] = False

        return patch, mask

    def read_patch(self, row: int, col: int, size: int, nodata_val: Optional[float]=None) -> np.ndarray:
        """Return (C, H, W) patch centered at (row, col)."""
        patch, mask = self._read_patch_arrays(row, col, size, nodata_val)
        patch = np.where(mask, patch, np.nan)
        patch = (patch - self.band_mean) / self.band_std
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        return patch.astype(np.float32)

    def read_patch_with_mask(
        self,
        row: int,
        col: int,
        size: int,
        nodata_val: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        patch, mask = self._read_patch_arrays(row, col, size, nodata_val)
        patch = np.where(mask, patch, np.nan)
        patch = (patch - self.band_mean) / self.band_std
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        mask_pixel_no_feature = ~mask.any(axis=0)
        return patch.astype(np.float32), mask_pixel_no_feature.astype(bool, copy=False)

    def grid_centers(self, stride: int) -> List[Tuple[int,int]]:
        rows = range(stride//2, self.height, stride)
        cols = range(stride//2, self.width, stride)
        return [(r,c) for r in rows for c in cols]

    def random_coord(self, patch: int, rng: np.random.Generator) -> Tuple[int, int]:
        half = patch // 2
        if self._valid_indices is not None and self._valid_indices.size:
            idx = int(rng.integers(0, self._valid_indices.shape[0]))
            r, c = self._valid_indices[idx]
            r = int(np.clip(r, half, max(self.height - half - 1, half)))
            c = int(np.clip(c, half, max(self.width - half - 1, half)))
            return r, c
        if (
            self._valid_rows is not None
            and self._valid_cols is not None
            and self._valid_rows.size
            and self._valid_cols.size
        ):
            for _ in range(128):
                r = int(self._valid_rows[int(rng.integers(0, self._valid_rows.size))])
                c = int(self._valid_cols[int(rng.integers(0, self._valid_cols.size))])
                if self._combined_valid_mask is None or self._combined_valid_mask[r, c]:
                    r = int(np.clip(r, half, max(self.height - half - 1, half)))
                    c = int(np.clip(c, half, max(self.width - half - 1, half)))
                    return r, c
        r = int(rng.integers(half, max(self.height - half, half + 1)))
        c = int(rng.integers(half, max(self.width - half, half + 1)))
        return r, c

    def coord_to_index(self, coord: Tuple[int, int]) -> int:
        r, c = coord
        return r * self.width + c

    # ------------------------------------------------------------------
    def _set_valid_mask(self, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            self._combined_valid_mask = None
            self._valid_rows = None
            self._valid_cols = None
            self._valid_indices = None
            return
        mask_bool = mask.astype(bool, copy=False)
        if not mask_bool.any():
            self._combined_valid_mask = None
            self._valid_rows = np.empty(0, dtype=np.int32)
            self._valid_cols = np.empty(0, dtype=np.int32)
            self._valid_indices = np.empty((0, 2), dtype=np.int32)
            return
        self._combined_valid_mask = mask_bool
        self._valid_rows = np.where(mask_bool.any(axis=1))[0]
        self._valid_cols = np.where(mask_bool.any(axis=0))[0]
        self._valid_indices = np.argwhere(mask_bool)

    def _update_combined_mask(self, boundary_mask: Optional[np.ndarray]) -> None:
        boundary_bool = boundary_mask.astype(bool, copy=False) if boundary_mask is not None else None
        if self._base_valid_mask is not None and boundary_bool is not None:
            combined = self._base_valid_mask & boundary_bool
        elif self._base_valid_mask is not None:
            combined = self._base_valid_mask
        elif boundary_bool is not None:
            combined = boundary_bool
        else:
            combined = None
        self._set_valid_mask(combined)

    def set_boundary_mask(self, mask: np.ndarray) -> None:
        if mask.shape != (self.height, self.width):
            raise ValueError("Boundary mask shape does not match raster dimensions.")
        mask_bool = mask.astype(bool, copy=False)
        self.boundary_mask = mask_bool
        if hasattr(self, "_valid_window_masks"):
            setattr(self, "_valid_window_masks", {})
        self._update_combined_mask(mask_bool)
        if self._valid_indices is None or self._valid_indices.shape[0] == 0:
            raise ValueError("Boundary mask removed all valid pixels for this stack.")

    def clear_boundary_mask(self) -> None:
        self.boundary_mask = None
        if hasattr(self, "_valid_window_masks"):
            setattr(self, "_valid_window_masks", {})
        self._update_combined_mask(None)

def load_label_pixels(geojson_path: str, stack: GeoStack) -> List[Tuple[int,int]]:
    """Convert label points (class=1) into pixel indices (row, col)."""
    pts = []
    with open(geojson_path, 'r') as f:
        gj = json.load(f)
    for feat in gj['features']:
        geom = shape(feat['geometry'])
        x, y = geom.x, geom.y
        row_, col_ = rowcol(stack.transform, x, y)
        if 0 <= row_ < stack.height and 0 <= col_ < stack.width:
            pts.append((int(row_), int(col_)))
    
    print(f'[info] Loaded {len(pts)} label points from {os.path.basename(geojson_path)}')
    return pts
