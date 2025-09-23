# src/gfm4mpm/data/geo_stack.py
from __future__ import annotations
import os, json
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

    def read_patch(self, row: int, col: int, size: int, nodata_val: Optional[float]=None) -> np.ndarray:
        """Return (C, H, W) patch centered at (row, col)."""
        half = size // 2
        r0, c0 = max(row-half, 0), max(col-half, 0)
        r1, c1 = min(r0+size, self.height), min(c0+size, self.width)
        window = Window(c0, r0, c1-c0, r1-r0)
        patch = np.stack([s.read(1, window=window) for s in self.srcs], axis=0).astype(np.float32)
        # pad if near edges
        pad_h, pad_w = size - patch.shape[1], size - patch.shape[2]
        if pad_h or pad_w:
            patch = np.pad(patch, ((0,0),(0,pad_h),(0,pad_w)), mode='edge')
        if nodata_val is not None:
            patch[:, patch[0] == nodata_val] = 0.0
        # per-band z-score
        mean = patch.mean(axis=(1,2), keepdims=True)
        std  = patch.std(axis=(1,2), keepdims=True) + 1e-6
        patch = (patch - mean) / std
        return patch

    def grid_centers(self, stride: int) -> List[Tuple[int,int]]:
        rows = range(stride//2, self.height, stride)
        cols = range(stride//2, self.width, stride)
        return [(r,c) for r in rows for c in cols]

def load_deposit_pixels(geojson_path: str, stack: GeoStack) -> List[Tuple[int,int]]:
    """Convert deposit points (class=1) into pixel indices (row, col)."""
    pts = []
    with open(geojson_path, 'r') as f:
        gj = json.load(f)
    for feat in gj['features']:
        geom = shape(feat['geometry'])
        x, y = geom.x, geom.y
        row_, col_ = rowcol(stack.transform, x, y)
        if 0 <= row_ < stack.height and 0 <= col_ < stack.width:
            pts.append((int(row_), int(col_)))
    return pts
