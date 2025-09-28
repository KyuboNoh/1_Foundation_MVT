#!/usr/bin/env python3
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube_temp.csv   --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --collection-id gsc-2021_temp   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate --check-raster --check-raster-features Magnetic_HGM
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube.csv   --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate   --check-raster --check-raster-features Magnetic_HGM
# python stacify_GCS_data.py   --csv C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_selected_Norm.csv   --out C:\Users\kyubo\Desktop\Research\Data\1_Stacify   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate   --check-raster --check-raster-features Magnetic_HGM
"""Convert a single GSC CSV table into a STAC catalog with tabular + raster assets."""

import argparse
import json
import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from unify.one_d import csv_to_parquet
from stacify_bc_data import (
    CHECKSUM_EXT,
    TABLE_EXT,
    add_raster_assets,
    build_catalog,
    build_collection,
    ensure_dir,
    sha256sum,
    _require_pystac,
    _install_local_schema_cache,
    _require_rasterio,
)

REGISTRY_RELATIVE = Path("unify") / "feature_registry.yaml"


def infer_columns(parquet_path: Path) -> list[Dict[str, str]]:
    table = pq.read_table(parquet_path)
    schema = table.schema
    cols: list[Dict[str, str]] = []
    for field in schema:
        logical = map_arrow_type(field.type)
        cols.append({"name": field.name, "type": logical})
    return cols


def map_arrow_type(dtype: pa.DataType) -> str:
    if pa.types.is_integer(dtype):
        return "integer"
    if pa.types.is_floating(dtype):
        return "number"
    if pa.types.is_boolean(dtype):
        return "boolean"
    if pa.types.is_timestamp(dtype):
        return "datetime"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"
    return "string"


def copy_feature_registry(collection_root: Path) -> Path:
    ensure_dir(collection_root)
    dst = collection_root / "feature_registry.yaml"
    if dst.exists():
        return dst
    src = Path(__file__).resolve().parent / REGISTRY_RELATIVE
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(
            "crs: \"EPSG:3005\"\n"
            "resolution: 100\n"
            "variables: {}\n",
            encoding="utf-8",
    )
    return dst




def _slugify_column(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    return slug or "column"

# TODO: not general normalization yet; works for GSC 2021 Zn-Pb csv file only yet.
def _normalize_column_key(name: Optional[str]) -> str:
    if not name:
        return ""
    clean = name.replace("\ufeff", "").replace("ï»¿", "").strip()
    if clean.startswith("\"") and clean.endswith("\"") and len(clean) >= 2:
        clean = clean[1:-1]
    if clean.startswith("'") and clean.endswith("'") and len(clean) >= 2:
        clean = clean[1:-1]
    return clean.strip().lower()


def _save_quicklook(array: np.ndarray, title: str, out_path: Path) -> None:
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("Matplotlib unavailable; skipping quicklook for %s (%s)", title, exc)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(array)
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(masked, origin="upper", cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def _parse_polygon_wkt(wkt: str) -> Optional[np.ndarray]:
    text = (wkt or "").strip()
    if not text or not text.upper().startswith("POLYGON"):
        return None
    start = text.find("((")
    end = text.rfind("))")
    if start == -1 or end == -1 or end <= start + 2:
        return None
    coord_text = text[start + 2 : end]
    coords: list[tuple[float, float]] = []
    for token in coord_text.split(','):
        parts = token.strip().split()
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        coords.append((lon, lat))
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return np.asarray(coords, dtype=np.float64)


def _polygon_centroid(coords: np.ndarray) -> Optional[tuple[float, float]]:
    if coords.shape[0] < 4:
        return None
    x = coords[:, 0]
    y = coords[:, 1]
    x0 = x[:-1]
    y0 = y[:-1]
    x1 = x[1:]
    y1 = y[1:]
    cross = x0 * y1 - x1 * y0
    area = cross.sum() * 0.5
    if math.isclose(area, 0.0):
        return float(np.nanmean(x0)), float(np.nanmean(y0))
    cx = ((x0 + x1) * cross).sum() / (6.0 * area)
    cy = ((y0 + y1) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def _point_in_polygon(lon: float, lat: float, polygon: np.ndarray) -> bool:
    inside = False
    for i in range(len(polygon) - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        if ((y0 > lat) != (y1 > lat)) and (
            lon < (x1 - x0) * (lat - y0) / (y1 - y0 + 1e-16) + x0
        ):
            inside = not inside
    return inside


def _coordinate_to_index(value: float, edges: np.ndarray) -> Optional[int]:
    if not math.isfinite(value):
        return None
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return int(edges.size - 2)
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    if idx < 0:
        return None
    return idx


def _parse_geometry_series(values: Sequence[str]) -> tuple[list[Optional[np.ndarray]], np.ndarray, np.ndarray]:
    polygons: list[Optional[np.ndarray]] = []
    lat_centroids = np.full(len(values), np.nan, dtype=np.float64)
    lon_centroids = np.full(len(values), np.nan, dtype=np.float64)
    for idx, text in enumerate(values):
        coords = _parse_polygon_wkt(text)
        polygons.append(coords)
        if coords is None:
            continue
        centroid = _polygon_centroid(coords)
        if centroid is None:
            continue
        lon_centroids[idx], lat_centroids[idx] = centroid
    return polygons, lat_centroids, lon_centroids


def _infer_grid_size_from_h3(
    frame: pd.DataFrame,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    resolution_column: Optional[str],
    default: int = 512,
) -> int:
    try:
        import h3
    except ImportError:
        logging.debug("h3 not available; using default grid size %d", default)
        return default

    if not resolution_column or resolution_column not in frame.columns:
        logging.debug("Resolution column missing; using default grid size %d", default)
        return default

    res_series = pd.to_numeric(frame[resolution_column], errors="coerce")
    if res_series.dropna().empty:
        logging.debug("No finite H3 resolutions; using default grid size %d", default)
        return default

    h3_res = int(res_series.dropna().mode().iat[0])
    edge_km = h3.edge_length(h3_res, unit="km")

    finite_lat = lat_vals[np.isfinite(lat_vals)]
    finite_lon = lon_vals[np.isfinite(lon_vals)]
    if finite_lat.size == 0 or finite_lon.size == 0:
        logging.debug("Insufficient coordinate coverage; using default grid size %d", default)
        return default

    lat_span = float(finite_lat.max() - finite_lat.min())
    lon_span = float(finite_lon.max() - finite_lon.min())
    if lat_span <= 0 or lon_span <= 0:
        return default

    lat_mean = float(finite_lat.mean())
    deg_per_km_lat = 1.0 / 110.574
    deg_per_km_lon = 1.0 / (111.320 * max(math.cos(math.radians(lat_mean)), 1e-6))

    pixel_height = max(edge_km * deg_per_km_lat, 1e-6)
    pixel_width = max(edge_km * deg_per_km_lon, 1e-6)

    rows = math.ceil(lat_span / pixel_height)
    cols = math.ceil(lon_span / pixel_width)
    size = int(np.clip(max(rows, cols), 16, 4096))
    logging.info(
        "Auto-selected grid size %d using H3 resolution %d (edge %.2f km)",
        size,
        h3_res,
        edge_km,
    )
    return size

def _select_grid_columns(
    schema_columns: Sequence[Dict[str, str]],
    requested: Optional[Iterable[str]],
    lat_column: Optional[str],
    lon_column: Optional[str],
    h3_column: Optional[str],
    geometry_column: Optional[str],
    resolution_column: Optional[str],
) -> List[str]:
    if requested:
        return list(dict.fromkeys(requested))
    exclude = {
        c
        for c in (lat_column, lon_column, h3_column, geometry_column, resolution_column)
        if c
    }
    numeric_kinds = {"number", "integer"}
    selected: List[str] = []
    for entry in schema_columns:
        name = entry.get("name")
        if not name or name in exclude:
            continue
        logical = (entry.get("type") or "").lower()
        if logical in numeric_kinds:
            selected.append(name)
    return selected


def _to_numpy(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _normalize_lat_lon(
    frame: pd.DataFrame,
    lat_column: Optional[str],
    lon_column: Optional[str],
    h3_column: Optional[str],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    lat_vals: Optional[np.ndarray] = None
    lon_vals: Optional[np.ndarray] = None

    if lat_column and lat_column in frame.columns:
        lat_vals = _to_numpy(frame[lat_column])
    if lon_column and lon_column in frame.columns:
        lon_vals = _to_numpy(frame[lon_column])

    if lat_vals is not None and lon_vals is not None:
        return lat_vals, lon_vals

    if not h3_column or h3_column not in frame.columns:
        return None

    try:
        import h3
    except ImportError:  # pragma: no cover - optional dependency
        logging.warning(
            "h3 library not available; cannot derive coordinates from H3 addresses."
        )
        return None

    h3_series = frame[h3_column].fillna("").astype(str).to_numpy()
    lat_vals = np.full(h3_series.shape[0], np.nan, dtype=np.float64)
    lon_vals = np.full(h3_series.shape[0], np.nan, dtype=np.float64)
    for idx, cell in enumerate(h3_series):
        if not cell:
            continue
        lat, lon = h3.h3_to_geo(cell)
        lat_vals[idx] = lat
        lon_vals[idx] = lon
    return lat_vals, lon_vals




def _generate_raster_assets(
    parquet_path: Path,
    collection,
    assets_dir: Path,
    lat_column: Optional[str],
    lon_column: Optional[str],
    h3_column: Optional[str],
    geometry_column: Optional[str],
    resolution_column: Optional[str],
    value_columns: Sequence[str],
    grid_size: int,
    check_raster: bool,
    check_features: Optional[Sequence[str]],
) -> tuple[List[Dict[str, Path]], Path]:
    assetization_dir = assets_dir / "assetization"
    ensure_dir(assetization_dir)

    if not value_columns:
        logging.info("No numeric columns selected for rasterization; skipping raster assets.")
        return [], assetization_dir

    try:
        schema = pq.read_schema(parquet_path)
    except FileNotFoundError:
        logging.warning("Parquet file not found for rasterization: %s", parquet_path)
        return [], assetization_dir

    available_names = list(schema.names)
    norm_lookup = {_normalize_column_key(real): real for real in available_names}

    def _resolve(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        key = _normalize_column_key(name)
        resolved = norm_lookup.get(key)
        if resolved is None:
            return name
        return resolved

    value_pairs: List[tuple[str, str]] = []
    for column in value_columns:
        resolved = _resolve(column)
        if resolved not in available_names:
            logging.warning("Column %s not found in table schema; skipping rasterization.", column)
            continue
        value_pairs.append((column, resolved))

    if not value_pairs:
        logging.info("No valid numeric columns remained after schema resolution; skipping raster assets.")
        return [], assetization_dir

    columns_to_read = {resolved for _, resolved in value_pairs}

    lat_resolved = _resolve(lat_column)
    if lat_resolved not in available_names:
        lat_resolved = None
    elif lat_resolved:
        columns_to_read.add(lat_resolved)

    lon_resolved = _resolve(lon_column)
    if lon_resolved not in available_names:
        lon_resolved = None
    elif lon_resolved:
        columns_to_read.add(lon_resolved)

    h3_resolved = _resolve(h3_column)
    if h3_resolved not in available_names:
        h3_resolved = None
    elif h3_resolved:
        columns_to_read.add(h3_resolved)

    geometry_resolved = _resolve(geometry_column)
    if geometry_resolved not in available_names:
        geometry_resolved = None
    elif geometry_resolved:
        columns_to_read.add(geometry_resolved)

    resolution_resolved = _resolve(resolution_column)
    if resolution_resolved not in available_names:
        resolution_resolved = None
    elif resolution_resolved:
        columns_to_read.add(resolution_resolved)

    table = pq.read_table(parquet_path, columns=list(columns_to_read))
    frame = table.to_pandas()
    row_count = len(frame)
    geometry_polygons: list[Optional[np.ndarray]]
    geometry_lat_centroids: Optional[np.ndarray] = None
    geometry_lon_centroids: Optional[np.ndarray] = None
    if geometry_resolved and geometry_resolved in frame.columns:
        geometry_polygons, geometry_lat_centroids, geometry_lon_centroids = _parse_geometry_series(
            frame[geometry_resolved].fillna("").astype(str).tolist()
        )
    else:
        geometry_polygons = [None] * row_count

    coords = _normalize_lat_lon(frame, lat_resolved, lon_resolved, h3_resolved)
    if coords is None:
        lat_vals = np.full(row_count, np.nan, dtype=np.float64)
        lon_vals = np.full(row_count, np.nan, dtype=np.float64)
    else:
        lat_vals, lon_vals = coords

    if geometry_lat_centroids is not None:
        missing_lat = ~np.isfinite(lat_vals)
        lat_vals[missing_lat] = geometry_lat_centroids[missing_lat]
    if geometry_lon_centroids is not None:
        missing_lon = ~np.isfinite(lon_vals)
        lon_vals[missing_lon] = geometry_lon_centroids[missing_lon]

    valid_xy = np.isfinite(lat_vals) & np.isfinite(lon_vals)
    if not valid_xy.any():
        logging.warning(
            "Unable to determine latitude/longitude for rasterization from lat/lon, H3, or geometry."
        )
        return [], assetization_dir

    if grid_size <= 0:
        grid_size = _infer_grid_size_from_h3(frame, lat_vals, lon_vals, resolution_resolved)

    if grid_size <= 1:
        logging.warning("Grid size must be greater than 1 to rasterize; skipping raster assets.")
        return [], assetization_dir

    lat_min = float(lat_vals[valid_xy].min())
    lat_max = float(lat_vals[valid_xy].max())
    lon_min = float(lon_vals[valid_xy].min())
    lon_max = float(lon_vals[valid_xy].max())

    if not math.isfinite(lat_min) or not math.isfinite(lat_max) or math.isclose(lat_min, lat_max):
        logging.warning("Latitude extent degenerate; skipping rasterization.")
        return [], assetization_dir
    if not math.isfinite(lon_min) or not math.isfinite(lon_max) or math.isclose(lon_min, lon_max):
        logging.warning("Longitude extent degenerate; skipping rasterization.")
        return [], assetization_dir

    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    rasterio = _require_rasterio()
    from rasterio.transform import from_origin

    raw_raster_dir = assets_dir / "tmp_rasters"
    target_asset_dir = assets_dir / "rasters"
    ensure_dir(raw_raster_dir)
    ensure_dir(target_asset_dir)

    lon_res = (lon_max - lon_min) / grid_size
    lat_res = (lat_max - lat_min) / grid_size
    transform = from_origin(lon_min, lat_max, lon_res, lat_res)

    quicklook_dir: Optional[Path] = None
    feature_allow: Optional[set[str]] = None
    slug_allow: Optional[set[str]] = None
    if check_raster:
        if check_features:
            quicklook_dir = assets_dir / "quicklooks"
            ensure_dir(quicklook_dir)
            feature_allow = {_normalize_column_key(name) for name in check_features}
            slug_allow = {_slugify_column(name) for name in check_features}
        else:
            logging.warning(
                "check_raster enabled but no check_raster_features provided; skipping quicklook generation."
            )

    raster_paths: List[Path] = []

    for original_name, resolved_name in value_pairs:
        if resolved_name not in frame.columns:
            logging.debug("Resolved column %s missing from frame; skipping.", resolved_name)
            continue

        values = pd.to_numeric(frame[resolved_name], errors="coerce").to_numpy(dtype=np.float64)
        value_mask = np.isfinite(values) & valid_xy
        if not value_mask.any():
            logging.debug("Column %s has no finite samples; skipping raster.", original_name)
            continue

        accum_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
        accum_count = np.zeros((grid_size, grid_size), dtype=np.float64)

        indices = np.nonzero(value_mask)[0]
        for idx in indices:
            val = float(values[idx])
            poly = geometry_polygons[idx]
            if poly is not None:
                lat_bounds = (np.nanmin(poly[:, 1]), np.nanmax(poly[:, 1]))
                lon_bounds = (np.nanmin(poly[:, 0]), np.nanmax(poly[:, 0]))
                row_start = _coordinate_to_index(lat_bounds[0], lat_edges)
                row_end = _coordinate_to_index(lat_bounds[1], lat_edges)
                col_start = _coordinate_to_index(lon_bounds[0], lon_edges)
                col_end = _coordinate_to_index(lon_bounds[1], lon_edges)
                if None not in (row_start, row_end, col_start, col_end):
                    for row in range(min(row_start, row_end), max(row_start, row_end) + 1):
                        lat_c = lat_centers[row]
                        for col in range(min(col_start, col_end), max(col_start, col_end) + 1):
                            lon_c = lon_centers[col]
                            if _point_in_polygon(lon_c, lat_c, poly):
                                accum_sum[row, col] += val
                                accum_count[row, col] += 1.0
                    continue

            row_idx = _coordinate_to_index(lat_vals[idx], lat_edges)
            col_idx = _coordinate_to_index(lon_vals[idx], lon_edges)
            if row_idx is None or col_idx is None:
                continue
            accum_sum[row_idx, col_idx] += val
            accum_count[row_idx, col_idx] += 1.0

        finite_mask = accum_count > 0
        if not finite_mask.any():
            logging.debug("Column %s produced all-NaN raster; skipping.", original_name)
            continue

        mean_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        mean_grid[finite_mask] = (accum_sum[finite_mask] / accum_count[finite_mask]).astype(np.float32)
        mean_grid = np.flipud(mean_grid)

        finite_after = np.isfinite(mean_grid)
        coverage = float(finite_after.sum()) / float(finite_after.size)
        finite_values = mean_grid[finite_after]
        min_val = float(finite_values.min()) if finite_values.size else float("nan")
        max_val = float(finite_values.max()) if finite_values.size else float("nan")
        logging.info(
            "Rasterized column %s (resolved %s): coverage=%.1f%% range=[%.3f, %.3f]",
            original_name,
            resolved_name,
            coverage * 100.0,
            min_val,
            max_val,
        )

        quicklook_requested = False
        if quicklook_dir is not None and feature_allow is not None and slug_allow is not None:
            col_key = _normalize_column_key(original_name)
            col_slug = _slugify_column(original_name)
            quicklook_requested = col_key in feature_allow or col_slug in slug_allow

        out_path = raw_raster_dir / f"{parquet_path.stem}_{_slugify_column(original_name)}_{grid_size}.tif"
        profile = {
            "driver": "GTiff",
            "height": mean_grid.shape[0],
            "width": mean_grid.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mean_grid, 1)

        if quicklook_requested and quicklook_dir is not None and finite_after.any():
            quicklook_path = quicklook_dir / f"{parquet_path.stem}_{_slugify_column(original_name)}_{grid_size}.png"
            _save_quicklook(mean_grid, original_name, quicklook_path)
            logging.info("Saved raster quicklook for %s -> %s", original_name, quicklook_path)

        raster_paths.append(out_path)
        logging.info("Built raster diagnostic for column %s -> %s", original_name, out_path.name)

    if not raster_paths:
        logging.info("No rasters generated from numeric columns; skipping asset registration.")
        return [], assetization_dir

    products: List[Dict[str, Path]] = []
    try:
        products = add_raster_assets(collection, raster_paths, target_asset_dir, cogify=True)
    finally:
        for path in raster_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logging.debug("Failed to remove temporary raster %s", path)

    return products, assetization_dir


def build_table_item(collection, parquet_path: Path, columns: list[Dict[str, str]], collection_root: Path) -> None:
    mods = _require_pystac()
    Item = mods["Item"]
    Asset = mods["Asset"]

    item = Item(
        id=parquet_path.stem,
        geometry=None,
        bbox=None,
        datetime=datetime.now(tz=timezone.utc),
        properties={"table:columns": columns},
        stac_extensions=[TABLE_EXT, CHECKSUM_EXT],
    )
    item.add_asset(
        "data",
        Asset(
            href=str(parquet_path),
            media_type="application/x-parquet",
            roles=["data"],
            title=parquet_path.name,
            extra_fields={"checksum:sha256": sha256sum(parquet_path)},
        ),
    )
    collection.add_item(item)


def parse_schema_hints(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a CSV table into a STAC catalog.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--out", type=str, required=True, help="Output directory for STAC catalog")
    parser.add_argument("--collection-id", type=str, default="gsc-table", help="Collection ID")
    parser.add_argument("--title", type=str, default="GSC Table", help="Catalog title")
    parser.add_argument("--description", type=str, default="GSC tabular data registered in STAC.")
    parser.add_argument("--license", type=str, default="proprietary")
    parser.add_argument("--keywords", type=str, nargs="*", default=[])
    parser.add_argument("--assets-subdir", type=str, default="assets")
    parser.add_argument("--schema", type=str, default=None, help="Optional JSON column:type hints")
    parser.add_argument("--validate", action="store_true", help="Run STAC validation")
    parser.add_argument("--source-url", type=str, default=None, help="Optional source data URL")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature columns to rasterize (defaults to numeric inference when omitted).",
    )
    parser.add_argument(
        "--lat-column",
        type=str,
        default=None,
        help="Column providing latitude values (degrees).",
    )
    parser.add_argument(
        "--lon-column",
        type=str,
        default=None,
        help="Column providing longitude values (degrees).",
    )
    parser.add_argument(
        "--h3-column",
        type=str,
        default="H3_Address",
        help="Column providing H3 addresses when lat/lon are unavailable.",
    )
    parser.add_argument(
        "--geometry-column",
        type=str,
        default="H3_Geometry",
        help="Column containing WKT polygons describing each H3 footprint.",
    )
    parser.add_argument(
        "--h3-resolution-column",
        type=str,
        default="H3_Resolution",
        help="Column containing H3 resolution for automatic grid sizing.",
    )
    parser.add_argument(
        "--grid-columns",
        nargs="+",
        default=None,
        help="Subset of columns to rasterize; defaults to all numeric columns.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=0,
        help="Resolution of the generated latitude/longitude grid (grid_size x grid_size). Set 0 to auto-compute from H3 resolution.",
    )
    parser.add_argument(
        "--check-raster",
        action="store_true",
        help="Generate quicklook PNGs for selected features after rasterization.",
    )
    parser.add_argument(
        "--check-raster-features",
        nargs="+",
        default=None,
        help="Feature columns to include in quicklook PNG generation (requires --check-raster).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)
    collection_root = out_dir / args.collection_id
    ensure_dir(collection_root)

    if args.grid_size == 1:
        parser.error("--grid-size must not be 1")

    # Save a copy of the raw CSV
    raw_dir = collection_root / "raw"
    ensure_dir(raw_dir)
    shutil.copy2(csv_path, raw_dir / csv_path.name)

    # Copy feature registry template
    copy_feature_registry(collection_root)

    # Convert CSV to Parquet
    assets_dir = collection_root / args.assets_subdir
    ensure_dir(assets_dir)
    tables_dir = assets_dir / "tables"
    ensure_dir(tables_dir)
    schema_hints = parse_schema_hints(args.schema)
    parquet_path = tables_dir / (csv_path.stem + ".parquet")
    csv_to_parquet(csv_path, parquet_path, schema_hints=schema_hints)

    columns = infer_columns(parquet_path)

    requested_columns = args.features if args.features else args.grid_columns
    grid_columns = _select_grid_columns(
        columns,
        requested_columns,
        args.lat_column,
        args.lon_column,
        args.h3_column,
        args.geometry_column,
        args.h3_resolution_column,
    )

    mods = _require_pystac()
    pystac = mods["pystac"]
    Provider = mods["Provider"]
    Link = mods["Link"]
    RelType = mods["RelType"]

    cat = build_catalog(out_dir, args.title, args.description)
    coll = build_collection(
        args.collection_id,
        args.description,
        license_str=args.license,
        keywords=args.keywords,
    )
    coll.stac_extensions = [TABLE_EXT, CHECKSUM_EXT]
    cat.stac_extensions = coll.stac_extensions

    providers = [
        Provider(name="GSC Data Source", roles=["producer"], url=args.source_url or csv_path.as_uri()),
        Provider(name="STAC Conversion Pipeline", roles=["processor"], url=f"file://{Path(__file__).resolve()}"),
    ]
    coll.providers = providers

    if args.source_url:
        coll.add_link(Link(rel=RelType.SOURCE, target=args.source_url, title="Source data portal"))
    else:
        coll.add_link(Link(rel="via", target=csv_path.as_uri(), title="Original CSV"))

    build_table_item(coll, parquet_path, columns, collection_root)
    table_item = coll.get_item(parquet_path.stem)
    if table_item is not None:
        table_meta_path = assets_dir / "assetization" / f"{table_item.id}.json"
        ensure_dir(table_meta_path.parent)
        table_item.set_self_href(str(table_meta_path))
        try:
            table_item.save_object(include_self_link=False)
        except Exception as exc:
            logging.warning("Failed to save table item %s: %s", table_item.id, exc)

    raster_products: List[Dict[str, Path]] = []
    assetization_dir: Path

    try:
        raster_targets = args.features if args.features else grid_columns
        raster_products, assetization_dir = _generate_raster_assets(
            parquet_path,
            coll,
            assets_dir,
            args.lat_column,
            args.lon_column,
            args.h3_column,
            args.geometry_column,
            args.h3_resolution_column,
            raster_targets,
            args.grid_size,
            args.check_raster,
            args.check_raster_features,
        )
    except Exception as exc:
        logging.exception("Failed to generate raster assets: %s", exc)
        assetization_dir = assets_dir / "assetization"
        ensure_dir(assetization_dir)

    coll.summaries = pystac.Summaries({"table:columns": columns})

    cat.add_child(coll)
    collection_href = collection_root / "collection.json"
    ensure_dir(collection_href.parent)
    coll.set_self_href(str(collection_href))

    for prod in raster_products:
        item = prod.get("item") if isinstance(prod, dict) else None
        if item is None:
            continue
        metadata_path = assetization_dir / f"{item.id}.json"
        ensure_dir(metadata_path.parent)
        item.set_self_href(str(metadata_path))
        try:
            item.save_object(include_self_link=False)
        except Exception as exc:
            logging.warning("Failed to save raster item %s: %s", item.id, exc)

    reset_io = None
    if args.validate:
        reset_io = _install_local_schema_cache()
    if args.validate:
        try:
            coll.validate()
            for it in coll.get_items():
                it.validate()
            logging.info("STAC validation passed.")
        except Exception as exc:
            logging.warning(f"Validation failed: {exc}")
        logging.info("STAC validation passed.")
    if reset_io:
        reset_io()

    catalog_href = out_dir / f"catalog_{args.collection_id}.json"
    cat.set_self_href(str(catalog_href))
    cat.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
    print(f"STAC catalog written to: {out_dir}")


if __name__ == "__main__":
    main()
