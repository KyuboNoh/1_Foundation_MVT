#!/usr/bin/env python3
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube_temp.csv   --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --collection-id gsc-2021_temp   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate --check-raster --check-raster-features Magnetic_HGM
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube.csv   --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate   --check-raster --check-raster-features Magnetic_HGM

# python stacify_GCS_data.py   --csv C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_selected_Norm.csv   --out C:\Users\kyubo\Desktop\Research\Data\1_Stacify   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate   --check-raster --check-raster-features Magnetic_HGM

# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_selected_Norm.csv   --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate --check-raster --check-raster-features Magnetic_HGM
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_selected_Norm.csv   --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --features Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority --label Training_MVT_Deposit --validate --check-raster --check-raster-features Magnetic_HGM Training_MVT_Deposit

"""
Convert a single GSC CSV table into a STAC catalog with tabular + raster assets. 
For GCS data specifically, 
we use the H3 geospatial index column to derive equal-area grids for rasterization;
we save raterized data for NA and AU separately to avoid extreme distortion, and use common/metadata generation_training.py to save the information;
"""

import argparse
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from affine import Affine
from pyproj import Transformer

from unify.one_d import csv_to_parquet
from Common.metadata_generation_training import generate_training_metadata
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

MAX_CATEGORICAL_LEVELS = 32


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


@dataclass
class H3GridSpec:
    name: str
    frame_mask: np.ndarray
    width: int
    height: int
    transform: Affine
    crs: str
    pixel_size_m: float


def infer_equal_area_grid_from_h3(
    frame: pd.DataFrame,
    h3_col: str = "H3_Address",
    h3_res_col: str = "H3_Resolution",
    crs_epsg: str = "EPSG:6933",
    scale: float = 1.0,
    min_dim: int = 64,
    max_dim: int = 50000,
) -> List[H3GridSpec]:
    try:
        import h3
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The h3 library is required for H3-based rasterization") from exc

    if h3_col not in frame.columns:
        raise ValueError(f"Missing column: {h3_col}")

    addresses = frame[h3_col].dropna().astype(str).str.strip()
    addresses = addresses[addresses != ""]
    if addresses.empty:
        raise ValueError("No H3 addresses available to infer grid geometry.")

    if h3_res_col in frame.columns and frame[h3_res_col].notna().any():
        res_series = pd.to_numeric(frame[h3_res_col], errors="coerce").dropna()
        if res_series.empty:
            res = None
        else:
            res = int(res_series.mode().iloc[0])
    else:
        res = None

    if res is None:
        sample_addrs = addresses.head(100)
        resolutions: list[int] = []
        for addr in sample_addrs:
            if hasattr(h3, "get_resolution"):
                resolutions.append(int(h3.get_resolution(addr)))
            else:
                resolutions.append(int(h3.h3_get_resolution(addr)))  # type: ignore[attr-defined]
        if not resolutions:
            raise ValueError("Unable to infer H3 resolution from addresses.")
        res = int(pd.Series(resolutions).mode().iloc[0])

    sample_addr = addresses.iloc[0]
    if hasattr(h3, "cell_area"):
        hex_area_m2 = float(h3.cell_area(sample_addr, unit="m^2"))
    elif hasattr(h3, "hex_area"):
        hex_area_m2 = float(h3.hex_area(res, unit="m^2"))  # type: ignore[attr-defined]
    else:  # pragma: no cover - legacy fallback
        if hasattr(h3, "edge_length"):
            edge_m = float(h3.edge_length(res, unit="m"))  # type: ignore[attr-defined]
        else:
            edge_m = float(h3.h3_edge_length(res, unit="m"))  # type: ignore[attr-defined]
        hex_area_m2 = (3.0 * math.sqrt(3.0) / 2.0) * (edge_m ** 2)

    pixel_size_m = max(math.sqrt(hex_area_m2) * float(scale), 1e-3)

    if hasattr(h3, "cell_to_latlng"):
        lats_lons = np.array([h3.cell_to_latlng(addr) for addr in addresses], dtype=np.float64)
    else:
        lats_lons = np.array([h3.h3_to_geo(addr) for addr in addresses], dtype=np.float64)  # type: ignore[attr-defined]
    if lats_lons.size == 0:
        raise ValueError("Could not decode any H3 centroids.")

    lats = lats_lons[:, 0]
    lons = lats_lons[:, 1]

    transformer = Transformer.from_crs("EPSG:4326", crs_epsg, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    addresses_index = addresses.index.to_numpy()
    frame_positions = frame.index.get_indexer(addresses_index)
    frame_positions = np.asarray(frame_positions, dtype=np.int64)
    valid_position_mask = frame_positions >= 0
    if not valid_position_mask.any():
        raise ValueError("Unable to match H3 addresses to dataframe rows.")
    if not valid_position_mask.all():
        logging.debug(
            "Dropping %d H3 records with missing dataframe alignment during grid inference.",
            int((~valid_position_mask).sum()),
        )
        addresses = addresses.iloc[valid_position_mask]
        lats = lats[valid_position_mask]
        lons = lons[valid_position_mask]
        xs = xs[valid_position_mask]
        ys = ys[valid_position_mask]
        frame_positions = frame_positions[valid_position_mask]

    if addresses.empty:
        raise ValueError("No H3 addresses available to infer grid geometry.")

    na_mask = (lons < 0.0) & (lats > 0.0)
    au_mask = (lons > 0.0) & (lats < 0.0)
    region_masks: list[tuple[str, np.ndarray]] = []
    if na_mask.any():
        region_masks.append(("NA", na_mask))
    if au_mask.any():
        region_masks.append(("AU", au_mask))

    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    if not finite_mask.any():
        raise ValueError("Invalid projected bounds; cannot derive raster dimensions.")

    used_mask = np.zeros_like(finite_mask, dtype=bool)
    for _, mask in region_masks:
        used_mask |= mask
    leftover_mask = (~used_mask) & finite_mask

    if not region_masks:
        if leftover_mask.any():
            logging.warning(
                "Found %d H3 records outside NA/AU; skipping leftover cells.",
                int(leftover_mask.sum()),
            )
        raise ValueError("Unable to classify any H3 coordinates into NA/AU regions.")
    elif leftover_mask.any():
        logging.info(
            "Ignoring %d H3 records outside NA/AU regions during grid inference.",
            int(leftover_mask.sum()),
        )

    grid_specs: List[H3GridSpec] = []
    full_frame_length = len(frame)

    for name, base_mask in region_masks:
        region_mask = base_mask & finite_mask
        if not region_mask.any():
            logging.debug("Skipping region %s with no finite projected coordinates.", name)
            continue

        region_x = xs[region_mask]
        region_y = ys[region_mask]
        xmin = float(np.nanmin(region_x))
        xmax = float(np.nanmax(region_x))
        ymin = float(np.nanmin(region_y))
        ymax = float(np.nanmax(region_y))
        if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmin == xmax or ymin == ymax:
            raise ValueError(f"Invalid projected bounds for region {name}; cannot derive raster dimensions.")

        width = int(math.ceil((xmax - xmin) / pixel_size_m))
        height = int(math.ceil((ymax - ymin) / pixel_size_m))

        width = int(np.clip(width, min_dim, max_dim))
        height = int(np.clip(height, min_dim, max_dim))

        if width <= 0 or height <= 0:
            raise ValueError(f"Derived raster dimensions are degenerate for region {name}.")

        px_x = (xmax - xmin) / width
        px_y = (ymax - ymin) / height
        transform = Affine(px_x, 0.0, xmin, 0.0, -px_y, ymax)

        frame_mask = np.zeros(full_frame_length, dtype=bool)
        region_positions = frame_positions[region_mask]
        region_positions = region_positions[(region_positions >= 0) & (region_positions < full_frame_length)]
        frame_mask[region_positions] = True

        grid_specs.append(
            H3GridSpec(
                name=name,
                frame_mask=frame_mask,
                width=width,
                height=height,
                transform=transform,
                crs=crs_epsg,
                pixel_size_m=pixel_size_m,
            )
        )

    if not grid_specs:
        raise ValueError("Unable to derive any grid definitions from H3 addresses.")

    return grid_specs


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


def _infer_grid_size_from_data(
    frame: pd.DataFrame,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    default: int = 256,
) -> int:
    finite_lat = lat_vals[np.isfinite(lat_vals)]
    finite_lon = lon_vals[np.isfinite(lon_vals)]
    if finite_lat.size == 0 or finite_lon.size == 0:
        logging.debug("Insufficient coordinate coverage; using default grid size %d", default)
        return default

    lat_span = float(finite_lat.max() - finite_lat.min())
    lon_span = float(finite_lon.max() - finite_lon.min())
    if lat_span <= 0 or lon_span <= 0:
        return default

    # Aim for roughly square cells
    lat_mean = float(finite_lat.mean())
    deg_per_km = 1.0 / (111.320 * max(math.cos(math.radians(lat_mean)), 1e-6))
    span_km = max(lat_span, lon_span) / deg_per_km

    # Target ~1 km resolution but cap at reasonable sizes
    size = int(np.clip(span_km, 16, 1024))

    logging.info("Auto-selected fallback grid size %d for %.1f km extent", size, span_km)
    return size

def _select_grid_columns(
    schema_columns: Sequence[Dict[str, str]],
    requested: Optional[Iterable[str]],
    lat_column: Optional[str],
    lon_column: Optional[str],
    h3_column,
    geometry_column,
    h3_resolution_column,
) -> List[str]:
    if requested:
        return list(dict.fromkeys(requested))
    exclude = {c for c in (lat_column, lon_column) if c}
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


def _sanitize_category(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "__empty__"
    return cleaned


def _encode_categorical_column(
    frame: pd.DataFrame,
    original_name: str,
    resolved_name: str,
    *,
    max_levels: int = MAX_CATEGORICAL_LEVELS,
) -> List[Tuple[str, str]]:
    if resolved_name not in frame.columns:
        return []

    series = frame[resolved_name]
    text_series = series.astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null", "na"}
    is_missing = text_series.isin(invalid_tokens) | series.isna()
    valid_series = text_series[~is_missing]

    if valid_series.empty:
        return []

    value_counts = valid_series.value_counts()
    if value_counts.empty:
        return []

    categories = list(value_counts.index[:max_levels])
    if len(value_counts) > max_levels:
        logging.warning(
            "Column %s has %d distinct categories; keeping top %d and grouping the rest as __other__.",
            original_name,
            len(value_counts),
            max_levels,
        )
    encoded: List[Tuple[str, str]] = []
    existing_cols = set(frame.columns)

    def _unique_column(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in existing_cols:
            candidate = f"{base}_{suffix}"
            suffix += 1
        existing_cols.add(candidate)
        return candidate

    for cat in categories:
        label = _sanitize_category(cat)
        col_slug = _slugify_column(label)
        new_col = _unique_column(f"{resolved_name}__cat__{col_slug}")
        mask = (text_series == cat).astype(np.float32)
        frame[new_col] = mask
        encoded.append((f"{original_name}={label}", new_col))

    if len(value_counts) > max_levels:
        others_mask = ~(text_series.isin(categories)) & ~is_missing
        if others_mask.any():
            other_col = _unique_column(f"{resolved_name}__cat__other")
            frame[other_col] = others_mask.astype(np.float32)
            encoded.append((f"{original_name}=__other__", other_col))

    if is_missing.any():
        missing_col = _unique_column(f"{resolved_name}__cat__missing")
        frame[missing_col] = is_missing.astype(np.float32)
        encoded.append((f"{original_name}=__missing__", missing_col))

    frame[resolved_name] = np.nan
    return encoded


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
        logging.warning(
            "h3 column not provided; cannot generate raster files."
        )
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
        # lat, lon = h3.h3_to_geo(cell)
        lat, lon = h3.cell_to_latlng(cell)
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
    label_column: Optional[str],
    value_columns: Sequence[str],
    grid_size: int,
    check_raster: bool,
    check_features: Optional[Sequence[str]],
) -> tuple[List[Dict[str, Path]], Path]:
    assetization_dir = assets_dir / "assetization"
    ensure_dir(assetization_dir)

    def _region_masks_from_latlon(lat_array: np.ndarray, lon_array: np.ndarray) -> List[tuple[str, np.ndarray]]:
        region_masks_local: List[tuple[str, np.ndarray]] = []
        finite = np.isfinite(lat_array) & np.isfinite(lon_array)
        if not finite.any():
            return region_masks_local
        na_mask = finite & (lon_array < 0.0) & (lat_array > 0.0)
        if na_mask.any():
            region_masks_local.append(("NA", na_mask))
        au_mask = finite & (lon_array > 0.0) & (lat_array < 0.0)
        if au_mask.any():
            region_masks_local.append(("AU", au_mask))
        return region_masks_local

    if not value_columns:
        logging.info("No numeric columns selected for rasterization; skipping raster assets.")
        return [], assetization_dir

    try:
        schema = pq.read_schema(parquet_path)
    except FileNotFoundError:
        logging.warning("Parquet file not found for rasterization: %s", parquet_path)
        return [], assetization_dir

    available_names = list(schema.names)
    logging.debug("Parquet schema columns: %s", available_names)
    norm_lookup = {_normalize_column_key(real): real for real in available_names}
    logging.debug("Normalized name lookup keys: %s", list(norm_lookup.keys()))

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
    # logging.info("Resolved value columns for rasterization: %s", value_pairs)

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

    label_resolved = _resolve(label_column)
    if label_resolved not in available_names:
        if label_column:
            logging.warning("Label column %s not found in table schema.", label_column)
        label_resolved = None
    elif label_resolved:
        columns_to_read.add(label_resolved)

    table = pq.read_table(parquet_path, columns=list(columns_to_read))
    frame = table.to_pandas()
    # Force numeric coercion for value columns (help when CSV columns are strings like 'N/A' or use comma decimals)
    import pandas as _pd
    if label_resolved and all(resolved_name != label_resolved for _, resolved_name in value_pairs):
        value_pairs.append((label_column or label_resolved, label_resolved))

    categorical_expansions: Dict[str, List[Tuple[str, str]]] = {}

    for orig_name, resolved_name in value_pairs:
        if label_resolved is not None and resolved_name == label_resolved:
            continue
        if resolved_name not in frame.columns:
            continue

        original_series = frame[resolved_name].copy()
        numeric_primary = _pd.to_numeric(original_series, errors='coerce')
        primary_finite = numeric_primary.notna().sum()
        numeric_series = numeric_primary
        used_alt = False

        if primary_finite == 0 and original_series.notna().sum() > 0:
            try:
                numeric_alt = _pd.to_numeric(
                    original_series.astype(str).str.replace(',', '.'),
                    errors='coerce',
                )
            except Exception:
                numeric_alt = None
            if numeric_alt is not None and numeric_alt.notna().sum() > 0:
                numeric_series = numeric_alt
                used_alt = True

        final_finite = numeric_series.notna().sum()
        logging.debug(
            "Column %s (resolved %s): primary_finite=%d final_finite=%d alt_used=%s",
            orig_name,
            resolved_name,
            int(primary_finite),
            int(final_finite),
            used_alt,
        )

        if final_finite > 0:
            frame[resolved_name] = numeric_series
            continue

        if original_series.notna().sum() > 0:
            encoded_pairs = _encode_categorical_column(
                frame,
                orig_name,
                resolved_name,
            )
            if encoded_pairs:
                categorical_expansions[resolved_name] = encoded_pairs
                logging.info(
                    "Expanded categorical column %s into %d indicator feature(s).",
                    orig_name,
                    len(encoded_pairs),
                )
                continue

        frame[resolved_name] = numeric_series

    feature_specs: List[Dict[str, Any]] = []
    feature_index: Dict[str, int] = {}

    def _feature_spec(name: str) -> Dict[str, Any]:
        if name not in feature_index:
            feature_index[name] = len(feature_specs)
            feature_specs.append(
                {
                    "name": name,
                    "columns": [],
                    "band_labels": [],
                    "categories": None,
                    "is_label": False,
                    "kind": None,
                }
            )
        return feature_specs[feature_index[name]]

    for orig_name, resolved_name in value_pairs:
        spec = _feature_spec(orig_name)
        if label_resolved is not None and resolved_name == label_resolved:
            spec["is_label"] = True

    for orig_name, resolved_name in value_pairs:
        spec = _feature_spec(orig_name)
        expansions = categorical_expansions.get(resolved_name)
        if expansions:
            categories: List[str] = []
            band_labels: List[str] = []
            spec["columns"].clear()
            for display_name, column_name in expansions:
                spec["columns"].append((display_name, column_name))
                label = display_name.split("=", 1)[1] if "=" in display_name else display_name
                categories.append(label)
                band_labels.append(display_name)
            spec["categories"] = categories
            spec["band_labels"] = band_labels
            spec["kind"] = "categorical"
        else:
            spec["columns"].append((orig_name, resolved_name))
            if spec["kind"] is None:
                spec["kind"] = "label" if spec["is_label"] else "numeric"
            if not spec["band_labels"]:
                spec["band_labels"].append(orig_name)

    feature_specs = [spec for spec in feature_specs if spec["columns"]]
    if not feature_specs:
        logging.info("No usable feature columns found after processing; skipping raster assets.")
        return [], assetization_dir

    label_total_ones: Optional[float] = None
    if label_resolved and label_resolved in frame.columns:
        series = frame[label_resolved]
        numeric = pd.to_numeric(series, errors="coerce")
        needs_mapping = numeric.isna() & series.notna()
        if needs_mapping.any():
            text_series = series.astype(str).str.strip().str.lower()
            mapping = {
                "present": 1.0,
                "absent": 0.0,
                "1": 1.0,
                "0": 0.0,
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
            }
            numeric = text_series.map(mapping)
        invalid_mask = (~numeric.isna()) & (~numeric.isin([0.0, 1.0]))
        if invalid_mask.any():
            logging.warning(
                "Label column %s has values outside {Absent, Present, 0, 1}; coercing others to NaN.",
                label_resolved,
            )
            numeric = numeric.where(~invalid_mask)
        frame[label_resolved] = numeric
        label_total_ones = float(frame[label_resolved].fillna(0.0).sum())
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

    valid_latlon = np.isfinite(lat_vals) & np.isfinite(lon_vals)
    if not valid_latlon.any():
        logging.warning(
            "Unable to determine latitude/longitude for rasterization from lat/lon, H3, or geometry."
        )
        return [], assetization_dir

    grid_specs: List[H3GridSpec] = []
    rasterio = None
    if h3_resolved:
        try:
            grid_specs = infer_equal_area_grid_from_h3(
                frame,
                h3_col=h3_resolved,
                h3_res_col=resolution_resolved,
            )
            if grid_specs and grid_size > 0:
                max_dim = max(max(spec.width, spec.height) for spec in grid_specs)
                if max_dim > 0:
                    scale_override = max(max_dim / float(grid_size), 1e-3)
                    grid_specs = infer_equal_area_grid_from_h3(
                        frame,
                        h3_col=h3_resolved,
                        h3_res_col=resolution_resolved,
                        scale=scale_override,
                    )
            if grid_specs:
                summary = ", ".join(f"{spec.name}:{spec.width}x{spec.height}" for spec in grid_specs)
                logging.info(
                    "Using equal-area grids %s (%s, pixel≈%.1fm)",
                    summary,
                    grid_specs[0].crs,
                    grid_specs[0].pixel_size_m,
                )
        except Exception as exc:
            logging.warning("Falling back to latitude/longitude grid: %s", exc)
            grid_specs = []

    if not grid_specs:
        region_masks = _region_masks_from_latlon(lat_vals, lon_vals)
        if not region_masks:
            logging.warning("Unable to derive NA/AU fallback grids; skipping raster assets.")
            return [], assetization_dir

        rasterio = _require_rasterio()
        from rasterio.transform import from_origin

        grid_specs = []
        for name, mask in region_masks:
            region_lat = lat_vals[mask]
            region_lon = lon_vals[mask]
            if region_lat.size == 0 or region_lon.size == 0:
                logging.debug("Region %s lacked valid lat/lon samples; skipping.", name)
                continue

            lat_min = float(np.nanmin(region_lat))
            lat_max = float(np.nanmax(region_lat))
            lon_min = float(np.nanmin(region_lon))
            lon_max = float(np.nanmax(region_lon))

            if not math.isfinite(lat_min) or not math.isfinite(lat_max) or math.isclose(lat_min, lat_max):
                logging.warning("Latitude extent degenerate for region %s; skipping rasterization.", name)
                continue
            if not math.isfinite(lon_min) or not math.isfinite(lon_max) or math.isclose(lon_min, lon_max):
                logging.warning("Longitude extent degenerate for region %s; skipping rasterization.", name)
                continue

            current_size = grid_size
            if current_size <= 0:
                current_size = _infer_grid_size_from_data(frame.loc[mask], region_lat, region_lon)
            if current_size <= 1:
                logging.warning("Grid size must be greater than 1 for region %s; skipping.", name)
                continue

            lon_res = (lon_max - lon_min) / current_size
            lat_res = (lat_max - lat_min) / current_size
            if math.isclose(lon_res, 0.0) or math.isclose(lat_res, 0.0):
                logging.warning("Fallback resolution degenerate for region %s; skipping.", name)
                continue

            transform = from_origin(lon_min, lat_max, lon_res, lat_res)
            frame_mask = mask.copy()
            grid_specs.append(
                H3GridSpec(
                    name=name,
                    frame_mask=frame_mask,
                    width=current_size,
                    height=current_size,
                    transform=transform,
                    crs="EPSG:4326",
                    pixel_size_m=float("nan"),
                )
            )

            logging.info(
                "Using fallback geographic grid %sx%s for region %s (EPSG:4326, pixel≈%.4f°×%.4f°)",
                current_size,
                current_size,
                name,
                lon_res,
                lat_res,
            )

        if not grid_specs:
            logging.warning("Failed to build fallback grids for NA/AU; skipping raster assets.")
            return [], assetization_dir
    else:
        rasterio = _require_rasterio()

    if rasterio is None:
        rasterio = _require_rasterio()

    raw_raster_dir = assets_dir / "tmp_rasters"
    target_asset_dir = assets_dir / "rasters"
    ensure_dir(raw_raster_dir)
    ensure_dir(target_asset_dir)

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

    column_cache: dict[str, np.ndarray] = {}
    raster_paths: List[Path] = []
    raster_details: dict[Path, dict] = {}

    for spec in grid_specs:
        region_mask = spec.frame_mask
        region_name = spec.name or "UNKNOWN"
        if region_mask.shape[0] != row_count:
            logging.warning("Region %s mask length mismatch; skipping.", region_name)
            continue
        if not region_mask.any():
            logging.debug("Region %s mask empty; skipping.", region_name)
            continue

        grid_width = spec.width
        grid_height = spec.height
        transform = spec.transform
        grid_crs = spec.crs

        pixel_width = float(transform.a)
        pixel_height = float(-transform.e)
        x_origin = float(transform.c)
        y_origin = float(transform.f)

        col_centers = x_origin + (np.arange(grid_width) + 0.5) * pixel_width
        row_centers = y_origin - (np.arange(grid_height) + 0.5) * pixel_height

        region_indices = np.nonzero(region_mask)[0]
        region_lat_vals = lat_vals[region_mask]
        region_lon_vals = lon_vals[region_mask]
        region_geometry_polygons = [geometry_polygons[idx] for idx in region_indices]

        if grid_crs.upper() == "EPSG:4326":
            region_transformer: Optional[Transformer] = None
        else:
            region_transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)

        region_x_vals = np.full(region_indices.size, np.nan, dtype=np.float64)
        region_y_vals = np.full(region_indices.size, np.nan, dtype=np.float64)
        region_valid_latlon = np.isfinite(region_lat_vals) & np.isfinite(region_lon_vals)
        if region_valid_latlon.any():
            if region_transformer is None:
                region_x_vals[region_valid_latlon] = region_lon_vals[region_valid_latlon]
                region_y_vals[region_valid_latlon] = region_lat_vals[region_valid_latlon]
            else:
                lon_subset = region_lon_vals[region_valid_latlon]
                lat_subset = region_lat_vals[region_valid_latlon]
                xs_subset, ys_subset = region_transformer.transform(lon_subset, lat_subset)
                region_x_vals[region_valid_latlon] = np.asarray(xs_subset, dtype=np.float64)
                region_y_vals[region_valid_latlon] = np.asarray(ys_subset, dtype=np.float64)

        region_valid_xy = np.isfinite(region_x_vals) & np.isfinite(region_y_vals)
        if not region_valid_xy.any():
            logging.debug("Region %s has no projectable coordinates; skipping rasterization.", region_name)
            continue

        region_projected_polygons: list[Optional[np.ndarray]]
        if region_transformer is None:
            region_projected_polygons = region_geometry_polygons
        else:
            region_projected_polygons = []
            for poly in region_geometry_polygons:
                if poly is None:
                    region_projected_polygons.append(None)
                    continue
                xs_poly, ys_poly = region_transformer.transform(poly[:, 0], poly[:, 1])
                region_projected_polygons.append(
                    np.column_stack(
                        [np.asarray(xs_poly, dtype=np.float64), np.asarray(ys_poly, dtype=np.float64)]
                    )
                )

        def _col_range_local(x0: float, x1: float) -> Optional[tuple[int, int]]:
            if not math.isfinite(x0) or not math.isfinite(x1):
                return None
            if x0 > x1:
                x0, x1 = x1, x0
            left = int(math.floor((x0 - x_origin) / pixel_width))
            right = int(math.floor((x1 - x_origin) / pixel_width))
            if right < 0 or left >= grid_width:
                return None
            left = max(left, 0)
            right = min(right, grid_width - 1)
            if left > right:
                return None
            return left, right

        def _row_range_local(y0: float, y1: float) -> Optional[tuple[int, int]]:
            if not math.isfinite(y0) or not math.isfinite(y1):
                return None
            if y0 > y1:
                y0, y1 = y1, y0
            top = int(math.floor((y_origin - y1) / pixel_height))
            bottom = int(math.floor((y_origin - y0) / pixel_height))
            if bottom < 0 or top >= grid_height:
                if bottom < 0 and top < 0:
                    return None
                if bottom >= grid_height and top >= grid_height:
                    return None
            top = max(top, 0)
            bottom = min(bottom, grid_height - 1)
            if top > bottom:
                return None
            return top, bottom

        def _col_index_local(x: float) -> Optional[int]:
            rng = _col_range_local(x, x)
            return None if rng is None else rng[0]

        def _row_index_local(y: float) -> Optional[int]:
            rng = _row_range_local(y, y)
            return None if rng is None else rng[0]

        region_label_totals: Dict[str, float] = {}
        if label_resolved is not None and label_resolved in frame.columns:
            region_label_totals[label_resolved] = float(
                frame.loc[region_mask, label_resolved].fillna(0.0).sum()
            )

        region_tag = f"_{region_name}" if region_name else ""
        for spec in feature_specs:
            feature_name = spec["name"]
            columns = spec["columns"]
            band_labels_spec = spec.get("band_labels") or [lbl for lbl, _ in columns]
            categories = spec.get("categories")
            is_label_feature = bool(spec.get("is_label"))
            kind = spec.get("kind") or ("label" if is_label_feature else "numeric")
            is_categorical = kind == "categorical"

            band_grids: List[np.ndarray] = []
            band_display: List[str] = []
            band_categories: List[str] = [] if categories else []

            for band_idx, (display_name, resolved_name) in enumerate(columns):
                if resolved_name not in frame.columns:
                    if is_categorical:
                        band_grids.append(np.zeros((grid_height, grid_width), dtype=np.float32))
                        band_display.append(display_name)
                        if categories:
                            band_categories.append(categories[band_idx])
                    else:
                        logging.debug(
                            "Resolved column %s missing from frame; skipping feature %s band %s.",
                            resolved_name,
                            feature_name,
                            display_name,
                        )
                    continue

                if resolved_name not in column_cache:
                    column_cache[resolved_name] = frame[resolved_name].to_numpy(dtype=np.float32, copy=False)
                values = column_cache[resolved_name]
                region_values = values[region_mask]
                value_mask = np.isfinite(region_values) & region_valid_xy

                accum_sum = np.zeros((grid_height, grid_width), dtype=np.float64)
                accum_count = np.zeros((grid_height, grid_width), dtype=np.float64)

                indices = np.nonzero(value_mask)[0]
                for idx in indices:
                    val = float(region_values[idx])
                    poly = region_projected_polygons[idx]
                    if poly is not None:
                        y_bounds = (np.nanmin(poly[:, 1]), np.nanmax(poly[:, 1]))
                        x_bounds = (np.nanmin(poly[:, 0]), np.nanmax(poly[:, 0]))
                        row_range = _row_range_local(y_bounds[0], y_bounds[1])
                        col_range = _col_range_local(x_bounds[0], x_bounds[1])
                        if row_range and col_range:
                            row_start, row_end = row_range
                            col_start, col_end = col_range
                            for row in range(row_start, row_end + 1):
                                y_c = row_centers[row]
                                for col in range(col_start, col_end + 1):
                                    x_c = col_centers[col]
                                    if _point_in_polygon(x_c, y_c, poly):
                                        accum_sum[row, col] += val
                                        accum_count[row, col] += 1.0
                            continue

                    row_idx = _row_index_local(region_y_vals[idx])
                    col_idx = _col_index_local(region_x_vals[idx])
                    if row_idx is None or col_idx is None:
                        continue
                    accum_sum[row_idx, col_idx] += val
                    accum_count[row_idx, col_idx] += 1.0

                finite_mask = accum_count > 0
                if not finite_mask.any():
                    if is_categorical:
                        band = np.zeros((grid_height, grid_width), dtype=np.float32)
                    else:
                        logging.debug(
                            "Feature %s band %s produced no samples in region %s; skipping band.",
                            feature_name,
                            display_name,
                            region_name,
                        )
                        continue
                else:
                    band = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
                    if is_label_feature and resolved_name in region_label_totals:
                        band[finite_mask] = accum_sum[finite_mask].astype(np.float32)
                    else:
                        band[finite_mask] = (accum_sum[finite_mask] / accum_count[finite_mask]).astype(np.float32)

                band_grids.append(band)
                band_display.append(display_name)
                if categories:
                    cat_name = categories[band_idx]
                    band_categories.append(cat_name)

            if not band_grids:
                continue

            stack = np.stack(band_grids, axis=0).astype(np.float32, copy=False)
            finite_mask_stack = np.isfinite(stack)
            coverage = (
                float(np.count_nonzero(finite_mask_stack)) / float(finite_mask_stack.size)
                if finite_mask_stack.size
                else 0.0
            )
            finite_vals = stack[finite_mask_stack]
            min_val = float(finite_vals.min()) if finite_vals.size else float("nan")
            max_val = float(finite_vals.max()) if finite_vals.size else float("nan")
            logging.info(
                "Rasterized feature %s (%d band%s) for region %s: coverage=%.1f%% range=[%.3f, %.3f]",
                feature_name,
                stack.shape[0],
                "s" if stack.shape[0] != 1 else "",
                region_name,
                coverage * 100.0,
                min_val,
                max_val,
            )

            quicklook_requested = False
            if quicklook_dir is not None and feature_allow is not None and slug_allow is not None:
                key = _normalize_column_key(feature_name)
                slug = _slugify_column(feature_name)
                quicklook_requested = key in feature_allow or slug in slug_allow

            grid_suffix = f"{grid_width}x{grid_height}{region_tag}"
            out_path = raw_raster_dir / f"{parquet_path.stem}_{_slugify_column(feature_name)}_{grid_suffix}.tif"
            profile = {
                "driver": "GTiff",
                "height": stack.shape[1],
                "width": stack.shape[2],
                "count": stack.shape[0],
                "dtype": stack.dtype.name,
                "crs": grid_crs,
                "transform": transform,
                "nodata": np.nan,
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(stack)

            if is_label_feature and label_resolved is not None:
                label_sum = region_label_totals.get(label_resolved)
                if label_sum is not None:
                    raster_sum = float(np.nansum(stack))
                    if not math.isclose(raster_sum, label_sum, rel_tol=1e-6, abs_tol=1e-6):
                        logging.warning(
                            "Label raster sum %.3f differs from region %s positive count %.3f; check coverage.",
                            raster_sum,
                            region_name,
                            label_sum,
                        )

            quicklook_path_obj: Optional[Path] = None
            if quicklook_requested and quicklook_dir is not None and stack.size:
                band_title = band_categories[0] if band_categories else band_display[0]
                quicklook_title = f"{feature_name} ({band_title})"
                quicklook_path_obj = quicklook_dir / f"{parquet_path.stem}_{_slugify_column(feature_name)}_{grid_suffix}.png"
                _save_quicklook(stack[0], quicklook_title, quicklook_path_obj)
                logging.info("Saved raster quicklook for %s [%s] -> %s", feature_name, region_name, quicklook_path_obj)

            raster_paths.append(out_path)
            raster_detail = {
                "region": region_name,
                "feature": feature_name,
                "is_label": bool(is_label_feature),
                "kind": kind,
                "band_display": band_display,
            }
            if quicklook_path_obj is not None:
                raster_detail["quicklook_path"] = quicklook_path_obj
            if categories:
                raster_detail["categories"] = categories
                raster_detail["band_display"] = categories
            raster_details[out_path] = raster_detail
            logging.info(
                "Built raster diagnostic for feature %s (region %s) -> %s",
                feature_name,
                region_name,
                out_path.name,
            )
    if not raster_paths:
        logging.info("No raster grids were generated; skipping asset registration.")
        return [], assetization_dir

    products: List[Dict[str, Path]] = []
    try:
        products = add_raster_assets(collection, raster_paths, target_asset_dir, cogify=True)
        for product in products:
            detail = raster_details.get(product.get("source_path"))
            if detail:
                product.update(detail)
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

    parser.add_argument("--features", nargs="+", default=None, help="Feature columns to rasterize (defaults to numeric inference when omitted).",)
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
        "--label",
        type=str,
        default=None,
        help="Binary label column (Present/Absent) to rasterize with 0/1 encoding.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Skip raster and thumbnail generation for faster debugging.",
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
    # logging.info("Inferred table columns: %s", [c.get("name") for c in columns])

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

    if args.label:
        if args.label not in grid_columns:
            grid_columns = list(dict.fromkeys(list(grid_columns) + [args.label]))

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

    if args.debug:
        logging.info("Debug mode enabled; skipping raster and thumbnail generation.")
        assetization_dir = assets_dir / "assetization"
        ensure_dir(assetization_dir)
    else:
        try:
            if args.features:
                raster_targets = list(args.features)
            else:
                raster_targets = list(grid_columns)
            if args.label and args.label not in raster_targets:
                raster_targets.append(args.label)
            logging.info("Raster targets requested: %s", raster_targets)
            logging.info("Grid columns selected for rasterization: %s", grid_columns)
            raster_products, assetization_dir = _generate_raster_assets(
                parquet_path,
                coll,
                assets_dir,
                args.lat_column,
                args.lon_column,
                args.h3_column,
                args.geometry_column,
                args.h3_resolution_column,
                args.label,
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
        prod["metadata_path"] = metadata_path
        prod["item_id"] = item.id
        kind = prod.get("kind")
        categories = prod.get("categories")
        band_display = prod.get("band_display")
        if kind:
            item.properties.setdefault("gfm:feature_kind", str(kind))
        if categories:
            item.properties.setdefault("gfm:categories", list(categories))
        if band_display:
            item.properties.setdefault("gfm:bands", list(band_display))
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

    training_metadata_path = generate_training_metadata(
        collection_root,
        raster_products,
        debug=args.debug,
    )
    logging.info("Training metadata summary written to %s", training_metadata_path)

    catalog_href = out_dir / f"catalog_{args.collection_id}.json"
    cat.set_self_href(str(catalog_href))
    cat.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
    print(f"STAC catalog written to: {out_dir}")


if __name__ == "__main__":
    main()
