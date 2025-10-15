"""Generate point GeoJSONs for binary label columns."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)

_BINARY_TEXT_MAP = {
    "present": 1.0,
    "absent": 0.0,
    "1": 1.0,
    "0": 0.0,
    "true": 1.0,
    "false": 0.0,
    "yes": 1.0,
    "no": 0.0,
    "y": 1.0,
    "n": 0.0,
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("_")
    return slug or "label"


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(_BINARY_TEXT_MAP)
    if mapped.notna().any():
        return mapped
    return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)


def _region_masks(lat: np.ndarray, lon: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        return masks
    na_mask = finite & (lon < 0.0) & (lat > 0.0)
    au_mask = finite & (lon > 0.0) & (lat < 0.0)
    if na_mask.any():
        masks["NA"] = na_mask
    if au_mask.any():
        masks["AU"] = au_mask
    return masks


def _coords_from_frame(
    frame: pd.DataFrame,
    lat_column: Optional[str],
    lon_column: Optional[str],
    geometry_column: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    lat = None
    lon = None
    if lat_column and lat_column in frame.columns:
        lat = pd.to_numeric(frame[lat_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if lon_column and lon_column in frame.columns:
        lon = pd.to_numeric(frame[lon_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if lat is not None and lon is not None:
        return lat, lon

    if geometry_column and geometry_column in frame.columns:
        try:
            import shapely.wkt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            _LOG.warning("Shapely unavailable; cannot derive coordinates from geometry column %s (%s).", geometry_column, exc)
        else:
            lat_vals = np.full(len(frame), np.nan, dtype=np.float64)
            lon_vals = np.full(len(frame), np.nan, dtype=np.float64)
            for idx, text in enumerate(frame[geometry_column].fillna("").astype(str)):
                if not text:
                    continue
                try:
                    geom = shapely.wkt.loads(text)
                except Exception:
                    continue
                if geom.is_empty:
                    continue
                point = geom.representative_point()
                lon_vals[idx], lat_vals[idx] = point.x, point.y
            return lat_vals, lon_vals

    length = len(frame)
    return (
        np.full(length, np.nan, dtype=np.float64),
        np.full(length, np.nan, dtype=np.float64),
    )


def _build_features(
    lat: Sequence[float],
    lon: Sequence[float],
    row_ids: Sequence[int],
    label_values: Sequence[float],
) -> List[Dict[str, object]]:
    feats: List[Dict[str, object]] = []
    for lat_val, lon_val, row_id, lab in zip(lat, lon, row_ids, label_values):
        if not (math.isfinite(lat_val) and math.isfinite(lon_val)):
            continue
        feature = {
            "type": "Feature",
            "properties": {
                "label": float(lab),
                "row": int(row_id),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon_val, lat_val],
            },
        }
        feats.append(feature)
    return feats


def _write_geojson(path: Path, features: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_label_geojson(
    parquet_path: Path,
    assets_dir: Path,
    *,
    label_column: str,
    lat_column: Optional[str],
    lon_column: Optional[str],
    geometry_column: Optional[str] = None,
    threshold: float = 0.5,
) -> List[Path]:
    """Create GeoJSON point files for rows with positive label values."""

    parquet_path = Path(parquet_path).resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    assets_dir = Path(assets_dir).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    label_column = str(label_column)
    columns: List[str] = [label_column]
    for col in (lat_column, lon_column, geometry_column):
        if col and col not in columns:
            columns.append(col)

    table = pq.read_table(parquet_path, columns=[c for c in columns if c])
    frame = table.to_pandas()
    if label_column not in frame.columns:
        _LOG.warning("Label column %s not present; skipping GeoJSON generation.", label_column)
        return []

    label_series = _coerce_binary(frame[label_column]).fillna(0.0)
    positive_mask = label_series > float(threshold)
    if not positive_mask.any():
        _LOG.info("No positive rows detected in %s using column %s.", parquet_path.name, label_column)
        return []

    lat_vals_all, lon_vals_all = _coords_from_frame(frame, lat_column, lon_column, geometry_column)
    if not np.isfinite(lat_vals_all).any() or not np.isfinite(lon_vals_all).any():
        _LOG.warning("Latitude/longitude could not be determined; skipping GeoJSON export for %s.", label_column)
        return []

    mask_np = positive_mask.to_numpy()
    lat_vals = lat_vals_all[mask_np]
    lon_vals = lon_vals_all[mask_np]
    row_ids = frame.index.to_numpy()[mask_np]
    label_values = label_series.to_numpy()[mask_np]

    features_all = _build_features(lat_vals, lon_vals, row_ids, label_values)
    if not features_all:
        _LOG.warning("No finite coordinates found for positives in %s.", label_column)
        return []

    written: List[Path] = []
    label_slug = _slugify(label_column)
    aggregated_path = assets_dir / f"{label_slug}.geojson"
    _write_geojson(aggregated_path, features_all)
    written.append(aggregated_path)

    region_masks = _region_masks(lat_vals, lon_vals)
    if region_masks:
        lat_pos = lat_vals
        lon_pos = lon_vals
        for region, mask in region_masks.items():
            features = _build_features(
                lat_pos,
                lon_pos,
                row_ids[mask],
                label_values[mask],
            )
            if not features:
                continue
            region_path = assets_dir / f"{label_slug}_{region}.geojson"
            _write_geojson(region_path, features)
            written.append(region_path)

    _LOG.info("Wrote %d label GeoJSON file(s) for %s.", len(written), label_column)
    return written
