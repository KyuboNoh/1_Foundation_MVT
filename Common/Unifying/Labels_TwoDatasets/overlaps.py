from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # optional raster IO
    import rasterio
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OverlapTile:
    dataset: str
    tile_id: Optional[str]
    row_col: Optional[Tuple[int, int]]
    native_point: Optional[Tuple[float, float]]
    window: Optional[Tuple[int, int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OverlapPair:
    a: OverlapTile
    b: OverlapTile
    centroid: Optional[Tuple[float, float]]
    notes: Dict[str, Any] = field(default_factory=dict)

    def dataset_names(self) -> Tuple[str, str]:
        return self.a.dataset, self.b.dataset


@dataclass
class OverlapSet:
    dataset_a: str
    dataset_b: str
    generated_from: Optional[str]
    dataset_resolutions: Dict[str, Optional[float]]
    approx_spacing: Dict[str, Optional[float]]
    search_radius: Optional[float]
    raw_pair_candidates: Optional[int]
    centroid_geometry: Optional[Dict[str, Any]]
    pairs: List[OverlapPair] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Summarise overlap coverage using robust statistics."""

        count = len(self.pairs)
        centroid_arr = []
        if count:
            for pair in self.pairs:
                if pair.centroid is None:
                    continue
                centroid_arr.append(pair.centroid)
        centroid_summary = None
        if centroid_arr:
            arr = np.asarray(centroid_arr, dtype=np.float64)
            centroid_summary = {
                "mean_x": float(np.nanmean(arr[:, 0])),
                "mean_y": float(np.nanmean(arr[:, 1])),
                "spread_x": float(np.nanstd(arr[:, 0])),
                "spread_y": float(np.nanstd(arr[:, 1])),
            }

        window_rows = []
        window_cols = []
        for pair in self.pairs:
            for tile in (pair.a, pair.b):
                if tile.window is None:
                    continue
                row, col = tile.window
                window_rows.append(float(row))
                window_cols.append(float(col))
        window_summary = None
        if window_rows and window_cols:
            window_summary = {
                "rows": _summary_array(window_rows),
                "cols": _summary_array(window_cols),
            }

        return {
            "datasets": [self.dataset_a, self.dataset_b],
            "pair_count": count,
            "generated_from": self.generated_from,
            "search_radius": self.search_radius,
            "approx_spacing": self.approx_spacing,
            "dataset_resolutions": self.dataset_resolutions,
            "centroid_stats": centroid_summary,
            "window_stats": window_summary,
        }


def _summary_array(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Overlap information not found at {path}.") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse overlap JSON {path}: {exc}") from exc


def load_overlap_pairs(path: Path) -> OverlapSet:
    """Parse an overlap pairs JSON file emitted by `integrate_stac.py`."""

    doc = _load_json(path)

    dataset_pair = doc.get("dataset_pair")
    if not isinstance(dataset_pair, (list, tuple)) or len(dataset_pair) < 2:
        raise ValueError(f"Overlap document {path} does not declare a valid 'dataset_pair'.")
    dataset_a, dataset_b = str(dataset_pair[0]), str(dataset_pair[1])

    generated_from = doc.get("generated_from")
    if generated_from is not None:
        generated_from = str(generated_from)

    dataset_resolutions = {}
    raw_resolutions = doc.get("dataset_resolutions")
    if isinstance(raw_resolutions, dict):
        for key, value in raw_resolutions.items():
            try:
                dataset_resolutions[str(key)] = float(value) if value is not None else None
            except Exception:
                dataset_resolutions[str(key)] = None

    approx_spacing = {}
    raw_spacing = doc.get("approx_spacing")
    if isinstance(raw_spacing, dict):
        for key, value in raw_spacing.items():
            try:
                approx_spacing[str(key)] = float(value) if value is not None else None
            except Exception:
                approx_spacing[str(key)] = None

    search_radius = doc.get("search_radius")
    if search_radius is not None:
        try:
            search_radius = float(search_radius)
        except Exception:
            search_radius = None

    raw_pair_candidates = doc.get("raw_pair_candidates")
    if raw_pair_candidates is not None:
        try:
            raw_pair_candidates = int(raw_pair_candidates)
        except Exception:
            raw_pair_candidates = None

    centroid_geometry = doc.get("overlap") if isinstance(doc.get("overlap"), dict) else None

    pair_entries = doc.get("pairs")
    if not isinstance(pair_entries, list):
        pair_entries = []

    parsed_pairs: List[OverlapPair] = []
    for entry in pair_entries:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_pair(entry, dataset_a, dataset_b)
        if parsed is not None:
            parsed_pairs.append(parsed)

    overlap_set = OverlapSet(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        generated_from=generated_from,
        dataset_resolutions=dataset_resolutions,
        approx_spacing=approx_spacing,
        search_radius=search_radius,
        raw_pair_candidates=raw_pair_candidates,
        centroid_geometry=centroid_geometry,
        pairs=parsed_pairs,
    )
    return overlap_set


def _parse_pair(entry: Dict[str, Any], dataset_a: str, dataset_b: str) -> Optional[OverlapPair]:
    tile_a = _extract_tile(entry, dataset_a, 0, suffixes=("0", "a"))
    tile_b = _extract_tile(entry, dataset_b, 1, suffixes=("1", "b"))
    if tile_a is None or tile_b is None:
        # Some records might encode lookup keys the other way round; try swapping
        tile_a_alt = _extract_tile(entry, dataset_a, 0, suffixes=("0", "a", "b"))
        tile_b_alt = _extract_tile(entry, dataset_b, 1, suffixes=("1", "b", "a"))
        if tile_a_alt is None or tile_b_alt is None:
            return None
        tile_a, tile_b = tile_a_alt, tile_b_alt

    centroid = _extract_centroid(entry)
    notes = {
        key: value
        for key, value in entry.items()
        if key not in _KNOWN_PAIR_KEYS
    }
    return OverlapPair(a=tile_a, b=tile_b, centroid=centroid, notes=notes)


def _extract_tile(
    entry: Dict[str, Any],
    dataset_name: str,
    index: int,
    suffixes: Tuple[str, ...],
) -> Optional[OverlapTile]:
    candidates: Dict[str, Any] = {}

    # Accept keys in priority order to cover multiple serialization variants.
    possible_keys: List[Tuple[str, str]] = []
    base_tokens = [
        f"dataset_{index}",
        f"dataset_{dataset_name}",
        dataset_name,
        f"dataset_{suffixes[0]}",
        f"dataset_{suffixes[-1]}",
    ]
    unique_tokens = []
    for token in base_tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    for token in unique_tokens:
        possible_keys.extend(
            [
                (token, "lookup"),
                (token, "tile"),
                (token, "tile_id"),
                (token, "row_col"),
                (token, "native_point"),
                (token, "window"),
                (token, "window_pixels"),
                (token, "window_size"),
                (token, "window_shape"),
            ]
        )

    resolved: Dict[str, Any] = {}
    for token, suffix in possible_keys:
        key = f"{token}_{suffix}"
        if key in entry and suffix not in resolved:
            resolved[suffix] = entry[key]

    tile_id = _normalise_tile_id(resolved)
    row_col = _normalise_row_col(resolved)
    native_point = _normalise_point(resolved)
    window = _normalise_window(resolved)
    extra = {
        key: value
        for key, value in resolved.items()
        if key not in {"lookup", "tile", "tile_id", "row_col", "native_point", "window", "window_pixels", "window_size", "window_shape"}
    }

    if tile_id is None and row_col is None and native_point is None:
        return None

    return OverlapTile(
        dataset=dataset_name,
        tile_id=tile_id,
        row_col=row_col,
        native_point=native_point,
        window=window,
        extra=extra,
    )


def _normalise_tile_id(resolved: Dict[str, Any]) -> Optional[str]:
    for key in ("lookup", "tile", "tile_id"):
        value = resolved.get(key)
        if value is None:
            continue
        return str(value)
    return None


def _normalise_row_col(resolved: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    value = resolved.get("row_col")
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    if arr.size >= 2:
        row_val, col_val = arr.flat[0], arr.flat[1]
    else:
        return None
    try:
        return int(row_val), int(col_val)
    except Exception:
        return None


def _normalise_point(resolved: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    value = resolved.get("native_point")
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.size < 2:
        return None
    return float(arr.flat[0]), float(arr.flat[1])


def _normalise_window(resolved: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    for key in ("window", "window_pixels", "window_size", "window_shape"):
        value = resolved.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size < 2:
            continue
        try:
            return int(arr.flat[0]), int(arr.flat[1])
        except Exception:
            continue
    return None


def _extract_centroid(entry: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("overlap_centerloid", "overlap_centroid", "centroid"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size >= 2:
            return float(arr.flat[0]), float(arr.flat[1])
    native_a = entry.get("dataset_0_native_point") or entry.get("dataset_a_native_point")
    native_b = entry.get("dataset_1_native_point") or entry.get("dataset_b_native_point")
    try:
        point_a = np.asarray(native_a, dtype=float) if native_a is not None else None
        point_b = np.asarray(native_b, dtype=float) if native_b is not None else None
    except Exception:
        point_a = point_b = None
    if point_a is not None and point_b is not None and point_a.size >= 2 and point_b.size >= 2:
        centroid = (point_a + point_b) / 2.0
        return float(centroid[0]), float(centroid[1])
    return None


_KNOWN_PAIR_KEYS = {
    "dataset_0_lookup",
    "dataset_0_row_col",
    "dataset_0_native_point",
    "dataset_0_window",
    "dataset_0_window_pixels",
    "dataset_0_window_size",
    "dataset_0_window_shape",
    "dataset_1_lookup",
    "dataset_1_row_col",
    "dataset_1_native_point",
    "dataset_1_window",
    "dataset_1_window_pixels",
    "dataset_1_window_size",
    "dataset_1_window_shape",
    "dataset_a_lookup",
    "dataset_a_row_col",
    "dataset_a_native_point",
    "dataset_b_lookup",
    "dataset_b_row_col",
    "dataset_b_native_point",
    "tile_a",
    "tile_b",
    "dataset_a_tile",
    "dataset_b_tile",
    "dataset_a_window",
    "dataset_b_window",
    "dataset_a_window_pixels",
    "dataset_b_window_pixels",
    "overlap_centerloid",
    "overlap_centroid",
    "centroid",
}


def _load_overlap_mask_data(mask_path: Optional[Path]) -> Optional[Dict[str, object]]:
    if mask_path is None:
        return None
    if rasterio is None:
        print(f"[warn] rasterio unavailable; cannot apply overlap mask filtering ({mask_path}).")
        return None
    try:
        with rasterio.open(mask_path) as src:
            mask_array = src.read(1)
            return {
                "array": np.asarray(mask_array),
                "transform": src.transform,
                "shape": mask_array.shape,
                "nodata": src.nodata,
            }
    except Exception as exc:
        print(f"[warn] Unable to load overlap mask {mask_path}: {exc}")
        return None


def _extract_reembedding_overlap(
    entry: Optional[Dict[str, object]],
    overlap_mask: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if entry is None:
        return None
    mask_flags = entry.get("mask_flags")
    if overlap_mask is None or mask_flags is None:
        return entry
    keep_idx = [idx for idx, flag in enumerate(mask_flags) if flag]
    if not keep_idx:
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required to process overlap re-embeddings.")
    index_tensor = torch.as_tensor(keep_idx, dtype=torch.long)
    features_field = entry.get("features")
    if isinstance(features_field, torch.Tensor):
        features = features_field.index_select(0, index_tensor)
    else:
        features = np.asarray(features_field)[keep_idx]

    labels = entry.get("labels")

    def _slice_list(values):
        if values is None:
            return None
        return [values[i] for i in keep_idx]

    indices_slice = _slice_list(entry.get("indices")) or []
    coords_slice = _slice_list(entry.get("coords")) or []
    metadata_slice = _slice_list(entry.get("metadata")) or []
    row_cols_mask_slice = _slice_list(entry.get("row_cols_mask")) or []
    if isinstance(labels, np.ndarray):
        labels_slice = labels[keep_idx]
    else:
        label_list = _slice_list(labels) or []
        labels_slice = np.asarray(label_list, dtype=np.int16)
    filtered = {
        "dataset": entry.get("dataset"),
        "features": features,
        "indices": list(indices_slice),
        "coords": list(coords_slice),
        "metadata": list(metadata_slice),
        "labels": labels_slice,
        "row_cols": list(row_cols_mask_slice),
        "mask_flags": [True] * len(keep_idx),
        "row_cols_mask": list(row_cols_mask_slice),
    }
    return filtered
