# Common/data_utils.py
from __future__ import annotations

import glob
import os
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from affine import Affine
try:  # Optional heavy imports; guard for environments without rasterio/shapely
    import rasterio  # type: ignore
    from rasterio.features import shapes as raster_shapes, rasterize  # type: ignore
    from rasterio.warp import Resampling, reproject  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore
    raster_shapes = None  # type: ignore
    rasterize = None  # type: ignore
    Resampling = None  # type: ignore
    reproject = None  # type: ignore

try:
    from shapely.geometry import Polygon, MultiPolygon, shape, mapping  # type: ignore
    from shapely.ops import unary_union  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Polygon = MultiPolygon = shape = mapping = None  # type: ignore
    unary_union = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_THIS_DIR.parent / "Methods" / "0_Benchmark_GFM4MPM").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.gfm4mpm.data.geo_stack import GeoStack  # noqa: E402
from src.gfm4mpm.data.stac_table import StacTableStack  # noqa: E402


@dataclass
class BoundaryInfo:
    path: Path
    mask: np.ndarray
    transform: Affine
    crs: Optional[str]
    geometry: Optional[Union[Polygon, MultiPolygon]]


@dataclass
class SplitDataLoad:
    stack: Any
    stack_root: Optional[Path]
    label_column: str
    feature_columns: Optional[List[str]]
    training_args: Optional[Dict]
    training_args_path: Optional[Path]
    region_stacks: Dict[str, GeoStack]
    label_paths: Dict[str, List[Path]]
    mode: str  # 'table' or 'raster'


def load_split_stack(
    stac_root: Optional[str],
    stac_table: Optional[str],
    bands_glob: Optional[str],
    encoder_path: Path,
    default_label: str,
    load_training_args: Callable[[Optional[Path], Path], Tuple[Optional[Dict], Optional[Path]]],
    load_training_metadata: Callable[[Optional[Path], Path], Tuple[Optional[Dict], Optional[Path]]],
    resolve_label_rasters: Callable[[Optional[Dict], Optional[Path], str], List[Dict[str, Any]]],
    collect_feature_rasters: Callable[[Optional[Dict], Optional[Path], Optional[Sequence[str]]], Dict[str, List[Path]]],
    infer_region: Callable[[Path], Optional[str]],
) -> SplitDataLoad:
    stack_root_path: Optional[Path] = None
    training_args_data: Optional[Dict] = None
    training_args_path: Optional[Path] = None
    training_metadata: Optional[Dict] = None
    training_metadata_path: Optional[Path] = None
    feature_columns: Optional[List[str]] = None
    label_column = default_label

    if stac_root or stac_table:
        stack_source = Path(stac_table or stac_root).resolve()
        stack_root_path = stack_source.parent if stack_source.is_file() else stack_source
        training_args_data, training_args_path = load_training_args(stack_root_path, encoder_path)
        training_metadata, training_metadata_path = load_training_metadata(stack_root_path, encoder_path)

        lat_column = None
        lon_column = None
        if training_args_data:
            features_raw = training_args_data.get("features")
            if isinstance(features_raw, list):
                feature_columns = list(dict.fromkeys(str(f) for f in features_raw))
            lat_column = training_args_data.get("lat_column")
            lon_column = training_args_data.get("lon_column")
            label_column = training_args_data.get("label_column", label_column)

        stack = StacTableStack(
            stack_source,
            label_columns=[label_column],
            feature_columns=feature_columns,
            latitude_column=lat_column,
            longitude_column=lon_column,
        )
        if feature_columns:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")

        return SplitDataLoad(
            stack=stack,
            stack_root=stack_root_path,
            label_column=label_column,
            feature_columns=feature_columns,
            training_args=training_args_data,
            training_args_path=training_args_path,
            region_stacks={},
            label_paths={},
            mode="table",
        )

    band_paths_glob = sorted(glob.glob(bands_glob or ""))
    if not band_paths_glob:
        raise RuntimeError(f"No raster bands matched pattern {bands_glob}")

    try:
        common = os.path.commonpath(band_paths_glob)  # type: ignore[name-defined]
        stack_root_path = Path(common)
    except ValueError:
        stack_root_path = Path(band_paths_glob[0]).resolve().parent

    metadata_root = find_metadata_root(stack_root_path)

    training_args_data, training_args_path = load_training_args(metadata_root, encoder_path)
    training_metadata, training_metadata_path = load_training_metadata(metadata_root, encoder_path)
    if training_args_data:
        features_raw = training_args_data.get("features")
        if isinstance(features_raw, list):
            feature_columns = list(dict.fromkeys(str(f) for f in features_raw))
        label_column = training_args_data.get("label_column", label_column)

    region_stacks, feature_names, metadata, feature_entries = load_region_stacks(metadata_root, feature_columns)
    if not region_stacks:
        raise RuntimeError("No region stacks could be constructed; check raster metadata")

    boundary_sources: List[Union[str, Path]] = []
    boundary_dir = (metadata_root / "boundaries").resolve()
    if boundary_dir.exists():
        boundary_sources.append(boundary_dir)

    if training_metadata:
        boundaries_section = training_metadata.get("boundaries")
        if isinstance(boundaries_section, dict):
            def _collect_boundary_records(records: Any) -> None:
                if not isinstance(records, list):
                    return
                for entry in records:
                    if not isinstance(entry, dict):
                        continue
                    path_val = entry.get("path") or entry.get("filename")
                    if not path_val:
                        continue
                    candidate = Path(path_val)
                    if not candidate.is_absolute():
                        candidate = (metadata_root / candidate).resolve()
                    boundary_sources.append(candidate)

            project_records = boundaries_section.get("project")
            if project_records:
                _collect_boundary_records(project_records)
            else:
                for value in boundaries_section.values():
                    _collect_boundary_records(value)

    boundary_infos: Dict[str, BoundaryInfo] = {}
    if boundary_sources:
        boundary_infos = load_region_boundaries(boundary_sources, project_only=True)
        if boundary_infos:
            applied_masks = apply_boundaries_to_region_stacks(region_stacks, boundary_infos)
            if applied_masks:
                applied_regions = ", ".join(sorted(applied_masks.keys()))
                source_desc = ", ".join({str(Path(src)) for src in boundary_sources})
                print(
                    f"[info] Applied boundary constraints for {len(applied_masks)} region(s) from {source_desc}: {applied_regions}"
                )

    label_paths_by_region = collect_label_paths_by_region(
        metadata,
        training_metadata_path,
        label_column,
        fallback_root=metadata_root,
    )

    if feature_columns is None:
        feature_columns = list(feature_names)

    base_stack, resolved_features = build_geostack_for_regions(feature_columns, region_stacks)
    feature_columns = list(resolved_features or feature_columns)

    return SplitDataLoad(
        stack=base_stack,
        stack_root=metadata_root,
        label_column=label_column,
        feature_columns=feature_columns,
        training_args=training_args_data,
        training_args_path=training_args_path,
        region_stacks=region_stacks,
        label_paths=label_paths_by_region,
        mode="raster",
    )


def TwoD_data_TwoD_label_neighbor_stats(values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbour sums and counts for a 2D grid (8-connected)."""
    padded_vals = np.pad(values, 1, mode="constant", constant_values=np.nan)
    padded_valid = np.pad(valid_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    sums = np.zeros_like(values, dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            window_vals = padded_vals[1 + dy : 1 + dy + values.shape[0], 1 + dx : 1 + dx + values.shape[1]]
            window_mask = padded_valid[1 + dy : 1 + dy + values.shape[0], 1 + dx : 1 + dx + values.shape[1]].astype(bool)
            sums += np.where(window_mask, window_vals, 0.0)
            counts += window_mask.astype(np.int16)
    return sums, counts


def TwoD_data_TwoD_label_infer_raster_kind(
    values: np.ndarray,
    *,
    dtype: Optional[np.dtype] = None,
    max_unique: int = 32,
    integer_tolerance: float = 1e-6,
) -> tuple[str, Optional[List[float]]]:
    """Infer whether a raster should be treated as categorical or continuous."""
    if values.size == 0:
        return "empty", None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "empty", None
    dtype = np.dtype(dtype) if dtype is not None else values.dtype
    unique_values = np.unique(finite)
    if unique_values.size <= max_unique:
        if np.issubdtype(dtype, np.integer):
            return "categorical", unique_values.astype(int).tolist()
        rounded = np.round(unique_values)
        if np.all(np.abs(unique_values - rounded) <= integer_tolerance):
            return "categorical", rounded.astype(int).tolist()
    return "continuous", None


def TwoD_data_TwoD_label_fill_missing(
    grid: np.ndarray,
    *,
    valid_mask: np.ndarray,
    max_iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing cells using neighbour averages."""
    filled = grid.copy()
    valid = valid_mask.copy()
    for _ in range(max_iterations):
        if valid.all():
            break
        neighbour_sum, neighbour_count = TwoD_data_TwoD_label_neighbor_stats(filled, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            neighbour_mean = np.divide(neighbour_sum, neighbour_count, where=neighbour_count > 0)
        fill_mask = (~valid) & (neighbour_count > 0)
        if not fill_mask.any():
            break
        filled[fill_mask] = neighbour_mean[fill_mask]
        valid[fill_mask] = True
    return filled, valid


def TwoD_data_TwoD_label_smooth_grid(
    grid: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Apply simple neighbour smoothing to a 2D grid."""
    smoothed = grid.copy()
    for _ in range(max(iterations, 0)):
        neighbour_sum, neighbour_count = TwoD_data_TwoD_label_neighbor_stats(smoothed, np.isfinite(smoothed))
        total_sum = neighbour_sum + smoothed
        total_count = neighbour_count + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.divide(total_sum, total_count, where=total_count > 0)
    return smoothed


def TwoD_data_TwoD_label_normalize_raster(
    array: np.ndarray,
    *,
    nodata: Optional[float] = None,
    dtype: Optional[np.dtype] = None,
    fill_iterations: int = 6,
    smoothing_iterations: int = 1,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Normalise a 2D raster band using outlier clipping, imputation, and scaling."""
    if array.ndim != 2:
        raise ValueError("TwoD_data_TwoD_label_normalize_raster expects a 2D array.")
    original = np.array(array, copy=True)
    dtype = np.dtype(dtype) if dtype is not None else original.dtype
    mask = np.isfinite(original)
    if nodata is not None:
        mask &= original != nodata
    finite_values = original[mask]
    if finite_values.size == 0:
        return original.astype(np.float32), {
            "kind": "empty",
            "dtype": str(dtype),
            "nodata": float(nodata) if nodata is not None else None,
            "scaling": None,
            "categories": None,
        }

    kind, categories = TwoD_data_TwoD_label_infer_raster_kind(finite_values, dtype=dtype)
    if kind == "categorical":
        categorical = original.copy()
        categorical[~mask] = nodata if nodata is not None else 0
        info: Dict[str, Any] = {
            "kind": "categorical",
            "dtype": str(dtype),
            "nodata": float(nodata) if nodata is not None else None,
            "categories": categories or [],
            "scaling": None,
        }
        return categorical.astype(dtype), info

    working = original.astype(np.float64)
    working[~mask] = np.nan

    q1 = np.nanpercentile(working, 25)
    q3 = np.nanpercentile(working, 75)
    iqr = q3 - q1
    if np.isfinite(iqr) and iqr > 0:
        upper = q3 + 1.5 * iqr
        p975 = np.nanpercentile(working, 97.5)
        capped = np.clip(working, None, upper)
        working = np.where(capped > upper, p975, capped)

    filled, valid = TwoD_data_TwoD_label_fill_missing(working, valid_mask=np.isfinite(working), max_iterations=fill_iterations)
    still_nan = np.isnan(filled)
    if still_nan.any():
        filled[still_nan] = np.nanmedian(filled)

    if smoothing_iterations > 0:
        filled = TwoD_data_TwoD_label_smooth_grid(filled, iterations=smoothing_iterations)

    mean = float(np.nanmean(filled))
    std = float(np.nanstd(filled))
    if std <= 0 or not np.isfinite(std):
        std = 1.0
    normalized = (filled - mean) / std
    normalized = normalized.astype(np.float32)

    info = {
        "kind": "continuous",
        "dtype": "float32",
        "nodata": None,
        "categories": None,
        "scaling": {"mean": mean, "std": std},
        "filled_fraction": float(valid.sum()) / float(valid.size),
    }
    return normalized, info


def TwoD_data_TwoD_label_rasterize_labels(
    geometries: Iterable,
    *,
    template_profile: Dict[str, Any],
    burn_value: float = 1.0,
    dtype: str = "uint8",
    fill_value: float = 0.0,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize vector geometries onto a template raster grid."""
    from rasterio import features

    height = template_profile.get("height")
    width = template_profile.get("width")
    transform = template_profile.get("transform")
    if height is None or width is None or transform is None:
        raise ValueError("Template profile must supply 'height', 'width', and 'transform'.")

    shapes = [(geom, burn_value) for geom in geometries if geom is not None]
    if not shapes:
        return np.zeros((height, width), dtype=np.dtype(dtype))

    raster = features.rasterize(
        shapes,
        out_shape=(int(height), int(width)),
        transform=transform,
        fill=fill_value,
        dtype=dtype,
        all_touched=all_touched,
    )
    return raster


def clamp_coords_to_window(
    coords: Sequence[CoordLike],
    stack: Any,
    window: int,
) -> Tuple[List[Union[Tuple[int, int], RegionCoord]], int]:

    
    if not coords:
        return [], 0

    def _split(coord: Union[Tuple[Any, ...], List[Any], Dict[str, Any]]) -> Tuple[Optional[str], int, int]:
        if isinstance(coord, dict):
            region_val = coord.get("region")
            row_val = coord.get("row")
            col_val = coord.get("col")
            if region_val is None or row_val is None or col_val is None:
                raise ValueError(f"Coordinate dictionary missing keys: {coord}")
            return str(region_val), int(row_val), int(col_val)
        if isinstance(coord, (tuple, list)):
            if len(coord) == 3:
                region_val, row_val, col_val = coord
                return str(region_val), int(row_val), int(col_val)
            if len(coord) == 2:
                row_val, col_val = coord
                return None, int(row_val), int(col_val)
        raise TypeError(f"Unsupported coordinate format: {coord!r}")

    window = int(window)
    if window <= 1:
        packed = [_split(coord) for coord in coords]
        if hasattr(stack, "resolve_region_stack"):
            return [(region or stack.regions[0], row, col) for region, row, col in packed], 0
        return [(row, col) for _, row, col in packed], 0

    half = max(window // 2, 0)

    if hasattr(stack, "resolve_region_stack"):
        clamped: List[RegionCoord] = []
        clamp_count = 0
        for coord in coords:
            region, row, col = _split(coord)
            region_key = region or getattr(stack, "default_region", None)
            if region_key is None:
                available = list(getattr(stack, "regions", []))
                if not available:
                    raise ValueError("Region-aware stack provided no regions to resolve coordinates against.")
                region_key = available[0]
            region_stack = stack.resolve_region_stack(region_key)
            if region_stack is None:
                raise KeyError(f"Unknown region '{region_key}' for coordinate {coord!r}")
            height = getattr(region_stack, "height", None)
            width = getattr(region_stack, "width", None)
            if height is None or width is None:
                clamped.append((region_key, int(row), int(col)))
                continue
            min_row = half
            max_row = max(height - half - 1, half)
            min_col = half
            max_col = max(width - half - 1, half)

            valid_rows = getattr(region_stack, "_valid_rows", None)
            valid_cols = getattr(region_stack, "_valid_cols", None)
            if isinstance(valid_rows, np.ndarray) and valid_rows.size:
                min_row = max(min_row, int(valid_rows.min()))
                max_row = min(max_row, int(valid_rows.max()))
            if isinstance(valid_cols, np.ndarray) and valid_cols.size:
                min_col = max(min_col, int(valid_cols.min()))
                max_col = min(max_col, int(valid_cols.max()))

            if max_row < min_row:
                clamp_val = max(0, min(height - 1, half))
                min_row = max_row = clamp_val
            if max_col < min_col:
                clamp_val = max(0, min(width - 1, half))
                min_col = max_col = clamp_val

            row_int = int(row)
            col_int = int(col)
            clamped_r = min(max(row_int, min_row), max_row)
            clamped_c = min(max(col_int, min_col), max_col)
            if clamped_r != row_int or clamped_c != col_int:
                clamp_count += 1
            clamped.append((region_key, clamped_r, clamped_c))
        return clamped, clamp_count

    # Single-region raster or table stack
    height = getattr(stack, "height", None)
    width = getattr(stack, "width", None)
    if height is None or width is None:
        return [(int(r), int(c)) for _, r, c in (_split(coord) for coord in coords)], 0

    min_row = half
    max_row = max(height - half - 1, half)
    min_col = half
    max_col = max(width - half - 1, half)

    valid_rows = getattr(stack, "_valid_rows", None)
    valid_cols = getattr(stack, "_valid_cols", None)
    if isinstance(valid_rows, np.ndarray) and valid_rows.size:
        min_row = max(min_row, int(valid_rows.min()))
        max_row = min(max_row, int(valid_rows.max()))
    if isinstance(valid_cols, np.ndarray) and valid_cols.size:
        min_col = max(min_col, int(valid_cols.min()))
        max_col = min(max_col, int(valid_cols.max()))

    if max_row < min_row:
        clamp_val = max(0, min(height - 1, half))
        min_row = max_row = clamp_val
    if max_col < min_col:
        clamp_val = max(0, min(width - 1, half))
        min_col = max_col = clamp_val

    clamped_rows: List[Tuple[int, int]] = []
    clamp_count = 0
    for _, row, col in (_split(coord) for coord in coords):
        row_int = int(row)
        col_int = int(col)
        clamped_r = min(max(row_int, min_row), max_row)
        clamped_c = min(max(col_int, min_col), max_col)
        if clamped_r != row_int or clamped_c != col_int:
            clamp_count += 1
        clamped_rows.append((clamped_r, clamped_c))

    return clamped_rows, clamp_count


def filter_valid_raster_coords(
    stack: Any,
    coords: Sequence[RegionCoord],
    window: int,
    *,
    min_valid_fraction: float = 0.0,
) -> Tuple[List[RegionCoord], List[RegionCoord]]:
    """Remove coordinates whose patches contain no valid data."""

    coords_list = list(coords)
    if not coords_list:
        return [], []

    min_valid_fraction = max(0.0, min(1.0, float(min_valid_fraction)))

    result = _filter_valid_raster_coords_integral(
        stack,
        coords_list,
        int(window),
        min_valid_fraction,
    )
    if result is not None:
        return result

    return _filter_valid_raster_coords_patch(
        stack,
        coords_list,
        int(window),
        min_valid_fraction,
    )


def _ensure_integral_mask(stack_obj: Any) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    if rasterio is None:
        return None
    integral = getattr(stack_obj, "_valid_integral", None)
    shape = getattr(stack_obj, "_valid_integral_shape", None)
    if integral is not None and shape is not None:
        return integral, shape

    srcs = getattr(stack_obj, "srcs", None)
    if not srcs:
        return None
    try:
        mask_arr = srcs[0].dataset_mask()
    except Exception:
        return None
    if mask_arr is None:
        return None
    mask_arr = np.asarray(mask_arr)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[0]
    mask_bool = mask_arr != 0
    if not mask_bool.any():
        return None

    valid = mask_bool.astype(np.int32, copy=False)
    integral = valid.cumsum(axis=0).cumsum(axis=1)
    stack_obj._valid_integral = integral
    stack_obj._valid_integral_shape = (valid.shape[0], valid.shape[1])
    return integral, (valid.shape[0], valid.shape[1])


def _window_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    """Return sum of values in [r0, r1) x [c0, c1)."""
    total = integral[r1 - 1, c1 - 1]
    if r0 > 0:
        total -= integral[r0 - 1, c1 - 1]
    if c0 > 0:
        total -= integral[r1 - 1, c0 - 1]
    if r0 > 0 and c0 > 0:
        total += integral[r0 - 1, c0 - 1]
    return int(total)


def _filter_valid_raster_coords_integral(
    stack: Any,
    coords_list: Sequence[RegionCoord],
    window: int,
    min_valid_fraction: float,
) -> Optional[Tuple[List[RegionCoord], List[RegionCoord]]]:
    if rasterio is None:
        return None

    half = window // 2
    use_progress = tqdm is not None and len(coords_list) > 0
    global_mask = None
    if not hasattr(stack, "resolve_region_stack"):
        global_mask = _get_valid_window_mask(stack, window)
        if global_mask is None:
            return coords_list

    global_mask = None
    if not hasattr(stack, "resolve_region_stack"):
        global_mask = _get_valid_window_mask(stack, window)
        if global_mask is None:
            return coords_list

    iterator = coords_list
    if use_progress:
        iterator = tqdm(coords_list, desc="Validate raster windows")

    cache: Dict[Any, bool] = {}
    valid: List[RegionCoord] = []
    invalid: List[RegionCoord] = []

    try:
        for coord in iterator:
            if hasattr(stack, "resolve_region_stack"):
                region, row, col = coord
                region_stack = stack.resolve_region_stack(region)
                if region_stack is None:
                    invalid.append(coord)
                    continue
            else:
                region_stack = stack
                if isinstance(coord, (tuple, list)):
                    if len(coord) == 3:
                        _, row, col = coord
                    elif len(coord) == 2:
                        row, col = coord
                    else:
                        invalid.append(coord)
                        continue
                else:
                    invalid.append(coord)
                    continue

            row_int = int(row)
            col_int = int(col)

            key = (region, row_int, col_int) if hasattr(stack, "resolve_region_stack") else (row_int, col_int)
            cached = cache.get(key)
            if cached is not None:
                (valid if cached else invalid).append(coord)
                continue

            integral_info = _ensure_integral_mask(region_stack)
            if integral_info is None:
                return None  # fallback to patch-based method
            integral, (height, width) = integral_info

            r0 = max(row_int - half, 0)
            r1 = min(r0 + window, height)
            if r1 <= r0:
                cache[key] = False
                invalid.append(coord)
                continue
            c0 = max(col_int - half, 0)
            c1 = min(c0 + window, width)
            if c1 <= c0:
                cache[key] = False
                invalid.append(coord)
                continue

            area = (r1 - r0) * (c1 - c0)
            if area <= 0:
                cache[key] = False
                invalid.append(coord)
                continue

            count_valid = _window_sum(integral, r0, c0, r1, c1)
            fraction = count_valid / float(area)
            is_valid = count_valid > 0 and fraction >= min_valid_fraction
            cache[key] = is_valid
            (valid if is_valid else invalid).append(coord)
    finally:
        if use_progress:
            print("[info] Validate raster windows completed (integral).")

    return valid, invalid


def _filter_valid_raster_coords_patch(
    stack: Any,
    coords_list: Sequence[RegionCoord],
    window: int,
    min_valid_fraction: float,
) -> Tuple[List[RegionCoord], List[RegionCoord]]:
    cache: Dict[Any, bool] = {}
    valid: List[RegionCoord] = []
    invalid: List[RegionCoord] = []

    use_progress = tqdm is not None and len(coords_list) > 0
    iterator = coords_list
    if use_progress:
        iterator = tqdm(coords_list, desc="Validate raster windows")

    for coord in iterator:
        try:
            if hasattr(stack, "resolve_region_stack"):
                region, row, col = coord
                region_stack = stack.resolve_region_stack(region)
            else:
                region_stack = stack
                if isinstance(coord, (tuple, list)) and len(coord) == 3:
                    _, row, col = coord
                elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                    row, col = coord
                else:
                    raise ValueError(f"Unsupported coordinate for validation: {coord!r}")
            row_int = int(row)
            col_int = int(col)

            key = (region, row_int, col_int) if hasattr(stack, "resolve_region_stack") else (row_int, col_int)
            cached = cache.get(key)
            if cached is not None:
                (valid if cached else invalid).append(coord)
                continue

            patch, mask = region_stack.read_patch_with_mask(row_int, col_int, int(window))
        except Exception:
            cache[key] = False
            invalid.append(coord)
            continue

        patch = np.asarray(patch)
        if patch.size == 0 or not np.isfinite(patch).any():
            cache[key] = False
            invalid.append(coord)
            continue

        mask_bool = np.asarray(mask).astype(bool, copy=False)
        if mask_bool.shape != patch.shape[-2:]:
            mask_bool = np.broadcast_to(mask_bool, patch.shape[-2:])
        valid_fraction = 1.0 - (mask_bool.sum() / mask_bool.size)
        is_valid = not mask_bool.all() and valid_fraction >= min_valid_fraction
        cache[key] = is_valid
        (valid if is_valid else invalid).append(coord)

    if use_progress:
        print("[info] Validate raster windows completed (patch).")

    return valid, invalid


def count_valid_window_centers(stack: Any, window: int) -> Optional[int]:
    window = int(window)
    if window <= 1:
        if hasattr(stack, "resolve_region_stack") and hasattr(stack, "iter_region_stacks"):
            total = 0
            for _, region_stack in stack.iter_region_stacks():
                h = getattr(region_stack, "height", None)
                w = getattr(region_stack, "width", None)
                if h is None or w is None:
                    return None
                total += int(h) * int(w)
            return total
        h = getattr(stack, "height", None)
        w = getattr(stack, "width", None)
        if h is not None and w is not None:
            return int(h) * int(w)
        return None

    def _count_for_stack(stack_obj: Any) -> Optional[int]:
        mask = _get_valid_window_mask(stack_obj, window)
        if mask is None:
            return None
        return int(mask.sum())

    if hasattr(stack, "resolve_region_stack") and hasattr(stack, "iter_region_stacks"):
        total = 0
        for _, region_stack in stack.iter_region_stacks():
            count = _count_for_stack(region_stack)
            if count is None:
                return None
            total += count
        return total

    return _count_for_stack(stack)


class MultiRegionStack:
    """Region-aware wrapper that round-robins across multiple GeoStacks."""

    def __init__(self, feature_names: Sequence[str], region_stacks: Dict[str, GeoStack]):
        if not region_stacks:
            raise ValueError("region_stacks must not be empty")

        self.feature_columns: List[str] = list(feature_names)
        ordered_regions = sorted(region_stacks.items(), key=lambda kv: kv[0])
        self._regions: List[Tuple[str, GeoStack]] = []
        self._region_map: Dict[str, GeoStack] = {}
        for region, stack in ordered_regions:
            if not isinstance(stack, GeoStack):
                raise TypeError(f"Region '{region}' must map to a GeoStack instance.")
            self._regions.append((region, stack))
            self._region_map[region] = stack
            print(f"[info] Region '{region}' prepared with {stack.count} feature map(s).")

        reference = self._regions[0][1]
        self.count = reference.count
        self.kind = "raster"
        self.is_table = False
        self.default_region = self._regions[0][0]
        self.feature_metadata = [
            {"regions": [region for region, _ in self._regions]}
            for _ in self.feature_columns
        ]
        self._cycle_index = int(np.random.default_rng().integers(0, len(self._regions)))
        self.band_paths_by_region: Dict[str, List[str]] = {
            region: list(getattr(stack, "band_paths", []))
            for region, stack in self._regions
        }
        self.band_paths: List[str] = list(self.band_paths_by_region.get(self.default_region, []))
        self.srcs = list(getattr(reference, "srcs", []))

    @property
    def regions(self) -> List[str]:
        return [region for region, _ in self._regions]

    def iter_region_stacks(self) -> Iterable[Tuple[str, GeoStack]]:
        return tuple(self._regions)

    def resolve_region_stack(self, region: str) -> GeoStack:
        try:
            return self._region_map[str(region)]
        except KeyError as exc:
            raise KeyError(f"Region '{region}' is not available in this stack.") from exc

    def region_shapes(self) -> Dict[str, Tuple[int, int]]:
        return {
            region: (stack.height, stack.width)
            for region, stack in self._regions
        }

    def _normalize_coord(self, coord: CoordLike, *, default_region: Optional[str] = None) -> RegionCoord:
        if isinstance(coord, dict):
            region = coord.get("region")
            row = coord.get("row")
            col = coord.get("col")
        elif isinstance(coord, (tuple, list)):
            if len(coord) == 3:
                region, row, col = coord
            elif len(coord) == 2:
                region = default_region or self.default_region
                row, col = coord
            else:
                raise ValueError(f"Unsupported coordinate tuple: {coord!r}")
        else:
            raise TypeError(f"Unsupported coordinate type: {type(coord)}")

        if region is None:
            region = default_region or self.default_region
        region_str = str(region)
        if region_str not in self._region_map:
            raise KeyError(f"Region '{region_str}' not present in stack.")
        return region_str, int(row), int(col)

    def _parse_coord_and_window(self, *args, **kwargs) -> Tuple[RegionCoord, int]:
        if args and isinstance(args[0], (tuple, list, dict)):
            coord_like = args[0]
            if len(args) < 2:
                raise ValueError("Window size missing for read_patch call.")
            window = int(args[1])
            coord = self._normalize_coord(coord_like)
            return coord, window

        if len(args) == 4:
            region, row, col, window = args
            coord = self._normalize_coord((region, row, col))
            return coord, int(window)

        if len(args) == 3:
            row, col, window = args
            coord = self._normalize_coord((self.default_region, row, col))
            return coord, int(window)

        window_kw = kwargs.get("window")
        coord_kw = kwargs.get("coord")
        if coord_kw is not None and window_kw is not None:
            coord = self._normalize_coord(coord_kw)
            return coord, int(window_kw)

        raise TypeError("Unsupported call signature for region-aware read_patch.")

    def random_coord(
        self,
        window: int,
        rng: np.random.Generator,
    ) -> RegionCoord:
        region_idx = int(rng.integers(0, len(self._regions)))
        region_name, region_stack = self._regions[region_idx]
        row, col = region_stack.random_coord(window, rng)
        return region_name, int(row), int(col)

    def grid_centers(self, stride: int) -> List[RegionCoord]:
        centers: List[RegionCoord] = []
        for region, stack in self._regions:
            region_centers = stack.grid_centers(stride)
            centers.extend((region, int(r), int(c)) for r, c in region_centers)
        return centers

    def read_patch(self, *args, **kwargs) -> np.ndarray:
        coord, window = self._parse_coord_and_window(*args, **kwargs)
        region, row, col = coord
        stack = self.resolve_region_stack(region)
        return stack.read_patch(row, col, window)

    def read_patch_with_mask(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        coord, window = self._parse_coord_and_window(*args, **kwargs)
        region, row, col = coord
        stack = self.resolve_region_stack(region)
        return stack.read_patch_with_mask(row, col, window)

    def sample_patch(
        self,
        window: int,
        rng: np.random.Generator,
        *,
        skip_nan: bool = True,
        max_attempts: int = 32,
        return_region: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, str]]:
        attempts = max(1, int(max_attempts))
        last_patch: Optional[np.ndarray] = None
        last_mask: Optional[np.ndarray] = None
        last_region: Optional[str] = None
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
            last_region = region_name
            if return_region:
                return patch, mask_no_feature, region_name
            return patch, mask_no_feature

        if skip_nan:
            raise RuntimeError(
                "Failed to sample a finite patch from any regional raster; "
                "consider disabling --skip-nan or inspecting raster coverage."
            )

        if last_patch is not None:
            fallback_mask = last_mask if last_mask is not None else np.zeros(last_patch.shape[-2:], dtype=bool)
            if return_region:
                return last_patch, fallback_mask, last_region or self.default_region
            return last_patch, fallback_mask

        raise RuntimeError(
            "Unable to sample a patch from regional rasters; verify raster coverage and metadata."
        )


def build_geostack_for_regions(
    feature_names: Sequence[str],
    region_stacks: Dict[str, GeoStack],
    numeric_filter: Optional[Callable[[GeoStack], GeoStack]] = None,
) -> Tuple[Any, List[str]]:
    feature_names = list(feature_names)
    if not region_stacks:
        raise ValueError("region_stacks must not be empty")

    if numeric_filter:
        filtered: Dict[str, GeoStack] = {}
        for region_key, stack in region_stacks.items():
            result = numeric_filter(stack)
            filtered[region_key] = result if isinstance(result, GeoStack) else stack
        region_stacks = filtered

    if len(region_stacks) == 1:
        region_name, stack = next(iter(region_stacks.items()))
        stack.feature_columns = list(feature_names)
        stack.feature_metadata = [{"regions": [region_name]} for _ in feature_names]
        return stack, list(getattr(stack, "feature_columns", feature_names))

    stack_multi = MultiRegionStack(feature_names, region_stacks)
    return stack_multi, list(getattr(stack_multi, "feature_columns", feature_names))


def load_stac_rasters(
    stac_root: Path,
    features: Optional[Sequence[str]] = None,
    numeric_filter: Optional[Callable[[GeoStack], GeoStack]] = None,
) -> Tuple[Any, Dict[str, GeoStack], List[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, List[Path]]]:
    stac_root = Path(stac_root).resolve()
    metadata_root = stac_root
    feature_names, region_paths, metadata, feature_entries = collect_region_feature_paths(stac_root, features)
    region_stacks: Dict[str, GeoStack] = {}
    for region, paths in region_paths.items():
        stack = GeoStack([str(p) for p in paths])
        stack.feature_columns = list(feature_names)
        stack.feature_metadata = [{"regions": [region]} for _ in feature_names]
        region_stacks[region] = stack

    boundary_dir = (metadata_root / "boundaries").resolve()
    if boundary_dir.exists():
        boundary_infos = load_region_boundaries(boundary_dir, project_only=True)
        if boundary_infos:
            applied_masks = apply_boundaries_to_region_stacks(region_stacks, boundary_infos)
            if applied_masks:
                applied_regions = ", ".join(sorted(applied_masks.keys()))
                print(
                    f"[info] Applied boundary constraints for {len(applied_masks)} region(s) from {boundary_dir}: {applied_regions}"
                )

    boundary_dir = (stac_root / "boundaries").resolve()
    if boundary_dir.exists():
        boundary_geoms = load_region_boundaries(boundary_dir, project_only=True)
        if boundary_geoms:
            applied_masks = apply_boundaries_to_region_stacks(region_stacks, boundary_geoms)
            if applied_masks:
                applied_regions = ", ".join(sorted(applied_masks.keys()))
                print(
                    f"[info] Applied boundary constraints for {len(applied_masks)} region(s) from {boundary_dir}: {applied_regions}"
                )

    base_stack, resolved_features = build_geostack_for_regions(
        feature_names,
        region_stacks,
        numeric_filter=numeric_filter,
    )
    feature_list = list(resolved_features or feature_names)
    return base_stack, region_stacks, feature_list, metadata, feature_entries, region_paths


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


def collect_region_feature_paths(
    stac_root: Path,
    features: Optional[Sequence[str]],
) -> Tuple[List[str], Dict[str, List[Path]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
                    f"[info] No explicit feature subset provided; using all {len(selected_pairs)} feature(s) from metadata: {summary_names}"
                )

            feature_names = [name for name, _ in selected_pairs]
            region_to_feature_paths: Dict[str, Dict[str, Path]] = defaultdict(dict)

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
                    region_to_feature_paths[region][feature_name] = tif_path
                    print(f"        - {tif_path} (region={region})")

            usable_regions: Dict[str, List[Path]] = {}
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
        ordered: List[Path] = []
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
                raise FileNotFoundError(
                    f"No raster asset matching feature '{feature}' (slug '{slug}') under {assets_dir}"
                )
            ordered.append(match)
            remaining.remove(match)
        candidates = ordered
        feature_names = list(features)
    else:
        candidates = sorted(candidates)
        feature_names = [Path(p).stem for p in candidates]

    region_paths = {"GLOBAL": [path.resolve() for path in candidates]}
    return feature_names, region_paths, metadata, None


def load_region_stacks(
    stac_root: Path,
    features: Optional[Sequence[str]],
) -> Tuple[Dict[str, GeoStack], List[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    feature_names, region_paths, metadata, entries = collect_region_feature_paths(stac_root, features)
    region_stacks: Dict[str, GeoStack] = {}
    for region, paths in region_paths.items():
        stack = GeoStack([str(p) for p in paths])
        stack.feature_columns = list(feature_names)
        stack.feature_metadata = [{"regions": [region]} for _ in feature_names]
        region_stacks[region] = stack
    return region_stacks, feature_names, metadata, entries


def collect_label_paths_by_region(
    metadata: Optional[Dict[str, Any]],
    metadata_path: Optional[Path],
    label_column: str,
    fallback_root: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    paths_by_region: Dict[str, List[Path]] = defaultdict(list)
    if metadata and metadata_path:
        labels_section = metadata.get("labels")
        if isinstance(labels_section, dict):
            entries = labels_section.get("entries")
            if isinstance(entries, dict):
                entry = entries.get(label_column)
                if isinstance(entry, dict):
                    tif_records = entry.get("tifs", [])
                    for record in tif_records:
                        rel_path = record.get("path") or record.get("filename")
                        if not rel_path:
                            continue
                        region = str(record.get("region") or "GLOBAL").upper()
                        path_obj = Path(rel_path)
                        if not path_obj.is_absolute():
                            path_obj = (metadata_path.parent / path_obj).resolve()
                        paths_by_region[region].append(path_obj)

    if not paths_by_region and fallback_root is not None:
        fallback = sorted(Path(fallback_root).glob(f"*{label_column.lower()}*.tif"))
        if fallback:
            paths_by_region["GLOBAL"] = [path.resolve() for path in fallback]

    return paths_by_region


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(name))
    slug = slug.strip("_").lower()
    return slug or "feature"


def find_metadata_root(path: Path) -> Path:
    candidates = [path] + list(path.parents)
    for candidate in candidates:
        if (candidate / "training_metadata.json").exists() or (candidate / "assetization" / "training_metadata.json").exists():
            return candidate
    return path
RegionCoord = Tuple[str, int, int]
CoordLike = Union[Tuple[int, int], Tuple[str, int, int], Dict[str, Any]]


def normalize_region_coord(coord: CoordLike, *, default_region: Optional[str] = None) -> RegionCoord:
    if isinstance(coord, dict):
        region = coord.get("region")
        row = coord.get("row")
        col = coord.get("col")
    elif isinstance(coord, (tuple, list)):
        if len(coord) == 3:
            region, row, col = coord
        elif len(coord) == 2:
            if default_region is None:
                raise ValueError(
                    "default_region must be provided when normalizing 2-tuple coordinates."
                )
            region = default_region
            row, col = coord
        else:
            raise ValueError(f"Unsupported coordinate tuple: {coord!r}")
    else:
        raise TypeError(f"Unsupported coordinate type: {type(coord)}")

    if region is None:
        if default_region is None:
            raise ValueError("Coordinate missing region and no default provided.")
        region = default_region

    return str(region), int(row), int(col)


def region_coord_to_dict(coord: RegionCoord) -> Dict[str, int]:
    region, row, col = coord
    return {"region": str(region), "row": int(row), "col": int(col)}


def resolve_search_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved


def load_training_args(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates: List[Path] = []
    root = resolve_search_root(stack_root)
    if root is not None:
        search_dirs = [root] + list(root.parents)
        for base in search_dirs:
            metadata_candidates = [base / 'training_metadata.json', base / 'assetization' / 'training_metadata.json']
            for meta_candidate in metadata_candidates:
                try:
                    resolved_meta = meta_candidate.resolve()
                except Exception:
                    continue
                if not resolved_meta.exists():
                    continue
                try:
                    with resolved_meta.open('r', encoding='utf-8') as fh:
                        meta_data = json.load(fh)
                    pretraining_entry = meta_data.get('pretraining')
                    if isinstance(pretraining_entry, dict):
                        args_data = pretraining_entry.get('args')
                        if isinstance(args_data, dict):
                            merged_args = dict(args_data)
                            features_override = pretraining_entry.get('features')
                            if isinstance(features_override, list):
                                merged_args['features'] = list(dict.fromkeys(str(f) for f in features_override))
                            output_dir_val = pretraining_entry.get('output_dir')
                            if output_dir_val:
                                try:
                                    out_path = Path(output_dir_val)
                                    if not out_path.is_absolute():
                                        out_path = (resolved_meta.parent / out_path).resolve()
                                    ta_candidate = out_path / 'training_args_1_pretrain.json'
                                    if ta_candidate.exists():
                                        with ta_candidate.open('r', encoding='utf-8') as ta_fh:
                                            ta_data = json.load(ta_fh)
                                        if isinstance(ta_data, dict):
                                            merged_args.update(
                                                {k: v for k, v in ta_data.items() if k not in merged_args or k == 'features'}
                                            )
                                            features_ta = ta_data.get('features')
                                            if isinstance(features_ta, list):
                                                merged_args['features'] = list(dict.fromkeys(str(f) for f in features_ta))
                                except Exception as exc:
                                    print(f"[warn] Failed to read training_args_1_pretrain.json from metadata output_dir: {exc}")
                            return merged_args, resolved_meta
                except Exception as exc:
                    print(f"[warn] Failed to read pretraining args from {resolved_meta}: {exc}")

        direct_pretrain = root / 'training_args_1_pretrain.json'
        direct_standard = root / 'training_args.json'
        candidates.extend([direct_pretrain, direct_standard])
        try:
            for candidate in root.rglob('training_args_1_pretrain.json'):
                candidates.append(candidate)
            for candidate in root.rglob('training_args.json'):
                candidates.append(candidate)
        except Exception:
            pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_args_1_pretrain.json')
    candidates.append(encoder_dir / 'training_args.json')

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
            return data, resolved
        except Exception as exc:
            print(f"[warn] Failed to load training args from {resolved}: {exc}")
    return None, None


def load_training_metadata(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates: List[Path] = []
    root = resolve_search_root(stack_root)
    if root is not None:
        search_dirs = [root] + list(root.parents)
        for base in search_dirs:
            for candidate in [base / 'training_metadata.json', base / 'assetization' / 'training_metadata.json']:
                candidates.append(candidate)
            try:
                for candidate in base.rglob('training_metadata.json'):
                    candidates.append(candidate)
            except Exception:
                pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_metadata.json')

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as fh:
                return json.load(fh), resolved
        except Exception as exc:
            print(f"[warn] Failed to load training metadata from {resolved}: {exc}")
    return None, None


def mae_kwargs_from_training_args(args_dict: Dict) -> Dict:
    mapping = {
        'encoder_dim': 'embed_dim',
        'encoder_depth': 'depth',
        'encoder_num_heads': 'encoder_num_heads',
        'mlp_ratio': 'mlp_ratio',
        'mlp_ratio_decoder': 'mlp_ratio_dec',
        'decoder_dim': 'dec_dim',
        'decoder_depth': 'dec_depth',
        'decoder_num_heads': 'decoder_num_heads',
        'mask_ratio': 'mask_ratio',
        'patch': 'patch_size',
    }
    mae_kwargs: Dict = {}
    for src_key, dest_key in mapping.items():
        if src_key not in args_dict or args_dict[src_key] is None:
            continue
        value = args_dict[src_key]
        if dest_key in {
            'embed_dim',
            'depth',
            'encoder_num_heads',
            'dec_dim',
            'dec_depth',
            'decoder_num_heads',
            'patch_size',
        }:
            value = int(value)
        elif dest_key in {'mlp_ratio', 'mlp_ratio_dec', 'mask_ratio'}:
            value = float(value)
        mae_kwargs[dest_key] = value
    window = args_dict.get('window')
    if window is not None:
        mae_kwargs['image_size'] = int(window)
    return mae_kwargs


def resolve_pretraining_patch(args_dict: Optional[Dict], fallback: int) -> int:
    if not args_dict:
        return fallback
    for key in ('patch', 'window'):
        value = args_dict.get(key)
        if value is None:
            continue
        try:
            patch_val = int(value)
        except (TypeError, ValueError):
            continue
        if patch_val > 0:
            return patch_val
    return fallback


def infer_region_from_name(path: Path) -> Optional[str]:
    stem_parts = path.stem.split('_')
    priority_codes = {'NA', 'AU', 'SA', 'EU', 'AS', 'AF', 'OC', 'GL', 'CA', 'US'}
    for token in reversed(stem_parts):
        up = token.upper()
        if up in priority_codes:
            return up
    for token in reversed(stem_parts):
        up = token.upper()
        if up in {'GLOBAL', 'WORLD'}:
            return 'GLOBAL'
        if up.isalpha() and 1 <= len(up) <= 4:
            return up
    return None


def resolve_label_rasters(
    metadata: Optional[Dict],
    metadata_path: Optional[Path],
    label_column: str,
) -> List[Dict[str, Any]]:
    if not metadata or not metadata_path:
        return []
    labels_section = metadata.get('labels')
    if not isinstance(labels_section, dict):
        return []
    entries = labels_section.get('entries')
    if not isinstance(entries, dict):
        return []
    entry = entries.get(label_column)
    if not isinstance(entry, dict):
        return []
    tif_entries = entry.get('tifs')
    if not isinstance(tif_entries, list):
        return []
    base_dir = metadata_path.parent
    resolved: List[Dict[str, Any]] = []
    for tif_info in tif_entries:
        if not isinstance(tif_info, dict):
            continue
        path_value = tif_info.get('path') or tif_info.get('filename')
        if not path_value:
            continue
        tif_path = Path(path_value)
        if not tif_path.is_absolute():
            tif_path = (base_dir / tif_path).resolve()
        if not tif_path.exists():
            print(f"[warn] Label raster not found: {tif_path}")
            continue
        region = tif_info.get('region')
        region_key = str(region or infer_region_from_name(tif_path) or 'GLOBAL').upper()
        resolved.append({"path": tif_path, "region": region_key})
    return resolved


def collect_feature_rasters(
    metadata: Optional[Dict],
    metadata_path: Optional[Path],
    feature_names: Optional[Sequence[str]],
) -> Dict[str, List[Path]]:
    if not metadata or not metadata_path:
        return {}
    features_section = metadata.get('features')
    if not isinstance(features_section, dict):
        return {}
    entries = features_section.get('entries')
    if not isinstance(entries, dict):
        return {}
    base_dir = metadata_path.parent

    def _lookup(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if name in entries:
            return name, entries[name]
        lowered = name.lower()
        for key, value in entries.items():
            if key.lower() == lowered:
                return key, value
        return None

    ordered_features: List[str]
    if feature_names:
        ordered_features = list(feature_names)
    else:
        ordered_features = sorted(entries.keys(), key=str.lower)

    region_map: Dict[str, Dict[str, List[Path]]] = {}
    for feature_name in ordered_features:
        match = _lookup(feature_name)
        if match is None:
            print(f"[warn] Feature '{feature_name}' missing from training_metadata.json entries.")
            continue
        canonical_name, entry = match
        tif_records = entry.get('tifs', [])
        if not isinstance(tif_records, list):
            continue
        for record in tif_records:
            if not isinstance(record, dict):
                continue
            path_value = record.get('path') or record.get('filename')
            if not path_value:
                continue
            tif_path = Path(path_value)
            if not tif_path.is_absolute():
                tif_path = (base_dir / tif_path).resolve()
            else:
                tif_path = tif_path.resolve()
            if not tif_path.exists():
                print(f"[warn] Feature raster not found: {tif_path}")
                continue
            region = record.get('region')
            region_key = str(region or infer_region_from_name(tif_path) or 'GLOBAL').upper()
            region_map.setdefault(region_key, {}).setdefault(canonical_name, []).append(tif_path)

    usable: Dict[str, List[Path]] = {}
    for region, feature_map in region_map.items():
        missing = [name for name in ordered_features if name not in feature_map]
        if missing:
            continue
        ordered_paths: List[Path] = []
        for name in ordered_features:
            paths = feature_map.get(name)
            if not paths:
                break
            ordered_paths.extend(sorted(paths, key=lambda p: p.name))
        if len(ordered_paths) == sum(len(feature_map[name]) for name in ordered_features if name in feature_map):
            usable[region] = ordered_paths
    return usable


def read_stack_patch(stack: Any, coord: Any, window: int) -> np.ndarray:

    if hasattr(stack, "resolve_region_stack"):
        default_region = getattr(stack, "default_region", None)
        if default_region is None:
            regions = getattr(stack, "regions", [])
            default_region = regions[0] if regions else "GLOBAL"
        region, row, col = normalize_region_coord(coord, default_region=default_region)
        region_stack = stack.resolve_region_stack(region)
        return region_stack.read_patch(row, col, window)

    if isinstance(coord, dict):
        row = coord.get("row")
        col = coord.get("col")
        if row is None or col is None:
            raise ValueError(f"Coordinate dictionary missing row/col: {coord}")
        return stack.read_patch(int(row), int(col), window)

    if isinstance(coord, (tuple, list)) and len(coord) == 3:
        _, row, col = coord
        return stack.read_patch(int(row), int(col), window)

    row, col = coord
    return stack.read_patch(int(row), int(col), window)


def _get_valid_window_mask(stack_obj: Any, window: int) -> Optional[np.ndarray]:
    if rasterio is None:
        return None
    window = int(window)
    if window <= 1:
        return None

    cache = getattr(stack_obj, "_valid_window_masks", None)
    if cache is None:
        cache = {}
        setattr(stack_obj, "_valid_window_masks", cache)
    mask = cache.get(window)
    if mask is not None:
        return mask

    base_mask: Optional[np.ndarray] = None
    srcs = getattr(stack_obj, "srcs", None)
    if srcs:
        try:
            mask_arr = srcs[0].dataset_mask()
        except Exception:
            mask_arr = None
        if mask_arr is not None:
            mask_arr = np.asarray(mask_arr)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            mask_bool = mask_arr != 0
            if mask_bool.any():
                base_mask = mask_bool.astype(bool, copy=False)
    if base_mask is None:
        combined = getattr(stack_obj, "_combined_valid_mask", None)
        if combined is not None and combined.any():
            base_mask = combined.astype(bool, copy=False)
    if base_mask is None:
        boundary = getattr(stack_obj, "boundary_mask", None)
        if boundary is not None and boundary.any():
            base_mask = boundary.astype(bool, copy=False)
    if base_mask is None or not base_mask.any():
        return None
    boundary_overlay = getattr(stack_obj, "boundary_mask", None)
    if boundary_overlay is not None:
        base_mask = base_mask & boundary_overlay.astype(bool, copy=False)
        if not base_mask.any():
            return None
    valid_mask = base_mask.astype(bool, copy=False)
    cache[window] = valid_mask
    return valid_mask


def prefilter_valid_window_coords(
    stack: Any,
    coords: Sequence[RegionCoord],
    window: int,
) -> List[RegionCoord]:
    """
    Quickly drop coordinates whose window would fall outside the valid raster mask.

    Uses an eroded dataset mask (window-sized) to approximate which centers are safe
    without reading patches from disk. Coordinates falling outside the mask are
    considered invalid and removed.
    """
    coords_list = list(coords)
    if not coords_list:
        return []

    window = int(window)
    if window <= 1:
        return coords_list

    iterator = coords_list
    use_progress = tqdm is not None and len(coords_list) > 0
    if use_progress:
        iterator = tqdm(coords_list, desc="Prefilter window coverage")

    filtered: List[RegionCoord] = []
    global_mask: Optional[np.ndarray] = None
    if not hasattr(stack, "resolve_region_stack"):
        global_mask = _get_valid_window_mask(stack, window)
        if global_mask is None:
            return coords_list

    for coord in iterator:
        if hasattr(stack, "resolve_region_stack"):
            region, row, col = coord
            region_stack = stack.resolve_region_stack(region)
            if region_stack is None:
                continue
            mask = _get_valid_window_mask(region_stack, window)
            if mask is None:
                filtered.append(coord)
                continue
            r, c = int(row), int(col)
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
                filtered.append(coord)
        else:
            if isinstance(coord, (tuple, list)):
                if len(coord) == 3:
                    _, row, col = coord
                elif len(coord) == 2:
                    row, col = coord
                else:
                    continue
            else:
                row = col = None
            if row is None or col is None:
                continue
            mask = global_mask
            if mask is None:
                continue
            r, c = int(row), int(col)
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
                filtered.append(coord)

    if use_progress:
        print("[info] Prefilter window coverage completed.")

    return filtered


def _sanitize_region_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    sanitized = sanitized.strip("_") or "region"
    return sanitized.lower()


def _write_boundary_geojson(
    geometry: Union[Polygon, MultiPolygon],
    out_path: Path,
    *,
    properties: Optional[Dict[str, Any]] = None,
    crs_name: Optional[str] = None,
) -> Path:
    feature = {
        "type": "Feature",
        "properties": properties or {},
        "geometry": mapping(geometry),
    }
    geojson: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    if crs_name:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": crs_name},
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(geojson), encoding="utf-8")
    return out_path


def compute_raster_geometry(
    raster_path: Union[str, Path]
) -> Tuple[Optional[Union[Polygon, MultiPolygon]], Optional[str]]:
    if rasterio is None or raster_shapes is None or shape is None or unary_union is None:
        return None, None

    raster_path = Path(raster_path)
    try:
        with rasterio.open(raster_path) as ds:
            mask = ds.dataset_mask()
            if mask is None:
                return None, None
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            mask_bool = mask_arr != 0
            if not mask_bool.any():
                return None, None

            polygons = []
            for geom, value in raster_shapes(
                mask_bool.astype(np.uint8), mask=mask_bool, transform=ds.transform
            ):
                if int(value) == 0:
                    continue
                poly = shape(geom).buffer(0)
                if not poly.is_empty:
                    polygons.append(poly)
            if not polygons:
                return None, ds.crs.to_string() if ds.crs else None
            merged = unary_union(polygons)
            if merged.is_empty:
                return None, ds.crs.to_string() if ds.crs else None
            if not isinstance(merged, (Polygon, MultiPolygon)):
                merged = merged.buffer(0)
            if merged.is_empty:
                return None, ds.crs.to_string() if ds.crs else None
            crs_name = ds.crs.to_string() if ds.crs else None
            return merged, crs_name
    except Exception:
        return None, None


def export_raster_boundary(
    raster_path: Union[str, Path],
    out_geojson: Union[str, Path],
    *,
    properties: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    geometry, crs_name = compute_raster_geometry(raster_path)
    if geometry is None:
        return None
    props = {"source_raster": str(Path(raster_path).resolve())}
    if properties:
        props.update(properties)
    out_path = Path(out_geojson)
    return _write_boundary_geojson(geometry, out_path, properties=props, crs_name=crs_name)


def export_region_boundaries(
    raster_paths_by_region: Dict[str, Sequence[Union[str, Path]]],
    out_dir: Union[str, Path],
    *,
    prefix: str = "boundary",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    results: Dict[str, Path] = {}
    if not raster_paths_by_region:
        return results

    global_geometries: List[Union[Polygon, MultiPolygon]] = []
    global_crs: Optional[str] = None

    for region, paths in raster_paths_by_region.items():
        region_geom: Optional[Union[Polygon, MultiPolygon]] = None
        region_crs: Optional[str] = None
        for path in paths:
            geom, crs_name = compute_raster_geometry(path)
            if geom is None:
                continue
            region_geom = geom if region_geom is None else region_geom.union(geom)
            region_crs = region_crs or crs_name
        if region_geom is None or region_geom.is_empty:
            continue
        region_slug = _sanitize_region_name(region)
        out_path = out_dir / f"{prefix}_{region_slug}.geojson"
        props = {"region": region}
        results[region] = _write_boundary_geojson(region_geom, out_path, properties=props, crs_name=region_crs)
        global_geometries.append(region_geom)
        if region_crs and not global_crs:
            global_crs = region_crs

    if global_geometries:
        try:
            global_geom = unary_union(global_geometries)
        except Exception:
            global_geom = None
        if global_geom and not global_geom.is_empty:
            out_path = out_dir / f"{prefix}.geojson"
            props = {"regions": list(results.keys())}
            results["GLOBAL"] = _write_boundary_geojson(global_geom, out_path, properties=props, crs_name=global_crs)

    return results


def export_raster_boundary(
    raster_path: Union[str, Path],
    out_geojson: Union[str, Path],
) -> Optional[Path]:
    """Export a coarse boundary polygon for the raster's valid data mask."""
    if rasterio is None or raster_shapes is None or shape is None or unary_union is None:
        return None

    raster_path = Path(raster_path)
    out_geojson = Path(out_geojson)

    try:
        with rasterio.open(raster_path) as ds:
            mask = ds.dataset_mask()
            if mask is None:
                return None
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            mask_bool = mask_arr != 0
            if not mask_bool.any():
                return None

            polygons = []
            for geom, value in raster_shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=ds.transform):
                if int(value) == 0:
                    continue
                poly = shape(geom).buffer(0)
                if not poly.is_empty:
                    polygons.append(poly)
    except Exception:
        return None

    if not polygons:
        return None
    try:
        merged = unary_union(polygons)
    except Exception:
        return None
    if merged.is_empty:
        return None
    if not isinstance(merged, (Polygon, MultiPolygon)):
        merged = merged.buffer(0)
    if merged.is_empty:
        return None

    try:
        out_geojson.parent.mkdir(parents=True, exist_ok=True)
        feature = {
            "type": "Feature",
            "properties": {
                "source_raster": str(raster_path.resolve()),
            },
            "geometry": mapping(merged),
        }
        geojson = {
            "type": "FeatureCollection",
            "features": [feature],
        }
        with rasterio.open(raster_path) as ds:
            if ds.crs:
                try:
                    geojson["crs"] = {
                        "type": "name",
                        "properties": {"name": ds.crs.to_string()},
                    }
                except Exception:
                    pass
        out_geojson.write_text(json.dumps(geojson), encoding="utf-8")
        return out_geojson
    except Exception:
        return None


def load_region_boundaries(
    boundary_source: Union[str, Path, Sequence[Union[str, Path]]],
    project_only: bool = False,
) -> Dict[str, BoundaryInfo]:
    """Load boundary rasters (GeoTIFF) and derive masks/geometry."""
    if rasterio is None:
        return {}

    def _normalize_path(item: Union[str, Path]) -> Optional[Path]:
        try:
            path_obj = Path(item)
        except Exception:
            return None
        try:
            return path_obj.resolve()
        except Exception:
            return path_obj

    candidate_paths: List[Path] = []

    if isinstance(boundary_source, (list, tuple, set)):
        items = list(boundary_source)
    else:
        items = [boundary_source]

    for entry in items:
        normalized = _normalize_path(entry)
        if normalized is None:
            continue
        if normalized.is_dir():
            candidate_paths.extend(sorted(normalized.glob("*.tif")))
            candidate_paths.extend(sorted(normalized.glob("*.tiff")))
        elif normalized.is_file():
            candidate_paths.append(normalized)

    if not candidate_paths:
        return {}

    # Deduplicate paths while preserving order
    seen: set[Path] = set()
    candidates: List[Path] = []
    for path in candidate_paths:
        try:
            real = path.resolve()
        except Exception:
            real = path
        if real in seen:
            continue
        seen.add(real)
        candidates.append(real)

    if not candidates:
        return {}

    results: Dict[str, BoundaryInfo] = {}
    for candidate in sorted(candidates):
        stem_lower = candidate.stem.lower()
        if project_only and "project" not in stem_lower:
            continue
        region_key = "PROJECT" if "project" in stem_lower else stem_lower.upper()
        try:
            with rasterio.open(candidate) as ds:
                band = ds.read(1, masked=True)
                if np.ma.isMaskedArray(band):
                    data = np.asarray(band.filled(0.0), dtype=np.float64)
                    validity = ~np.asarray(band.mask, dtype=bool)
                else:
                    data = np.asarray(band, dtype=np.float64)
                    validity = np.ones_like(data, dtype=bool)
                mask = validity & np.isfinite(data) & (data != 0)
                if not mask.any():
                    continue
                crs_name = ds.crs.to_string() if ds.crs else None
                geometry = None
                if raster_shapes is not None and shape is not None:
                    try:
                        polygons: List[Union[Polygon, MultiPolygon]] = []
                        for geom, value in raster_shapes(
                            mask.astype(np.uint8),
                            mask=mask,
                            transform=ds.transform,
                        ):
                            if int(value) == 0:
                                continue
                            poly = shape(geom)
                            if not poly.is_empty:
                                polygons.append(poly)
                        if polygons:
                            geometry = unary_union(polygons) if unary_union is not None else polygons[0]
                            if isinstance(geometry, (list, tuple)) and polygons:
                                geometry = polygons[0]
                    except Exception:
                        geometry = None
                info = BoundaryInfo(
                    path=candidate.resolve(),
                    mask=mask.astype(bool, copy=False),
                    transform=ds.transform,
                    crs=crs_name,
                    geometry=geometry,
                )
        except Exception:
            continue
        results[region_key] = info
    return results


def _resample_boundary_mask(
    info: BoundaryInfo,
    target_transform: Affine,
    target_crs: Optional[str],
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    if reproject is None or Resampling is None:
        return None
    dst = np.zeros((height, width), dtype=np.float32)
    try:
        reproject(
            source=info.mask.astype(np.uint8),
            destination=dst,
            src_transform=info.transform,
            src_crs=info.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
    except Exception:
        return None
    return (dst >= 0.5)


def apply_boundaries_to_region_stacks(
    region_stacks: Dict[str, GeoStack],
    boundary_infos: Dict[str, BoundaryInfo],
) -> Dict[str, np.ndarray]:
    if not boundary_infos:
        return {}

    applied_masks: Dict[str, np.ndarray] = {}
    for region, stack in region_stacks.items():
        region_keys = [
            region,
            str(region).upper(),
            str(region).lower(),
            str(region).title(),
        ]
        info = None
        for key in region_keys:
            info = boundary_infos.get(key)
            if info is not None:
                break
        if info is None:
            info = boundary_infos.get("PROJECT") or boundary_infos.get("GLOBAL")
        if info is None:
            continue
        mask = info.mask
        if mask.shape != (stack.height, stack.width):
            target_crs = None
            stack_crs = getattr(stack, "crs", None)
            if stack_crs is not None:
                try:
                    target_crs = stack_crs.to_string()
                except Exception:
                    target_crs = stack_crs
            resampled = _resample_boundary_mask(
                info,
                stack.transform,
                target_crs,
                stack.height,
                stack.width,
            )
            if resampled is None:
                print(
                    f"[warn] Boundary raster {info.path} could not be resampled to match stack "
                    f"{region} dimensions {(stack.height, stack.width)}; skipping."
                )
                continue
            mask = resampled
        try:
            stack.set_boundary_mask(mask)
        except ValueError as exc:
            print(f"[warn] Boundary mask for region '{region}' skipped: {exc}")
            continue
        applied_masks[region] = mask
    return applied_masks


def export_window_center_visualizations(
    stack: Any,
    window: int,
    out_dir: Union[str, Path],
    *,
    boundaries: Optional[Dict[str, BoundaryInfo]] = None,
    prefix: str = "window_centers",
    max_scatter_points: int = 250_000,
) -> Dict[str, Dict[str, Path]]:
    """
    Export GeoTIFF/PNG diagnostics showing valid window centers relative to dataset boundaries.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is required to export window center diagnostics.")

    from rasterio.transform import array_bounds  # type: ignore
    from rasterio import transform as rio_transform  # type: ignore

    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is None:
        print("[warn] Matplotlib unavailable; PNG window center previews will be skipped.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _iter_regions():
        if hasattr(stack, "iter_region_stacks"):
            yield from stack.iter_region_stacks()
        else:
            region_name = getattr(stack, "default_region", None) or "GLOBAL"
            yield (region_name, stack)

    def _resolve_boundary(region_name: str) -> Optional[BoundaryInfo]:
        if not boundaries:
            return None
        keys = [
            region_name,
            str(region_name).upper(),
            str(region_name).lower(),
            str(region_name).title(),
            "PROJECT",
            "GLOBAL",
        ]
        for key in keys:
            info = boundaries.get(key)
            if info is not None:
                return info
        return None

    def _plot_geometry(ax, geom):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for sub in geom.geoms:
                _plot_geometry(ax, sub)
            return
        try:
            exterior = geom.exterior
        except Exception:
            exterior = None
        if exterior is not None:
            x, y = exterior.xy
            ax.plot(x, y, color="crimson", linewidth=1.0, alpha=0.9)
        interiors = getattr(geom, "interiors", [])
        for ring in interiors:
            xi, yi = ring.xy
            ax.plot(xi, yi, color="crimson", linewidth=0.6, alpha=0.8)

    def _mask_to_geometry(mask_arr, transform):
        if raster_shapes is None or shape is None:
            return None
        polygons = []
        try:
            for geom, value in raster_shapes(mask_arr.astype(np.uint8), mask=mask_arr, transform=transform):
                if int(value) == 0:
                    continue
                poly = shape(geom)
                if not poly.is_empty:
                    polygons.append(poly)
        except Exception:
            return None
        if not polygons:
            return None
        try:
            merged = unary_union(polygons) if unary_union is not None else None
        except Exception:
            merged = None
        if merged is None:
            merged = polygons[0] if len(polygons) == 1 else None
        return merged

    results: Dict[str, Dict[str, Path]] = {}
    for region_name, region_stack in _iter_regions():
        mask = _get_valid_window_mask(region_stack, window)
        if mask is None:
            continue
        srcs = getattr(region_stack, "srcs", None)
        if not srcs:
            continue
        primary = srcs[0]
        profile = primary.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
        )

        region_key = str(region_name or "GLOBAL")
        slug = _sanitize_region_name(region_key or "GLOBAL")
        tif_path = out_path / f"{prefix}_{slug}.tif"
        png_path = out_path / f"{prefix}_{slug}.png"

        data = mask.astype(np.uint8, copy=False)
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(data, 1)

        entry: Dict[str, Path] = {"tif": tif_path}

        if plt is not None:
            try:
                bounds = array_bounds(mask.shape[0], mask.shape[1], primary.transform)
                extent = (bounds[0], bounds[2], bounds[3], bounds[1])
                fig, ax = plt.subplots(figsize=(8, 8))
                boundary_info = _resolve_boundary(region_key)
                geometry = None
                if boundary_info is not None:
                    overlay_mask = boundary_info.mask
                    overlay_transform = boundary_info.transform
                    if overlay_mask.shape != data.shape:
                        target_crs = None
                        if primary.crs is not None:
                            try:
                                target_crs = primary.crs.to_string()
                            except Exception:
                                target_crs = primary.crs
                        resampled_overlay = _resample_boundary_mask(
                            boundary_info,
                            primary.transform,
                            target_crs,
                            data.shape[0],
                            data.shape[1],
                        )
                        if resampled_overlay is not None:
                            overlay_mask = resampled_overlay
                            overlay_transform = primary.transform
                    geometry = boundary_info.geometry
                    if geometry is None and overlay_mask is not None:
                        geometry = _mask_to_geometry(overlay_mask, overlay_transform)
                if geometry is None:
                    geometry = _mask_to_geometry(mask, primary.transform)
                if geometry is not None:
                    _plot_geometry(ax, geometry)

                rows, cols = np.where(mask)
                if rows.size > 0:
                    if rows.size > max_scatter_points:
                        rng = np.random.default_rng(0)
                        indices = rng.choice(rows.size, size=max_scatter_points, replace=False)
                        rows = rows[indices]
                        cols = cols[indices]
                    xs, ys = rio_transform.xy(primary.transform, rows, cols, offset="center")
                    ax.scatter(xs, ys, s=1, c="tab:blue", alpha=0.6, linewidths=0)

                ax.set_title(f"Window centers (window={window}) — {region_key}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal")
                fig.tight_layout()
                fig.savefig(png_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                entry["png"] = png_path
            except Exception:
                if plt is not None:
                    plt.close("all")

        results[region_key] = entry
    return results
