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

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_THIS_DIR / "Methods" / "0_Benchmark_GFM4MPM").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.gfm4mpm.data.geo_stack import GeoStack  # noqa: E402
from src.gfm4mpm.data.stac_table import StacTableStack  # noqa: E402


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
    feature_names, region_paths, metadata, feature_entries = collect_region_feature_paths(stac_root, features)
    region_stacks: Dict[str, GeoStack] = {}
    for region, paths in region_paths.items():
        stack = GeoStack([str(p) for p in paths])
        stack.feature_columns = list(feature_names)
        stack.feature_metadata = [{"regions": [region]} for _ in feature_names]
        region_stacks[region] = stack

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
