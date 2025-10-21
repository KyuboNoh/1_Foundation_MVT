#!/usr/bin/env python3
"""
Utilities to combine metadata from multiple STAC collections into a unified
feature schema. 
This is the first step for the Universal GFM4MPM pipeline 
which will ingest different regional STAC exports (e.g., CAN/US/AU, BC) and
produce a compatible superset of channels.
We generate additional "overlap" features to utilize the information in injested
collections optimally.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import combinations

import json

try:
    from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape as shapely_shape
    from shapely.ops import transform as shapely_transform, unary_union
except Exception:  # pragma: no cover - optional dependency guard
    shapely_shape = None  # type: ignore[assignment]
    Polygon = MultiPolygon = None  # type: ignore[assignment]
    shapely_transform = unary_union = None  # type: ignore[assignment]

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency guard
    CRS = Transformer = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio import features as rio_features
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover - optional dependency guard
    rasterio = None  # type: ignore[assignment]
    rio_features = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class DatasetSummary:
    dataset_id: str
    root: Path
    collection_id: Optional[str]
    features: Dict[str, Dict]
    labels: Dict[str, Dict]
    boundaries: Dict[str, List[Dict]]
    regions: Dict[str, Dict[str, Dict]]  # region -> {"features": {...}, "labels": {...}}
    label_geojsons: Dict[str, List[Path]]

    @property
    def feature_names(self) -> List[str]:
        return sorted(self.features.keys())

    @property
    def label_names(self) -> List[str]:
        return sorted(self.labels.keys())

    @property
    def region_keys(self) -> List[str]:
        return sorted(self.regions.keys())

    @property
    def is_multi_region(self) -> bool:
        return len(self.region_keys) > 1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file {path}: {exc}")


def _locate_training_metadata(root: Path) -> Tuple[Optional[dict], Optional[Path]]:
    """
    Returns (metadata_dict, path) for the first training_metadata.json discovered
    under the provided root. Searches common locations written by stacify_*.
    """
    candidates: List[Path] = [
        root / "training_metadata.json",
        root / "assetization" / "training_metadata.json",
    ]
    try:
        for candidate in root.rglob("training_metadata.json"):
            if candidate not in candidates:
                candidates.append(candidate)
    except Exception:
        pass

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        meta = _load_json(resolved)
        if isinstance(meta, dict):
            return meta, resolved
    return None, None


def _canonical_path(base: Path, value: str) -> Path:
    path_obj = Path(str(value))
    if not path_obj.is_absolute():
        return (base / path_obj).resolve()
    return path_obj.resolve()


def _augment_record(record: Dict, base: Path) -> Dict:
    rec = dict(record)
    for key in ("path", "item_json", "quicklook_png"):
        value = rec.get(key)
        if not value:
            continue
        try:
            resolved = _canonical_path(base, value)
            rec[f"{key}_resolved"] = str(resolved)
        except Exception:
            pass
    return rec


def _augment_entry(entry: Dict, base: Path) -> Dict:
    if not isinstance(entry, dict):
        return {}
    new_entry = {k: v for k, v in entry.items() if k != "tifs"}
    tifs = entry.get("tifs")
    augmented: List[Dict] = []
    if isinstance(tifs, list):
        for record in tifs:
            if not isinstance(record, dict):
                continue
            augmented.append(_augment_record(record, base))
    new_entry["tifs"] = augmented
    return new_entry


def _parse_bridge_entry(entry: str, dataset_ids: Sequence[str]) -> Dict[str, List[str]]:
    raw = entry.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError(f"Bridge entry must be wrapped in braces: {entry!r}")
    content = raw[1:-1]
    parts = [part.strip() for part in content.split(";")]
    if len(parts) != len(dataset_ids):
        raise ValueError(
            f"Bridge entry {entry!r} provides {len(parts)} segments but {len(dataset_ids)} dataset(s) were supplied."
        )
    mapping: Dict[str, List[str]] = {}
    for ds_id, segment in zip(dataset_ids, parts):
        if not segment:
            mapping[ds_id] = []
            continue
        tokens = [token.strip() for token in segment.split(",")]
        mapping[ds_id] = [token for token in tokens if token]
    return mapping


def _load_bridge_visualizer():
    from importlib import import_module

    module = import_module("Methods.0_Benchmark_GFM4MPM.src.gfm4mpm.infer.infer_maps")
    return getattr(module, "generate_bridge_visualizations")


def _parse_region_selection(entry: str, dataset_ids: Sequence[str]) -> Dict[str, List[str]]:
    raw = entry.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError(f"Region selection must be wrapped in braces: {entry!r}")
    content = raw[1:-1]
    parts = [part.strip() for part in content.split(";")]
    if len(parts) != len(dataset_ids):
        raise ValueError(
            f"Region selection {entry!r} provides {len(parts)} segments but {len(dataset_ids)} dataset(s) were supplied."
        )
    mapping: Dict[str, List[str]] = {}
    for ds_id, segment in zip(dataset_ids, parts):
        if not segment:
            mapping[ds_id] = []
            continue
        tokens = [token.strip() for token in segment.split(",")]
        cleaned = [token for token in tokens if token]
        if not cleaned:
            mapping[ds_id] = []
        else:
            mapping[ds_id] = cleaned
    return mapping




def _load_collection_id(root: Path) -> Optional[str]:
    # Collection JSON is emitted when stacify_* runs. We only need the ID.
    collection_json = root / "collection.json"
    if not collection_json.exists():
        return None
    doc = _load_json(collection_json)
    if not isinstance(doc, dict):
        return None
    collection_id = doc.get("id") or doc.get("collection", {}).get("id")
    if collection_id:
        return str(collection_id)
    return None


def _summarise_dataset(root: Path, dataset_id: Optional[str] = None) -> DatasetSummary:
    root = root.resolve()
    metadata, metadata_path = _locate_training_metadata(root)
    if metadata is None:
        raise FileNotFoundError(
            f"Could not locate training_metadata.json under {root}. "
            "Ensure the STAC collection was generated with stacify_*."
        )

    collection_id = _load_collection_id(root)
    dataset_name = dataset_id or collection_id or root.name

    features_section = metadata.get("features", {})
    raw_feature_entries = features_section.get("entries", {}) if isinstance(features_section, dict) else {}
    labels_section = metadata.get("labels", {})
    raw_label_entries = labels_section.get("entries", {}) if isinstance(labels_section, dict) else {}

    feature_entries = {str(name): _augment_entry(entry, root) for name, entry in raw_feature_entries.items()}
    label_entries = {str(name): _augment_entry(entry, root) for name, entry in raw_label_entries.items()}

    boundaries_section: Dict[str, List[Dict]] = {}
    raw_boundaries = metadata.get("boundaries")
    if isinstance(raw_boundaries, dict):
        for key, records in raw_boundaries.items():
            if isinstance(records, list):
                filtered = [_augment_record(rec, root) for rec in records if isinstance(rec, dict)]
                if filtered:
                    boundaries_section[key] = filtered

    region_map: Dict[str, Dict[str, Dict[str, Dict]]] = {}

    def _accumulate_region_entry(
        entry_name: str,
        entry_payload: Dict,
        record: Optional[Dict],
        bucket: str,
    ) -> None:
        region_key = "GLOBAL"
        if isinstance(record, dict):
            region_key = str(record.get("region") or "GLOBAL")
        region_bucket = region_map.setdefault(region_key, {"features": {}, "labels": {}})[bucket]
        target_entry = region_bucket.setdefault(entry_name, {k: v for k, v in entry_payload.items() if k != "tifs"})
        target_entry.setdefault("tifs", [])
        if isinstance(record, dict):
            target_entry["tifs"].append(record)

    for feature_name, entry in feature_entries.items():
        tifs = entry.get("tifs") if isinstance(entry, dict) else None
        if isinstance(tifs, list) and tifs:
            for record in tifs:
                if isinstance(record, dict):
                    _accumulate_region_entry(feature_name, entry, record, "features")
        else:
            _accumulate_region_entry(feature_name, entry if isinstance(entry, dict) else {}, None, "features")

    for label_name, entry in label_entries.items():
        tifs = entry.get("tifs") if isinstance(entry, dict) else None
        if isinstance(tifs, list) and tifs:
            for record in tifs:
                if isinstance(record, dict):
                    _accumulate_region_entry(label_name, entry, record, "labels")
        else:
            _accumulate_region_entry(label_name, entry if isinstance(entry, dict) else {}, None, "labels")

    region_names_upper = {name.upper() for name in region_map}
    label_geojsons: Dict[str, List[Path]] = {}
    label_dir = root / "assets" / "labels" / "geojson"
    if label_dir.exists():
        for path in sorted(label_dir.glob("*.geojson")):
            region_key = "GLOBAL"
            stem_upper = path.stem.upper()
            parts = stem_upper.split("_")
            if parts:
                candidate = parts[-1]
                if candidate in region_names_upper:
                    region_key = candidate
            label_geojsons.setdefault(region_key, []).append(path.resolve())

    return DatasetSummary(
        dataset_id=dataset_name,
        root=root,
        collection_id=collection_id,
        features={str(k): v for k, v in feature_entries.items()},
        labels={str(k): v for k, v in label_entries.items()},
        boundaries=boundaries_section,
        regions=region_map,
        label_geojsons=label_geojsons,
    )


def _load_boundary_geometry(path: Path) -> Optional[Polygon]:
    if shapely_shape is None:
        return None
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in {".geojson", ".json"}:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if data.get("type") == "FeatureCollection":
                    features = data.get("features") or []
                    geoms = [shapely_shape(feat.get("geometry")) for feat in features if isinstance(feat, dict) and feat.get("geometry")]
                    geoms = [geom for geom in geoms if geom and not geom.is_empty]
                    if geoms:
                        return unary_union(geoms)
                elif "geometry" in data:
                    geom = shapely_shape(data["geometry"])
                    return geom if geom and not geom.is_empty else None
            return None
        elif suffix in {".tif", ".tiff"}:
            import numpy as np
            import rasterio
            from rasterio.features import shapes

            with rasterio.open(path) as ds:
                mask = ds.dataset_mask()
                if mask is None:
                    return None
                mask_arr = mask != 0
                if not mask_arr.any():
                    return None
                transform = ds.transform
                geoms = []
                for geom, val in shapes(mask.astype(np.uint8), mask=mask_arr, transform=transform):
                    if int(val) == 0:
                        continue
                    shape_obj = shapely_shape(geom)
                    if shape_obj.is_empty:
                        continue
                    geoms.append(shape_obj)
                if geoms:
                    return unary_union(geoms)
    except Exception:
        return None
    return None


def _geometry_from_raster_data(path: Path) -> Optional[Polygon]:
    if shapely_shape is None or unary_union is None:
        return None
    try:
        import numpy as np
        import rasterio
        from rasterio.features import shapes
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as ds:
            data = ds.read(masked=True)
            if isinstance(data, np.ma.MaskedArray):
                valid_mask = ~data.mask
            else:
                valid_mask = np.isfinite(data)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.any(axis=0)
            if not np.any(valid_mask):
                try:
                    dataset_mask = ds.dataset_mask()
                except Exception:
                    dataset_mask = None
                if dataset_mask is None:
                    return None
                valid_mask = dataset_mask != 0
                if not np.any(valid_mask):
                    return None
            valid_mask = np.asarray(valid_mask, dtype=bool)
            data_arr = valid_mask.astype(np.uint8)
            geoms = []
            for geom, val in shapes(data_arr, mask=valid_mask, transform=ds.transform):
                if int(val) == 0:
                    continue
                shape_obj = shapely_shape(geom)
                if shape_obj.is_empty:
                    continue
                geoms.append(shape_obj)
            if geoms:
                return unary_union(geoms)
    except Exception:
        return None
    return None


def _clean_projected_geometry(geom: Polygon, *, min_area: float = 0.0) -> Optional[Polygon]:
    if shapely_shape is None or unary_union is None:
        return None
    if geom is None or geom.is_empty:
        return None
    try:
        cleaned = geom.buffer(0)
    except Exception:
        return None
    if cleaned.is_empty:
        return None
    if cleaned.geom_type == "GeometryCollection":
        parts = [
            part
            for part in cleaned.geoms
            if getattr(part, "geom_type", "") in {"Polygon", "MultiPolygon"} and not part.is_empty
        ]
        if not parts:
            return None
        try:
            cleaned = unary_union(parts)
        except Exception:
            return None
        if cleaned.is_empty:
            return None
    if min_area > 0.0 and cleaned.area < min_area:
        return None
    return cleaned


def _iter_polygon_parts(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return [part for part in geom.geoms if not part.is_empty]
    if geom_type == "GeometryCollection":
        parts: List[Polygon] = []
        for part in geom.geoms:
            parts.extend(_iter_polygon_parts(part))
        return parts
    return []


def _transform_geometry(transformer, geom: Polygon, *, min_area: float = 0.0) -> Optional[Polygon]:
    if shapely_shape is None or unary_union is None:
        return None
    parts = []
    for part in _iter_polygon_parts(geom):
        try:
            projected = shapely_transform(transformer.transform, part)
        except Exception:
            continue
        cleaned = _clean_projected_geometry(projected, min_area=min_area)
        if cleaned is not None:
            parts.append(cleaned)
    if not parts:
        return None
    try:
        combined = unary_union(parts)
    except Exception:
        combined = parts[0]
        for additional in parts[1:]:
            try:
                combined = combined.union(additional)
            except Exception:
                continue
    return _clean_projected_geometry(combined, min_area=min_area)


def _detect_dataset_crs(summary: DatasetSummary) -> Optional[CRS]:
    if CRS is None:
        return None
    try:
        import rasterio
    except Exception:
        return None

    candidates: List[Path] = []
    for feature_entry in summary.features.values():
        if not isinstance(feature_entry, dict):
            continue
        for record in feature_entry.get("tifs", []) or []:
            path_value = record.get("path_resolved") or record.get("path")
            if not path_value:
                continue
            candidate = Path(path_value)
            if candidate.exists():
                candidates.append(candidate)
        if candidates:
            break

    if not candidates:
        for records in summary.boundaries.values():
            for record in records:
                path_value = record.get("path_resolved") or record.get("path")
                if not path_value:
                    continue
                candidate = Path(path_value)
                if candidate.exists() and candidate.suffix.lower() in {".tif", ".tiff"}:
                    candidates.append(candidate)
            if candidates:
                break

    for path in candidates:
        try:
            with rasterio.open(path) as ds:
                if ds.crs:
                    return CRS.from_user_input(ds.crs)
        except Exception:
            continue
    return None


def _collect_dataset_geometry(summary: DatasetSummary, dataset_crs: CRS) -> Optional[Polygon]:
    if shapely_shape is None or unary_union is None:
        return None

    geoms: List[Polygon] = []
    for feature_entry in summary.features.values():
        if not isinstance(feature_entry, dict):
            continue
        for record in feature_entry.get("tifs", []) or []:
            path_value = record.get("path_resolved") or record.get("path")
            if not path_value:
                continue
            geom = _geometry_from_raster_data(Path(path_value))
            if geom is not None and not geom.is_empty:
                geoms.append(geom)

    if not geoms:
        for records in summary.boundaries.values():
            for record in records:
                path_value = record.get("path_resolved") or record.get("path")
                if not path_value:
                    continue
                geom = _load_boundary_geometry(Path(path_value))
                if geom is not None and not geom.is_empty:
                    geoms.append(geom)

    if not geoms:
        import rasterio
        from shapely.geometry import box

        for feature_entry in summary.features.values():
            if not isinstance(feature_entry, dict):
                continue
            for record in feature_entry.get("tifs", []) or []:
                path_value = record.get("path_resolved") or record.get("path")
                if not path_value:
                    continue
                path = Path(path_value)
                if not path.exists():
                    continue
                try:
                    with rasterio.open(path) as ds:
                        bounds = ds.bounds
                        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                        geoms.append(geom)
                except Exception:
                    continue
                break
            if geoms:
                break

    if not geoms:
        return None

    merged = unary_union(geoms)
    if merged.is_empty:
        return None
    return merged


def _collect_dataset_tiles(summary: DatasetSummary, target_crs: Optional[CRS]) -> List[Dict[str, object]]:
    if shapely_shape is None or rasterio is None or box is None:
        return []
    tiles: List[Dict[str, object]] = []
    seen_paths: set[Path] = set()
    for region_name, region_info in summary.regions.items():
        feature_entries = region_info.get("features", {}) if isinstance(region_info, dict) else {}
        for feature_name, entry in feature_entries.items():
            tifs = entry.get("tifs", []) if isinstance(entry, dict) else []
            for record in tifs:
                if not isinstance(record, dict):
                    continue
                path_value = record.get("path_resolved") or record.get("path")
                if not path_value:
                    continue
                tile_path = Path(str(path_value))
                if not tile_path.is_absolute():
                    tile_path = (summary.root / tile_path).resolve()
                else:
                    tile_path = tile_path.resolve()
                if tile_path in seen_paths or not tile_path.exists():
                    continue
                try:
                    with rasterio.open(tile_path) as ds:
                        if ds.crs is None:
                            continue
                        bounds = ds.bounds
                        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                        src_crs = ds.crs
                        if target_crs is not None and Transformer is not None and shapely_transform is not None:
                            if src_crs != target_crs:
                                transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
                                geom = shapely_transform(transformer.transform, geom)
                        elif target_crs is not None:
                            continue
                except Exception:
                    continue
                tiles.append(
                    {
                        "tile_id": record.get("tile_id") or tile_path.stem,
                        "path": str(tile_path),
                        "region": region_name,
                        "feature": feature_name,
                        "geom": geom,
                    }
                )
                seen_paths.add(tile_path)
    return tiles


def _write_overlap_pairs(
    tile_geoms: Dict[str, List[Dict[str, object]]],
    overlap_geom: Optional[Polygon],
    target_dir: Path,
) -> Dict[str, str]:
    if shapely_shape is None or not tile_geoms:
        return {}
    data_dir = target_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pair_outputs: Dict[str, str] = {}
    for ds_a, ds_b in combinations(sorted(tile_geoms.keys()), 2):
        records: List[Dict[str, object]] = []
        for tile_a in tile_geoms[ds_a]:
            geom_a = tile_a.get("geom")
            if geom_a is None:
                continue
            for tile_b in tile_geoms[ds_b]:
                geom_b = tile_b.get("geom")
                if geom_b is None:
                    continue
                inter = geom_a.intersection(geom_b)
                if overlap_geom is not None:
                    inter = inter.intersection(overlap_geom)
                if inter.is_empty:
                    continue
                area = float(inter.area)
                if area <= 0:
                    continue
                centroid = inter.centroid
                area_a = float(geom_a.area) if geom_a.area else None
                area_b = float(geom_b.area) if geom_b.area else None
                records.append(
                    {
                        "dataset_a": ds_a,
                        "dataset_a_tile": tile_a.get("tile_id"),
                        "dataset_a_feature": tile_a.get("feature"),
                        "dataset_a_region": tile_a.get("region"),
                        "dataset_a_path": tile_a.get("path"),
                        "dataset_b": ds_b,
                        "dataset_b_tile": tile_b.get("tile_id"),
                        "dataset_b_feature": tile_b.get("feature"),
                        "dataset_b_region": tile_b.get("region"),
                        "dataset_b_path": tile_b.get("path"),
                        "overlap_area": area,
                        "overlap_fraction_a": (area / area_a) if area_a else None,
                        "overlap_fraction_b": (area / area_b) if area_b else None,
                        "overlap_centroid": [float(centroid.x), float(centroid.y)],
                    }
                )
        if records:
            pair_name = f"{ds_a}_{ds_b}_overlap_pairs.json"
            output_path = data_dir / pair_name
            try:
                with output_path.open("w", encoding="utf-8") as fh:
                    json.dump(records, fh, indent=2)
                pair_outputs[f"{ds_a}-{ds_b}"] = str(output_path)
            except Exception as exc:
                print(f"[warn] Failed to write overlap pairs for {ds_a}-{ds_b}: {exc}")
    return pair_outputs


def _build_union_feature_table(datasets: Sequence[DatasetSummary]) -> Dict[str, List[str]]:
    """Map each feature name to the dataset IDs that provide it."""
    table: Dict[str, List[str]] = {}
    for ds in datasets:
        for feature in ds.feature_names:
            table.setdefault(feature, []).append(ds.dataset_id)
    for feature, owners in table.items():
        owners.sort()
    return dict(sorted(table.items(), key=lambda kv: kv[0].lower()))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate multiple STAC collections into a unified metadata summary.")
    parser.add_argument("--collections", nargs="+", required=True, help="Paths to STAC collection roots produced by stacify_*.",)
    parser.add_argument("--output", type=str, default=None, help="Optional path to write combined metadata JSON. If omitted, prints to stdout.",)
    parser.add_argument("--dataset-ids", nargs="*",
        default=None,
        help="Optional friendly IDs corresponding to --collections (must match in length).",
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default=None,
        help="Optional project name. When provided with --output, the combined metadata "
             "is written to <output>/<projectname>/combined_metadata.json",
    )
    parser.add_argument(
        "--bridge-guess-number",
        type=int,
        default=0,
        help="Optional count of heuristic feature bridges supplied via --bridge.",
    )
    parser.add_argument(
        "--bridge",
        action="append",
        default=[],
        help=(
            "Feature bridge specification across datasets. Use braces with segments separated by ';', "
            "one segment per dataset, e.g. \"{featA1, featA2; featB1}\" for two datasets. "
            "Repeat the flag for multiple bridge guesses."
        ),
    )
    parser.add_argument(
        "--region-select",
        type=str,
        default=None,
        help=(
            "Optional region selection string matching the number of datasets, e.g. \"{NA,AU; GLOBAL}\" "
            "to keep regions NA and AU from dataset 1 and region GLOBAL from dataset 2. "
            "Use ALL or * to retain every region for a dataset."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate preview PNG/TIF copies for bridge features/labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_paths = [Path(p).expanduser().resolve() for p in args.collections]
    if args.dataset_ids:
        if len(args.dataset_ids) != len(root_paths):
            raise ValueError("--dataset-ids must match the number of --collections")
        dataset_ids = list(args.dataset_ids)
    else:
        dataset_ids = [None] * len(root_paths)
    datasets: List[DatasetSummary] = []
    dataset_crs_map: Dict[str, Optional[CRS]] = {}
    for root, ds_id in zip(root_paths, dataset_ids):
        summary = _summarise_dataset(root, ds_id)
        datasets.append(summary)
        dataset_crs_map[summary.dataset_id] = None

    ordered_dataset_ids = [ds.dataset_id for ds in datasets]

    region_selection_map: Optional[Dict[str, List[str]]] = None
    if args.region_select:
        try:
            region_selection_map = _parse_region_selection(args.region_select, ordered_dataset_ids)
        except ValueError as exc:
            raise SystemExit(f"Error parsing --region-select entry '{args.region_select}': {exc}") from exc

    if region_selection_map:
        filtered_datasets: List[DatasetSummary] = []
        for summary in datasets:
            selections = region_selection_map.get(summary.dataset_id)
            if not selections:
                filtered_datasets.append(summary)
                continue
            normalized = {token.upper() for token in selections}
            if any(token in {"ALL", "*"} for token in normalized):
                filtered_datasets.append(summary)
                continue
            kept_regions = {name: info for name, info in summary.regions.items() if name.upper() in normalized}
            if not kept_regions:
                raise SystemExit(
                    f"Region selection removed all regions for dataset '{summary.dataset_id}'."
                )
            summary.regions = kept_regions
            used_feature_names: set[str] = set()
            used_label_names: set[str] = set()
            for region_info in kept_regions.values():
                used_feature_names.update(region_info.get("features", {}).keys())
                used_label_names.update(region_info.get("labels", {}).keys())
            summary.features = {
                name: entry
                for name, entry in summary.features.items()
                if name in used_feature_names
            }
            summary.labels = {
                name: entry
                for name, entry in summary.labels.items()
                if name in used_label_names
            }
            if summary.boundaries:
                filtered_boundaries: Dict[str, List[Dict]] = {}
                for key, records in summary.boundaries.items():
                    filtered_records = [
                        rec
                        for rec in records
                        if str(rec.get("region") or "GLOBAL").upper() in normalized
                    ]
                    if filtered_records:
                        filtered_boundaries[key] = filtered_records
                summary.boundaries = filtered_boundaries
            filtered_datasets.append(summary)
        datasets = filtered_datasets
        ordered_dataset_ids = [ds.dataset_id for ds in datasets]

    overlap_geoms_dataset: Dict[str, Polygon] = {}
    overlap_geom_equal = None
    target_crs = CRS.from_epsg(8857) if CRS is not None else None
    if shapely_shape is None or CRS is None or Transformer is None or target_crs is None:
        print("[warn] CRS transformation requested but shapely/pyproj unavailable; skipping overlap detection.")
    else:
        geom_pairs: List[Tuple[str, Polygon, Polygon]] = []
        for summary in datasets:
            dataset_crs = _detect_dataset_crs(summary)
            if dataset_crs is None:
                print(f"[warn] Unable to detect CRS for dataset {summary.dataset_id}; skipping overlap computation.")
                dataset_crs_map[summary.dataset_id] = None
                continue
            crs_str = dataset_crs.to_string() if hasattr(dataset_crs, "to_string") else str(dataset_crs)
            print(f"[info] Collection '{summary.dataset_id}' CRS detected {crs_str}")
            dataset_crs_map[summary.dataset_id] = dataset_crs
            native_geom = _collect_dataset_geometry(summary, dataset_crs)
            native_geom = _clean_projected_geometry(native_geom)
            if native_geom is None:
                print(f"[warn] Unable to derive geometry for dataset {summary.dataset_id}; skipping overlap computation.")
                continue
            transformer = Transformer.from_crs(dataset_crs, target_crs, always_xy=True)
            equal_geom = _transform_geometry(transformer, native_geom)
            if equal_geom is None:
                print(f"[warn] Projected geometry for dataset {summary.dataset_id} collapsed during cleaning; skipping.")
                continue
            geom_pairs.append((summary.dataset_id, native_geom, equal_geom))

        if len(geom_pairs) >= 2:
            max_area = max(eq_geom.area for _, _, eq_geom in geom_pairs if not eq_geom.is_empty)
            area_epsilon = max(max_area * 1e-8, 1e-6)
            overlap_geom_equal = geom_pairs[0][2]
            for _, _, eq_geom in geom_pairs[1:]:
                candidate = overlap_geom_equal.intersection(eq_geom)
                candidate = _clean_projected_geometry(candidate, min_area=area_epsilon)
                if candidate is None:
                    overlap_geom_equal = None
                    break
                overlap_geom_equal = candidate
            if overlap_geom_equal is None:
                print("[warn] No overlap area detected across datasets.")
            else:
                print("[info] Overlap area detected across datasets.")
                for dataset_id, _, eq_geom in geom_pairs:
                    dataset_overlap = _clean_projected_geometry(
                        eq_geom.intersection(overlap_geom_equal),
                        min_area=area_epsilon,
                    )
                    if dataset_overlap is None:
                        continue
                    overlap_geoms_dataset[dataset_id] = dataset_overlap
        elif geom_pairs:
            print("[warn] Overlap detection requires at least two datasets with valid footprints.")

    bridge_mappings: List[Dict[str, List[str]]] = []
    for raw_bridge in args.bridge:
        try:
            bridge_mapping = _parse_bridge_entry(raw_bridge, ordered_dataset_ids)
        except ValueError as exc:
            raise SystemExit(f"Error parsing --bridge entry '{raw_bridge}': {exc}") from exc
        bridge_mappings.append(bridge_mapping)

    dataset_lookup = {ds.dataset_id: ds for ds in datasets}
    label_geojson_map = {
        ds.dataset_id: {
            region: [str(path) for path in paths]
            for region, paths in ds.label_geojsons.items()
        }
        for ds in datasets
        if ds.label_geojsons
    }

    tile_geoms_index: Dict[str, List[Dict[str, object]]] = {}
    if shapely_shape is not None and rasterio is not None:
        for ds in datasets:
            tiles = _collect_dataset_tiles(ds, target_crs)
            if tiles:
                tile_geoms_index[ds.dataset_id] = tiles

    label_union = sorted({label for ds in datasets for label in ds.label_names})
    region_union = sorted({region for ds in datasets for region in ds.region_keys})

    payload = {
        "dataset_count": len(datasets),
        "region_union": region_union,
        "bridge_guess_number": int(args.bridge_guess_number or len(bridge_mappings)),
        "bridge_guesses": bridge_mappings,
        "datasets": [
            {
                "dataset_id": ds.dataset_id,
                "collection_id": ds.collection_id,
                "root": str(ds.root),
                "feature_names": ds.feature_names,
                "label_names": ds.label_names,
                "is_multi_region": ds.is_multi_region,
                "label_geojsons": {
                    region: [str(path) for path in paths]
                    for region, paths in ds.label_geojsons.items()
                },
                "regions": {
                    region: {
                        "feature_names": sorted(region_info.get("features", {}).keys()),
                        "label_names": sorted(region_info.get("labels", {}).keys()),
                        "features": region_info.get("features", {}),
                        "labels": region_info.get("labels", {}),
                        "feature_paths": {
                            feat: [
                                path
                                for rec in (entry.get("tifs", []) if isinstance(entry, dict) else [])
                                if isinstance(rec, dict)
                                for path in [rec.get("path_resolved") or rec.get("path")]
                                if path
                            ]
                            for feat, entry in region_info.get("features", {}).items()
                        },
                        "label_paths": {
                            lab: [
                                path
                                for rec in (entry.get("tifs", []) if isinstance(entry, dict) else [])
                                if isinstance(rec, dict)
                                for path in [rec.get("path_resolved") or rec.get("path")]
                                if path
                            ]
                            for lab, entry in region_info.get("labels", {}).items()
                        },
                    }
                    for region, region_info in ds.regions.items()
                },
                "boundaries": ds.boundaries,
            }
            for ds in datasets
        ],
        "label_union": label_union,
    }

    overlap_boundary_path: Optional[Path] = None
    if overlap_geom_equal is not None and shapely_shape is not None and mapping is not None and not overlap_geom_equal.is_empty:
        payload["study_area_overlap_detected"] = True
    else:
        payload["study_area_overlap_detected"] = False
        payload["study_area_overlap_boundary"] = None

    output_path = args.output
    project_name = args.projectname

    if project_name and not output_path:
        raise ValueError("--projectname requires --output to specify the base directory.")

    output_file: Optional[Path] = None
    visual_base: Optional[Path] = None

    if output_path:
        base_path = Path(output_path).expanduser()
        if project_name:
            target_dir = (base_path / project_name).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / "combined_metadata.json"
            visual_base = target_dir
        else:
            resolved = base_path.resolve()
            if resolved.suffix:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                output_file = resolved
                visual_base = resolved.parent
            else:
                resolved.mkdir(parents=True, exist_ok=True)
                output_file = resolved / "combined_metadata.json"
                visual_base = resolved

    if output_file is None and args.visualize and visual_base is None:
        visual_base = Path.cwd() / "integrate_stac_output"

    overlap_mask_path: Optional[Path] = None

    if (
        payload.get("study_area_overlap_detected")
        and overlap_geom_equal is not None
        and shapely_shape is not None
        and mapping is not None
    ):
        boundary_dir = visual_base if visual_base is not None else Path.cwd()
        boundary_dir.mkdir(parents=True, exist_ok=True)
        overlap_boundary_path = boundary_dir / "study_area_overlap.geojson"
        try:
            geom_mapping = mapping(overlap_geom_equal)
            geojson_doc = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "crs": "EPSG:8857"
                        },
                        "geometry": geom_mapping,
                    }
                ],
            }
            overlap_boundary_path.write_text(json.dumps(geojson_doc, indent=2), encoding="utf-8")
            payload["study_area_overlap_boundary"] = str(overlap_boundary_path)
        except Exception as exc:
            print(f"[warn] Failed to write overlap boundary GeoJSON: {exc}")
            payload["study_area_overlap_boundary"] = None

        overlap_mask_path = boundary_dir / "study_area_overlap.tif"
        if rasterio is not None and rio_features is not None and from_origin is not None and target_crs is not None:
            try:
                minx, miny, maxx, maxy = overlap_geom_equal.bounds
                if maxx > minx and maxy > miny:
                    resolution = 500.0
                    width = max(1, int(np.ceil((maxx - minx) / resolution)))
                    height = max(1, int(np.ceil((maxy - miny) / resolution)))
                    transform = from_origin(minx, maxy, resolution, resolution)
                    mask_array = rio_features.rasterize(
                        [(overlap_geom_equal, 1)],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        default_value=1,
                        all_touched=True,
                        dtype="uint8",
                    )
                    with rasterio.open(
                        overlap_mask_path,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=1,
                        dtype="uint8",
                        crs=target_crs,
                        transform=transform,
                    ) as dst:
                        dst.write(mask_array, 1)
                    payload["study_area_overlap_mask"] = str(overlap_mask_path)
            except Exception as exc:
                print(f"[warn] Failed to write overlap boundary GeoTIFF: {exc}")
                if overlap_mask_path.exists():
                    try:
                        overlap_mask_path.unlink()
                    except Exception:
                        pass
                overlap_mask_path = None
                payload["study_area_overlap_mask"] = None
        else:
            print("[warn] Rasterio unavailable; skipping overlap mask generation.")
            payload["study_area_overlap_mask"] = None
    else:
        payload["study_area_overlap_mask"] = None

    pair_output_root = visual_base if visual_base is not None else Path.cwd()
    overlap_pairs_outputs: Dict[str, str] = {}
    if tile_geoms_index:
        overlap_pairs_outputs = _write_overlap_pairs(tile_geoms_index, overlap_geom_equal, pair_output_root)
    payload["study_area_overlap_pairs"] = overlap_pairs_outputs

    if args.visualize:
        if visual_base is None:
            visual_base = Path.cwd() / "integrate_stac_output"
        generate_bridge_visualizations = _load_bridge_visualizer()
        generate_bridge_visualizations(
            bridge_mappings,
            dataset_lookup,
            visual_base / "bridge_visualizations",
            label_geojson_map,
            overlap_geoms_dataset if overlap_geoms_dataset else None,
            dataset_crs_map,
            target_crs,
            overlap_geom_equal,
            overlap_mask_path,
        )

    if output_file is not None:
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[info] Wrote integrated metadata to {output_file}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
