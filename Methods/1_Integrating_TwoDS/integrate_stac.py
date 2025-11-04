#!/usr/bin/env python3
"""
Utilities to combine metadata from multiple STAC collections into a unified feature schema. 
This is the first step for the Universal GFM4MPM pipeline which will ingest different regional STAC exports (e.g., CAN/US/AU, BC) and
produce a compatible superset of channels.
We generate additional "overlap" features to utilize the information in injested collections optimally.
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from affine import Affine
from Common.overlap_debug_plot import save_overlap_debug_plot
from itertools import combinations

try:
    from shapely.geometry import MultiPolygon, Polygon, Point, box, mapping, shape as shapely_shape
    from shapely.ops import transform as shapely_transform, unary_union
except Exception:  # pragma: no cover - optional dependency guard
    shapely_shape = None  # type: ignore[assignment]
    Polygon = MultiPolygon = Point = None  # type: ignore[assignment]
    shapely_transform = unary_union = None  # type: ignore[assignment]

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency guard
    CRS = Transformer = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio import features as rio_features
    from rasterio.transform import from_origin, rowcol
except Exception:  # pragma: no cover - optional dependency guard
    rasterio = None  # type: ignore[assignment]
    rio_features = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - optional dependency guard
    NearestNeighbors = None  # type: ignore[assignment]

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


def _pixel_resolution(transform) -> Optional[float]:
    if transform is None:
        return None
    try:
        res_x = math.sqrt(transform.a * transform.a + transform.b * transform.b)
        res_y = math.sqrt(transform.d * transform.d + transform.e * transform.e)
    except AttributeError:
        return None
    res_candidates = [res for res in (res_x, res_y) if isinstance(res, (int, float)) and res > 0]
    if not res_candidates:
        return None
    return float(min(res_candidates))


def _normalise_tile_id(
    dataset_id: str,
    tile: Dict[str, object],
    row: Optional[int],
    col: Optional[int],
    centroid_key: Tuple[float, float],
) -> str:
    region = str(tile.get("region") or "GLOBAL").upper()
    if row is not None and col is not None:
        return f"{region}_{row}_{col}"
    centroid_int = (int(round(centroid_key[0] * 1000)), int(round(centroid_key[1] * 1000)))
    return f"{dataset_id}_CENTROID_{centroid_int[0]}_{centroid_int[1]}"


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

    module = import_module("Common.cls.infer.infer_maps")
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


def _write_geojson_feature(path: Path, geom: Polygon, source_path: Path) -> None:
    feature = {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "source_path": str(source_path),
        },
    }
    payload = {"type": "FeatureCollection", "features": [feature]}
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _rasterize_geometry_to_path(
    geom: Polygon,
    out_path: Path,
    *,
    transform: Affine,
    width: int,
    height: int,
    target_crs,
    source_path: Path,
) -> None:
    if rasterio is None or rio_features is None:
        return
    try:
        shapes = [(mapping(geom), 1)]
        raster = rio_features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            transform=transform,
            crs=target_crs,
        ) as dst:
            dst.write(raster, 1)
            dst.update_tags(source_path=str(source_path))
    except Exception:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass


def _convert_boundary_to_unified_crs(
    src_path: Path,
    *,
    transformer,
    target_crs,
    dataset_root: Path,
    role: Optional[str],
    region: Optional[str],
    integration_root: Path,
    default_resolution: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    if shapely_shape is None or Transformer is None or target_crs is None:
        return None
    geom_native = _load_boundary_geometry(src_path)
    if geom_native is None or geom_native.is_empty:
        return None
    projected_geom = _transform_geometry(transformer, geom_native)
    if projected_geom is None or projected_geom.is_empty:
        return None

    minx, miny, maxx, maxy = projected_geom.bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return None
    span_x = maxx - minx
    span_y = maxy - miny
    if span_x <= 0 or span_y <= 0:
        return None

    resolution = default_resolution
    if resolution is None and rasterio is not None:
        suffix = src_path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            try:
                with rasterio.open(src_path) as src:
                    res_val = _pixel_resolution(src.transform)
                    if res_val is not None and res_val > 0:
                        resolution = float(res_val)
            except Exception:
                pass
    if resolution is None or not math.isfinite(resolution) or resolution <= 0:
        resolution = max(span_x, span_y) / 2048.0
        if not math.isfinite(resolution) or resolution <= 0:
            resolution = 1000.0

    width = max(1, int(math.ceil(span_x / resolution)))
    height = max(1, int(math.ceil(span_y / resolution)))
    transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)

    base_dir = integration_root / "data" / "Boundary_UnifiedCRS"
    dataset_dir = base_dir / dataset_root.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_name = src_path.stem + "_UnifiedCRS"
    out_tif = dataset_dir / f"{base_name}.tif"
    out_geojson = dataset_dir / f"{base_name}.geojson"

    _write_geojson_feature(out_geojson, projected_geom, src_path)
    _rasterize_geometry_to_path(
        projected_geom,
        out_tif,
        transform=transform,
        width=width,
        height=height,
        target_crs=target_crs,
        source_path=src_path,
    )

    record: Dict[str, object] = {
        "path": str(out_tif),
        "filename": out_tif.name,
        "role": role or "UnifiedCRS",
        "path_resolved": str(out_tif),
        "geojson_path": str(out_geojson),
        "geojson_path_resolved": str(out_geojson),
        "source_path": str(src_path),
    }
    if region is not None:
        record["region"] = region
    return record


def _ensure_unified_crs_boundaries(
    summary: DatasetSummary,
    dataset_crs: Optional[CRS],
    target_crs,
    *,
    integration_root: Path,
) -> None:
    if dataset_crs is None or Transformer is None or shapely_shape is None or target_crs is None:
        return
    transformer = Transformer.from_crs(dataset_crs, target_crs, always_xy=True)
    processed: set[Path] = set()
    new_records: List[Dict[str, object]] = []

    def _handle(path: Path, role: Optional[str], region: Optional[str]) -> None:
        if not path.exists():
            return
        resolved = path.resolve()
        if resolved.suffix.lower() not in {".tif", ".tiff", ".geojson", ".json"}:
            return
        if resolved.name.endswith("_UnifiedCRS.tif") or resolved.name.endswith("_UnifiedCRS.geojson"):
            return
        if resolved in processed:
            return
        processed.add(resolved)
        record = _convert_boundary_to_unified_crs(
            resolved,
            transformer=transformer,
            target_crs=target_crs,
            dataset_root=summary.root,
            role=role,
            region=region,
            integration_root=integration_root,
        )
        if record is not None:
            new_records.append(record)

    for records in summary.boundaries.values():
        for record in records:
            path_value = record.get("path_resolved") or record.get("path")
            if not path_value:
                continue
            src_path = Path(path_value)
            if not src_path.is_absolute():
                src_path = (summary.root / src_path).resolve()
            role = record.get("role") if isinstance(record, dict) else None
            region = record.get("region") if isinstance(record, dict) else None
            _handle(src_path, role, region)

    boundaries_dir = summary.root / "boundaries"
    if boundaries_dir.exists():
        for candidate in boundaries_dir.glob("*"):
            if not candidate.is_file():
                continue
            _handle(candidate, None, None)

    if new_records:
        unified_records = summary.boundaries.setdefault("Unified CRS", [])
        existing_paths = {
            Path(record.get("path_resolved", "")).resolve()
            for record in unified_records
            if isinstance(record, dict) and record.get("path_resolved")
        }
        for record in new_records:
            record_path = Path(record.get("path_resolved", "")).resolve()
            if record_path not in existing_paths:
                unified_records.append(record)

def _collect_dataset_tiles_v0(summary: DatasetSummary, target_crs: Optional[CRS]) -> List[Dict[str, object]]:
    # _collect_dataset_tiles_v0 assumes consistent data coverage (region) per region/dataset.
    if shapely_shape is None or rasterio is None or box is None:
        return []
    tiles: List[Dict[str, object]] = []
    seen_paths: set[Path] = set()
    for region_name, region_info in summary.regions.items():
        feature_entries = region_info.get("features", {}) if isinstance(region_info, dict) else {}
        # use the first feature entry per region as representative coverage
        representative = None
        for feat_name, entry in feature_entries.items():
            if isinstance(entry, dict):
                representative = (feat_name, entry)
                break
        if representative is None:
            continue
        feature_name, entry = representative
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
                    transform = ds.transform
                    width = ds.width
                    height = ds.height
                    resolution = _pixel_resolution(transform)
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
                    "crs": src_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "pixel_resolution": resolution,
                }
            )
            seen_paths.add(tile_path)
    return tiles


def _load_embedding_windows(bundle_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not bundle_path.exists():
        print(f"[warn] Embedding bundle not found at {bundle_path}; skipping.")
        return records
    try:
        with np.load(bundle_path, allow_pickle=True) as data:
            tile_ids = data.get("tile_ids")
            coords = data.get("coords")
            metadata = data.get("metadata")
            if tile_ids is None or coords is None:
                print(f"[warn] Embedding bundle {bundle_path} lacks tile_ids/coords; skipping.")
                return records
            tile_ids = np.asarray(tile_ids).astype(str)
            coords_arr = np.asarray(coords, dtype=np.float64)
            if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
                print(f"[warn] Embedding bundle {bundle_path} provides malformed coords array; skipping.")
                return records
            meta_entries: Optional[np.ndarray]
            if metadata is not None:
                meta_entries = np.asarray(metadata, dtype=object)
            else:
                meta_entries = None
            count = tile_ids.shape[0]
            for idx in range(count):
                tile_id = str(tile_ids[idx])
                coord = coords_arr[idx]
                if coord.size < 2 or not np.all(np.isfinite(coord[:2])):
                    coord_tuple: Optional[Tuple[float, float]] = None
                else:
                    coord_tuple = (float(coord[0]), float(coord[1]))
                meta_raw = None
                if meta_entries is not None:
                    meta_raw = meta_entries[idx]
                    if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == object and meta_raw.ndim == 0:
                        meta_raw = meta_raw.item()
                    elif hasattr(meta_raw, "item") and not isinstance(meta_raw, dict):
                        try:
                            meta_raw = meta_raw.item()
                        except Exception:
                            pass
                meta_dict = meta_raw if isinstance(meta_raw, dict) else {}
                region = str(meta_dict.get("region") or meta_dict.get("Region") or "GLOBAL")
                row_val = meta_dict.get("row")
                col_val = meta_dict.get("col")
                try:
                    row_int = int(row_val) if row_val is not None else None
                except Exception:
                    row_int = None
                try:
                    col_int = int(col_val) if col_val is not None else None
                except Exception:
                    col_int = None
                if row_int is None or col_int is None:
                    parts = tile_id.split("_")
                    if len(parts) >= 3:
                        try:
                            col_candidate = int(parts[-1])
                            row_candidate = int(parts[-2])
                            region_candidate = "_".join(parts[:-2]) or region
                            row_int = row_int if row_int is not None else row_candidate
                            col_int = col_int if col_int is not None else col_candidate
                            region = region_candidate or region
                        except Exception:
                            pass
                pixel_resolution = meta_dict.get("pixel_resolution") or meta_dict.get("resolution")
                try:
                    pixel_resolution_float = float(pixel_resolution) if pixel_resolution is not None else None
                except Exception:
                    pixel_resolution_float = None
                label_val = meta_dict.get("label") if isinstance(meta_dict, dict) else None
                try:
                    label_int = int(label_val) if label_val is not None else None
                except Exception:
                    label_int = None
                records.append(
                    {
                        "tile_id": tile_id,
                        "coord": coord_tuple,
                        "region": region,
                        "row": row_int,
                        "col": col_int,
                        "path": meta_dict.get("path") or meta_dict.get("source_path"),
                        "pixel_resolution": pixel_resolution_float,
                        "metadata": meta_dict,
                        "label": label_int,
                        "is_augmented": False,
                    }
                )
    except Exception as exc:
        print(f"[warn] Failed to load embedding bundle {bundle_path}: {exc}")
    return records


def _load_augmented_embedding_windows(bundle_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not bundle_path.exists():
        print(f"[warn] Augmentation bundle not found at {bundle_path}; skipping.")
        return records
    try:
        with np.load(bundle_path, allow_pickle=True) as data:
            tile_ids = data.get("tile_ids")
            coords = data.get("coords")
            metadata = data.get("metadata")
            flags = data.get("is_augmented")
            if tile_ids is None or coords is None or metadata is None or flags is None:
                print(f"[warn] Augmentation bundle {bundle_path} lacks required arrays; skipping.")
                return records
            tile_ids = np.asarray(tile_ids).astype(str)
            coords_arr = np.asarray(coords, dtype=np.float64)
            flags = np.asarray(flags, dtype=bool)
            meta_entries = np.asarray(metadata, dtype=object)
            count = tile_ids.shape[0]
            for idx in range(count):
                if not bool(flags[idx]):
                    continue
                coord = coords_arr[idx]
                if coord.size < 2 or not np.all(np.isfinite(coord[:2])):
                    continue
                coord_tuple = (float(coord[0]), float(coord[1]))
                meta_raw = meta_entries[idx]
                if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == object and meta_raw.ndim == 0:
                    meta_raw = meta_raw.item()
                elif hasattr(meta_raw, "item") and not isinstance(meta_raw, dict):
                    try:
                        meta_raw = meta_raw.item()
                    except Exception:
                        pass
                meta_dict = meta_raw if isinstance(meta_raw, dict) else {}
                region = str(meta_dict.get("region") or meta_dict.get("Region") or "GLOBAL")
                row_val = meta_dict.get("row")
                col_val = meta_dict.get("col")
                try:
                    row_int = int(row_val) if row_val is not None else None
                except Exception:
                    row_int = None
                try:
                    col_int = int(col_val) if col_val is not None else None
                except Exception:
                    col_int = None
                source_tile_id = meta_dict.get("source_tile_id") or tile_ids[idx].split("__")[0]
                label_val = meta_dict.get("label") if isinstance(meta_dict, dict) else None
                try:
                    label_int = int(label_val) if label_val is not None else 1
                except Exception:
                    label_int = 1
                records.append(
                    {
                        "tile_id": str(tile_ids[idx]),
                        "coord": coord_tuple,
                        "region": region,
                        "row": row_int,
                        "col": col_int,
                        "source_tile_id": source_tile_id,
                        "is_augmented": True,
                        "label": label_int,
                    }
                )
    except Exception as exc:
        print(f"[warn] Failed to parse augmentation bundle {bundle_path}: {exc}")
        return []
    return records


def _compute_embedding_spacing_map(windows_map: Dict[str, List[Dict[str, object]]]) -> Dict[str, float]:
    spacing: Dict[str, float] = {}
    if NearestNeighbors is None:
        return spacing
    for dataset_id, records in windows_map.items():
        coords = np.asarray(
            [rec["coord"] for rec in records if rec.get("coord") is not None],
            dtype=np.float64,
        )
        if coords.shape[0] < 2 or coords.shape[1] < 2:
            continue
        try:
            n_neighbors = min(coords.shape[0], 8)
            model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
            model.fit(coords)
            distances, _ = model.kneighbors(coords)
            if distances.shape[1] < 2:
                continue
            nn_dist = distances[:, 1]
            nn_dist = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
            if nn_dist.size == 0:
                continue
            spacing[dataset_id] = float(np.median(nn_dist))
        except Exception:
            continue
    return spacing


def _determine_alignment_roles(
    ds_a: str,
    ds_b: str,
    spacing_map: Optional[Dict[str, Optional[float]]],
    dataset_resolutions: Optional[Dict[str, float]] = None,
) -> Tuple[str, str]:
    spacing_map = spacing_map or {}
    spacing_a = spacing_map.get(ds_a)
    spacing_b = spacing_map.get(ds_b)
    if spacing_a is None and dataset_resolutions is not None:
        spacing_a = dataset_resolutions.get(ds_a)
    if spacing_b is None and dataset_resolutions is not None:
        spacing_b = dataset_resolutions.get(ds_b)
    if spacing_a is not None and spacing_b is not None and spacing_a != spacing_b:
        return (ds_a, ds_b) if spacing_a <= spacing_b else (ds_b, ds_a)
    return (ds_a, ds_b) if ds_a.lower() <= ds_b.lower() else (ds_b, ds_a)


def _record_label_value(record: Optional[Dict[str, object]]) -> Optional[int]:
    if not isinstance(record, dict):
        return None
    label_val = record.get("label")
    if label_val is None and isinstance(record.get("metadata"), dict):
        meta = record["metadata"]
        label_val = meta.get("label") or meta.get("Label")
    try:
        return int(label_val) if label_val is not None else None
    except Exception:
        return None


def _record_is_augmented(record: Optional[Dict[str, object]]) -> bool:
    if not isinstance(record, dict):
        return False
    flag = record.get("is_augmented")
    if isinstance(flag, bool):
        return flag
    if flag in {1, "1", "true", "True"}:
        return True
    return False


def _build_embedding_pair_record(
    ds_a: str,
    ds_b: str,
    fine_ds: str,
    coarse_ds: str,
    fine_rec: Dict[str, object],
    coarse_rec: Dict[str, object],
    centroid_list: List[float],
) -> Dict[str, object]:
    rec_map = {
        fine_ds: fine_rec,
        coarse_ds: coarse_rec,
    }

    rec_a = rec_map.get(ds_a)
    rec_b = rec_map.get(ds_b)

    def _row_col(rec: Optional[Dict[str, object]]) -> Optional[List[int]]:
        if rec is None:
            return None
        row = rec.get("row")
        col = rec.get("col")
        try:
            row_int = int(row) if row is not None else None
            col_int = int(col) if col is not None else None
        except Exception:
            row_int = col_int = None
        if row_int is None or col_int is None:
            return None
        return [row_int, col_int]

    def _native_point(rec: Optional[Dict[str, object]]) -> Optional[List[float]]:
        if rec is None:
            return None
        coord = rec.get("coord")
        if coord is None:
            return None
        try:
            return [float(coord[0]), float(coord[1])]
        except Exception:
            return None

    def _lookup(rec: Optional[Dict[str, object]]) -> Optional[str]:
        if rec is None:
            return None
        val = rec.get("tile_id")
        return str(val) if val is not None else None

    native_a = _native_point(rec_a)
    native_b = _native_point(rec_b)
    centroid_override = None
    if native_a is not None and native_b is not None:
        centroid_override = [
            float((native_a[0] + native_b[0]) / 2.0),
            float((native_a[1] + native_b[1]) / 2.0),
        ]
    return {
        "dataset_0_row_col": _row_col(rec_a),
        "dataset_0_native_point": native_a,
        "dataset_0_lookup": _lookup(rec_a),
        "dataset_0_label": _record_label_value(rec_a),
        "dataset_0_is_augmented": _record_is_augmented(rec_a),
        "dataset_1_row_col": _row_col(rec_b),
        "dataset_1_native_point": native_b,
        "dataset_1_lookup": _lookup(rec_b),
        "dataset_1_label": _record_label_value(rec_b),
        "dataset_1_is_augmented": _record_is_augmented(rec_b),
        "overlap_centerloid": centroid_override or centroid_list,
    }


def _generate_embedding_overlap_pairs(
    ds_a: str,
    ds_b: str,
    embedding_windows: Dict[str, List[Dict[str, object]]],
    spacing_map: Dict[str, Optional[float]],
    dataset_resolutions: Dict[str, float],
) -> Optional[Tuple[List[Dict[str, object]], Dict[str, object]]]:
    if NearestNeighbors is None:
        return None
    if ds_a not in embedding_windows or ds_b not in embedding_windows:
        return None

    fine_ds, coarse_ds = _determine_alignment_roles(ds_a, ds_b, spacing_map, dataset_resolutions)
    fine_records = [rec for rec in embedding_windows.get(fine_ds, []) if rec.get("coord") is not None]
    coarse_records = [rec for rec in embedding_windows.get(coarse_ds, []) if rec.get("coord") is not None]
    if not fine_records or not coarse_records:
        return None

    coords_fine = np.asarray([rec["coord"] for rec in fine_records], dtype=np.float64)
    coords_coarse = np.asarray([rec["coord"] for rec in coarse_records], dtype=np.float64)
    if coords_fine.shape[0] == 0 or coords_coarse.shape[0] == 0:
        return None

    spacing_fine = spacing_map.get(fine_ds)
    spacing_coarse = spacing_map.get(coarse_ds)
    radius_candidates = [val for val in (spacing_coarse, spacing_fine) if val is not None and np.isfinite(val) and val > 0]
    if not radius_candidates:
        for dataset_id in (fine_ds, coarse_ds):
            val = dataset_resolutions.get(dataset_id)
            if val is not None and np.isfinite(val) and val > 0:
                radius_candidates.append(val)
    base_radius = float(np.median(radius_candidates)) if radius_candidates else 1.0
    search_radius = float(max(base_radius * 0.75, 1.0))

    try:
        radius_model = NearestNeighbors(radius=search_radius, algorithm="ball_tree")
        radius_model.fit(coords_coarse)
        neighbors_radius = radius_model.radius_neighbors(coords_fine, return_distance=False)
        nearest_model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nearest_model.fit(coords_coarse)
        nearest_dist, nearest_idx = nearest_model.kneighbors(coords_fine, return_distance=True)
    except Exception as exc:
        print(f"[warn] Failed to compute embedding-based overlaps for {ds_a}-{ds_b}: {exc}")
        return None

    aggregated: Dict[Tuple[str, str], Dict[str, object]] = {}
    raw_candidates = 0
    for fine_idx, coarse_candidates in enumerate(neighbors_radius):
        candidate_indices = np.array(coarse_candidates, dtype=int)
        if candidate_indices.size == 0:
            candidate_indices = nearest_idx[fine_idx]
        candidate_indices = np.unique(candidate_indices)
        for coarse_idx in candidate_indices:
            raw_candidates += 1
            distance_val = None
            if isinstance(nearest_dist, np.ndarray) and nearest_dist.size > fine_idx:
                distance_val = float(nearest_dist[fine_idx][0])
            else:
                distance_val = float(np.linalg.norm(coords_fine[fine_idx] - coords_coarse[coarse_idx]))
            fine_rec = fine_records[fine_idx]
            coarse_rec = coarse_records[int(coarse_idx)]
            key = (str(fine_rec.get("tile_id")), str(coarse_rec.get("tile_id")))
            if key in aggregated:
                continue
            centroid_coords = fine_rec.get("coord") or coarse_rec.get("coord")
            if centroid_coords is not None:
                try:
                    centroid_list = [float(centroid_coords[0]), float(centroid_coords[1])]
                except Exception:
                    centroid_list = [float("nan"), float("nan")]
            else:
                centroid_list = [float("nan"), float("nan")]
            aggregated[key] = _build_embedding_pair_record(
                ds_a,
                ds_b,
                fine_ds,
                coarse_ds,
                fine_rec,
                coarse_rec,
                centroid_list,
            )

    records = list(aggregated.values())
    if not records:
        return None

    doc_meta: Dict[str, object] = {
        "alignment_roles": {
            "fine": fine_ds,
            "coarse": coarse_ds,
            ds_a: "fine" if ds_a == fine_ds else "coarse",
            ds_b: "fine" if ds_b == fine_ds else "coarse",
        },
        "approx_spacing": {
            ds_a: spacing_map.get(ds_a),
            ds_b: spacing_map.get(ds_b),
        },
        "search_radius": search_radius,
        "raw_pair_candidates": raw_candidates,
        "generated_from": "embedding_windows",
        "dataset_pair": [ds_a, ds_b],
    }
    return records, doc_meta


def _generate_raster_overlap_pairs(
    ds_a: str,
    ds_b: str,
    tile_geoms: Dict[str, List[Dict[str, object]]],
    overlap_geom: Optional[Polygon],
    target_crs: Optional[CRS],
    dataset_resolutions: Dict[str, float],
) -> Optional[Tuple[List[Dict[str, object]], Dict[str, object]]]:
    tiles_a = tile_geoms.get(ds_a, [])
    tiles_b = tile_geoms.get(ds_b, [])
    if not tiles_a or not tiles_b or shapely_shape is None:
        return None

    raw_pairs = 0
    aggregated: Dict[Tuple[str, str], Dict[str, object]] = {}

    def _native_representative_point(tile: Dict[str, object], geom_equal: Polygon) -> Optional[Point]:
        if geom_equal.is_empty or shapely_transform is None or Transformer is None:
            return None
        crs_src = tile.get("crs")
        if crs_src is None or target_crs is None or crs_src == target_crs:
            try:
                return geom_equal.representative_point()
            except Exception:
                return None
        try:
            transformer = Transformer.from_crs(target_crs, crs_src, always_xy=True)
        except Exception:
            return None
        try:
            native_geom = shapely_transform(transformer.transform, geom_equal)
        except Exception:
            return None
        if native_geom.is_empty:
            return None
        try:
            return native_geom.representative_point()
        except Exception:
            return None

    def _locate_pixel(
        tile: Dict[str, object],
        native_point: Optional[Point],
        fallback_centroid: Tuple[float, float],
    ) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]]]:
        if rowcol is None:
            return None, None, None
        transform = tile.get("transform")
        crs_src = tile.get("crs")
        width = tile.get("width")
        height = tile.get("height")
        if transform is None or crs_src is None or width is None or height is None:
            return None, None, None
        if native_point is not None and not native_point.is_empty:
            x_native = float(native_point.x)
            y_native = float(native_point.y)
        else:
            x_native, y_native = fallback_centroid
            if target_crs is not None and Transformer is not None and crs_src != target_crs:
                try:
                    transformer = Transformer.from_crs(target_crs, crs_src, always_xy=True)
                    x_native, y_native = transformer.transform(*fallback_centroid)
                except Exception:
                    return None, None, None
        try:
            inv_transform = ~transform
            col_float, row_float = inv_transform * (x_native, y_native)
        except Exception:
            try:
                col_float, row_float = rowcol(transform, x_native, y_native, op=float)
            except Exception:
                return None, None, (x_native, y_native)
        if any(math.isnan(val) for val in (row_float, col_float)):
            return None, None, (x_native, y_native)
        width_int = int(width)
        height_int = int(height)
        row = int(round(row_float))
        col = int(round(col_float))
        if not (0 <= row < height_int and 0 <= col < width_int):
            pixel_tol = tile.get("pixel_resolution") or 0.0
            tol_rows = max(0.5, float(pixel_tol) / max(abs(transform.e), 1e-6))
            tol_cols = max(0.5, float(pixel_tol) / max(abs(transform.a), 1e-6))
            if (-tol_rows <= row_float <= height_int - 1 + tol_rows) and (
                -tol_cols <= col_float <= width_int - 1 + tol_cols
            ):
                row = min(max(int(round(row_float)), 0), height_int - 1)
                col = min(max(int(round(col_float)), 0), width_int - 1)
            else:
                return None, None, (x_native, y_native)
        return row, col, (x_native, y_native)

    for tile_a in tiles_a:
        geom_a = tile_a.get("geom")
        if geom_a is None:
            continue
        area_a = float(geom_a.area) if geom_a.area else None
        for tile_b in tiles_b:
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
            raw_pairs += 1
            centroid = inter.centroid
            centroid_x = float(centroid.x)
            centroid_y = float(centroid.y)
            rep_a = _native_representative_point(tile_a, inter)
            rep_b = _native_representative_point(tile_b, inter)
            row_a, col_a, native_a = _locate_pixel(tile_a, rep_a, (centroid_x, centroid_y))
            row_b, col_b, native_b = _locate_pixel(tile_b, rep_b, (centroid_x, centroid_y))

            def _tile_identifier(tile: Dict[str, object], row: Optional[int], col: Optional[int]) -> Optional[str]:
                if row is not None and col is not None:
                    region = str(tile.get("region") or "GLOBAL").upper()
                    return f"{region}_{row}_{col}"
                value = tile.get("tile_id")
                return str(value) if value is not None else None

            tile_id_a = _tile_identifier(tile_a, row_a, col_a)
            tile_id_b = _tile_identifier(tile_b, row_b, col_b)
            key = (str(tile_id_a), str(tile_id_b))
            if key in aggregated:
                continue

            def _row_col(row_val: Optional[int], col_val: Optional[int]) -> Optional[List[int]]:
                if row_val is None or col_val is None:
                    return None
                return [int(row_val), int(col_val)]

            def _native_list(point: Optional[Tuple[float, float]]) -> Optional[List[float]]:
                if point is None:
                    return None
                try:
                    return [float(point[0]), float(point[1])]
                except Exception:
                    return None

            native_list_a = _native_list(native_a)
            native_list_b = _native_list(native_b)
            centroid_override = None
            if native_list_a is not None and native_list_b is not None:
                centroid_override = [
                    float((native_list_a[0] + native_list_b[0]) / 2.0),
                    float((native_list_a[1] + native_list_b[1]) / 2.0),
                ]

            aggregated[key] = {
                "dataset_0_row_col": _row_col(row_a, col_a),
                "dataset_0_native_point": native_list_a,
                "dataset_0_lookup": tile_id_a,
                "dataset_0_label": None,
                "dataset_0_is_augmented": False,
                "dataset_1_row_col": _row_col(row_b, col_b),
                "dataset_1_native_point": native_list_b,
                "dataset_1_lookup": tile_id_b,
                "dataset_1_label": None,
                "dataset_1_is_augmented": False,
                "overlap_centerloid": centroid_override or [centroid_x, centroid_y],
            }

    if not aggregated:
        return None

    records = list(aggregated.values())
    doc_meta = {
        "alignment_roles": {
            "fine": ds_a,
            "coarse": ds_b,
            ds_a: "fine",
            ds_b: "coarse",
        },
        "approx_spacing": {
            ds_a: dataset_resolutions.get(ds_a),
            ds_b: dataset_resolutions.get(ds_b),
        },
        "search_radius": None,
        "raw_pair_candidates": raw_pairs,
        "generated_from": "tile_bounds",
        "dataset_pair": [ds_a, ds_b],
    }
    return records, doc_meta

def _compute_dataset_resolutions(tile_geoms: Dict[str, List[Dict[str, object]]]) -> Dict[str, float]:
    resolutions: Dict[str, float] = {}
    for dataset_id, tiles in tile_geoms.items():
        candidates: List[float] = []
        for tile in tiles:
            value = tile.get("pixel_resolution")
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(float(value))
        if candidates:
            resolutions[dataset_id] = min(candidates)
    return resolutions


@dataclass
class _PairVariantSpec:
    key: str
    suffix: str = ""
    pair_label: Optional[str] = None
    filter_fn: Optional[Callable[[Dict[str, object]], bool]] = None


def _write_overlap_pairs(
    tile_geoms: Dict[str, List[Dict[str, object]]],
    overlap_geom: Optional[Polygon],
    target_dir: Path,
    target_crs: Optional[CRS],
    dataset_resolutions: Optional[Dict[str, float]] = None,
    embedding_windows: Optional[Dict[str, List[Dict[str, object]]]] = None,
    embedding_spacing: Optional[Dict[str, Optional[float]]] = None,
    variants: Optional[Sequence[_PairVariantSpec]] = None,
    dataset_order: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, str]]:
    if dataset_resolutions is None:
        dataset_resolutions = _compute_dataset_resolutions(tile_geoms)
    data_dir = target_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    embedding_windows = embedding_windows or {}
    embedding_spacing = embedding_spacing or {}
    if not variants:
        variants = [_PairVariantSpec(key="default", suffix="", pair_label=None, filter_fn=None)]
    outputs: Dict[str, Dict[str, str]] = {spec.key: {} for spec in variants}

    all_dataset_ids: List[str] = []
    if dataset_order:
        seen = set()
        for ds_id in dataset_order:
            if ds_id in seen:
                continue
            seen.add(ds_id)
            all_dataset_ids.append(ds_id)
    remaining = (set(tile_geoms.keys()) | set(embedding_windows.keys())) - set(all_dataset_ids)
    if remaining:
        all_dataset_ids.extend(sorted(remaining))

    for idx_a in range(len(all_dataset_ids)):
        ds_a = all_dataset_ids[idx_a]
        for idx_b in range(idx_a + 1, len(all_dataset_ids)):
            ds_b = all_dataset_ids[idx_b]
            if ds_a == ds_b:
                continue

            use_embedding = (
                ds_a in embedding_windows
                and ds_b in embedding_windows
                and NearestNeighbors is not None
            )

            result: Optional[Tuple[List[Dict[str, object]], Dict[str, object]]]
            if use_embedding:
                result = _generate_embedding_overlap_pairs(
                    ds_a,
                    ds_b,
                    embedding_windows,
                    embedding_spacing,
                    dataset_resolutions,
                )
            else:
                result = _generate_raster_overlap_pairs(
                    ds_a,
                    ds_b,
                    tile_geoms,
                    overlap_geom,
                    target_crs,
                    dataset_resolutions,
                )

            if not result:
                continue

            records, meta = result
            if not records:
                continue

            raw_candidates = int(meta.get("raw_pair_candidates") or len(records))
            alignment_roles = meta.get("alignment_roles") if isinstance(meta.get("alignment_roles"), dict) else None
            approx_spacing = meta.get("approx_spacing") if isinstance(meta.get("approx_spacing"), dict) else None
            search_radius = meta.get("search_radius") if isinstance(meta.get("search_radius"), (int, float)) else None
            generated_from = meta.get("generated_from") if isinstance(meta.get("generated_from"), str) else None

            for spec in variants:
                filtered_records = records
                if spec.filter_fn is not None:
                    filtered_records = [rec for rec in records if spec.filter_fn(rec)]
                if not filtered_records:
                    continue
                suffix_token = spec.suffix or ""
                if suffix_token and not suffix_token.startswith("_"):
                    suffix_token = f"_{suffix_token}"
                label_suffix = suffix_token
                pair_name = f"{ds_a}_{ds_b}_overlap_pairs{suffix_token}.json"
                output_path = data_dir / pair_name
                try:
                    with output_path.open("w", encoding="utf-8") as fh:
                        overlap_info = None
                        if overlap_geom is not None and mapping is not None and not overlap_geom.is_empty:
                            try:
                                geom_mapping = mapping(overlap_geom)
                            except Exception:
                                geom_mapping = None
                            overlap_info = {
                                "crs": target_crs.to_string() if target_crs is not None else None,
                                "area": float(overlap_geom.area),
                                "bounds": list(overlap_geom.bounds),
                                "geometry": geom_mapping,
                            }
                        doc = {
                            "dataset_pair": meta.get("dataset_pair", [ds_a, ds_b]),
                            "alignment_roles": alignment_roles,
                            "dataset_resolutions": {
                                ds_a: float(dataset_resolutions.get(ds_a)) if ds_a in dataset_resolutions else None,
                                ds_b: float(dataset_resolutions.get(ds_b)) if ds_b in dataset_resolutions else None,
                            },
                            "approx_spacing": approx_spacing,
                            "search_radius": search_radius,
                            "raw_pair_candidates": raw_candidates,
                            "generated_from": generated_from or ("embedding_windows" if use_embedding else "tile_bounds"),
                            "overlap": overlap_info,
                            "pairs": filtered_records,
                        }
                        if spec.pair_label is not None:
                            doc["pair_variant"] = spec.pair_label
                        json.dump(doc, fh, indent=2)
                        outputs.setdefault(spec.key, {})[f"{ds_a}-{ds_b}{label_suffix}"] = str(output_path)
                except Exception as exc:
                    print(f"[warn] Failed to write overlap pairs for {ds_a}-{ds_b} ({spec.key}): {exc}")

    return outputs


def _build_union_feature_table(datasets: Sequence[DatasetSummary]) -> Dict[str, List[str]]:
    """Map each feature name to the dataset IDs that provide it."""
    table: Dict[str, List[str]] = {}
    for ds in datasets:
        for feature in ds.feature_names:
            table.setdefault(feature, []).append(ds.dataset_id)
    for feature, owners in table.items():
        owners.sort()
    return dict(sorted(table.items(), key=lambda kv: kv[0].lower()))


def _pair_label(record: Dict[str, object], index: int) -> Optional[int]:
    key = f"dataset_{index}_label"
    value = record.get(key)
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _pair_is_augmented(record: Dict[str, object], index: int) -> bool:
    key = f"dataset_{index}_is_augmented"
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if value in {1, "1", "true", "True"}:
        return True
    return False


def _filter_positive_pair(record: Dict[str, object]) -> bool:
    return _pair_label(record, 0) == 1 and _pair_label(record, 1) == 1


def _filter_positive_non_augmented(record: Dict[str, object]) -> bool:
    if not _filter_positive_pair(record):
        return False
    return (not _pair_is_augmented(record, 0)) and (not _pair_is_augmented(record, 1))


def _filter_positive_augmented(record: Dict[str, object]) -> bool:
    if not _filter_positive_pair(record):
        return False
    return _pair_is_augmented(record, 0) or _pair_is_augmented(record, 1)


def _extract_overlap_samples(doc: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Tuple[float, float]], str, str]:
    dataset_pair = doc.get("dataset_pair") or ["dataset_0", "dataset_1"]
    anchor_label = str(dataset_pair[0])
    target_label = str(dataset_pair[1])
    anchor_samples: List[Dict[str, object]] = []
    target_samples: List[Dict[str, object]] = []
    centroids: List[Tuple[float, float]] = []
    for entry in doc.get("pairs", []):
        for idx, container in ((0, anchor_samples), (1, target_samples)):
            coord = entry.get(f"dataset_{idx}_native_point")
            if coord is None:
                continue
            try:
                x, y = float(coord[0]), float(coord[1])
            except Exception:
                continue
            container.append(
                {
                    "coord": (x, y),
                    "dataset": anchor_label if idx == 0 else target_label,
                    "is_augmented": bool(entry.get(f"dataset_{idx}_is_augmented")),
                }
            )
        centroid = entry.get("overlap_centerloid")
        if centroid is not None:
            try:
                cx, cy = float(centroid[0]), float(centroid[1])
                centroids.append((cx, cy))
            except Exception:
                pass
    return anchor_samples, target_samples, centroids, anchor_label, target_label


def _render_debug_plots(
    path_map: Dict[str, str],
    boundary_geometry: Optional[Dict[str, object]],
    output_dir: Path,
    base_filename: str,
    title_prefix: str,
    *,
    include_centroids: bool = False,
) -> None:
    if not path_map:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(path_map.items())
    multi = len(entries) > 1
    for label, path_str in entries:
        path = Path(path_str)
        try:
            with path.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except Exception:
            continue
        anchor_samples, target_samples, centroids, anchor_label, target_label = _extract_overlap_samples(doc)
        if not anchor_samples and not target_samples:
            continue
        if multi:
            safe_label = label.replace("/", "_").replace(" ", "_")
            filename = f"{Path(base_filename).stem}_{safe_label}.png"
        else:
            filename = f"{Path(base_filename).stem}.png"
        output_path = output_dir / filename
        title = f"{title_prefix}: {anchor_label} vs {target_label}"
        save_overlap_debug_plot(
            output_path,
            boundary_geometry,
            anchor_samples,
            target_samples,
            title=title,
            anchor_label=anchor_label,
            target_label=target_label,
            centroid_points=centroids if include_centroids else None,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate multiple STAC collections into a unified metadata summary.")
    parser.add_argument("--collections", nargs="+", required=True, help="Paths to STAC collection roots produced by stacify_*.",)
    parser.add_argument("--output", type=str, default=None, help="Optional path to write combined metadata JSON. If omitted, prints to stdout.",)
    parser.add_argument("--dataset-ids", nargs="*", default=None, help="Optional friendly IDs corresponding to --collections (must match in length).",)
    parser.add_argument("--embedding-path", nargs="+", required=True, help=".npz embedding bundles aligned with --collections order for window-level overlap detection.",)
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Include positive augmentation bundles when computing overlap pairs.")
    parser.add_argument("--pos-aug-path", nargs="*", default=None, help="Optional positive augmentation .npz paths aligned with --collections order.")
    parser.add_argument("--projectname", type=str, default=None, help="Optional project name. When provided with --output, the combined metadata is written to <output>/<projectname>/combined_metadata.json"    )
    parser.add_argument("--bridge-guess-number", type=int, default=0, help="Optional count of heuristic feature bridges supplied via --bridge.",)
    parser.add_argument("--bridge", action="append", default=[], help=( "Feature bridge specification across datasets. Use braces with segments separated by ';', one segment per dataset, e.g. \"{featA1, featA2; featB1}\" for two datasets. Repeat the flag for multiple bridge guesses."), )
    parser.add_argument("--region-select", type=str, default=None,help=("Optional region selection string matching the number of datasets, e.g. \"{NA,AU; GLOBAL}\" to keep regions NA and AU from dataset 1 and region GLOBAL from dataset 2. Use ALL or * to retain every region for a dataset."),)
    parser.add_argument("--visualize", action="store_true", help="Generate preview PNG/TIF copies for bridge features/labels.",)
    parser.add_argument("--debug", action="store_true", dest="debug_overlap", help="Generate overlap debug plots for positive pairs.")
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
    if len(args.embedding_path) != len(root_paths):
        raise ValueError("--embedding-path must match the number of --collections")
    embedding_bundle_paths: List[Path] = [
        Path(p).expanduser().resolve() for p in args.embedding_path
    ]

    requested_aug_paths = list(args.pos_aug_path or [])
    use_positive_augmentation = bool(args.use_positive_augmentation or requested_aug_paths)
    if requested_aug_paths and len(requested_aug_paths) != len(root_paths):
        raise ValueError("--pos-aug-path must align with the number of --collections")

    output_path = args.output
    project_name = args.projectname

    if project_name and not output_path:
        raise ValueError("--projectname requires --output to specify the base directory.")

    output_file: Optional[Path] = None
    visual_base: Optional[Path] = None
    target_dir: Optional[Path] = None

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
                target_dir = resolved
                output_file = resolved / "combined_metadata.json"
                visual_base = resolved

    if output_file is None and args.visualize and visual_base is None:
        visual_base = Path.cwd() / "integrate_stac_output"

    integration_root = target_dir or visual_base or (Path.cwd() / "integrate_stac_output")
    integration_root.mkdir(parents=True, exist_ok=True)

    datasets: List[DatasetSummary] = []
    dataset_crs_map: Dict[str, Optional[CRS]] = {}
    embedding_windows_map: Dict[str, List[Dict[str, object]]] = {}
    embedding_path_map: Dict[str, Path] = {}
    augmentation_windows_map: Dict[str, List[Dict[str, object]]] = {}
    augmentation_stats: Dict[str, Dict[str, object]] = {}

    for idx, (root, ds_id) in enumerate(zip(root_paths, dataset_ids)):
        summary = _summarise_dataset(root, ds_id)
        datasets.append(summary)
        dataset_crs_map[summary.dataset_id] = None
        bundle_path = embedding_bundle_paths[idx]
        windows = _load_embedding_windows(bundle_path)
        base_window_count = len(windows)
        if windows:
            embedding_windows_map[summary.dataset_id] = windows
        else:
            print(f"[warn] Embedding bundle {bundle_path} produced no window entries for dataset {summary.dataset_id}")
        embedding_path_map[summary.dataset_id] = bundle_path

        if use_positive_augmentation:
            raw_aug_entry = requested_aug_paths[idx] if idx < len(requested_aug_paths) else None
            resolved_aug_path: Optional[Path] = None
            aug_records: List[Dict[str, object]] = []
            if raw_aug_entry:
                token = str(raw_aug_entry).strip()
                if token and token.lower() not in {"none", "null"}:
                    candidate = Path(token).expanduser()
                    if not candidate.is_absolute():
                        candidate = _canonical_path(root, token)
                    try:
                        resolved_aug_path = candidate.resolve()
                    except Exception:
                        resolved_aug_path = candidate
            else:
                resolved_aug_path = (bundle_path.parent / "positive_augmentation.npz").resolve()

            if resolved_aug_path is not None:
                aug_records = _load_augmented_embedding_windows(resolved_aug_path)
                if aug_records:
                    augmentation_windows_map[summary.dataset_id] = aug_records
                    print(
                        f"[info] Positive augmentation: dataset {summary.dataset_id} -> {len(aug_records)} augmented windows ({resolved_aug_path})"
                    )
                elif resolved_aug_path.exists():
                    print(
                        f"[info] Positive augmentation bundle {resolved_aug_path} contains no augmented records for dataset {summary.dataset_id}"
                    )
            else:
                print(
                    f"[warn] Positive augmentation requested but no bundle path configured for dataset {summary.dataset_id}; skipping."
                )

            augmentation_stats[summary.dataset_id] = {
                "bundle_path": str(resolved_aug_path) if resolved_aug_path is not None else None,
                "base_windows": base_window_count,
                "augmented_windows": len(aug_records),
                "total_windows_with_augmented": base_window_count + len(aug_records),
            }

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

        for summary in datasets:
            dataset_crs = dataset_crs_map.get(summary.dataset_id)
            if dataset_crs is not None:
                _ensure_unified_crs_boundaries(
                    summary,
                    dataset_crs,
                    target_crs,
                    integration_root=integration_root,
                )

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
            tiles = _collect_dataset_tiles_v0(ds, target_crs)
            if tiles:
                tile_geoms_index[ds.dataset_id] = tiles

    dataset_resolution_map = _compute_dataset_resolutions(tile_geoms_index) if tile_geoms_index else {}
    embedding_spacing_map = _compute_embedding_spacing_map(embedding_windows_map) if embedding_windows_map else {}
    embedding_windows_with_aug: Optional[Dict[str, List[Dict[str, object]]]] = None
    embedding_spacing_with_aug: Optional[Dict[str, float]] = None
    if use_positive_augmentation and augmentation_windows_map:
        embedding_windows_with_aug = {
            key: list(records) for key, records in embedding_windows_map.items()
        }
        for dataset_id, aug_records in augmentation_windows_map.items():
            combined = embedding_windows_with_aug.setdefault(dataset_id, [])
            combined.extend(aug_records)
        embedding_spacing_with_aug = (
            _compute_embedding_spacing_map(embedding_windows_with_aug) if embedding_windows_with_aug else {}
        )

    label_union = sorted({label for ds in datasets for label in ds.label_names})
    region_union = sorted({region for ds in datasets for region in ds.region_keys})

    payload = {
        "dataset_count": len(datasets),
        "native_crs": {
            ds.dataset_id: (dataset_crs_map.get(ds.dataset_id).to_string() if dataset_crs_map.get(ds.dataset_id) is not None and hasattr(dataset_crs_map.get(ds.dataset_id), "to_string") else str(dataset_crs_map.get(ds.dataset_id)))
            for ds in datasets
        },
        "unified_crs": "EPSG:8857",
        "region_union": region_union,
        "dataset_min_resolution": {key: float(value) for key, value in dataset_resolution_map.items()},
        "dataset_window_spacing": {
            key: float(value)
            for key, value in embedding_spacing_map.items()
            if value is not None
        },
        "bridge_guess_number": int(args.bridge_guess_number or len(bridge_mappings)),
        "bridge_guesses": bridge_mappings,
        "datasets": [
            {
                "dataset_id": ds.dataset_id,
                "collection_id": ds.collection_id,
                "root": str(ds.root),
                "embedding_path": str(embedding_path_map.get(ds.dataset_id)) if ds.dataset_id in embedding_path_map else None,
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

    if use_positive_augmentation:
        payload["positive_augmentation"] = {
            "enabled": True,
            "datasets": {
                dataset_id: {
                    "bundle_path": stats.get("bundle_path"),
                    "base_windows": int(stats.get("base_windows", 0)),
                    "augmented_windows": int(stats.get("augmented_windows", 0)),
                    "total_windows_with_augmented": int(stats.get("total_windows_with_augmented", 0)),
                }
                for dataset_id, stats in augmentation_stats.items()
            },
        }
    else:
        payload["positive_augmentation"] = {"enabled": False}

    overlap_boundary_path: Optional[Path] = None
    if overlap_geom_equal is not None and shapely_shape is not None and mapping is not None and not overlap_geom_equal.is_empty:
        payload["study_area_overlap_detected"] = True

        overlap_dir = integration_root / "data" / "Boundary_UnifiedCRS" / "Overlap"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        overlap_geojson = overlap_dir / "study_area_overlap.geojson"
        overlap_tif = overlap_dir / "study_area_overlap.tif"

        _write_geojson_feature(overlap_geojson, overlap_geom_equal, Path("overlap"))

        span_x = overlap_geom_equal.bounds[2] - overlap_geom_equal.bounds[0]
        span_y = overlap_geom_equal.bounds[3] - overlap_geom_equal.bounds[1]
        resolution = max(span_x, span_y) / 4096.0
        if not math.isfinite(resolution) or resolution <= 0:
            resolution = 1000.0
        width = max(1, int(math.ceil(span_x / resolution)))
        height = max(1, int(math.ceil(span_y / resolution)))
        transform = Affine(resolution, 0.0, overlap_geom_equal.bounds[0], 0.0, -resolution, overlap_geom_equal.bounds[3])
        _rasterize_geometry_to_path(
            overlap_geom_equal,
            overlap_tif,
            transform=transform,
            width=width,
            height=height,
            target_crs=target_crs,
            source_path=overlap_geojson,
        )

        payload["study_area_overlap_boundary"] = {
            "geojson_path": str(overlap_geojson),
            "tif_path": str(overlap_tif),
        }
    else:
        payload["study_area_overlap_detected"] = False
        payload["study_area_overlap_boundary"] = None

    overlap_mask_path: Optional[Path] = None
    if payload.get("study_area_overlap_boundary"):
        overlap_entry = payload["study_area_overlap_boundary"]
        overlap_mask_path = Path(overlap_entry.get("tif_path")) if isinstance(overlap_entry, dict) else None
        payload["study_area_overlap_mask"] = overlap_entry.get("tif_path") if isinstance(overlap_entry, dict) else None
    else:
        payload["study_area_overlap_mask"] = None

    pairs_dir = integration_root / "data" / "Pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pair_output_root = pairs_dir
    overlap_pairs_outputs: Dict[str, str] = {}
    overlap_pairs_positive: Dict[str, str] = {}
    if tile_geoms_index or embedding_windows_map:
        base_variants = [
            _PairVariantSpec(key="default", suffix="", pair_label=None, filter_fn=None),
            _PairVariantSpec(
                key="positive",
                suffix="_positive",
                pair_label="positive_only",
                filter_fn=_filter_positive_non_augmented,
            ),
        ]
        base_outputs = _write_overlap_pairs(
            tile_geoms_index,
            overlap_geom_equal,
            pair_output_root,
            target_crs,
            dataset_resolution_map,
            embedding_windows_map,
            embedding_spacing_map,
            variants=base_variants,
            dataset_order=ordered_dataset_ids,
        )
        overlap_pairs_outputs = base_outputs.get("default", {})
        overlap_pairs_positive = base_outputs.get("positive", {})
        if overlap_pairs_outputs:
            for pair_label, pair_path in sorted(overlap_pairs_outputs.items()):
                print(f"[info] Wrote overlap pair summary for {pair_label} to {pair_path}")
        if overlap_pairs_positive:
            for pair_label, pair_path in sorted(overlap_pairs_positive.items()):
                print(f"[info] Wrote positive-overlap summary for {pair_label} to {pair_path}")
    payload["study_area_overlap_pairs"] = overlap_pairs_outputs
    payload["study_area_overlap_pairs_positive"] = overlap_pairs_positive if overlap_pairs_positive else None

    overlap_pairs_positive_aug: Dict[str, str] = {}
    if use_positive_augmentation and embedding_windows_with_aug:
        aug_variants = [
            _PairVariantSpec(
                key="positive_aug",
                suffix="_positive_aug",
                pair_label="positive_augmentation",
                filter_fn=_filter_positive_augmented,
            )
        ]
        augmented_outputs = _write_overlap_pairs(
            tile_geoms_index,
            overlap_geom_equal,
            pair_output_root,
            target_crs,
            dataset_resolution_map,
            embedding_windows_with_aug,
            embedding_spacing_with_aug,
            variants=aug_variants,
            dataset_order=ordered_dataset_ids,
        )
        overlap_pairs_positive_aug = augmented_outputs.get("positive_aug", {})
        if overlap_pairs_positive_aug:
            for pair_label, pair_path in sorted(overlap_pairs_positive_aug.items()):
                print(f"[info] Wrote positive-aug overlap summary for {pair_label} to {pair_path}")
    payload["study_area_overlap_pairs_positive_aug"] = overlap_pairs_positive_aug if overlap_pairs_positive_aug else None
    payload["study_area_overlap_pairs_augmented"] = payload["study_area_overlap_pairs_positive_aug"]

    if getattr(args, "debug_overlap", False):
        boundary_geometry = None
        if overlap_geom_equal is not None and mapping is not None and not overlap_geom_equal.is_empty:
            try:
                boundary_geometry = mapping(overlap_geom_equal)
            except Exception:
                boundary_geometry = None
        debug_output_root = visual_base if visual_base is not None else Path.cwd()
        debug_dir = debug_output_root / "overlap_debug"
        _render_debug_plots(
            overlap_pairs_positive,
            boundary_geometry,
            debug_dir,
            "debug_positive_overlap.png",
            "Positive overlap",
        )
        _render_debug_plots(
            overlap_pairs_positive_aug,
            boundary_geometry,
            debug_dir,
            "debug_positive_aug_overlap.png",
            "Positive overlap (augmented)",
            include_centroids=True,
        )

    if args.visualize:
        if visual_base is None:
            visual_base = Path.cwd() / "integrate_stac_output"
        generate_bridge_visualizations = _load_bridge_visualizer()
        bridge_output_dir = visual_base / "bridge_visualizations"
        generate_bridge_visualizations(
            bridge_mappings,
            dataset_lookup,
            bridge_output_dir,
            label_geojson_map,
            overlap_geoms_dataset if overlap_geoms_dataset else None,
            dataset_crs_map,
            target_crs,
            overlap_geom_equal,
            overlap_mask_path,
        )
        print(f"[info] Wrote bridge visualizations to {bridge_output_dir}")

    if output_file is not None:
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[info] Wrote integrated metadata to {output_file}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
