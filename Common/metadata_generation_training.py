"""Utilities for summarising generated raster assets for training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def _rel_or_abs(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target.resolve())
    except Exception:
        return str(target)


def _canonical_path(base: Path, value: Any) -> Path:
    path_obj = value if isinstance(value, Path) else Path(str(value))
    if not path_obj.is_absolute():
        path_obj = (base / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()
    return path_obj


def _dedup_records(existing: Iterable[Mapping[str, Any]], base: Path) -> set[Path]:
    records: set[Path] = set()
    for record in existing:
        rel = record.get("path") or record.get("filename")
        if not rel:
            continue
        try:
            records.add(_canonical_path(base, rel))
        except Exception:
            continue
    return records


def _ensure_section(summary: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    section = summary.get(key)
    if not isinstance(section, dict):
        section = {"entries": {}, "total": 0}
        summary[key] = section
    section.setdefault("entries", {})
    return section  # type: ignore[return-value]


def generate_training_metadata(
    collection_root: Path,
    raster_products: Sequence[Mapping],
    *,
    output_filename: str = "training_metadata.json",
    debug: bool = False,
) -> Path:
    """Persist a JSON summary of raster assets grouped by region and feature."""

    collection_root = collection_root.resolve()
    metadata_path = collection_root / output_filename

    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except Exception:
        existing = {}

    if not isinstance(existing, dict):
        existing = {}

    summary: MutableMapping[str, Any] = existing
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["debug"] = bool(debug)

    features_section = _ensure_section(summary, "features")
    labels_section = _ensure_section(summary, "labels")
    summary.setdefault(
        "pretraining",
        {
            "args": None,
            "output_dir": None,
            "updated_at": None,
        },
    )

    bucket_lookup = {
        "features": features_section,
        "labels": labels_section,
    }

    base_dir = metadata_path.parent if metadata_path.parent.exists() else collection_root
    any_updates = False

    for product in raster_products:
        bucket_key = "labels" if product.get("is_label") else "features"
        section = bucket_lookup[bucket_key]
        entries: MutableMapping[str, Any] = section["entries"]  # type: ignore[assignment]

        feature_name = str(product.get("feature") or product.get("item_id") or "feature")
        feature_entry = entries.setdefault(feature_name, {})
        feature_entry.setdefault("tifs", [])
        existing_paths = _dedup_records(feature_entry["tifs"], base_dir)  # type: ignore[arg-type]

        kind = product.get("kind")
        if kind and "kind" not in feature_entry:
            feature_entry["kind"] = str(kind)
        categories = product.get("categories")
        if categories and "categories" not in feature_entry:
            feature_entry["categories"] = list(categories)

        tif_info: MutableMapping[str, Any] = {}
        region = product.get("region")
        if region:
            tif_info["region"] = str(region)

        asset_path = product.get("asset_path")
        if asset_path:
            asset_path = _canonical_path(base_dir, asset_path)
            tif_info["path"] = _rel_or_abs(base_dir, asset_path)
            tif_info["filename"] = asset_path.name
        metadata_path_asset = product.get("metadata_path")
        if metadata_path_asset:
            metadata_path_asset = _canonical_path(base_dir, metadata_path_asset)
            tif_info["item_json"] = _rel_or_abs(base_dir, metadata_path_asset)
        quicklook_path = product.get("quicklook_path")
        if quicklook_path:
            quicklook_path = _canonical_path(base_dir, quicklook_path)
            tif_info["quicklook_png"] = _rel_or_abs(base_dir, quicklook_path)
        band_display = product.get("band_display")
        if band_display:
            tif_info["bands"] = list(band_display)
        item_obj = product.get("item")
        if getattr(item_obj, "id", None):
            tif_info["stac_item_id"] = str(item_obj.id)
        valid_pixels = product.get("valid_pixel_count")
        total_pixels = product.get("total_pixel_count")
        if valid_pixels is not None:
            tif_info["valid_pixels"] = int(valid_pixels)
        if total_pixels is not None:
            tif_info["total_pixels"] = int(total_pixels)
        if valid_pixels is not None and total_pixels:
            try:
                fraction = float(valid_pixels) / float(total_pixels) if total_pixels else None
            except Exception:
                fraction = None
            if fraction is not None:
                tif_info["valid_fraction"] = fraction

        abs_path = None
        rel_path = tif_info.get("path")
        if rel_path:
            try:
                abs_path = _canonical_path(base_dir, rel_path)
            except Exception:
                abs_path = None

        if abs_path is None or abs_path not in existing_paths:
            feature_entry["tifs"].append(dict(tif_info))  # type: ignore[index]
            any_updates = True

        feature_entry["num_tifs"] = len(feature_entry["tifs"])  # type: ignore[index]

    features_entries = features_section["entries"]  # type: ignore[assignment]
    labels_entries = labels_section["entries"]  # type: ignore[assignment]
    features_section["total_features"] = len(features_entries)
    labels_section["total_labels"] = len(labels_entries)

    if not any_updates and not raster_products:
        summary.setdefault(
            "note",
            "Debug mode - raster generation skipped." if debug else "No raster products were generated.",
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metadata_path
