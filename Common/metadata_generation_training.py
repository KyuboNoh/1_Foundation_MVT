"""Utilities for summarising generated raster assets for training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


def _rel_or_abs(base: Path, target) -> str:
    path_obj = target if isinstance(target, Path) else Path(str(target))
    try:
        return str(path_obj.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path_obj.resolve())
    except Exception:
        return str(path_obj)


def generate_training_metadata(
    collection_root: Path,
    raster_products: Sequence[Mapping],
    *,
    output_filename: str = "training_metadata.json",
    debug: bool = False,
) -> Path:
    """Persist a JSON summary of raster assets grouped by region and feature."""

    collection_root = collection_root.resolve()
    summary: MutableMapping[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "debug": bool(debug),
        "features": {"total_features": 0, "entries": {}},
        "labels": {"total_labels": 0, "entries": {}},
    }

    bucket_lookup: MutableMapping[str, MutableMapping[str, object]]
    bucket_lookup = {  # type: ignore[assignment]
        "features": summary["features"],
        "labels": summary["labels"],
    }

    has_entries = False

    for product in raster_products:
        bucket_key = "labels" if product.get("is_label") else "features"
        bucket_map = bucket_lookup[bucket_key]["entries"]  # type: ignore[index]

        feature_name = str(product.get("feature") or product.get("item_id") or "feature")
        feature_entry = bucket_map.setdefault(feature_name, {"num_tifs": 0, "tifs": []})
        kind = product.get("kind")
        if kind and "kind" not in feature_entry:
            feature_entry["kind"] = str(kind)
        categories = product.get("categories")
        if categories and "categories" not in feature_entry:
            feature_entry["categories"] = list(categories)

        tif_info: MutableMapping[str, object] = {}
        region = product.get("region")
        if region:
            tif_info["region"] = str(region)

        asset_path = product.get("asset_path")
        if asset_path:
            asset_path = Path(asset_path)
            tif_info["path"] = _rel_or_abs(collection_root, asset_path)
            tif_info["filename"] = asset_path.name
        metadata_path = product.get("metadata_path")
        if metadata_path:
            metadata_path = Path(metadata_path)
            tif_info["item_json"] = _rel_or_abs(collection_root, metadata_path)
        quicklook_path = product.get("quicklook_path")
        if quicklook_path:
            quicklook_path = Path(quicklook_path)
            tif_info["quicklook_png"] = _rel_or_abs(collection_root, quicklook_path)
        band_display = product.get("band_display")
        if band_display:
            tif_info["bands"] = list(band_display)
        item_obj = product.get("item")
        if getattr(item_obj, "id", None):
            tif_info["stac_item_id"] = str(item_obj.id)

        if not tif_info:
            tif_info["note"] = "No asset references were recorded."

        feature_entry.setdefault("tifs", []).append(dict(tif_info))
        feature_entry["num_tifs"] = len(feature_entry["tifs"])  # keep count in sync
        has_entries = True

    if not has_entries:
        summary["note"] = (
            "Debug mode - raster generation skipped."
            if debug
            else "No raster products were generated."
        )
    else:
        summary["features"]["total_features"] = len(summary["features"]["entries"])  # type: ignore[index]
        summary["labels"]["total_labels"] = len(summary["labels"]["entries"])  # type: ignore[index]

    output_path = collection_root / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path
