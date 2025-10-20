#!/usr/bin/env python3
"""
Utilities to combine metadata from multiple STAC collections into a unified
feature schema. This is the first step for the Universal GFM4MPM pipeline
which will ingest different regional STAC exports (e.g., CAN/US/AU, BC) and
produce a compatible superset of channels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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

    return DatasetSummary(
        dataset_id=dataset_name,
        root=root,
        collection_id=collection_id,
        features={str(k): v for k, v in feature_entries.items()},
        labels={str(k): v for k, v in label_entries.items()},
        boundaries=boundaries_section,
        regions=region_map,
    )


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
    parser = argparse.ArgumentParser(
        description="Integrate multiple STAC collections into a unified metadata summary."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        required=True,
        help="Paths to STAC collection roots produced by stacify_*.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write combined metadata JSON. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="*",
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
    for root, ds_id in zip(root_paths, dataset_ids):
        summary = _summarise_dataset(root, ds_id)
        datasets.append(summary)

    ordered_dataset_ids = [ds.dataset_id for ds in datasets]
    bridge_mappings: List[Dict[str, List[str]]] = []
    for raw_bridge in args.bridge:
        try:
            mapping = _parse_bridge_entry(raw_bridge, ordered_dataset_ids)
        except ValueError as exc:
            raise SystemExit(f"Error parsing --bridge entry '{raw_bridge}': {exc}") from exc
        bridge_mappings.append(mapping)

    label_union = sorted({label for ds in datasets for label in ds.label_names})
    region_union = sorted({region for ds in datasets for region in ds.region_keys})

    payload = {
        "dataset_count": len(datasets),
        "region_union": region_union,
        "bridge_guess_number": int(args.bridge_guess_number),
        "bridge_guesses": bridge_mappings,
        "datasets": [
            {
                "dataset_id": ds.dataset_id,
                "collection_id": ds.collection_id,
                "root": str(ds.root),
                "feature_names": ds.feature_names,
                "label_names": ds.label_names,
                "is_multi_region": ds.is_multi_region,
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

    output_path = args.output
    project_name = args.projectname

    if project_name and not output_path:
        raise ValueError("--projectname requires --output to specify the base directory.")

    if output_path:
        base_path = Path(output_path).expanduser()
        if project_name:
            target_dir = (base_path / project_name).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / "combined_metadata.json"
        else:
            output_file = base_path.resolve()
            if output_file.suffix:
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file.mkdir(parents=True, exist_ok=True)
                output_file = output_file / "combined_metadata.json"

        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[info] Wrote integrated metadata to {output_file}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
