"""Utility helpers for preparing classifier labels across two overlapping datasets."""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np

from .overlaps import OverlapSet
from Common.data_utils import EmbeddingRecord
from .fusion_utils.workspace import OverlapAlignmentWorkspace

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]

try:  # optional raster IO
    import rasterio
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore[assignment]

def _normalise_cross_matches(
    matches: Sequence[Dict[str, object]],
    anchor_name: str,
    target_name: str,
) -> List[Dict[str, object]]:
    normalised: List[Dict[str, object]] = []
    for entry in matches:
        anchor_ds = entry.get("anchor_dataset")
        target_ds = entry.get("target_dataset")
        if anchor_ds == anchor_name and target_ds == target_name:
            normalised.append(entry)
            continue
        swapped = copy.deepcopy(entry)
        if anchor_ds == target_name and target_ds == anchor_name:
            swapped["anchor_dataset"], swapped["target_dataset"] = anchor_name, target_name
            swapped["anchor"], swapped["target"] = copy.deepcopy(entry.get("target")), copy.deepcopy(entry.get("anchor"))
            swapped["label_in_dataset_1"], swapped["label_in_dataset_2"] = (
                entry.get("label_in_dataset_2"),
                entry.get("label_in_dataset_1"),
            )
        else:
            swapped["anchor_dataset"] = anchor_name
            swapped["target_dataset"] = target_name
        normalised.append(swapped)
    return normalised


def _prepare_classifier_labels(
    *,
    meta_anchor_all: Optional[Dict[str, object]],
    meta_anchor_overlap: Optional[Dict[str, object]],
    meta_target_all: Optional[Dict[str, object]],
    meta_target_overlap: Optional[Dict[str, object]],
    overlapregion_label: Optional[str],
    label_cross_matches: Sequence[Dict[str, object]],
    sample_sets: Dict[str, Dict[str, object]],
    pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]],
    anchor_name: str,
    target_name: str,
    debug: bool,
    run_logger: "_RunLogger",
    label_matcher: Optional["OverlapAlignmentLabels"] = None,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    def _normalise_label(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        return text

    def _build_lookup(entry: Optional[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
        if not entry:
            return {}
        indices = entry.get("indices") or []
        metadata = entry.get("metadata") or []
        coords = entry.get("coords") or []
        row_cols_list = entry.get("row_cols") or entry.get("row_cols_mask") or []
        lookup: Dict[int, Dict[str, object]] = {}
        bound = max(len(indices), len(metadata), len(coords), len(row_cols_list))
        for pos in range(bound):
            meta_val = metadata[pos] if pos < len(metadata) else {}
            dataset_idx_raw: Optional[object] = None
            if isinstance(meta_val, dict):
                dataset_idx_raw = (
                    meta_val.get("index")
                    or meta_val.get("embedding_index")
                    or meta_val.get("dataset_index")
                    or meta_val.get("record_index")
                )
            if dataset_idx_raw is None and pos < len(indices):
                dataset_idx_raw = indices[pos]
            if dataset_idx_raw is None:
                continue
            try:
                dataset_idx = int(dataset_idx_raw)
            except Exception:
                continue
            coord_val = coords[pos] if pos < len(coords) else None
            if coord_val is None and isinstance(meta_val, dict):
                coord_val = meta_val.get("coord")
            row_col_val = None
            if isinstance(meta_val, dict):
                row_col_val = (
                    meta_val.get("row_col_mask")
                    or meta_val.get("row_col")
                    or meta_val.get("rowcol")
                )
            if row_col_val is None and pos < len(row_cols_list):
                row_col_val = row_cols_list[pos]
            if isinstance(row_col_val, (list, tuple)) and len(row_col_val) >= 2:
                try:
                    row_col_val = (int(row_col_val[0]), int(row_col_val[1]))
                except Exception:
                    row_col_val = None
            elif row_col_val is not None:
                row_col_val = None
            lookup[dataset_idx] = {
                "position": pos,
                "coord": coord_val,
                "metadata": meta_val if isinstance(meta_val, dict) else {},
                "row_col": row_col_val,
            }
        return lookup

    def _extract_sample_map(dataset_name: str) -> Dict[int, Dict[str, object]]:
        sample_entry = sample_sets.get(dataset_name)
        label_map: Dict[int, Dict[str, object]] = {}
        if not sample_entry:
            pn_lookup = pn_label_maps.get(dataset_name)
            if pn_lookup:
                run_logger.log(
                    f"[cls] Sample set unavailable for {dataset_name}; PN labels may be unusable for classifier indices."
                )
            else:
                run_logger.log(f"[cls] PN labels missing for dataset {dataset_name}; produced empty label set.")
            return label_map
        indices = sample_entry.get("indices") or []
        labels_raw = sample_entry.get("labels")
        coords = sample_entry.get("coords") or []
        metadata = sample_entry.get("metadata") or []
        for pos, raw_idx in enumerate(indices):
            try:
                dataset_idx = int(raw_idx)
            except Exception:
                continue
            label_val = None
            if labels_raw is not None:
                try:
                    val = labels_raw[pos]
                    if hasattr(val, "item"):
                        val = val.item()
                    label_val = 1 if int(val) > 0 else 0
                except Exception:
                    label_val = None
            coord_val = coords[pos] if pos < len(coords) else None
            meta_val = metadata[pos] if pos < len(metadata) else {}
            label_map[dataset_idx] = {
                "label": label_val,
                "coord": coord_val,
                "metadata": meta_val,
                "source_index": dataset_idx,
            }
        return label_map

    def _build_cross_maps(
        entries: Sequence[Dict[str, object]],
        anchor_dataset: str,
        target_dataset: str,
    ) -> Tuple[
        Dict[int, Dict[str, object]],
        Dict[int, Dict[str, object]],
        Dict[Tuple[int, int], Dict[str, object]],
        Dict[Tuple[int, int], Dict[str, object]],
    ]:
        anchor_to_target_idx: Dict[int, Dict[str, object]] = {}
        target_to_anchor_idx: Dict[int, Dict[str, object]] = {}
        anchor_to_target_rowcol: Dict[Tuple[int, int], Dict[str, object]] = {}
        target_to_anchor_rowcol: Dict[Tuple[int, int], Dict[str, object]] = {}

        def _extract_row_col(payload: Dict[str, object]) -> Optional[Tuple[int, int]]:
            raw = payload.get("row_col")
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                try:
                    return (int(raw[0]), int(raw[1]))
                except Exception:
                    return None
            return None

        for entry in entries:
            entry_anchor = entry.get("anchor_dataset")
            entry_target = entry.get("target_dataset")
            anchor_payload = entry.get("anchor") or {}
            target_payload = entry.get("target") or {}

            if entry_anchor != anchor_dataset or entry_target != target_dataset:
                if entry_anchor == target_dataset and entry_target == anchor_dataset:
                    anchor_payload, target_payload = target_payload, anchor_payload
                    entry_anchor, entry_target = entry_target, entry_anchor
                else:
                    continue

            anchor_idx_raw = anchor_payload.get("index")
            target_idx_raw = target_payload.get("index")
            target_label = target_payload.get("pn_label")
            anchor_label = anchor_payload.get("pn_label")
            anchor_rowcol = _extract_row_col(anchor_payload)
            target_rowcol = _extract_row_col(target_payload)

            if target_label is not None:
                lbl = 1 if int(target_label) > 0 else 0
                if anchor_idx_raw is not None:
                    try:
                        idx = int(anchor_idx_raw)
                        anchor_to_target_idx[idx] = {
                            "label": lbl,
                            "coord": target_payload.get("coord"),
                            "metadata": target_payload,
                            "source_index": int(target_idx_raw) if target_idx_raw is not None else None,
                            "distance": entry.get("distance"),
                            "row_col": anchor_rowcol,
                        }
                    except Exception:
                        pass
                if anchor_rowcol is not None:
                    anchor_to_target_rowcol[anchor_rowcol] = {
                        "label": lbl,
                        "coord": target_payload.get("coord"),
                        "metadata": target_payload,
                        "source_index": int(target_idx_raw) if target_idx_raw is not None else None,
                        "distance": entry.get("distance"),
                    }

            if anchor_label is not None:
                lbl = 1 if int(anchor_label) > 0 else 0
                if target_idx_raw is not None:
                    try:
                        idx = int(target_idx_raw)
                        target_to_anchor_idx[idx] = {
                            "label": lbl,
                            "coord": anchor_payload.get("coord"),
                            "metadata": anchor_payload,
                            "source_index": int(anchor_idx_raw) if anchor_idx_raw is not None else None,
                            "distance": entry.get("distance"),
                            "row_col": target_rowcol,
                        }
                    except Exception:
                        pass
                if target_rowcol is not None:
                    target_to_anchor_rowcol[target_rowcol] = {
                        "label": lbl,
                        "coord": anchor_payload.get("coord"),
                        "metadata": anchor_payload,
                        "source_index": int(anchor_idx_raw) if anchor_idx_raw is not None else None,
                        "distance": entry.get("distance"),
                    }

        return anchor_to_target_idx, target_to_anchor_idx, anchor_to_target_rowcol, target_to_anchor_rowcol

    def _assemble_result(
        dataset_label: str,
        lookup: Dict[int, Dict[str, object]],
        label_map: Dict[int, Dict[str, object]],
        source: str,
        *,
        cross_idx_map: Optional[Dict[int, Dict[str, object]]] = None,
        cross_rowcol_map: Optional[Dict[Tuple[int, int], Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        positions: List[int] = []
        dataset_indices: List[int] = []
        labels: List[int] = []
        coords: List[Optional[Tuple[float, float]]] = []
        meta_list: List[Dict[str, object]] = []
        source_indices: List[Optional[int]] = []
        cross_labels: List[Optional[int]] = []
        cross_coords: List[Optional[Tuple[float, float]]] = []
        cross_metadata: List[Optional[Dict[str, object]]] = []
        cross_source_indices: List[Optional[int]] = []
        cross_distances: List[Optional[float]] = []

        for dataset_idx, label_payload in label_map.items():
            if label_payload.get("label") is None:
                continue
            info = lookup.get(dataset_idx)
            if info is None:
                continue
            positions.append(int(info["position"]))
            dataset_indices.append(int(dataset_idx))
            labels.append(int(label_payload["label"]))
            coord_val = info.get("coord")
            if coord_val is None:
                coord_val = label_payload.get("coord")
            coords.append(coord_val)
            meta_list.append(info.get("metadata") or {})
            source_indices.append(label_payload.get("source_index"))

            cross_entry: Optional[Dict[str, object]] = None
            if cross_idx_map is not None:
                cross_entry = cross_idx_map.get(dataset_idx)
            if cross_entry is None and cross_rowcol_map is not None:
                row_col_val = info.get("row_col")
                if isinstance(row_col_val, tuple):
                    cross_entry = cross_rowcol_map.get(row_col_val)

            if cross_entry is not None:
                cross_labels.append(int(cross_entry.get("label", 0)))
                cross_coords.append(cross_entry.get("coord"))
                cross_metadata.append(cross_entry.get("metadata"))
                cross_source_indices.append(cross_entry.get("source_index"))
                cross_distances.append(cross_entry.get("distance"))
            else:
                cross_labels.append(None)
                cross_coords.append(None)
                cross_metadata.append(None)
                cross_source_indices.append(None)
                cross_distances.append(None)

        return {
            "dataset": dataset_label,
            "positions": positions,
            "dataset_indices": dataset_indices,
            "labels": labels,
            "coords": coords,
            "source": source,
            "source_indices": source_indices,
            "metadata": meta_list,
            "cross_labels": cross_labels,
            "cross_coords": cross_coords,
            "cross_metadata": cross_metadata,
            "cross_source_indices": cross_source_indices,
            "cross_distances": cross_distances,
        }

    def _filter_label_map(label_map: Dict[int, Dict[str, object]], lookup: Dict[int, Dict[str, object]]) -> Dict[int, Dict[str, object]]:
        if not label_map:
            return {}
        filtered: Dict[int, Dict[str, object]] = {}
        for idx in lookup.keys():
            entry = label_map.get(idx)
            if entry is not None and entry.get("label") is not None:
                filtered[int(idx)] = entry
        return filtered

    def _build_cross_label_map(
        lookup: Dict[int, Dict[str, object]],
        cross_idx_map: Dict[int, Dict[str, object]],
        cross_rowcol_map: Dict[Tuple[int, int], Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        label_map: Dict[int, Dict[str, object]] = {}
        for idx, info in lookup.items():
            payload = cross_idx_map.get(idx)
            if payload is None and cross_rowcol_map:
                row_col_val = info.get("row_col")
                if isinstance(row_col_val, (list, tuple)) and len(row_col_val) >= 2:
                    try:
                        key = (int(row_col_val[0]), int(row_col_val[1]))
                        payload = cross_rowcol_map.get(key)
                    except Exception:
                        payload = None
            if payload is None or payload.get("label") is None:
                continue
            label_map[int(idx)] = {
                "label": int(payload["label"]),
                "coord": payload.get("coord"),
                "metadata": payload.get("metadata") or {},
                "source_index": payload.get("source_index"),
            }
        return label_map

    def _empty_payload(dataset_label: str, source: str) -> Dict[str, object]:
        return {
            "dataset": dataset_label,
            "positions": [],
            "dataset_indices": [],
            "labels": [],
            "coords": [],
            "source": source,
            "source_indices": [],
            "metadata": [],
        }

    anchor_all_lookup = _build_lookup(meta_anchor_all)
    anchor_overlap_lookup = _build_lookup(meta_anchor_overlap)
    target_all_lookup = _build_lookup(meta_target_all)
    target_overlap_lookup = _build_lookup(meta_target_overlap)

    overlap_exists = bool(anchor_overlap_lookup or target_overlap_lookup)
    overlap_label = _normalise_label(overlapregion_label)
    if overlap_exists:
        if overlap_label is None:
            raise ValueError(
                "Overlap region detected but cls_training.overlapregion_label is not specified. "
                "Set overlapregion_label to the dataset name supplying PN labels for overlap regions."
            )
        if overlap_label not in {anchor_name, target_name}:
            raise ValueError(
                f"overlapregion_label must match either '{anchor_name}' or '{target_name}', got '{overlap_label}'."
            )

    anchor_to_target_map: Dict[int, Dict[str, object]] = {}
    target_to_anchor_map: Dict[int, Dict[str, object]] = {}
    anchor_to_target_rowcol_map: Dict[Tuple[int, int], Dict[str, object]] = {}
    target_to_anchor_rowcol_map: Dict[Tuple[int, int], Dict[str, object]] = {}

    if overlap_exists:
        (
            anchor_to_target_map,
            target_to_anchor_map,
            anchor_to_target_rowcol_map,
            target_to_anchor_rowcol_map,
        ) = _build_cross_maps(label_cross_matches, anchor_name, target_name)
        if label_matcher is not None:
            extra_matches = label_matcher.build_label_matches()
            (
                extra_anchor_to_target_map,
                extra_target_to_anchor_map,
                extra_anchor_to_target_rowcol,
                extra_target_to_anchor_rowcol,
            ) = _build_cross_maps(extra_matches, anchor_name, target_name)
            for key, value in extra_anchor_to_target_map.items():
                anchor_to_target_map.setdefault(key, value)
            for key, value in extra_target_to_anchor_map.items():
                target_to_anchor_map.setdefault(key, value)
            for key, value in extra_anchor_to_target_rowcol.items():
                anchor_to_target_rowcol_map.setdefault(key, value)
            for key, value in extra_target_to_anchor_rowcol.items():
                target_to_anchor_rowcol_map.setdefault(key, value)

    anchor_sample_map_full = _extract_sample_map(anchor_name)
    target_sample_map_full = _extract_sample_map(target_name)

    anchor_native_all = anchor_sample_map_full
    target_native_all = target_sample_map_full

    anchor_native_overlap = _filter_label_map(anchor_sample_map_full, anchor_overlap_lookup)
    target_native_overlap = _filter_label_map(target_sample_map_full, target_overlap_lookup)

    anchor_cross_overlap = _build_cross_label_map(anchor_overlap_lookup, anchor_to_target_map, anchor_to_target_rowcol_map)
    target_cross_overlap = _build_cross_label_map(target_overlap_lookup, target_to_anchor_map, target_to_anchor_rowcol_map)

    if overlap_label == anchor_name:
        anchor_label_map = anchor_native_overlap or anchor_cross_overlap
        anchor_source_tag = f"{anchor_name}_pn" if anchor_label_map is anchor_native_overlap else f"{target_name}_pn"
        target_label_map = target_cross_overlap or target_native_overlap
        target_source_tag = f"{anchor_name}_pn" if target_label_map is target_cross_overlap else f"{target_name}_pn"
    else:
        anchor_label_map = anchor_cross_overlap or anchor_native_overlap
        anchor_source_tag = f"{target_name}_pn" if anchor_label_map is anchor_cross_overlap else f"{anchor_name}_pn"
        target_label_map = target_native_overlap or target_cross_overlap
        target_source_tag = f"{target_name}_pn" if target_label_map is target_native_overlap else f"{anchor_name}_pn"

    def _resolve_lookup_for_labels(
        label_map: Dict[int, Dict[str, object]],
        primary: Dict[int, Dict[str, object]],
        fallback: Dict[int, Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        if not label_map:
            return primary or fallback
        resolved: Dict[int, Dict[str, object]] = {}
        for idx in label_map.keys():
            if idx in primary:
                resolved[idx] = primary[idx]
            elif idx in fallback:
                resolved[idx] = fallback[idx]
        return resolved

    anchor_all_lookup_resolved = _resolve_lookup_for_labels(anchor_native_all, anchor_all_lookup, anchor_all_lookup)
    target_all_lookup_resolved = _resolve_lookup_for_labels(target_native_all, target_all_lookup, target_all_lookup)

    anchor_overlap_lookup_resolved = _resolve_lookup_for_labels(anchor_label_map, anchor_overlap_lookup, anchor_all_lookup)
    target_overlap_lookup_resolved = _resolve_lookup_for_labels(target_label_map, target_overlap_lookup, target_all_lookup)

    anchor_all_result = _assemble_result(
        anchor_name,
        anchor_all_lookup_resolved,
        anchor_native_all,
        f"{anchor_name}_pn",
        cross_idx_map=anchor_to_target_map,
        cross_rowcol_map=anchor_to_target_rowcol_map,
    )
    target_all_result = _assemble_result(
        target_name,
        target_all_lookup_resolved,
        target_native_all,
        f"{target_name}_pn",
        cross_idx_map=target_to_anchor_map,
        cross_rowcol_map=target_to_anchor_rowcol_map,
    )

    anchor_overlap_result = (
        _assemble_result(
            anchor_name,
            anchor_overlap_lookup_resolved,
            anchor_label_map,
            anchor_source_tag,
            cross_idx_map=anchor_to_target_map,
            cross_rowcol_map=anchor_to_target_rowcol_map,
        )
        if anchor_label_map
        else _empty_payload(anchor_name, anchor_source_tag)
    )
    target_overlap_result = (
        _assemble_result(
            target_name,
            target_overlap_lookup_resolved,
            target_label_map,
            target_source_tag,
            cross_idx_map=target_to_anchor_map,
            cross_rowcol_map=target_to_anchor_rowcol_map,
        )
        if target_label_map
        else _empty_payload(target_name, target_source_tag)
    )
    return anchor_all_result, anchor_overlap_result, target_all_result, target_overlap_result


__all__ = [
    "_normalise_cross_matches",
    "_prepare_classifier_labels",
    "_build_aligned_pairs_OneToOne",
    "_build_aligned_pairs_SetToSet",
]




def _build_aligned_pairs_OneToOne(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    pn_label_maps: Optional[Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]]] = None,
    overlap_set: Optional[OverlapSet] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
    Dict[str, int],
    List[Dict[str, object]],
    List[torch.Tensor],
    Dict[str, Dict[str, Dict[str, object]]],
]:
    return _build_aligned_pairs_core(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=use_positive_only,
        aggregator=aggregator,
        gaussian_sigma=gaussian_sigma,
        debug=debug,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=anchor_augment_map,
        target_augment_map=target_augment_map,
        pn_label_maps=pn_label_maps,
        pairing_mode="one_to_one",
        overlap_set=overlap_set,
    )


def _build_aligned_pairs_SetToSet(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    pn_label_maps: Optional[Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]]] = None,
    overlap_set: Optional[OverlapSet] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
    Dict[str, int],
    List[Dict[str, object]],
    List[torch.Tensor],
    Dict[str, Dict[str, Dict[str, object]]],
]:
    return _build_aligned_pairs_core(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=use_positive_only,
        aggregator=aggregator,
        gaussian_sigma=gaussian_sigma,
        debug=debug,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=anchor_augment_map,
        target_augment_map=target_augment_map,
        pn_label_maps=pn_label_maps,
        pairing_mode="set_to_set",
        overlap_set=overlap_set,
    )

def _build_aligned_pairs_core(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    pn_label_maps: Optional[Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]]] = None,
    pairing_mode: str,
    overlap_set: Optional[OverlapSet] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
    Dict[str, int],
    List[Dict[str, object]],
    List[torch.Tensor],
    Dict[str, Dict[str, Dict[str, object]]],
]:
    anchor_augment_map = anchor_augment_map or {}
    target_augment_map = target_augment_map or {}
    pn_index_tracker: Dict[str, Dict[str, Dict[str, set[int]]]] = {}

    pn_label_maps = pn_label_maps or {}
    pairing_mode = (pairing_mode or "one_to_one").lower()

    def _register_index(record, dataset_key: str) -> None:
        if not dataset_key:
            return
        lookup = pn_label_maps.get(dataset_key)
        if not lookup:
            return
        region_val = getattr(record, "region", None)
        row_col_val = getattr(record, "row_col", None)
        if region_val is None or row_col_val is None:
            return
        try:
            row_val = int(row_col_val[0])
            col_val = int(row_col_val[1])
        except Exception:
            return
        region_key = str(region_val).upper()
        key = (region_key, row_val, col_val)
        label_bucket: Optional[str] = None
        pos_lookup = lookup.get("pos")
        if pos_lookup and key in pos_lookup:
            label_bucket = "pos"
        neg_lookup = lookup.get("neg")
        if neg_lookup and key in neg_lookup:
            label_bucket = "neg"
        if label_bucket is None:
            return
        idx_val = getattr(record, "index", None)
        if idx_val is None:
            return
        try:
            idx_int = int(idx_val)
        except Exception:
            return
        dataset_bucket = pn_index_tracker.setdefault(dataset_key, {})
        region_bucket = dataset_bucket.setdefault(region_key, {"pos": set(), "neg": set()})
        region_bucket[label_bucket].add(idx_int)

    def _finalise_pn_index_summary() -> Dict[str, Dict[str, Dict[str, object]]]:
        summary: Dict[str, Dict[str, Dict[str, object]]] = {}
        for dataset_name, regions in pn_index_tracker.items():
            region_summary: Dict[str, Dict[str, object]] = {}
            for region_name, label_sets in regions.items():
                pos_sorted = sorted(label_sets.get("pos", ()))
                neg_sorted = sorted(label_sets.get("neg", ()))
                region_summary[region_name] = {
                    "pos_count": len(pos_sorted),
                    "neg_count": len(neg_sorted),
                    "pos_original_indices": pos_sorted,
                    "neg_original_indices": neg_sorted,
                    "pos_reindexed_pairs": [
                        {"reindexed": re_idx, "original": original_idx}
                        for re_idx, original_idx in enumerate(pos_sorted)
                    ],
                    "neg_reindexed_pairs": [
                        {"reindexed": re_idx, "original": original_idx}
                        for re_idx, original_idx in enumerate(neg_sorted)
                    ],
                }
            summary[dataset_name] = region_summary
        return summary

    resolved_pairs = list(pairs)

    def _normalise_key(dataset_name: str, tile_id: Optional[str], row_col: Optional[Tuple[int, int]], coord: Optional[Tuple[float, float]], index: Optional[int] = None) -> Optional[Tuple[str, str]]:
        if tile_id:
            return dataset_name, f"tile_{tile_id}"
        if row_col and all(val is not None for val in row_col):
            return dataset_name, f"rowcol_{row_col[0]}_{row_col[1]}"
        if index is not None:
            return dataset_name, f"idx_{index}"
        if coord and len(coord) >= 2:
            key = (int(round(coord[0] * 1000)), int(round(coord[1] * 1000)))
            return dataset_name, f"coord_{key[0]}_{key[1]}"
        return None

    if overlap_set is not None:
        pair_map: Dict[Tuple[str, str], OverlapAlignmentPair] = {}
        for resolved in resolved_pairs:
            anchor_key = _normalise_key(
                resolved.anchor_dataset,
                getattr(resolved.anchor_record, "tile_id", None),
                getattr(resolved.anchor_record, "row_col", None),
                getattr(resolved.anchor_record, "coord", None),
                getattr(resolved.anchor_record, "index", None),
            )
            target_key = _normalise_key(
                resolved.target_dataset,
                getattr(resolved.target_record, "tile_id", None),
                getattr(resolved.target_record, "row_col", None),
                getattr(resolved.target_record, "coord", None),
                getattr(resolved.target_record, "index", None),
            )
            if anchor_key is None or target_key is None:
                continue
            pair_map[(anchor_key, target_key)] = resolved

        ordered: List[OverlapAlignmentPair] = []
        for raw_pair in overlap_set.pairs:
            first_tile, second_tile = raw_pair.a, raw_pair.b
            anchor_tile = first_tile if first_tile.dataset == anchor_name else second_tile if second_tile.dataset == anchor_name else None
            target_tile = second_tile if first_tile.dataset == anchor_name else first_tile if second_tile.dataset == anchor_name else None
            if anchor_tile is None or target_tile is None:
                continue
            anchor_key = _normalise_key(anchor_name, anchor_tile.tile_id, anchor_tile.row_col, anchor_tile.native_point)
            target_key = _normalise_key(target_name, target_tile.tile_id, target_tile.row_col, target_tile.native_point)
            if anchor_key is None or target_key is None:
                continue
            resolved = pair_map.get((anchor_key, target_key))
            if resolved is None:
                continue
            ordered.append(resolved)
        if ordered:
            resolved_pairs = ordered

    grouped: Dict[str, Dict[str, object]] = {}
    group_entries: List[Dict[str, object]] = []
    anchor_aug_added = 0
    target_aug_added = 0
    label_hist: Counter = Counter()
    debug_data: Optional[Dict[str, object]] = None
    if debug:
        debug_data = {
            "anchor_positive": {},
            "target_positive": {},
            "selected_pairs": [],
            "anchor_name": anchor_name,
            "target_name": target_name,
        }

    for pair in resolved_pairs:
        anchor_record, target_record = pair.anchor_record, pair.target_record
        anchor_ds, target_ds = pair.anchor_dataset, pair.target_dataset

        if anchor_ds != anchor_name or target_ds != target_name:
            if pair.target_dataset == anchor_name and pair.anchor_dataset == target_name:
                anchor_record, target_record = pair.target_record, pair.anchor_record
                anchor_ds, target_ds = pair.target_dataset, pair.anchor_dataset
            else:
                continue

        anchor_label = int(anchor_record.label)
        target_label = int(target_record.label)

        if debug_data is not None:
            if anchor_label == 1:
                _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
            if target_label == 1:
                _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))

        if use_positive_only and anchor_label != 1:
            continue
        if use_positive_only and target_label != 1:
            continue

        _register_index(anchor_record, anchor_ds)
        _register_index(target_record, target_ds)

        label_key = _label_key(anchor_label, target_label, anchor_name, target_name)
        label_hist[label_key] += 1

        if debug_data is not None:
            if anchor_label == 1 and target_label == 1:
                anchor_sample = _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
                target_sample = _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))
                if anchor_sample is not None and target_sample is not None:
                    debug_data["selected_pairs"].append({"anchor": anchor_sample, "target": target_sample})

        if pairing_mode == "one_to_one":
            entry = {
                "anchor": anchor_record,
                "targets": [target_record],
                "weights": [_pair_weight(anchor_record, target_record, gaussian_sigma)],
                "label_key": label_key,
            }
            group_entries.append(entry)
            continue

        tile_id = anchor_record.tile_id or f"anchor_{anchor_record.index}"
        grouped.setdefault(tile_id, {
            "anchor": anchor_record,
            "targets": [],
            "weights": [],
        })
        grouped[tile_id]["targets"].append(target_record)
        grouped[tile_id]["weights"].append(_pair_weight(anchor_record, target_record, gaussian_sigma))
        grouped[tile_id]["label_key"] = label_key

    anchor_vecs: List[torch.Tensor] = []
    target_vecs: List[torch.Tensor] = []
    target_stack_per_anchor: List[torch.Tensor] = []
    pair_metadata: List[Dict[str, object]] = []

    if pairing_mode == "set_to_set":
        group_entries = list(grouped.values())

    for entry in group_entries:
        targets: List = entry["targets"]
        if not targets:
            continue
        aggregated, normalized_weights, target_stack = _aggregate_targets(entry['anchor'], targets, entry['weights'], aggregator)
        if aggregated is None:
            continue
        anchor_tensor = torch.from_numpy(entry['anchor'].embedding).float()
        anchor_vecs.append(anchor_tensor)
        target_vecs.append(aggregated.clone())
        target_stack_per_anchor.append(target_stack.clone())
        meta_entry = _make_pair_meta(
            anchor_record=entry["anchor"],
            targets=targets,
            weights=normalized_weights,
            is_augmented=False,
        )
        meta_entry["pairing_mode"] = pairing_mode
        meta_entry["target_set_size"] = len(targets)
        if pairing_mode == "set_to_set":
            meta_entry["target_stack_shape"] = list(target_stack.shape)
        pair_metadata.append(meta_entry)
        entry_label_key = entry.get("label_key", "unlabelled")
        anchor_aug_list = anchor_augment_map.get(entry['anchor'].tile_id, [])
        for aug_emb in anchor_aug_list:
            aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
            if aug_tensor.shape != anchor_tensor.shape:
                continue
            anchor_vecs.append(aug_tensor)
            target_vecs.append(aggregated.clone())
            target_stack_per_anchor.append(target_stack.clone())
            label_hist[entry_label_key] += 1
            anchor_aug_added += 1
            meta_anchor_aug = _make_pair_meta(
                anchor_record=entry["anchor"],
                targets=targets,
                weights=normalized_weights,
                is_augmented=True,
                augmentation_role="anchor",
            )
            meta_anchor_aug["pairing_mode"] = pairing_mode
            meta_anchor_aug["target_set_size"] = len(targets)
            pair_metadata.append(meta_anchor_aug)

        for idx_target, target_record in enumerate(targets):
            aug_list = target_augment_map.get(target_record.tile_id, [])
            if not aug_list:
                continue
            for aug_emb in aug_list:
                aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
                if aug_tensor.shape != target_stack[idx_target].shape:
                    continue
                modified_stack = target_stack.clone()
                modified_stack[idx_target] = aug_tensor
                aggregated_aug = torch.matmul(normalized_weights.unsqueeze(0), modified_stack).squeeze(0)
                anchor_vecs.append(anchor_tensor.clone())
                target_vecs.append(aggregated_aug)
                target_stack_per_anchor.append(modified_stack.clone())
                label_hist[entry_label_key] += 1
                target_aug_added += 1
                meta_target_aug = _make_pair_meta(
                    anchor_record=entry["anchor"],
                    targets=targets,
                    weights=normalized_weights,
                    is_augmented=True,
                    augmentation_role="target",
                )
                meta_target_aug["pairing_mode"] = pairing_mode
                meta_target_aug["target_set_size"] = len(targets)
                pair_metadata.append(meta_target_aug)


    aug_stats = {
        "anchor_augmented_pairs": int(anchor_aug_added),
        "target_augmented_pairs": int(target_aug_added),
        "selected_pairs_augmented": int(anchor_aug_added + target_aug_added),
    }

    pn_index_summary = _finalise_pn_index_summary()

    if debug_data is not None:
        anchor_serialised = [_serialise_sample(sample) for sample in debug_data["anchor_positive"].values()]
        target_serialised = [_serialise_sample(sample) for sample in debug_data["target_positive"].values()]
        pair_serialised = [
            {
                "anchor": _serialise_sample(pair["anchor"]),
                "target": _serialise_sample(pair["target"]),
            }
            for pair in debug_data["selected_pairs"]
        ]
        debug_payload = {
            "anchor_positive": anchor_serialised,
            "target_positive": target_serialised,
            "selected_pairs": pair_serialised,
            "anchor_name": anchor_name,
            "target_name": target_name,
            "anchor_augmented_pairs": int(anchor_aug_added),
            "target_augmented_pairs": int(target_aug_added),
            "pn_index_summary": pn_index_summary,
        }
    else:
        debug_payload = None

    return anchor_vecs, target_vecs, label_hist, debug_payload, aug_stats, pair_metadata, target_stack_per_anchor, pn_index_summary


def _add_debug_sample(
    store: Dict[str, Dict[str, object]],
    record,
    dataset_name: str,
    dataset_meta: Optional[Dict[str, object]],
    *,
    is_augmented: bool = False,
) -> Optional[Dict[str, object]]:
    if record.coord is None or any(math.isnan(c) for c in record.coord[:2]):
        return None
    coord = (float(record.coord[0]), float(record.coord[1]))
    sample = store.get(record.tile_id)
    if sample is None:
        sample = {
            "tile_id": record.tile_id,
            "coord": coord,
            "dataset": dataset_name,
        }
        window = record.window_size
        pixel_res = record.pixel_resolution
        if window is None and dataset_meta:
            spacing = dataset_meta.get("window_spacing") or dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
            if spacing is not None:
                pixel_res = float(spacing)
                window = (1, 1)
        if window is not None:
            sample["window_size"] = [int(window[0]), int(window[1])]
        if pixel_res is None and dataset_meta:
            pixel_res = dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
        if pixel_res is not None:
            sample["pixel_resolution"] = float(pixel_res)
        if window is not None and pixel_res is not None:
            width = window[1] * float(pixel_res)
            height = window[0] * float(pixel_res)
            sample["footprint"] = [width, height]
        sample["is_augmented"] = bool(is_augmented)
        store[record.tile_id] = sample
    return sample


def _serialise_sample(sample: Dict[str, object]) -> Dict[str, object]:
    coord = sample.get("coord")
    coord_list = [float(coord[0]), float(coord[1])] if coord is not None else None
    data: Dict[str, object] = {
        "tile_id": sample.get("tile_id"),
    }
    if coord_list is not None:
        data["coord"] = coord_list
    if sample.get("dataset") is not None:
        data["dataset"] = sample["dataset"]
    if "window_size" in sample and sample["window_size"] is not None:
        data["window_size"] = [int(sample["window_size"][0]), int(sample["window_size"][1])]
    if "pixel_resolution" in sample and sample["pixel_resolution"] is not None:
        data["pixel_resolution"] = float(sample["pixel_resolution"])
    if sample.get("footprint") is not None:
        data["footprint"] = [float(sample["footprint"][0]), float(sample["footprint"][1])]
    if sample.get("is_augmented"):
        data["is_augmented"] = True
    return data

def _label_key(anchor_label: int, target_label: int, anchor_name: str, target_name: str) -> str:
    if anchor_label == 1 and target_label == 1:
        return "positive_common"
    if anchor_label == 1 and target_label != 1:
        return f"positive_{anchor_name}"
    if anchor_label != 1 and target_label == 1:
        return f"positive_{target_name}"
    return "unlabelled"



def _pair_weight(anchor_record, target_record, gaussian_sigma: Optional[float]) -> float:
    if gaussian_sigma is None:
        return 1.0
    if anchor_record.coord is None or target_record.coord is None:
        return 1.0
    anchor_xy = np.asarray(anchor_record.coord, dtype=float)
    target_xy = np.asarray(target_record.coord, dtype=float)
    if anchor_xy.size < 2 or target_xy.size < 2:
        return 1.0
    dist = float(np.linalg.norm(anchor_xy[:2] - target_xy[:2]))
    if dist <= 0.0:
        return 1.0
    weight = math.exp(- (dist ** 2) / (2.0 * (gaussian_sigma ** 2)))
    return float(weight + 1e-8)


def _aggregate_targets(anchor_record, targets: Sequence, weights: Sequence[float], aggregator: str) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if aggregator != "weighted_pool":
        raise NotImplementedError(f"Aggregator '{aggregator}' is not implemented yet.")

    embeddings = [torch.from_numpy(target.embedding).float() for target in targets]
    if not embeddings:
        return None, torch.empty(0), torch.empty(0)
    stacked = torch.stack(embeddings)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.isnan(weight_tensor).any() or float(weight_tensor.sum()) <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    aggregated = torch.matmul(weight_tensor.unsqueeze(0), stacked).squeeze(0)
    return aggregated, weight_tensor, stacked


def _make_pair_meta(
    anchor_record,
    targets: Sequence,
    weights: Optional[torch.Tensor],
    is_augmented: bool,
    augmentation_role: Optional[str] = None,
) -> Dict[str, object]:
    anchor_row_col = _normalise_row_col(getattr(anchor_record, "row_col", None))
    target_row_cols = [_normalise_row_col(getattr(target, "row_col", None)) for target in targets]
    target_coords = [_normalise_coord(getattr(target, "coord", None)) for target in targets]
    target_tile_ids = [getattr(target, "tile_id", None) for target in targets]
    target_labels = [int(getattr(target, "label", 0)) for target in targets]
    target_regions = [getattr(target, "region", None) for target in targets]
    try:
        anchor_index = int(getattr(anchor_record, "index"))
    except Exception:
        anchor_index = None
    target_indices: List[Optional[int]] = []
    for target in targets:
        try:
            target_indices.append(int(getattr(target, "index")))
        except Exception:
            target_indices.append(None)
    weights_list: Optional[List[float]] = None
    weighted_coord: Optional[Tuple[float, float]] = None
    if isinstance(weights, torch.Tensor) and weights.numel() == len(targets):
        weights_cpu = weights.detach().cpu()
        weights_list = [float(w) for w in weights_cpu.tolist()]
        weighted_coord = _weighted_average_coord(target_coords, weights_list)

    meta: Dict[str, object] = {
        "anchor_tile_id": getattr(anchor_record, "tile_id", None),
        "anchor_coord": _normalise_coord(getattr(anchor_record, "coord", None)),
        "anchor_region": getattr(anchor_record, "region", None),
        "anchor_label": int(getattr(anchor_record, "label", 0)),
        "anchor_row_col": anchor_row_col,
        "target_tile_ids": target_tile_ids,
        "target_coords": target_coords,
        "target_labels": target_labels,
        "target_regions": target_regions,
        "target_row_cols": target_row_cols,
        "target_weights": weights_list,
        "target_weighted_coord": weighted_coord,
        "is_augmented": bool(is_augmented),
        "anchor_index": anchor_index,
        "target_indices": target_indices,
    }
    if augmentation_role is not None:
        meta["augmentation_role"] = augmentation_role
    return meta

def _normalise_coord(coord: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if coord is None:
        return None
    try:
        x = float(coord[0])
        y = float(coord[1])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _normalise_row_col(value: Optional[Sequence[object]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        row = int(value[0])
        col = int(value[1])
    except Exception:
        return None
    return (row, col)





def _weighted_average_coord(
    coords: Sequence[Optional[Tuple[float, float]]],
    weights: Sequence[float],
) -> Optional[Tuple[float, float]]:
    total_weight = 0.0
    accum_x = 0.0
    accum_y = 0.0
    for coord, weight in zip(coords, weights):
        if coord is None:
            continue
        try:
            w = float(weight)
        except Exception:
            continue
        if not math.isfinite(w) or w <= 0:
            continue
        accum_x += coord[0] * w
        accum_y += coord[1] * w
        total_weight += w
    if total_weight <= 0:
        return None
    return (accum_x / total_weight, accum_y / total_weight)


def _lookup_pn_label(
    region: Optional[str],
    row_col: Optional[Tuple[int, int]],
    lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
) -> Optional[int]:
    if lookup is None or region is None or row_col is None:
        return None
    region_key = str(region).upper()
    row, col = row_col
    key = (region_key, int(row), int(col))
    if key in lookup.get("pos", set()):
        return 1
    if key in lookup.get("neg", set()):
        return 0
    return None


def _mask_contains_coord(coord: Optional[Tuple[float, float]], mask_info: Optional[Dict[str, object]]) -> bool:
    if mask_info is None:
        return True
    if coord is None or rasterio is None:
        return False
    transform = mask_info.get("transform")
    array = mask_info.get("array")
    shape = mask_info.get("shape")
    if transform is None or array is None or shape is None:
        return True
    try:
        row, col = rasterio.transform.rowcol(transform, coord[0], coord[1])
    except Exception:
        return False
    height, width = shape
    if not (0 <= row < height and 0 <= col < width):
        return False
    value = array[row, col]
    nodata = mask_info.get("nodata")
    try:
        if nodata is not None and np.isfinite(nodata):
            if np.isfinite(value) and np.isclose(value, nodata):
                return False
            if not np.isfinite(value):
                return False
    except Exception:
        pass
    return bool(value)

def _apply_projector_based_PUNlabels(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    pn_lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
    projector: nn.Module,
    batch_size: int,
    device: torch.device,
    run_logger: Any,
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
    apply_overlap_filter: bool = False,
) -> Optional[Dict[str, object]]:
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None:
        run_logger.log(f"[cls] dataset {dataset_name} not found in workspace; skipping classifier samples.")
        return None
    if apply_overlap_filter and overlap_mask is None:
        run_logger.log(f"[cls] overlap mask unavailable; cannot filter samples for dataset {dataset_name}.")
        apply_overlap_filter = False
    matched_records: List[EmbeddingRecord] = []
    labels: List[int] = []
    coords: List[Optional[Tuple[float, float]]] = []
    regions: List[Optional[str]] = []
    row_cols: List[Optional[Tuple[int, int]]] = []
    indices: List[int] = []
    metadata: List[Dict[str, object]] = []
    
    for record in bundle.records:
        if pn_lookup is not None:
            label = _lookup_pn_label(record.region, record.row_col, pn_lookup)
            if label is None:
                label_int = -1  # Unlabelled
            elif int(label) == 1:
                label_int = 1
            elif int(label) == 0:
                label_int = 0
            else:
                raise ValueError(f"Invalid PN label value: {label}")
        else:
            label_int = -1  # Unlabelled

        coord_norm = _normalise_coord(record.coord)
        if apply_overlap_filter and not _mask_contains_coord(coord_norm, overlap_mask):
            continue
        matched_records.append(record)
        labels.append(label_int)
        coords.append(coord_norm)
        regions.append(getattr(record, "region", None))
        row_cols.append(getattr(record, "row_col", None))
        indices.append(int(getattr(record, "index", len(indices))))
        metadata.append(
            {
                "dataset": dataset_name,
                "label": label_int,
                "coord": coord_norm,
                "region": getattr(record, "region", None),
                "row_col": getattr(record, "row_col", None),
                "embedding_index": int(getattr(record, "index", len(indices))),
                "tile_id": getattr(record, "tile_id", None),
                "overlap_filtered": bool(apply_overlap_filter),
            }
        )
    if not matched_records:
        run_logger.log(f"[cls] No PN-labelled samples found for dataset {dataset_name}; skipping classifier samples.")
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required to collect classifier samples.")
    
    # ✅ MEMORY FIX: Process records in chunks to avoid loading entire dataset into memory
    projector = projector.to(device)
    projector.eval()
    
    # Use the batch_size parameter passed from DCCA (now memory-efficient)
    chunk_size = max(batch_size, 2048)  # Use larger chunks for better performance (minimum 2048)
    num_samples = len(matched_records)
    projected_list = []
    
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    run_logger.log(f"[cls] Processing {num_samples} samples for {dataset_name} in {num_chunks} chunks of {chunk_size}")
    
    # Memory monitoring helper
    def _log_memory_status(stage: str, chunk_idx: int = None):
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            chunk_info = f" chunk {chunk_idx}" if chunk_idx is not None else ""
            run_logger.log(f"[cls-memory{chunk_info}] {stage}: {memory_mb:.1f}MB")
        except:
            pass
    
    # _log_memory_status("start")  # Commented out to reduce verbosity
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            
            # ✅ MEMORY FIX: Load only the current chunk into memory
            chunk_records = matched_records[start_idx:end_idx]
            chunk_embeddings = np.stack([np.asarray(rec.embedding, dtype=np.float32) for rec in chunk_records])
            
            embed_tensor = torch.from_numpy(chunk_embeddings).to(device)
            
            if hasattr(projector, 'aggregator'):
                # For AggregatorTargetHead, use self-attention: each sample attends to itself
                # This reduces dimensionality (512→256) via aggregator before final projection
                target_tensor = embed_tensor.unsqueeze(1)  # [B, 1, D] - single-item sequence per sample
                batch_projected = projector(embed_tensor, target_tensor).detach().cpu()
            else:
                batch_projected = projector(embed_tensor).detach().cpu()
            
            projected_list.append(batch_projected)
            
            # ✅ MEMORY FIX: Explicit cleanup after each chunk
            del chunk_embeddings, embed_tensor, batch_projected
            if hasattr(projector, 'aggregator'):
                del target_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ✅ MEMORY FIX: Force garbage collection every few chunks  
            if start_idx % (chunk_size * 200) == 0:  # Much less frequent GC for better performance
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _log_memory_status("after gc", start_idx // chunk_size)
    
    projected = torch.cat(projected_list, dim=0)
    # Clear intermediate lists to free memory
    del projected_list
    
    # Final cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # label_tensor = torch.tensor(labels, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.int16)
    return {
        "dataset": dataset_name,
        "features": projected,
        "labels": label_tensor,
        "metadata": metadata,
        "coords": coords,
        "regions": regions,
        "row_cols": row_cols,
        "indices": indices,
    }

def _apply_projector_based_PNlabels(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    pn_lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
    projector: nn.Module,
    device: torch.device,
    run_logger: Any,
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
    apply_overlap_filter: bool = False,
) -> Optional[Dict[str, object]]:
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None:
        run_logger.log(f"[cls] dataset {dataset_name} not found in workspace; skipping classifier samples.")
        return None
    if pn_lookup is None or (not pn_lookup.get("pos") and not pn_lookup.get("neg")):
        run_logger.log(f"[cls] PN lookup unavailable for dataset {dataset_name}; skipping classifier samples.")
        return None
    if apply_overlap_filter and overlap_mask is None:
        run_logger.log(f"[cls] overlap mask unavailable; cannot filter samples for dataset {dataset_name}.")
        apply_overlap_filter = False
    matched_records: List[EmbeddingRecord] = []
    labels: List[int] = []
    coords: List[Optional[Tuple[float, float]]] = []
    regions: List[Optional[str]] = []
    row_cols: List[Optional[Tuple[int, int]]] = []
    indices: List[int] = []
    metadata: List[Dict[str, object]] = []
    for record in bundle.records:
        label = _lookup_pn_label(record.region, record.row_col, pn_lookup)
        if label is None:
            continue
        label_int = 1 if int(label) > 0 else 0
        coord_norm = _normalise_coord(record.coord)
        if apply_overlap_filter and not _mask_contains_coord(coord_norm, overlap_mask):
            continue
        matched_records.append(record)
        labels.append(label_int)
        coords.append(coord_norm)
        regions.append(getattr(record, "region", None))
        row_cols.append(getattr(record, "row_col", None))
        indices.append(int(getattr(record, "index", len(indices))))
        metadata.append(
            {
                "dataset": dataset_name,
                "label": label_int,
                "coord": coord_norm,
                "region": getattr(record, "region", None),
                "row_col": getattr(record, "row_col", None),
                "embedding_index": int(getattr(record, "index", len(indices))),
                "tile_id": getattr(record, "tile_id", None),
                "overlap_filtered": bool(apply_overlap_filter),
            }
        )
    if not matched_records:
        run_logger.log(f"[cls] No PN-labelled samples found for dataset {dataset_name}; skipping classifier samples.")
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required to collect classifier samples.")
    embeddings = np.stack([np.asarray(rec.embedding, dtype=np.float32) for rec in matched_records])
    projector = projector.to(device)
    projector.eval()
    with torch.no_grad():
        embed_tensor = torch.from_numpy(embeddings).to(device)
        if hasattr(projector, 'aggregator'):
            # For AggregatorTargetHead, use self-attention: each sample attends to itself
            # This reduces dimensionality (512→256) via aggregator before final projection
            target_tensor = embed_tensor.unsqueeze(1)  # [B, 1, D] - single-item sequence per sample
            projected = projector(embed_tensor, target_tensor).detach().cpu()
        else:
            projected = projector(embed_tensor).detach().cpu()
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return {
        "dataset": dataset_name,
        "features": projected,
        "labels": label_tensor,
        "metadata": metadata,
        "coords": coords,
        "regions": regions,
        "row_cols": row_cols,
        "indices": indices,
    }

class AggregatorTargetHead(nn.Module):
    def __init__(self, aggregator: CrossAttentionAggregator, proj_target: nn.Module, *, use_positional_encoding: bool) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.proj_target = proj_target
        self.use_positional_encoding = bool(use_positional_encoding)

    def forward(
        self,
        anchor_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = self.aggregator(
            anchor_batch,
            target_batch,
            key_padding_mask=key_padding_mask,
            pos_encoding=pos_encoding if self.aggregator.use_positional_encoding else None,
        )
        return self.proj_target(fused)

def _subset_classifier_sample(
    sample: Dict[str, object],
    keep_mask: Sequence[bool],
    *,
    subset_tag: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to subset classifier samples.")
    features = sample.get("features")
    if not isinstance(features, torch.Tensor):
        raise TypeError("Classifier sample 'features' must be a torch.Tensor.")
    mask_tensor = torch.as_tensor(keep_mask, dtype=torch.bool, device=features.device)
    if mask_tensor.numel() != features.size(0):
        raise ValueError("Subset mask size mismatch for classifier sample.")
    if not bool(mask_tensor.any()):
        return None
    filtered_metadata: List[Dict[str, object]] = []
    metadata_entries = sample.get("metadata", [])
    for entry, keep in zip(metadata_entries, keep_mask):
        if not keep:
            continue
        entry_copy = dict(entry)
        if subset_tag is not None:
            entry_copy["subset_role"] = subset_tag
        filtered_metadata.append(entry_copy)
    filtered_sample: Dict[str, object] = {
        "dataset": sample.get("dataset"),
        "features": features[mask_tensor].clone(),
        "labels": sample["labels"][mask_tensor].clone(),
        "metadata": filtered_metadata,
        "coords": [coord for coord, keep in zip(sample.get("coords", []), keep_mask) if keep],
        "regions": [region for region, keep in zip(sample.get("regions", []), keep_mask) if keep],
        "row_cols": [row_col for row_col, keep in zip(sample.get("row_cols", []), keep_mask) if keep],
        "indices": [idx for idx, keep in zip(sample.get("indices", []), keep_mask) if keep],
    }
    if subset_tag is not None:
        filtered_sample["subset"] = subset_tag
    return filtered_sample
