"""Reusable helpers for constructing fused overlap datasets (v1 style)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional torch dependency
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]

from Common.metrics_logger import save_metrics_json
from Common.cls.infer.infer_maps import (write_prediction_outputs,)


Coord = Tuple[int, int]

def fusion_export_results(
    fusion_dir: Path,
    mlp: nn.Module,
    history_payload: List[Dict[str, object]],
    evaluation_summary: Dict[str, Dict[str, float]],
    metrics_summary: Dict[str, object],
    inference_outputs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    fusion_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = dict(metrics_summary)
    metrics_payload["evaluation"] = evaluation_summary
    metrics_payload["history"] = history_payload
    metrics_path = fusion_dir / "metrics.json"
    try:
        save_metrics_json(metrics_payload, metrics_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save fusion metrics: {exc}")
    state_path = fusion_dir / "classifier.pt"
    torch.save({"state_dict": mlp.state_dict()}, state_path)
    inference_summary: Dict[str, object] = {}
    for dataset_name, payload in inference_outputs.items():
        out_dir = fusion_dir / dataset_name
        write_prediction_outputs(
            payload["prediction"],
            payload["default_reference"],
            out_dir,
            pos_coords_by_region=payload["pos_map"],
            neg_coords_by_region=payload["neg_map"],
        )
        # predictions_path = out_dir / "predictions.npy"
        predictions_path = out_dir / "predictions.npz"
        try:
            prediction_payload = payload.get("prediction")
            if isinstance(prediction_payload, dict):
                global_payload = prediction_payload.get("GLOBAL") or {}
                mean_map = global_payload.get("mean")
                std_map = global_payload.get("std")
            else:
                mean_map = prediction_payload
                std_map = None
            pos_map = payload.get("pos_map") or {}
            neg_map = payload.get("neg_map") or {}
            pos_coords = [
                (region, int(r), int(c))
                for region, coords in pos_map.items()
                for r, c in coords
            ]
            neg_coords = [
                (region, int(r), int(c))
                for region, coords in neg_map.items()
                for r, c in coords
            ]
            row_cols = [tuple(rc) if isinstance(rc, (list, tuple)) else rc for rc in (payload.get("row_cols") or [])]
            coords_list = payload.get("coords") or [None] * len(row_cols)
            labels = payload.get("labels") or [0] * len(row_cols)
            metadata = payload.get("metadata") or [None] * len(row_cols)
            mean_values = payload.get("mean_values")
            std_values = payload.get("std_values")
            if mean_values is None and mean_map is not None and row_cols:
                mean_values = [float(mean_map[r, c]) for r, c in row_cols]
            if std_values is None and std_map is not None and row_cols:
                std_values = [float(std_map[r, c]) for r, c in row_cols]
            data_payload = {
                "mean": np.asarray(mean_map, dtype=np.float32) if mean_map is not None else None,
                "std": np.asarray(std_map, dtype=np.float32) if std_map is not None else None,
                "row_cols": row_cols,
                "coords": coords_list,
                "labels": labels,
                "metadata": metadata,
                "mean_values": np.asarray(mean_values, dtype=np.float32) if mean_values is not None else None,
                "std_values": np.asarray(std_values, dtype=np.float32) if std_values is not None else None,
                "pos_coords": pos_coords,
                "neg_coords": neg_coords,
            }
            # np.save(predictions_path, data_payload, allow_pickle=True)
            np.savez_compressed(predictions_path, predictions=data_payload)

        except Exception as exc:
            print(f"[warn] Failed to save prediction array for {dataset_name}: {exc}")
        inference_summary[dataset_name] = {
            "output_dir": str(out_dir),
            "npy_path": str(predictions_path),
            "positive_count": payload["counts"]["pos"],
            "negative_count": payload["counts"]["neg"],
        }
    return {
        "metrics_path": str(metrics_path),
        "state_dict_path": str(state_path),
        "evaluation": evaluation_summary,
        "history": history_payload,
        "outputs": inference_summary,
    }

def _ensure_numpy_features(value: object) -> np.ndarray:
    if torch is not None and isinstance(value, Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[0])
    return arr.astype(np.float32, copy=False)


def _resolve_row_col(entry: Dict[str, object], idx: int) -> Optional[Coord]:
    row_cols = entry.get("row_cols") or entry.get("row_cols_mask") or []
    if idx < len(row_cols):
        rc = row_cols[idx]
        if isinstance(rc, (list, tuple)) and len(rc) >= 2:
            try:
                return (int(rc[0]), int(rc[1]))
            except Exception:
                return None
    metadata = entry.get("metadata") or []
    if idx < len(metadata) and isinstance(metadata[idx], dict):
        rc_meta = metadata[idx].get("row_col") or metadata[idx].get("row_col_mask")
        if isinstance(rc_meta, (list, tuple)) and len(rc_meta) >= 2:
            try:
                return (int(rc_meta[0]), int(rc_meta[1]))
            except Exception:
                return None
    return None


def _resolve_coord(entry: Dict[str, object], idx: int) -> Optional[Tuple[float, float]]:
    coords = entry.get("coords") or []
    if idx < len(coords):
        coord = coords[idx]
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            return (float(coord[0]), float(coord[1]))
    metadata = entry.get("metadata") or []
    if idx < len(metadata) and isinstance(metadata[idx], dict):
        coord_meta = metadata[idx].get("coord")
        if isinstance(coord_meta, (list, tuple)) and len(coord_meta) >= 2:
            return (float(coord_meta[0]), float(coord_meta[1]))
    return None


def _collect_records(
    entry: Optional[Dict[str, object]],
    labels_override: Optional[Sequence[int]],
    dataset_tag: str,
) -> Tuple[Dict[int, Dict[str, object]], int]:
    """Normalise samples into a keyed dictionary for fusion assembly."""
    if entry is None:
        return {}, 0
    feats = entry.get("features")
    if feats is None:
        return {}, 0
    feats_np = _ensure_numpy_features(feats)
    count = feats_np.shape[0]
    dim = feats_np.shape[1] if feats_np.ndim == 2 else 0
    if count == 0 or dim == 0:
        return {}, dim
    if labels_override is not None and len(labels_override) == count:
        labels_np = np.asarray(labels_override, dtype=np.int16)
    else:
        raw_labels = entry.get("labels")
        if torch is not None and isinstance(raw_labels, Tensor):
            labels_np = raw_labels.detach().cpu().numpy().astype(np.int16, copy=False)
        elif isinstance(raw_labels, np.ndarray):
            labels_np = raw_labels.astype(np.int16, copy=False)
        else:
            labels_np = np.asarray(raw_labels or np.zeros(count, dtype=np.int16), dtype=np.int16)
        if labels_np.shape[0] != count:
            labels_np = np.resize(labels_np, count).astype(np.int16, copy=False)
    pair_ids_raw = entry.get("pair_ids") or list(range(count))
    pair_sources_raw = entry.get("pair_sources") or []
    metadata_list = entry.get("metadata") or []
    records: Dict[int, Dict[str, object]] = {}
    for idx in range(count):
        pair_id = int(pair_ids_raw[idx]) if idx < len(pair_ids_raw) else idx
        if pair_id in records:
            continue
        feature_vec = feats_np[idx].astype(np.float32, copy=False)
        label_val = int(labels_np[idx]) if idx < len(labels_np) else 0
        source_val = pair_sources_raw[idx] if idx < len(pair_sources_raw) else dataset_tag
        meta_entry = metadata_list[idx] if idx < len(metadata_list) and isinstance(metadata_list[idx], dict) else {}
        records[pair_id] = {
            "feature": feature_vec,
            "label": label_val,
            "metadata": dict(meta_entry),
            "row_col": _resolve_row_col(entry, idx),
            "coord": _resolve_coord(entry, idx),
            "source": str(source_val) if source_val is not None else dataset_tag,
            "dataset": dataset_tag,
        }
    return records, dim


def _pad_vector(vec: Optional[np.ndarray], length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    padded = np.zeros(length, dtype=np.float32)
    if vec is None or vec.size == 0:
        return padded
    limit = min(length, vec.shape[0])
    padded[:limit] = vec[:limit]
    return padded


def prepare_fusion_overlap_dataset_one_to_one(
    anchor_data: Optional[Dict[str, object]],
    target_data: Optional[Dict[str, object]],
    anchor_labels_override: Optional[Sequence[int]] = None,
    target_labels_override: Optional[Sequence[int]] = None,
    *,
    anchor_name: str,
    target_name: str,
    method_id: str = "simple",
) -> Optional[Dict[str, object]]:
    """Construct the fused training table used by the PN head."""
    method_key = method_id.lower()
    if method_key not in {"simple", "strong"}:
        method_key = "simple"
    if anchor_data is None and target_data is None:
        return None

    anchor_records, dim_u = _collect_records(anchor_data, anchor_labels_override, anchor_name)
    target_records, dim_v = _collect_records(target_data, target_labels_override, target_name)

    if dim_u <= 0 and dim_v <= 0:
        return None
    if dim_u <= 0:
        dim_u = dim_v
    if dim_v <= 0:
        dim_v = dim_u
    max_dim = max(dim_u, dim_v)

    union_pair_ids = sorted(set(anchor_records.keys()) | set(target_records.keys()))
    if not union_pair_ids:
        return None

    features_rows: List[np.ndarray] = []
    labels_rows: List[int] = []
    metadata_rows: List[Dict[str, object]] = []
    row_cols: List[Optional[Coord]] = []
    coords: List[Optional[Tuple[float, float]]] = []
    pair_groups: List[int] = []
    pair_sources: List[str] = []
    anchor_vectors: List[np.ndarray] = []
    target_vectors: List[np.ndarray] = []
    anchor_flags: List[bool] = []
    target_flags: List[bool] = []

    for pair_id in union_pair_ids:
        anchor_rec = anchor_records.get(pair_id)
        target_rec = target_records.get(pair_id)
        anchor_present = anchor_rec is not None
        target_present = target_rec is not None

        u_vec = _pad_vector(anchor_rec["feature"] if anchor_present else None, dim_u)
        v_vec = _pad_vector(target_rec["feature"] if target_present else None, dim_v)
        anchor_vectors.append(u_vec)
        target_vectors.append(v_vec)

        if method_key == "simple":
            phi_parts = [u_vec, v_vec]
        else:
            u_common = _pad_vector(anchor_rec["feature"] if anchor_present else None, max_dim)
            v_common = _pad_vector(target_rec["feature"] if target_present else None, max_dim)
            diff_vec = np.abs(u_common - v_common)
            prod_vec = u_common * v_common
            norm_u = float(np.linalg.norm(u_common))
            norm_v = float(np.linalg.norm(v_common))
            cosine_val = float(np.dot(u_common, v_common) / (norm_u * norm_v + 1e-8)) if norm_u > 0 and norm_v > 0 else 0.0
            missing_flag = 0.0 if anchor_present and target_present else 1.0
            phi_parts = [
                u_vec,
                v_vec,
                diff_vec,
                prod_vec,
                np.asarray([cosine_val], dtype=np.float32),
                np.asarray([missing_flag], dtype=np.float32),
            ]
        phi_vec = np.concatenate(phi_parts, dtype=np.float32)
        features_rows.append(phi_vec)

        label_source = anchor_name
        label_val = 0
        if anchor_present and anchor_rec.get("label") is not None:
            label_val = int(anchor_rec["label"])
            label_source = anchor_rec.get("source", anchor_name)
        elif target_present and target_rec.get("label") is not None:
            label_val = int(target_rec["label"])
            label_source = target_rec.get("source", target_name)
        labels_rows.append(label_val)
        pair_sources.append(str(label_source))
        pair_groups.append(int(pair_id))

        meta_entry: Dict[str, object] = {
            "pair_id": int(pair_id),
            "pair_label_source": str(label_source),
        }
        if anchor_present:
            meta_entry["anchor_metadata"] = anchor_rec.get("metadata", {})
        if target_present:
            meta_entry["target_metadata"] = target_rec.get("metadata", {})
        row_col = anchor_rec.get("row_col") if anchor_present else None
        if row_col is None and target_present:
            row_col = target_rec.get("row_col")
        coord = anchor_rec.get("coord") if anchor_present else None
        if coord is None and target_present:
            coord = target_rec.get("coord")
        metadata_rows.append(meta_entry)
        row_cols.append(row_col if isinstance(row_col, tuple) else None)
        coords.append(coord if isinstance(coord, tuple) else None)
        anchor_flags.append(anchor_present)
        target_flags.append(target_present)

    if not features_rows:
        return None

    features_arr = np.vstack(features_rows).astype(np.float32, copy=False)
    labels_arr = np.asarray(labels_rows, dtype=np.int16)
    pair_groups_arr = np.asarray(pair_groups, dtype=np.int32)
    anchor_matrix = (
        np.vstack(anchor_vectors).astype(np.float32, copy=False)
        if anchor_vectors
        else np.empty((0, dim_u), dtype=np.float32)
    )
    target_matrix = (
        np.vstack(target_vectors).astype(np.float32, copy=False)
        if target_vectors
        else np.empty((0, dim_v), dtype=np.float32)
    )

    return {
        "features": features_arr,
        "labels": labels_arr,
        "metadata": metadata_rows,
        "row_cols": row_cols,
        "coords": coords,
        "pair_groups": pair_groups_arr,
        "pair_sources": pair_sources,
        "anchor_vectors": anchor_matrix,
        "target_vectors": target_matrix,
        "anchor_present": anchor_flags,
        "target_present": target_flags,
        "dim_u": dim_u,
        "dim_v": dim_v,
        "name": f"fusion_{method_key}",
    }


def _trim_reembedding_entry(
    entry: Optional[Dict[str, object]],
    positions: Sequence[int],
) -> Optional[Dict[str, object]]:
    if entry is None or not positions:
        return None
    trimmed: Dict[str, object] = dict(entry)

    def _take_sequence(seq: Sequence[object]) -> List[object]:
        return [seq[pos] for pos in positions]

    features = entry.get("features")
    if torch is not None and isinstance(features, Tensor):
        selector = torch.as_tensor(positions, dtype=torch.long)
        trimmed["features"] = features.index_select(0, selector)
    elif isinstance(features, np.ndarray):
        trimmed["features"] = features[positions]

    labels = entry.get("labels")
    if torch is not None and isinstance(labels, Tensor):
        selector = torch.as_tensor(positions, dtype=torch.long)
        trimmed["labels"] = labels.index_select(0, selector)
    elif isinstance(labels, np.ndarray):
        trimmed["labels"] = labels[positions]
    elif labels is not None:
        trimmed["labels"] = np.asarray(_take_sequence(labels), dtype=np.int16)

    for key in ("indices", "coords", "metadata", "row_cols", "row_cols_mask", "mask_flags"):
        if key in entry and entry[key] is not None:
            trimmed[key] = _take_sequence(entry[key])

    return trimmed


def align_overlap_embeddings_for_pn_one_to_one(
    anchor_entry: Optional[Dict[str, object]],
    target_entry: Optional[Dict[str, object]],
    *,
    anchor_name: str,
    target_name: str,
    index_label_anchor_overlap: Optional[Dict[str, object]] = None,
    index_label_target_overlap: Optional[Dict[str, object]] = None,
) -> Tuple[
    Optional[Dict[str, object]],
    Optional[Dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    """Synchronise anchor/target overlap embeddings with PN labels."""
    if anchor_entry is None and target_entry is None:
        empty = np.asarray([], dtype=np.int16)
        return None, None, empty, empty

    def _parse_source_tag(value: Optional[str], default: str) -> str:
        raw = str(value or "")
        if not raw:
            return default
        tag = raw.split("_", 1)[0]
        return tag or default

    def _prepare_slice(
        entry: Optional[Dict[str, object]],
        index_payload: Optional[Dict[str, object]],
        default_source: str,
        *,
        is_anchor: bool,
    ) -> Tuple[Optional[Dict[str, object]], np.ndarray, List[Tuple[str, int]]]:
        if entry is None or not index_payload:
            return None, np.asarray([], dtype=np.int16), []

        source_dataset = _parse_source_tag(index_payload.get("source"), default_source)
        positions_raw = index_payload.get("positions") or []
        labels_raw = index_payload.get("labels") or []
        dataset_indices_raw = index_payload.get("dataset_indices") or []
        cross_indices_raw = index_payload.get("cross_source_indices") or []

        valid_items: List[Tuple[int, int, Optional[int], Optional[int]]] = []
        max_len = min(len(positions_raw), len(labels_raw))
        for idx in range(max_len):
            try:
                position = int(positions_raw[idx])
                label_int = 1 if int(labels_raw[idx]) > 0 else 0
            except Exception:
                continue
            dataset_idx = None
            if idx < len(dataset_indices_raw):
                try:
                    dataset_idx = int(dataset_indices_raw[idx])
                except Exception:
                    dataset_idx = None
            cross_idx = None
            if idx < len(cross_indices_raw):
                try:
                    cross_idx = int(cross_indices_raw[idx])
                except Exception:
                    cross_idx = None
            valid_items.append((position, label_int, dataset_idx, cross_idx))

        if not valid_items:
            return None, np.asarray([], dtype=np.int16), []

        positions = [item[0] for item in valid_items]
        trimmed = _trim_reembedding_entry(entry, positions)
        if trimmed is None:
            return None, np.asarray([], dtype=np.int16), []

        labels_arr = np.asarray([item[1] for item in valid_items], dtype=np.int16)
        pair_keys: List[Tuple[str, int]] = []
        for position, _, dataset_idx, cross_idx in valid_items:
            if source_dataset == anchor_name:
                ref_idx = dataset_idx if is_anchor else cross_idx
            elif source_dataset == target_name:
                ref_idx = cross_idx if is_anchor else dataset_idx
            else:
                ref_idx = dataset_idx if dataset_idx is not None else cross_idx
            if ref_idx is None:
                ref_idx = position
            pair_keys.append((source_dataset, int(ref_idx)))

        trimmed["pair_sources"] = [key[0] for key in pair_keys]
        return trimmed, labels_arr, pair_keys

    anchor_slice, anchor_labels, anchor_keys = _prepare_slice(
        anchor_entry,
        index_label_anchor_overlap,
        anchor_name,
        is_anchor=True,
    )
    target_slice, target_labels, target_keys = _prepare_slice(
        target_entry,
        index_label_target_overlap,
        target_name,
        is_anchor=False,
    )

    pair_key_to_id: Dict[Tuple[str, int], int] = {}
    next_pair_id = 0

    def _assign_pair_ids(slice_entry: Optional[Dict[str, object]], keys: List[Tuple[str, int]]) -> None:
        nonlocal next_pair_id
        if slice_entry is None or not keys:
            return
        pair_ids: List[int] = []
        for key in keys:
            if key not in pair_key_to_id:
                pair_key_to_id[key] = next_pair_id
                next_pair_id += 1
            pair_ids.append(pair_key_to_id[key])
        slice_entry["pair_ids"] = pair_ids
        pair_sources = slice_entry.get("pair_sources") or []
        metadata_list = slice_entry.get("metadata") or []
        for idx, meta in enumerate(metadata_list):
            if not isinstance(meta, dict):
                continue
            if idx < len(pair_ids):
                meta.setdefault("pair_id", pair_ids[idx])
            if idx < len(pair_sources):
                meta.setdefault("pair_label_source", pair_sources[idx])

    _assign_pair_ids(anchor_slice, anchor_keys)
    _assign_pair_ids(target_slice, target_keys)

    if (
        anchor_slice is None
        or target_slice is None
        or anchor_labels.size == 0
        or target_labels.size == 0
    ):
        empty = np.asarray([], dtype=np.int16)
        return None, None, empty, empty

    return anchor_slice, target_slice, anchor_labels, target_labels


def prepare_fusion_overlap_dataset_for_inference(
    anchor_entry: Optional[Dict[str, object]],
    target_entry: Optional[Dict[str, object]],
    *,
    method_id: str = "v1_simple",
) -> Optional[Dict[str, object]]:
    """Build the inference-time fused dataset from overlap embeddings."""
    method_key = method_id.lower()
    if anchor_entry is None and target_entry is None:
        return None

    def _entry_to_numpy(entry: Optional[Dict[str, object]]) -> Tuple[np.ndarray, int]:
        if entry is None:
            return np.empty((0, 0), dtype=np.float32), 0
        feats = entry.get("features")
        if feats is None:
            return np.empty((0, 0), dtype=np.float32), 0
        if torch is not None and isinstance(feats, Tensor):
            arr = feats.detach().cpu().numpy()
        elif isinstance(feats, np.ndarray):
            arr = feats
        else:
            arr = np.asarray(feats)
        if arr.ndim == 1:
            arr = arr.reshape(-1, arr.shape[0])
        arr = arr.astype(np.float32, copy=False)
        dim = arr.shape[1] if arr.ndim == 2 else 0
        return arr, dim

    def _build_lookup(entry: Optional[Dict[str, object]], tag: str) -> Tuple[Dict[Tuple[int, int], Dict[str, object]], int]:
        features_np, dim = _entry_to_numpy(entry)
        lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
        if entry is None or dim == 0 or features_np.size == 0:
            return lookup, dim
        labels_arr = entry.get("labels")
        if torch is not None and isinstance(labels_arr, Tensor):
            labels_np = labels_arr.detach().cpu().numpy().astype(np.int16, copy=False)
        elif isinstance(labels_arr, np.ndarray):
            labels_np = labels_arr.astype(np.int16, copy=False)
        else:
            labels_np = np.zeros(features_np.shape[0], dtype=np.int16)
        metadata_list = entry.get("metadata") or []
        for idx in range(features_np.shape[0]):
            row_col = _resolve_row_col(entry, idx)
            if row_col is None or row_col in lookup:
                continue
            coord = _resolve_coord(entry, idx)
            meta_entry = metadata_list[idx] if idx < len(metadata_list) and isinstance(metadata_list[idx], dict) else {}
            lookup[row_col] = {
                "feature": features_np[idx],
                "label": int(labels_np[idx]) if idx < len(labels_np) else 0,
                "coord": coord,
                "metadata": dict(meta_entry),
                "source": tag,
            }
        return lookup, dim

    anchor_lookup, dim_u = _build_lookup(anchor_entry, "anchor")
    target_lookup, dim_v = _build_lookup(target_entry, "target")
    if dim_u <= 0 and dim_v <= 0:
        return None
    if dim_u <= 0:
        dim_u = dim_v
    if dim_v <= 0:
        dim_v = dim_u
    max_dim = max(dim_u, dim_v)
    union_keys = sorted(set(anchor_lookup.keys()) | set(target_lookup.keys()))
    if not union_keys:
        return None

    def _pad(vec: Optional[np.ndarray], length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        out = np.zeros(length, dtype=np.float32)
        if vec is None or vec.size == 0:
            return out
        limit = min(length, vec.shape[0])
        out[:limit] = vec[:limit]
        return out

    features_rows: List[np.ndarray] = []
    labels_rows: List[int] = []
    metadata_rows: List[Dict[str, object]] = []
    row_cols: List[Tuple[int, int]] = []
    coords: List[Optional[Tuple[float, float]]] = []

    for key in union_keys:
        anchor_rec = anchor_lookup.get(key)
        target_rec = target_lookup.get(key)
        anchor_vec = _pad(anchor_rec["feature"] if anchor_rec else None, dim_u)
        target_vec = _pad(target_rec["feature"] if target_rec else None, dim_v)
        if method_key.endswith("strong"):
            u_common = _pad(anchor_rec["feature"] if anchor_rec else None, max_dim)
            v_common = _pad(target_rec["feature"] if target_rec else None, max_dim)
            diff_vec = np.abs(u_common - v_common)
            prod_vec = u_common * v_common
            norm_u = float(np.linalg.norm(u_common))
            norm_v = float(np.linalg.norm(v_common))
            cosine_val = float(np.dot(u_common, v_common) / (norm_u * norm_v + 1e-8)) if norm_u > 0 and norm_v > 0 else 0.0
            missing_flag = 0.0 if anchor_rec and target_rec else 1.0
            phi_parts = [
                anchor_vec,
                target_vec,
                diff_vec,
                prod_vec,
                np.asarray([cosine_val], dtype=np.float32),
                np.asarray([missing_flag], dtype=np.float32),
            ]
        else:
            phi_parts = [anchor_vec, target_vec]
        features_rows.append(np.concatenate(phi_parts, dtype=np.float32))
        label_val = 0
        if anchor_rec and anchor_rec.get("label") is not None:
            label_val = int(anchor_rec["label"])
        elif target_rec and target_rec.get("label") is not None:
            label_val = int(target_rec["label"])
        labels_rows.append(label_val)
        meta_entry: Dict[str, object] = {
            "row_col": key,
            "anchor_metadata": anchor_rec.get("metadata", {}) if anchor_rec else None,
            "target_metadata": target_rec.get("metadata", {}) if target_rec else None,
        }
        metadata_rows.append(meta_entry)
        row_cols.append(key)
        coord = anchor_rec.get("coord") if anchor_rec else None
        if coord is None and target_rec:
            coord = target_rec.get("coord")
        coords.append(coord if isinstance(coord, tuple) else None)

    if not features_rows:
        return None

    return {
        "features": np.vstack(features_rows).astype(np.float32, copy=False),
        "labels": np.asarray(labels_rows, dtype=np.int16),
        "metadata": metadata_rows,
        "row_cols": row_cols,
        "coords": coords,
        "dim_u": dim_u,
        "dim_v": dim_v,
        "name": f"fusion_{method_key}",
    }
