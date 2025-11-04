from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor
else:  # pragma: no cover - avoid optional torch dependency at runtime
    Tensor = None  # type: ignore[assignment]

from ..datasets import DatasetConfig, EmbeddingRecord, load_embedding_records, summarise_records
from ..overlaps import OverlapPair, OverlapSet, OverlapTile, load_overlap_pairs

AlignmentConfig = Any


@dataclass
class DatasetBundle:
    config: DatasetConfig
    records: List[EmbeddingRecord]
    summary: Dict[str, object]
    embeddings: Optional["Tensor"]
    class_prior: float

    def __post_init__(self):
        self._tile_index = {record.tile_id: record for record in self.records if record.tile_id}
        self._rowcol_index = {record.row_col: record for record in self.records if record.row_col is not None}
        self._coord_records: List[EmbeddingRecord] = [record for record in self.records if record.coord is not None]
        if self._coord_records:
            self._coord_array = np.asarray([record.coord for record in self._coord_records], dtype=np.float64)
        else:
            self._coord_array = None

    @property
    def name(self) -> str:
        return self.config.name

    def lookup_tile(self, tile_id: Optional[str]) -> Optional[EmbeddingRecord]:
        if tile_id is None:
            return None
        return self._tile_index.get(tile_id)

    def lookup_rowcol(self, row_col: Optional[Tuple[int, int]]) -> Optional[EmbeddingRecord]:
        if row_col is None:
            return None
        return self._rowcol_index.get(row_col)

    def lookup_coord(self, coord: Optional[Tuple[float, float]], max_distance: Optional[float] = None) -> Optional[EmbeddingRecord]:
        if coord is None or self._coord_array is None or self._coord_array.size == 0:
            return None
        point = np.asarray(coord, dtype=np.float64)
        if point.size < 2 or np.any(~np.isfinite(point[:2])):
            return None
        diffs = self._coord_array - point[:2]
        dist_sq = np.sum(diffs * diffs, axis=1)
        best_idx = int(np.argmin(dist_sq))
        best_dist = float(np.sqrt(dist_sq[best_idx]))
        if max_distance is not None and best_dist > max_distance:
            return None
        return self._coord_records[best_idx]

    def resolve(self, tile: OverlapTile, max_distance: Optional[float] = None) -> Optional[EmbeddingRecord]:
        record = self.lookup_tile(tile.tile_id)
        if record is not None:
            return record
        record = self.lookup_rowcol(tile.row_col)
        if record is not None:
            return record
        record = self.lookup_coord(tile.native_point, max_distance=max_distance)
        return record


@dataclass
class OverlapAlignmentPair:
    anchor_dataset: str
    anchor_record: EmbeddingRecord
    target_dataset: str
    target_record: EmbeddingRecord
    centroid: Optional[Tuple[float, float]]
    notes: Dict[str, object]

    def resolution_ratio(self) -> Optional[float]:
        a_res = self.anchor_record.pixel_resolution
        b_res = self.target_record.pixel_resolution
        if a_res is None or b_res is None or b_res == 0:
            return None
        return float(a_res) / float(b_res)

    def window_ratio(self) -> Optional[Tuple[float, float]]:
        a_win = self.anchor_record.window_size
        b_win = self.target_record.window_size
        if a_win is None or b_win is None:
            return None
        try:
            return float(a_win[0]) / float(b_win[0]), float(a_win[1]) / float(b_win[1])
        except Exception:
            return None

    def label_type(self) -> str:
        a_label = self.anchor_record.label
        b_label = self.target_record.label
        if a_label == 1 and b_label == 1:
            return "positive_common"
        if a_label == 1 and b_label != 1:
            return f"positive_{self.anchor_dataset}"
        if a_label != 1 and b_label == 1:
            return f"positive_{self.target_dataset}"
        return "unlabelled"


class OverlapAlignmentWorkspace:
    """Utility that organises overlap information and pairing metadata."""

    def __init__(self, config: AlignmentConfig):
        self.cfg = config
        self.integration_dataset_meta = {}
        if config.integration_metadata_path is not None:
            self.integration_dataset_meta = _load_integration_metadata(config.integration_metadata_path)

        self.datasets: Dict[str, DatasetBundle] = {}
        for dataset_cfg in config.datasets:
            dataset_meta = self.integration_dataset_meta.get(dataset_cfg.name)
            records = load_embedding_records(dataset_cfg, dataset_meta=dataset_meta)
            summary = summarise_records(records)
            embeddings = None
            if torch is not None:
                embeddings = torch.stack([torch.from_numpy(rec.embedding).float() for rec in records])
            labels = np.asarray([rec.label for rec in records], dtype=np.int32)
            if labels.size:
                class_prior = float(labels.mean())
            else:
                class_prior = dataset_cfg.class_prior if dataset_cfg.class_prior is not None else 0.1
            bundle = DatasetBundle(
                config=dataset_cfg,
                records=records,
                summary=summary,
                embeddings=embeddings,
                class_prior=class_prior,
            )
            self.datasets[bundle.name] = bundle

        self.overlap: Optional[OverlapSet] = None
        overlap_path = config.overlap_pairs_path
        augmented_path = getattr(config, "overlap_pairs_augmented_path", None)
        if config.use_positive_augmentation and augmented_path is not None:
            if augmented_path.exists():
                overlap_path = augmented_path
                print(f"[info] Using augmented overlap pairs from {augmented_path}")
            else:
                print(
                    f"[warn] Requested positive augmentation but augmented overlap file {augmented_path} "
                    "was not found; falling back to base overlap pairs."
                )
        if overlap_path is None:
            raise FileNotFoundError("Alignment config did not specify overlap_pairs_path; run integrate_stac first.")
        if not overlap_path.exists():
            raise FileNotFoundError(f"Overlap pairs path {overlap_path} does not exist. Ensure integrate_stac was executed and paths are correct.")
        self.overlap = load_overlap_pairs(overlap_path)

    def dataset_summaries(self) -> Dict[str, Dict[str, object]]:
        return {name: bundle.summary for name, bundle in self.datasets.items()}

    def overlap_summary(self) -> Optional[Dict[str, object]]:
        if self.overlap is None:
            return None
        return self.overlap.summary()

    def unresolved_pairs(self, max_coord_error: Optional[float] = None) -> int:
        if self.overlap is None:
            return 0
        unresolved = 0
        for pair in self.overlap.pairs:
            if self._resolve_pair(pair, max_coord_error=max_coord_error) is None:
                unresolved += 1
        return unresolved

    def iter_pairs(self, max_coord_error: Optional[float] = None) -> Iterable[OverlapAlignmentPair]:
        if self.overlap is None:
            return
        for pair in self.overlap.pairs:
            resolved_pair = self._resolve_pair(pair, max_coord_error=max_coord_error)
            if resolved_pair is not None:
                yield resolved_pair

    def pair_diagnostics(self, max_coord_error: Optional[float] = None, pairs: Optional[Sequence[OverlapAlignmentPair]] = None) -> Dict[str, object]:
        if pairs is None:
            pairs = list(self.iter_pairs(max_coord_error=max_coord_error)) if self.overlap else []
        total_pairs = len(self.overlap.pairs) if self.overlap is not None else 0
        resolution_ratios = [
            ratio for ratio in (pair.resolution_ratio() for pair in pairs) if ratio is not None
        ]
        window_ratios = [
            ratio for ratio in (pair.window_ratio() for pair in pairs) if ratio is not None
        ]
        return {
            "total_overlap_pairs": total_pairs,
            "resolved_pairs": len(pairs),
            "resolution_ratio_stats": _summary(resolution_ratios),
            "window_ratio_stats": None if not window_ratios else {
                "rows": _summary([r[0] for r in window_ratios]),
                "cols": _summary([r[1] for r in window_ratios]),
            },
        }

    def _resolve_pair(self, pair: OverlapPair, max_coord_error: Optional[float]) -> Optional[OverlapAlignmentPair]:
        bundle_a = self.datasets.get(pair.a.dataset)
        bundle_b = self.datasets.get(pair.b.dataset)
        if bundle_a is None or bundle_b is None:
            return None
        record_a = bundle_a.resolve(pair.a, max_distance=max_coord_error)
        record_b = bundle_b.resolve(pair.b, max_distance=max_coord_error)
        if record_a is None or record_b is None:
            return None
        return OverlapAlignmentPair(
            anchor_dataset=bundle_a.name,
            anchor_record=record_a,
            target_dataset=bundle_b.name,
            target_record=record_b,
            centroid=pair.centroid,
            notes=pair.notes,
        )


class OverlapAlignmentLabels:
    """
    Utility that aligns PN-labelled samples across datasets using resolved overlap pairs.

    This helper iterates over the resolved ``OverlapAlignmentPair`` entries exposed by an
    ``OverlapAlignmentWorkspace`` instance and emits only those pairs where at least one side
    corresponds to a PN-labelled tile. Each labelled tile is emitted at most once.
    """

    def __init__(
        self,
        workspace: "OverlapAlignmentWorkspace",
        pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]],
    ) -> None:
        self.workspace = workspace
        self.pn_label_maps = pn_label_maps

    def build_label_matches(
        self,
        *,
        pairs: Optional[Sequence[OverlapAlignmentPair]] = None,
        max_coord_error: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """
        Generate a list of cross-dataset label matches within the provided coordinate tolerance.

        Parameters
        ----------
        pairs:
            Optional precomputed ``OverlapAlignmentPair`` sequence. When omitted, the helper will
            iterate the workspace directly, applying ``max_coord_error``.
        max_coord_error:
            Distance tolerance used if pairs need to be resolved from the workspace.

        Returns
        -------
        List[Dict[str, object]]
            A list of label match payloads. Each item contains anchor/target metadata, PN label
            assignments (if available), and the Euclidean distance between the matched coordinates.
        """
        resolved_pairs: Sequence[OverlapAlignmentPair]
        if pairs is not None:
            resolved_pairs = pairs
        else:
            resolved_pairs = list(self.workspace.iter_pairs(max_coord_error=max_coord_error))

        matches: List[Dict[str, object]] = []
        seen_anchor: set[Tuple[str, str, int, int]] = set()
        seen_target: set[Tuple[str, str, int, int]] = set()

        for pair in resolved_pairs:
            anchor_info = self._resolve_record_info(pair.anchor_dataset, pair.anchor_record, seen_anchor)
            target_info = self._resolve_record_info(pair.target_dataset, pair.target_record, seen_target)

            if anchor_info is None and target_info is None:
                continue

            anchor_payload = anchor_info or self._basic_record_payload(pair.anchor_dataset, pair.anchor_record)
            target_payload = target_info or self._basic_record_payload(pair.target_dataset, pair.target_record)

            distance = self._coordinate_distance(
                anchor_payload.get("coord"),
                target_payload.get("coord"),
            )

            matches.append(
                {
                    "anchor_dataset": pair.anchor_dataset,
                    "target_dataset": pair.target_dataset,
                    "label_in_dataset_1": anchor_payload.get("pn_label"),
                    "label_in_dataset_2": target_payload.get("pn_label"),
                    "anchor": anchor_payload,
                    "target": target_payload,
                    "distance": distance,
                }
            )

        return matches

    def _resolve_record_info(
        self,
        dataset_name: str,
        record: "EmbeddingRecord",
        seen: set[Tuple[str, str, int, int]],
    ) -> Optional[Dict[str, object]]:
        pn_map = self.pn_label_maps.get(dataset_name)
        if not pn_map:
            return None

        row_col = getattr(record, "row_col", None)
        region = getattr(record, "region", None)
        key = self._normalise_key(dataset_name, region, row_col)
        if key is None:
            return None
        _, region_key, row, col = key
        pn_label = self._lookup_pn_label(pn_map, (region_key, row, col))
        if pn_label is None:
            return None

        if key in seen:
            # Already yielded this PN-labelled tile.
            return None
        seen.add(key)

        payload = self._basic_record_payload(dataset_name, record)
        payload["pn_label"] = pn_label
        payload["pn_key"] = key
        return payload

    @staticmethod
    def _normalise_key(
        dataset_name: str,
        region: Optional[object],
        row_col: Optional[Sequence[object]],
    ) -> Optional[Tuple[str, str, int, int]]:
        if row_col is None:
            return None
        try:
            row = int(row_col[0])
            col = int(row_col[1])
        except Exception:
            return None
        region_key = str(region).upper() if region is not None else "NONE"
        return (dataset_name, region_key, row, col)

    @staticmethod
    def _lookup_pn_label(
        pn_map: Dict[str, set[Tuple[str, int, int]]],
        key: Tuple[str, int, int],
    ) -> Optional[int]:
        region_key, row, col = key
        candidates: Sequence[Tuple[str, int, int]] = [
            (region_key, row, col),
            ("NONE", row, col),
        ]
        pos_set = pn_map.get("pos", set())
        neg_set = pn_map.get("neg", set())
        for candidate in candidates:
            if candidate in pos_set:
                return 1
            if candidate in neg_set:
                return 0
        return None

    @staticmethod
    def _basic_record_payload(dataset_name: str, record: "EmbeddingRecord") -> Dict[str, object]:
        coord = getattr(record, "coord", None)
        if coord is not None:
            try:
                coord_payload = (float(coord[0]), float(coord[1]))
            except Exception:
                coord_payload = None
        else:
            coord_payload = None

        try:
            index_val = int(getattr(record, "index"))
        except Exception:
            index_val = None

        payload: Dict[str, object] = {
            "dataset": dataset_name,
            "tile_id": getattr(record, "tile_id", None),
            "region": getattr(record, "region", None),
            "row_col": getattr(record, "row_col", None),
            "coord": coord_payload,
            "index": index_val,
            "record_label": int(getattr(record, "label", 0))
            if getattr(record, "label", None) is not None
            else None,
        }
        return payload

    @staticmethod
    def _coordinate_distance(
        coord_a: Optional[Tuple[float, float]],
        coord_b: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        if coord_a is None or coord_b is None:
            return None
        try:
            ax, ay = float(coord_a[0]), float(coord_a[1])
            bx, by = float(coord_b[0]), float(coord_b[1])
        except Exception:
            return None
        return float(np.linalg.norm([ax - bx, ay - by]))


def _summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _load_integration_metadata(path: Path) -> Dict[str, Dict[str, object]]:
    info: Dict[str, Dict[str, object]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        return info
    except json.JSONDecodeError:
        return info

    min_res_map = doc.get("dataset_min_resolution")
    if isinstance(min_res_map, dict):
        for key, value in min_res_map.items():
            dataset_key = str(key)
            try:
                resolution = float(value) if value is not None else None
            except Exception:
                resolution = None
            info.setdefault(dataset_key, {})["pixel_resolution"] = resolution
            info[dataset_key]["min_resolution"] = resolution

    window_spacing_map = doc.get("dataset_window_spacing")
    if isinstance(window_spacing_map, dict):
        for key, value in window_spacing_map.items():
            dataset_key = str(key)
            try:
                spacing = float(value) if value is not None else None
            except Exception:
                spacing = None
            info.setdefault(dataset_key, {})["window_spacing"] = spacing

    base_dir = path.parent

    def _resolve_path(candidate: Optional[str], dataset_root: Optional[str] = None) -> Optional[str]:
        if not candidate:
            return None
        try:
            candidate_path = Path(candidate)
        except (TypeError, ValueError):
            return None
        if candidate_path.is_absolute():
            return str(candidate_path)
        if dataset_root:
            root_path = Path(dataset_root)
            resolved = (root_path / candidate_path).resolve()
            if resolved.exists():
                return str(resolved)
        return str((base_dir / candidate_path).resolve())

    datasets_payload = doc.get("datasets")
    if isinstance(datasets_payload, list):
        for entry in datasets_payload:
            if not isinstance(entry, dict):
                continue
            dataset_id = entry.get("dataset_id") or entry.get("dataset") or entry.get("name")
            if not dataset_id:
                continue
            dataset_key = str(dataset_id)
            meta = info.setdefault(dataset_key, {})
            root_value = entry.get("root")
            if isinstance(root_value, str) and root_value:
                meta.setdefault("root", root_value)
            # store boundaries if present so downstream plotting can use them
            boundaries = entry.get("boundaries")
            if isinstance(boundaries, dict) and boundaries:
                meta["boundaries"] = copy.deepcopy(boundaries)

    overlap_mask_path = doc.get("study_area_overlap_mask")
    fallback_entry: Optional[Dict[str, object]] = None
    resolved_overlap_mask = _resolve_path(overlap_mask_path)
    if resolved_overlap_mask:
        fallback_entry = {
            "path": resolved_overlap_mask,
            "filename": Path(resolved_overlap_mask).name,
            "role": "project",
            "path_resolved": resolved_overlap_mask,
        }

    if fallback_entry is not None:
        for dataset_key, meta in info.items():
            boundaries = meta.get("boundaries")
            project_entries: Optional[Sequence[Dict[str, object]]] = None
            if isinstance(boundaries, dict):
                project_entries = boundaries.get("project")
            if project_entries:
                continue
            meta.setdefault("boundaries", {})["project"] = [copy.deepcopy(fallback_entry)]

    return info
