from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from .config import AlignmentConfig
from .datasets import EmbeddingRecord, DatasetConfig, load_embedding_records, summarise_records
from .overlaps import OverlapPair, OverlapSet, OverlapTile, load_overlap_pairs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor


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
        if overlap_path is not None:
            if overlap_path.exists():
                self.overlap = load_overlap_pairs(overlap_path)
            else:
                print(f"[warn] Overlap pairs path {overlap_path} does not exist.")

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

    return info
