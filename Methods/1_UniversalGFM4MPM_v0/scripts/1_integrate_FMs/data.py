from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EmbeddingRecord:
    embedding: np.ndarray
    label: int
    tile_id: str
    coord: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None


class EmbeddingDataset(Dataset):
    """Simple dataset wrapping pre-computed embeddings and metadata."""

    def __init__(self, records: Sequence[EmbeddingRecord], class_prior: Optional[float] = None):
        self.records = list(records)
        self.class_prior = class_prior
        if not self.records:
            raise ValueError("EmbeddingDataset expects at least one record.")
        emb_dim = self.records[0].embedding.shape[-1]
        for rec in self.records:
            if rec.embedding.shape[-1] != emb_dim:
                raise ValueError("All embeddings must share identical dimensionality.")
        self._embedding_dim = emb_dim

    def __len__(self) -> int:
        return len(self.records)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        emb = torch.from_numpy(rec.embedding).float()
        label = torch.tensor(rec.label, dtype=torch.long)
        meta = {
            "tile_id": rec.tile_id,
            "coord": rec.coord,
            "metadata": rec.metadata,
        }
        return emb, label, meta


def load_npz_embeddings(path: Path, positive_label: int = 1) -> List[EmbeddingRecord]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        raise ValueError(
            f"Embedding file {path} is a plain .npy array. Convert to .npz with 'embeddings', 'labels', etc., before using integrate_fms."
        )
    embeddings = data["embeddings"]
    labels = data["labels"].astype(int)
    tile_ids = data["tile_ids"].astype(str) if "tile_ids" in data else [f"tile_{i}" for i in range(len(embeddings))]
    coords = data.get("coords")
    metas = data.get("metadata")
    records: List[EmbeddingRecord] = []
    for idx, emb in enumerate(embeddings):
        coord_tuple = tuple(coords[idx]) if coords is not None else None
        metadata = None
        if metas is not None:
            entry = metas[idx]
            if isinstance(entry, dict):
                metadata = entry
            else:
                try:
                    metadata = entry.item()
                except Exception:
                    metadata = entry
        label = int(labels[idx])
        label = 1 if label == positive_label else 0
        records.append(EmbeddingRecord(embedding=emb, label=label, tile_id=str(tile_ids[idx]), coord=coord_tuple, metadata=metadata))
    return records


def load_records(embedding_path: Path, metadata_path: Optional[Path] = None, region_filter: Optional[Iterable[str]] = None) -> List[EmbeddingRecord]:
    suffix = embedding_path.suffix.lower()
    if suffix != ".npz":
        raise ValueError(f"Unsupported embedding file extension for {embedding_path}. Expected '.npz'.")
    records = load_npz_embeddings(embedding_path)

    original_count = len(records)
    if metadata_path is None:
        return records
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata_doc = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse metadata file {metadata_path}: {exc}") from exc

    region_lookup: Dict[str, str] = {}
    if isinstance(metadata_doc, dict):
        for region, tiles in metadata_doc.get("regions", {}).items():
            if isinstance(tiles, dict):
                for feat in tiles.get("feature_paths", {}).values():
                    for path in feat:
                        region_lookup[Path(path).stem] = region

    if region_filter:
        region_filter_upper = {str(region).upper() for region in region_filter}
        filtered: List[EmbeddingRecord] = []
        for rec in records:
            region_value = None
            if isinstance(rec.metadata, dict):
                region_value = rec.metadata.get("region")
            if region_value is not None:
                region_upper = str(region_value).upper()
            else:
                region_upper = region_lookup.get(rec.tile_id, "GLOBAL").upper()
            if region_upper in region_filter_upper:
                filtered.append(rec)
        records = filtered
        print(f"[info] load_records: dataset={embedding_path} regions filter={sorted(region_filter_upper)} original={original_count} kept={len(records)}")
    return records
