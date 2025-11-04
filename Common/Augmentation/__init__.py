from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_augmented_embeddings(path: Path, region_filter: Optional[Sequence[str]]) -> Tuple[Dict[str, List[np.ndarray]], int]:
    mapping: Dict[str, List[np.ndarray]] = {}
    count = 0
    region_filter_set: Optional[set[str]] = None
    if region_filter:
        region_filter_set = {str(entry).upper() for entry in region_filter}
    try:
        with np.load(path, allow_pickle=True) as bundle:
            embeddings = bundle.get("embeddings")
            metadata = bundle.get("metadata")
            flags = bundle.get("is_augmented")
            tile_ids = bundle.get("tile_ids")
            if embeddings is None or metadata is None or flags is None or tile_ids is None:
                print(f"[warn] positive augmentation bundle {path} is missing required arrays; ignoring.")
                return {}, 0
            for emb, meta, flag, tile_id in zip(embeddings, metadata, flags, tile_ids):
                if not bool(flag):
                    continue
                meta_dict: Dict[str, object] = {}
                if isinstance(meta, dict):
                    meta_dict = meta
                    try:
                        meta_dict = meta.item()
                    except Exception:
                        meta_dict = {}
                region_value = str(meta_dict.get("region") or meta_dict.get("Region") or "").upper()
                if region_filter_set and region_value and region_value not in region_filter_set:
                    continue
                source_tile_id = meta_dict.get("source_tile_id")
                if source_tile_id is None and isinstance(tile_id, str):
                    source_tile_id = tile_id.split("__")[0]
                if source_tile_id is None:
                    continue
                emb_arr = np.asarray(emb, dtype=np.float32)
                mapping.setdefault(str(source_tile_id), []).append(emb_arr)
                count += 1
    except FileNotFoundError:
        print(f"[warn] positive augmentation file {path} not found.")
        return {}, 0
    except Exception as exc:
        print(f"[warn] Failed to load positive augmentation file {path}: {exc}")
        return {}, 0
    return mapping, count


def _print_augmentation_usage(
    anchor_name: str,
    target_name: str,
    loaded_counts: Dict[str, int],
    usage_stats: Dict[str, int],
    total_pairs: int,
) -> None:
    anchor_loaded = loaded_counts.get(anchor_name, 0)
    target_loaded = loaded_counts.get(target_name, 0)
    anchor_used = int(usage_stats.get("anchor_augmented_pairs", 0))
    target_used = int(usage_stats.get("target_augmented_pairs", 0))
    print(
        "[info] augmentation usage: "
        f"{anchor_name} loaded={anchor_loaded} used={anchor_used}; "
        f"{target_name} loaded={target_loaded} used={target_used}; "
        f"total_pairs_with_aug={total_pairs}"
    )


__all__ = [
    "_load_augmented_embeddings",
    "_print_augmentation_usage",
]
