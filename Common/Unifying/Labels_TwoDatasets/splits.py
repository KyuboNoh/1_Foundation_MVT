"""Train/validation splitting utilities that maintain overlap integrity."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def _coerce_int_label(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        return 1 if int(value) > 0 else 0
    except Exception:
        return None


def _normalise_pair_id(value: object, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _overlap_split_indices(
    dataset: Dict[str, object],
    validation_fraction: float,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split indices while keeping overlap pairs together.

    The dataset is expected to contain ``pair_groups`` (one entry per sample)
    describing which anchor/target rows originate from the same overlap label,
    and optionally ``pair_sources`` describing which dataset supplied the PN
    label. Split decisions are performed on the pair level: all samples that
    share a group are forced into the same fold so that anchor / target views
    never diverge between train and validation.
    """

    features = dataset.get("features")
    labels = dataset.get("labels")
    if features is None or labels is None:
        return None

    labels = np.asarray(labels)
    total = int(labels.shape[0])
    if total <= 1 or np.unique(labels).size <= 1:
        return None

    pair_groups_raw = dataset.get("pair_groups")
    if pair_groups_raw is None or len(pair_groups_raw) != total:
        # Fallback: treat every entry as its own independent group.
        pair_groups = np.arange(total, dtype=int)
    else:
        pair_groups = np.asarray(pair_groups_raw)

    pair_sources_raw = dataset.get("pair_sources") or [None] * total
    if len(pair_sources_raw) != total:
        pair_sources_raw = list(pair_sources_raw) + [None] * (total - len(pair_sources_raw))
    pair_sources_raw = pair_sources_raw[:total]

    # Build mapping from group id -> sample indices.
    group_to_indices: Dict[int, List[int]] = defaultdict(list)
    group_to_source: Dict[int, Optional[str]] = {}
    for idx in range(total):
        group_id = _normalise_pair_id(pair_groups[idx], fallback=idx)
        group_to_indices[group_id].append(idx)
        existing = group_to_source.get(group_id)
        source_val = pair_sources_raw[idx]
        if existing is None and source_val is not None:
            group_to_source[group_id] = str(source_val)

    unique_groups = np.asarray(sorted(group_to_indices.keys()), dtype=int)
    if unique_groups.size <= 1:
        return None

    # Determine a stratification label per group using both the PN label and source.
    strat_tokens: List[str] = []
    for group_id in unique_groups:
        member_indices = group_to_indices[group_id]
        label_val = None
        for idx in member_indices:
            label_val = _coerce_int_label(labels[idx])
            if label_val is not None:
                break
        if label_val is None:
            label_val = 0
        source_tag = group_to_source.get(group_id) or "unknown"
        strat_tokens.append(f"{label_val}:{source_tag}")
    strat_tokens = np.asarray(strat_tokens, dtype=object)
    stratify = strat_tokens if np.unique(strat_tokens).size > 1 else None

    validation_fraction = max(0.0, min(float(validation_fraction), 0.9))
    if validation_fraction <= 0.0 or unique_groups.size < 4:
        train_groups = unique_groups
        val_groups = np.empty(0, dtype=int)
    else:
        train_groups, val_groups = train_test_split(
            unique_groups,
            test_size=validation_fraction,
            stratify=stratify,
            random_state=int(seed),
        )

    train_indices = (
        np.concatenate(
            [np.asarray(group_to_indices[int(g)], dtype=int) for g in train_groups]
        )
        if train_groups.size
        else np.empty(0, dtype=int)
    )
    val_indices = (
        np.concatenate(
            [np.asarray(group_to_indices[int(g)], dtype=int) for g in val_groups]
        )
        if val_groups.size
        else np.empty(0, dtype=int)
    )

    print(f"Split {total} samples into {train_indices.size} train and {val_indices.size} val maintaining overlap integrity."    )
    return train_indices, val_indices


__all__ = [
    "_overlap_split_indices",
]
