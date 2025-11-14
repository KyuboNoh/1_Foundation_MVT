"""
Utility classes and functions for transformer-aggregated DCCA.

This module contains:
- Dataset classes for transformer training
- Batch samplers for PU learning  
- Data collation functions for variable-length sequences
- Helper functions for logits/probability conversion
"""

from __future__ import annotations
import random
from typing import List, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler


def _ensure_logits(y: torch.Tensor) -> torch.Tensor:
    """
    Accepts network output that might be logits (preferred) or probs in [0,1].
    Returns logits tensor of shape [B].
    """
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    # if looks like probs, convert to logits
    if torch.isfinite(y).all() and y.min() >= 0.0 and y.max() <= 1.0:
        y = torch.logit(y.clamp(1e-6, 1 - 1e-6))
    return y


class TargetSetDataset(Dataset):
    """Dataset for transformer aggregator training with variable-length target sets."""
    
    def __init__(self, anchor_vecs: Sequence[torch.Tensor], target_stack_per_anchor: Sequence[torch.Tensor], metadata: Sequence[Dict[str, object]]):
        self.anchor_vecs = anchor_vecs
        self.target_stack_per_anchor = target_stack_per_anchor
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.anchor_vecs)

    def __getitem__(self, idx: int):
        return self.anchor_vecs[idx], self.target_stack_per_anchor[idx], self.metadata[idx]


class PUBatchSampler(Sampler[list]):
    """Batch sampler for PU learning that ensures balanced positive/unlabeled sampling."""
    
    def __init__(self, pos_idx, unl_idx, k_pos, k_unl, epoch_len=None, seed=0):
        self.pos_idx = pos_idx
        self.unl_idx = unl_idx
        self.k_pos = k_pos
        self.k_unl = k_unl
        self.seed = seed
        # how many batches per epoch (by unlabeled supply)
        self.epoch_len = epoch_len or (len(unl_idx) // k_unl)

    def __iter__(self):
        rng = random.Random(self.seed)
        # reshuffle every epoch
        pos = self.pos_idx[:] ; unl = self.unl_idx[:]
        rng.shuffle(pos) ; rng.shuffle(unl)
        # cycle positives if we run out
        p_ptr = 0
        for b in range(self.epoch_len):
            if (b * self.k_unl + self.k_unl) > len(unl):
                break
            batch_unl = unl[b*self.k_unl : (b+1)*self.k_unl]
            # take k_pos positives, cycling if necessary
            if p_ptr + self.k_pos > len(pos):
                rng.shuffle(pos)
                p_ptr = 0
            batch_pos = pos[p_ptr : p_ptr + self.k_pos]
            p_ptr += self.k_pos
            yield batch_pos + batch_unl

    def __len__(self):
        return self.epoch_len


def _collate_target_sets(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]], use_positional_encoding: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Collate function for variable-length target sets in transformer training.
    
    Args:
        batch: List of (anchor_vec, target_stack, metadata) tuples
        use_positional_encoding: Whether to compute positional encodings
        
    Returns:
        (anchor_batch, target_batch, mask_batch, pos_batch)
    """
    anchors, target_stack_per_anchor, metadata = zip(*batch)
    anchor_batch = torch.stack(anchors, dim=0)
    max_len = max(t.size(0) for t in target_stack_per_anchor)
    feature_dim = target_stack_per_anchor[0].size(1)
    target_batch = anchor_batch.new_zeros((len(batch), max_len, feature_dim))
    mask_batch = torch.ones(len(batch), max_len, dtype=torch.bool)
    pos_batch: Optional[torch.Tensor] = None
    if use_positional_encoding:
        pos_batch = anchor_batch.new_zeros((len(batch), max_len, 2))
    for idx, (stack, meta) in enumerate(zip(target_stack_per_anchor, metadata)):
        length = stack.size(0)
        target_batch[idx, :length] = stack
        mask_batch[idx, :length] = False
        if use_positional_encoding and pos_batch is not None:
            anchor_coord = meta.get("anchor_coord")
            target_coords = meta.get("target_coords") or []
            diffs: List[List[float]] = []
            for pos_idx in range(length):
                coord = target_coords[pos_idx] if pos_idx < len(target_coords) else None
                if anchor_coord is not None and coord is not None:
                    diffs.append([float(coord[0] - anchor_coord[0]), float(coord[1] - anchor_coord[1])])
                else:
                    diffs.append([0.0, 0.0])
            pos_batch[idx, :length] = torch.tensor(diffs, dtype=anchor_batch.dtype)
    return anchor_batch, target_batch, mask_batch, pos_batch