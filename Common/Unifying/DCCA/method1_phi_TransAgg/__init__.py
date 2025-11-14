"""
Transformer-Aggregated DCCA with Phi Feature Construction (Method 1)

This module contains the transformer-based aggregation approach for DCCA with
phi feature construction: φ(u,v,b_missing) = [u, v, |u-v|, u*v, cos(u,v), b_missing]

Key Components:
- PNHeadUnified: Unified PN classifier using phi features
- CrossAttentionAggregator: Transformer aggregator for set-to-set mapping
- AggregatorTargetHead: Wrapper combining aggregator + projection
- build_phi: Core phi feature construction function
- Training and inference functions for transformer-aggregated DCCA
"""

from .models import (
    PNHeadUnified,
    PNHeadAOnly,
    CrossAttentionAggregator,
    AggregatorTargetHead,
    cosine_sim,
    build_phi,
)

from .training import (
    fit_unified_head_OVERLAP_from_uv,
    _train_transformer_aggregator,
    nnpu_basic_loss_from_logits,
    nnpu_weighted_loss_from_logits,
)

from .inference import (
    run_overlap_inference_gAB_from_pairs,
    build_cls2_dataset_from_dcca_pairs,
    infer_in_bc,
    infer_outside_bc,
    _apply_transformer_target_head,
)

from .utils import (
    TargetSetDataset,
    PUBatchSampler,
    _collate_target_sets,
    _ensure_logits,
)

__all__ = [
    # Core models
    "PNHeadUnified",
    "PNHeadAOnly", 
    "CrossAttentionAggregator",
    "AggregatorTargetHead",
    "cosine_sim",
    "build_phi",
    
    # Training functions
    "fit_unified_head_OVERLAP_from_uv",
    "_train_transformer_aggregator",
    "nnpu_basic_loss_from_logits",
    "nnpu_weighted_loss_from_logits",
    
    # Inference functions
    "run_overlap_inference_gAB_from_pairs",
    "build_cls2_dataset_from_dcca_pairs", 
    "infer_in_bc",
    "infer_outside_bc",
    "_apply_transformer_target_head",
    
    # Utilities
    "TargetSetDataset",
    "PUBatchSampler",
    "_collate_target_sets",
    "_ensure_logits",
]