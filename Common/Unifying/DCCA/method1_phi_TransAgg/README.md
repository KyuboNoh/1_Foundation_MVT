# Transformer-Aggregated DCCA with Phi Features (Method 1)

This module implements the transformer-based aggregation approach for DCCA with phi feature construction.

## Overview

The key innovation is the phi feature construction: 
```
φ(u,v,b_missing) = [u, v, |u-v|, u*v, cos(u,v), b_missing]
```

This creates a rich 4d+2 dimensional feature vector that captures:
- **u**: Anchor projection 
- **v**: Target projection (aggregated via transformer)
- **|u-v|**: Element-wise absolute difference
- **u*v**: Element-wise product  
- **cos(u,v)**: Cosine similarity
- **b_missing**: Missing target indicator

## Architecture

### Core Components

1. **CrossAttentionAggregator**: Transformer for set-to-set mapping
2. **PNHeadUnified**: Unified classifier using phi features
3. **AggregatorTargetHead**: Combines aggregator + projection
4. **build_phi()**: Core phi feature construction

### Training Pipeline

```python
# 1. Train transformer aggregator with DCCA loss
proj_anchor, target_head, history = _train_transformer_aggregator(
    anchor_vecs, target_stack_per_anchor, pair_metadata,
    device=device, batch_size=32, steps=50, lr=1e-3
)

# 2. Build U,V,Bmiss for cls-2 training  
U, V, Bmiss = build_cls2_dataset_from_dcca_pairs(
    anchor_vecs, target_stack_per_anchor, 
    proj_anchor, target_head, device
)

# 3. Train unified head with knowledge distillation
gAB = fit_unified_head_OVERLAP_from_uv(
    gA_teacher, data_loader_uv, d_u=projection_dim,
    device=device, steps=10
)
```

### Inference Pipeline

```python
# Overlap inference using transformer aggregation + phi features
results = run_overlap_inference_gAB_from_pairs(
    anchor_vecs=anchor_vecs,
    target_stack_per_anchor=target_stack_per_anchor,
    pair_metadata=pair_metadata,
    projector_a=proj_anchor,
    projector_b=target_head,  # AggregatorTargetHead
    gAB=unified_classifier,
    device=device,
    passes=10  # MC dropout passes
)
```

## Key Features

- **Transformer Aggregation**: Handles variable-length target sets per anchor
- **Phi Feature Construction**: Rich interaction features between u and v
- **Knowledge Distillation**: Unified head learns from A-only teacher
- **Focal Loss**: Emphasizes hard positive examples during training
- **Multi-view Inference**: Handles both overlap (A+B) and non-overlap (A-only) regions

## Integration

This module integrates with the existing DCCA framework:
- Uses existing `ProjectionHead` and `dcca_loss` from `Common.Unifying.DCCA`
- Compatible with `OverlapAlignmentWorkspace` and `AlignmentConfig`
- Maintains backward compatibility with non-transformer DCCA approaches

## Usage Example

```python
from Common.Unifying.DCCA.method1_phi_TransAgg import (
    PNHeadUnified, CrossAttentionAggregator, build_phi,
    fit_unified_head_OVERLAP_from_uv, run_overlap_inference_gAB_from_pairs
)

# The phi features are automatically constructed inside PNHeadUnified:
gAB = PNHeadUnified(d=projection_dim, hidden=256)
prediction = gAB(u_anchor, v_target, b_missing_mask)
```