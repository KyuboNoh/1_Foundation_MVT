"""
Core models for transformer-aggregated DCCA with phi features.

This module contains the neural network architectures for:
- Unified PN classification using phi features  
- Transformer-based aggregation for set-to-set mapping
- Phi feature construction: [u, v, |u-v|, u*v, cos(u,v), b_missing]
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_sim(u, v, eps=1e-6):
    """Compute cosine similarity between u and v tensors."""
    return F.cosine_similarity(u, v, dim=-1, eps=eps).unsqueeze(-1)


def build_phi(u, v, b_missing):
    """
    Build phi feature vector: [u, v, |u-v|, u*v, cos(u,v), b_missing]
    
    Args:
        u: Anchor projection [B, d]
        v: Target projection [B, d] 
        b_missing: Missing target indicator [B, 1] in {0,1}
        
    Returns:
        Concatenated feature vector [B, 4d + 2]
    """
    # u,v: [B, d], b_missing: [B,1] in {0,1}
    if v is None:  # A-only fallback (outside BC)
        v = torch.zeros_like(u)
        b_missing = torch.ones(u.size(0), 1, device=u.device)
    feats = [u, v, torch.abs(u - v), u * v, cosine_sim(u, v), b_missing]
    return torch.cat(feats, dim=-1)  # [B, 4d + 2]


class PNHeadUnified(nn.Module):
    """Unified PN head on φ(u,v,mask) features."""
    
    def __init__(self, d, hidden=256):
        super().__init__()
        in_dim = 4*d + 2  # [u, v, |u-v|, u*v, cos, mask]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, u, v=None, b_missing=None):
        if b_missing is None:
            b_missing = torch.zeros(u.size(0), 1, device=u.device)
        phi = build_phi(u, v, b_missing)
        return torch.sigmoid(self.net(phi)).squeeze(-1)


class PNHeadAOnly(nn.Module):
    """A-only PN head for regions without target data."""
    
    def __init__(self, d, hidden=256, out_logits=True, prior_pi=None):
        super().__init__()
        self.out_logits = out_logits
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)
        # Optional: initialize bias to the prior logit to avoid all-negative start
        if prior_pi is not None and 0.0 < prior_pi < 1.0:
            with torch.no_grad():
                self.out.bias.fill_(math.log(prior_pi / (1.0 - prior_pi)))

    def forward(self, u):
        z = self.out(self.mlp(u)).squeeze(-1)  # logits
        return z if self.out_logits else torch.sigmoid(z)  # toggle if needed


class CrossAttentionAggregator(nn.Module):
    """Cross-attention aggregator for set-to-set mapping in transformer DCCA."""
    
    def __init__(
        self,
        anchor_dim: int,
        target_dim: int,
        agg_dim: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        if agg_dim % num_heads != 0:
            raise ValueError("Aggregator hidden dimension must be divisible by the number of heads.")
        self.use_positional_encoding = bool(use_positional_encoding)
        pos_dim = 2 if self.use_positional_encoding else 0
        kv_dim = target_dim + pos_dim
        self.query_proj = nn.Linear(anchor_dim, agg_dim)
        self.key_proj = nn.Linear(kv_dim, agg_dim)
        self.value_proj = nn.Linear(kv_dim, agg_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(agg_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(max(1, num_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(agg_dim) for _ in range(len(self.layers))])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(agg_dim, agg_dim)

    def forward(
        self,
        anchor_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_positional_encoding and pos_encoding is not None:
            kv_input = torch.cat([target_batch, pos_encoding], dim=-1)
        else:
            kv_input = target_batch
        query = self.query_proj(anchor_batch).unsqueeze(1)
        keys = self.key_proj(kv_input)
        values = self.value_proj(kv_input)
        x = query
        for attn_layer, norm in zip(self.layers, self.norms):
            attn_out, _ = attn_layer(x, keys, values, key_padding_mask=key_padding_mask)
            x = norm(x + self.dropout(attn_out))
        fused = self.output_proj(x.squeeze(1))
        return fused


class AggregatorTargetHead(nn.Module):
    """Wrapper combining aggregator + target projection head."""
    
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