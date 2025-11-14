"""
Inference functions for transformer-aggregated DCCA with phi features.

This module contains:
- Overlap inference using transformer aggregation and phi features
- Dataset construction for cls-2 training
- Multi-view inference functions (in BC vs outside BC)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .models import PNHeadUnified, build_phi
from .utils import TargetSetDataset, _collate_target_sets


@torch.no_grad()
def build_cls2_dataset_from_dcca_pairs(
    anchor_vecs,                 # List[Tensor], length = num_pairs
    target_stack_per_anchor,     # List[Tensor], each [M_i, d_B]
    projector_a,                 # nn.Module, A head
    projector_b,                 # nn.Module, B head (AggregatorTargetHead if used)
    device: torch.device,
):
    """
    Build U, V, Bmiss tensors for cls-2 training from DCCA pairs.
    
    Returns:
        (U, V, Bmiss) tensors for overlap region classification
    """
    u_list, v_list, bmiss_list = [], [], []
    for a_emb, b_stack in zip(anchor_vecs, target_stack_per_anchor):
        a = a_emb.to(device).unsqueeze(0)          # [1, dA]
        if hasattr(projector_b, "forward") and b_stack is not None and b_stack.numel() > 0:
            # aggregator uses (anchor, target_stack)
            v = projector_b(a, b_stack.to(device).unsqueeze(0)).squeeze(0)  # [d]
            b_missing = 0.0
        else:
            v = torch.zeros_like(a_emb, device=device)  # fallback
            b_missing = 1.0
        u = projector_a(a).squeeze(0)                   # [d]
        u_list.append(u)
        v_list.append(v)
        bmiss_list.append(torch.tensor([b_missing], device=device, dtype=torch.float32))
    U = torch.stack(u_list)                # [N_A, d]
    V = torch.stack(v_list)                # [N_A, d]
    Bmiss = torch.stack(bmiss_list)        # [N_A, 1]
    return U, V, Bmiss


def run_overlap_inference_gAB_from_pairs(
    *,
    anchor_vecs: List[torch.Tensor],
    target_stack_per_anchor: List[torch.Tensor],
    pair_metadata: List[Dict[str, object]],
    projector_a: nn.Module,
    projector_b: nn.Module,      # AggregatorTargetHead if you trained with the transformer aggregator; ProjectionHead otherwise
    gAB: nn.Module,
    device: torch.device,
    cfg,  # AlignmentConfig
    output_dir: Path,
    run_logger,  # _RunLogger
    passes: int = 10,
    target_vecs: Optional[List[torch.Tensor]] = None,  # optional fast-path if you already have fused B vectors
    batch_size: int = 256,
    pos_crd: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """
    Inference on overlap using the SAME pairing/aggregation as DCCA:
      U = projector_a(anchor_vec)
      V = projector_b(anchor_vec, target_stack)  # if aggregator-wrapped
        or projector_b(target_vec)               # if plain ProjectionHead and target_vecs provided
    """

    # ----- 1) Build U (A-projected) in batches -----
    run_logger.log(f"[inference-gAB] Preparing A/B projections for {len(anchor_vecs)} overlap pairs")
    A_all = torch.stack([a.detach().clone() for a in anchor_vecs]).to(device)  # [N, dA]

    def _proj_in_batches(X, head):
        outs = []
        for s in range(0, X.size(0), batch_size):
            outs.append(head(X[s:s+batch_size]).detach())
        return torch.cat(outs, dim=0)

    with torch.no_grad():
        projector_a.eval()
        U = _proj_in_batches(A_all, projector_a)  # [N, d]

    # ----- 2) Build V (B-projected) using the SAME path as training -----
    with torch.no_grad():
        projector_b.eval()
        if hasattr(projector_b, "aggregator"):  # AggregatorTargetHead path
            # Use the same helper you used during training to handle variable-length stacks
            V_list = _apply_transformer_target_head(
                projector_b,
                anchor_vecs,
                target_stack_per_anchor,
                pair_metadata,
                batch_size=batch_size,
                device=device,
            )
            if not V_list or len(V_list) != len(anchor_vecs):
                raise RuntimeError("[inference-gAB] Aggregator output mismatch; got "
                                   f"{0 if not V_list else len(V_list)} for N={len(anchor_vecs)}.")
            V = torch.stack([v.detach().to(device) for v in V_list])  # [N, d]
        else:
            # Plain ProjectionHead: expect pre-fused target vectors (same as DCCA input)
            if target_vecs is None or len(target_vecs) != len(anchor_vecs):
                raise RuntimeError("[inference-gAB] target_vecs must be provided (and match N) "
                                   "when projector_b is a plain ProjectionHead.")
            T_all = torch.stack([t.detach().clone() for t in target_vecs]).to(device)  # [N, dB]
            V = _proj_in_batches(T_all, projector_b)  # [N, d]

    # ----- 3) Coordinates & (optional) labels pulled from the DCCA pairing metadata -----
    coords = []
    labels = None
    maybe_labels = []
    for meta in pair_metadata:
        c = meta.get("anchor_coord")
        coords.append(c if c is not None else (None, None))
        # Try to recover labels if present; otherwise keep None
        lb = meta.get("anchor_label", None)
        maybe_labels.append(int(lb) if isinstance(lb, (int, np.integer)) else None)
    if any(l is not None for l in maybe_labels):
        labels = np.array([(-1 if l is None else l) for l in maybe_labels], dtype=np.int32)  # -1 means unknown

    run_logger.log(f"[inference-gAB] Projected shapes: U={tuple(U.shape)}, V={tuple(V.shape)}")

    # ----- 4) MC-Dropout inference on the unified head -----
    model_dir = output_dir / "gAB"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if unified model outputs probabilities or logits
    outputs_probs = isinstance(gAB, PNHeadUnified)  # PNHeadUnified outputs probabilities
    
    gAB.train()  # enable dropout
    preds_mc = []

    with torch.no_grad():
        N = U.size(0)
        zeros = torch.zeros(N, 1, device=device)  # B is present → b_missing=0
        for _ in range(int(max(1, passes))):
            batch_out = []
            for s in range(0, N, batch_size):
                u_b = U[s:s+batch_size]
                v_b = V[s:s+batch_size]
                bm_b = zeros[s:s+batch_size]
                
                output = gAB(u_b, v_b, bm_b)  # [B]
                
                if outputs_probs:
                    # Model already outputs probabilities
                    probs = output
                else:
                    # Model outputs logits, apply sigmoid
                    probs = torch.sigmoid(output)
                
                batch_out.append(probs.detach().cpu())
            preds_mc.append(torch.cat(batch_out, dim=0).numpy())

    gAB.eval()

    preds_mc = np.stack(preds_mc, axis=0)          # (passes, N)
    mean_pred = preds_mc.mean(axis=0)              # (N,)
    std_pred  = preds_mc.std(axis=0)               # (N,)

    # ----- 5) Save results -----
    np.save(model_dir / "predictions_mean.npy", mean_pred)
    np.save(model_dir / "predictions_std.npy",  std_pred)
    np.save(model_dir / "coordinates.npy",      np.asarray(coords, dtype=object))

    _create_inference_plots(
        model_dir=model_dir,
        model_name="gAB",
        coords=coords,
        mean_pred=mean_pred,
        std_pred=std_pred,
        pos_crd=pos_crd
    )

    summary = _compute_inference_summary(
        model_name="gAB",
        mean_pred=mean_pred,
        std_pred=std_pred,
        labels=labels
    )
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    run_logger.log(f"[inference-gAB] Saved results to {model_dir}")

    return {
        "predictions_mean": mean_pred,
        "predictions_std": std_pred,
        "coordinates": coords,
        "labels": labels,
        "summary": summary,
    }


@torch.no_grad()
def infer_in_bc(encA, encB, agg, pA, pB, gAB, xA, xB_set, bmask=None):
    """Inference in overlap region using A+B data with transformer aggregation."""
    zA = encA(xA.cuda()); u = pA(zA)
    zB = encB(xB_set.view(-1, *xB_set.shape[2:]).cuda()).view(xB_set.size(0), xB_set.size(1), -1)
    bar_zB = agg(zA, zB, key_padding_mask=bmask.cuda() if bmask is not None else None)
    v = pB(bar_zB)
    p = gAB(u, v, torch.zeros(u.size(0),1, device=u.device))
    return p  # PN probability in BC using A+B


@torch.no_grad()
def infer_outside_bc(encA, pA, gA, xA):
    """Inference outside overlap region using A-only data."""
    zA = encA(xA.cuda()); u = pA(zA)
    return gA(u)  # PN probability outside BC using A-only


def _apply_transformer_target_head(
    target_head,  # AggregatorTargetHead
    anchor_vecs: List[torch.Tensor],
    target_stack_per_anchor: List[torch.Tensor],
    pair_metadata: List[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Apply transformer target head in batches."""
    dataset = TargetSetDataset(anchor_vecs, target_stack_per_anchor, pair_metadata)
    collate_fn = lambda batch: _collate_target_sets(batch, target_head.aggregator.use_positional_encoding)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, drop_last=False, collate_fn=collate_fn)
    fused_outputs: List[torch.Tensor] = []
    target_head.eval()
    with torch.no_grad():
        for anchor_batch, target_batch, mask_batch, pos_batch in loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            if pos_batch is not None:
                pos_batch = pos_batch.to(device)
            output = target_head(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
            fused_outputs.extend(output.detach().cpu())
    return fused_outputs


def _create_inference_plots(
    model_dir: Path,
    model_name: str,
    coords: List[Tuple[float, float]],
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    pos_crd: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """Create and save scatter plots for inference results."""
    if plt is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mean predictions
    coords_array = np.array(coords)
    sc1 = axes[0].scatter(coords_array[:, 0], coords_array[:, 1], c=mean_pred, cmap='RdYlGn', s=2, marker='s', alpha=1.0)
    axes[0].set_title(f'{model_name} - Mean Prediction')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_aspect('equal', adjustable='datalim')
    if pos_crd is not None:
        pos_crd_array = np.array(pos_crd)
        axes[0].scatter(pos_crd_array[:, 0], pos_crd_array[:, 1], facecolors='none', edgecolors='blue', s=5, marker='o', alpha=1.0, label='Known Positives')
        axes[0].legend(loc='best')

    plt.colorbar(sc1, ax=axes[0], label='Probability')
    
    # Plot uncertainty (std)
    sc2 = axes[1].scatter(coords_array[:, 0], coords_array[:, 1], c=std_pred, cmap='viridis', s=2, marker='s', alpha=1.0)
    axes[1].set_title(f'{model_name} - Uncertainty (Std)')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].set_aspect('equal', adjustable='datalim')
    if pos_crd is not None:
        pos_crd_array = np.array(pos_crd)
        axes[1].scatter(pos_crd_array[:, 0], pos_crd_array[:, 1], facecolors='none', edgecolors='blue', s=5, marker='o', alpha=1.0, label='Known Positives')
        axes[1].legend(loc='best')    
    plt.colorbar(sc2, ax=axes[1], label='Std Dev')
    
    plt.tight_layout()
    plt.savefig(model_dir / "predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()


def _compute_inference_summary(
    model_name: str,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    labels: Optional[np.ndarray]
) -> Dict[str, object]:
    """Compute summary statistics for inference results."""
    summary = {
        "model": model_name,
        "n_points": len(mean_pred),
        "mean_prediction": float(mean_pred.mean()),
        "std_prediction": float(std_pred.mean()),
        "min_prediction": float(mean_pred.min()),
        "max_prediction": float(mean_pred.max()),
        "median_prediction": float(np.median(mean_pred)),
        "mean_uncertainty": float(std_pred.mean()),
        "max_uncertainty": float(std_pred.max()),
    }
    
    if labels is not None:
        pos_mask = labels > 0
        neg_mask = labels <= 0
        
        summary["n_positive"] = int(pos_mask.sum())
        summary["n_negative"] = int(neg_mask.sum())
        
        if pos_mask.any():
            summary["mean_pred_positive"] = float(mean_pred[pos_mask].mean())
            summary["std_pred_positive"] = float(std_pred[pos_mask].mean())
        else:
            summary["mean_pred_positive"] = None
            summary["std_pred_positive"] = None
            
        if neg_mask.any():
            summary["mean_pred_negative"] = float(mean_pred[neg_mask].mean())
            summary["std_pred_negative"] = float(std_pred[neg_mask].mean())
        else:
            summary["mean_pred_negative"] = None
            summary["std_pred_negative"] = None
    
    return summary