"""
Training functions for transformer-aggregated DCCA with phi features.

This module contains:
- Unified head training with knowledge distillation and focal loss
- Transformer aggregator training with DCCA loss
- PU learning loss functions for positive-unlabeled scenarios
"""

from __future__ import annotations
import itertools
import math
from typing import List, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Common.cls.models.mlp_dropout import MLPDropout
from Common.Unifying.DCCA import dcca_loss, ProjectionHead

from .models import PNHeadUnified, CrossAttentionAggregator, AggregatorTargetHead
from .utils import TargetSetDataset, _collate_target_sets


def fit_unified_head_OVERLAP_from_uv(
    gA: nn.Module,
    data_loader_uv: DataLoader,
    d_u: int,
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 10,
    lambda_cons: float = 0.5,
    lambda_focal: float = 2.0,    # NEW: focal weight for positives
    view_dropout: float = 0.1,
    noise_sigma: float = 0.0
) -> nn.Module:
    """
    Train unified PN head using knowledge distillation from A-only teacher.
    
    Args:
        gA: Teacher model (A-only)
        data_loader_uv: DataLoader with (u, v, b_missing) tuples
        d_u: Dimension of u and v projections
        device: Training device
        lr: Learning rate
        steps: Training epochs
        lambda_cons: Consistency loss weight
        lambda_focal: Focal loss weight for positives
        view_dropout: Probability of dropping target view
        noise_sigma: Noise level for robustness
        
    Returns:
        Trained PNHeadUnified model
    """
    
    # Check what type of outputs gA produces
    gA_outputs_probs = isinstance(gA, MLPDropout)  # MLPDropout outputs probs
    
    # Build unified head
    gAB = PNHeadUnified(d=d_u).to(device)
    gAB_outputs_probs = True  # PNHeadUnified outputs probabilities
    
    opt = torch.optim.AdamW(gAB.parameters(), lr=lr, weight_decay=1e-4)
    eps = 1e-6
    
    for epoch in range(1, steps + 1):
        epoch_loss = 0.0
        epoch_loss_kd = 0.0
        epoch_loss_focal = 0.0
        epoch_loss_cons = 0.0
        batch_count = 0
        
        pbar = tqdm(data_loader_uv, desc=f"    [cls - 2] Epoch {epoch:3d}/{steps}", leave=False)
        
        for u, v, bmiss in pbar:
            u, v, bmiss = u.to(device), v.to(device), bmiss.to(device)
            
            with torch.no_grad():
                # Optional noise for robustness
                if noise_sigma > 0:
                    u_noisy = u + noise_sigma * torch.randn_like(u)
                    v_noisy = v + noise_sigma * torch.randn_like(v)
                else:
                    u_noisy, v_noisy = u, v
                
                # Get teacher predictions in PROBABILITY space
                teacher_output = gA(u_noisy)
                if gA_outputs_probs:
                    teacher = teacher_output.detach()
                else:
                    teacher = torch.sigmoid(teacher_output).detach()  # Convert logits to probs
            
            # View dropout: teach robustness when B is absent
            if torch.rand(1).item() < view_dropout:
                v_in = torch.zeros_like(v_noisy)
                b_in = torch.ones(u_noisy.size(0), 1, device=device)
            else:
                v_in = v_noisy
                b_in = bmiss
            
            # Student predictions (ensure probabilities)
            student_output = gAB(u_noisy, v_in, b_in)
            if gAB_outputs_probs:
                p_student = student_output
            else:
                p_student = torch.sigmoid(student_output)
            
            # Consistency target (ensure probabilities)
            cons_output = gAB(u_noisy, v, torch.zeros_like(b_in))
            if gAB_outputs_probs:
                p_cons = cons_output
            else:
                p_cons = torch.sigmoid(cons_output)

            # NEW: Focal-weighted KD loss
            # Upweight samples where teacher predicts high probability
            with torch.no_grad():
                # Focal weight: higher for teacher's positive predictions
                focal_weight = teacher.pow(2.0)  # Weight = teacher²
                # Normalize so mean weight ≈ 1 (prevents loss scale issues)
                focal_weight = focal_weight / (focal_weight.mean() + eps)

            # Debug logging for first batch of epoch
            if batch_count == 0:  # First batch of epoch
                print(f"    [debug] Teacher predictions: mean={teacher.mean().item():.3f}, "
                    f"std={teacher.std().item():.3f}, min={teacher.min().item():.3f}, max={teacher.max().item():.3f}")
                print(f"    [debug] Student predictions: mean={p_student.mean().item():.3f}, "
                    f"std={p_student.std().item():.3f}, min={p_student.min().item():.3f}, max={p_student.max().item():.3f}")

            # All losses now operate on probabilities [0,1]
            kd_errors = (p_student - teacher).pow(2)
            loss_kd_base = kd_errors.mean()  # Base MSE
            loss_focal = (focal_weight * kd_errors).mean()  # Focal MSE
            loss_cons = F.mse_loss(p_student, p_cons.detach())
            
            # Total loss (removed entropy, added focal)
            loss = loss_kd_base + lambda_focal * loss_focal + lambda_cons * loss_cons
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            epoch_loss_kd += loss_kd_base.item()
            epoch_loss_focal += loss_focal.item()
            epoch_loss_cons += loss_cons.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'kd': f'{loss_kd_base.item():.4e}',
                'focal': f'{loss_focal.item():.4e}',
                'cons': f'{loss_cons.item():.4e}'
            })
        
        pbar.close()
        
        # Print epoch summary
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        avg_kd = epoch_loss_kd / batch_count if batch_count > 0 else 0.0
        avg_focal = epoch_loss_focal / batch_count if batch_count > 0 else 0.0
        avg_cons = epoch_loss_cons / batch_count if batch_count > 0 else 0.0
        
        print(f"    [cls - 2] Epoch {epoch:3d} | "
              f"loss={avg_loss:.4e} | "
              f"kd={avg_kd:.4e} | "
              f"focal={avg_focal:.4e} | "
              f"cons={avg_cons:.4e}")
    
    print(f"\n    [cls - 2] Training completed after {steps} epochs")
    return gAB.eval()


def _train_transformer_aggregator(
    anchor_vecs: Sequence[torch.Tensor],
    target_stack_per_anchor: Sequence[torch.Tensor],
    pair_metadata: Sequence[Dict[str, object]],
    *,
    validation_anchor_vecs: Optional[Sequence[torch.Tensor]] = None,
    validation_target_stack: Optional[Sequence[torch.Tensor]] = None,
    validation_metadata: Optional[Sequence[Dict[str, object]]] = None,
    device: torch.device,
    batch_size: int,
    steps: int,
    lr: float,
    agg_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    dcca_eps: float,
    drop_ratio: float,
    use_positional_encoding: bool,
    run_logger,
) -> Tuple[bool, Optional[nn.Module], Optional[AggregatorTargetHead], List[Dict[str, object]], int, Optional[str]]:
    """
    Train transformer aggregator using DCCA loss.
    
    Returns:
        (success, proj_anchor, target_head, history, agg_dim, failure_reason)
    """

    if not anchor_vecs or not target_stack_per_anchor:
        failure = "Transformer aggregator requires non-empty anchor/target stacks."
        run_logger.log(f"[agg] {failure}")
        return False, None, None, [], agg_dim, failure

    train_dataset = TargetSetDataset(anchor_vecs, target_stack_per_anchor, pair_metadata)
    if len(train_dataset) == 0:
        failure = "Transformer aggregator received empty training dataset."
        run_logger.log(f"[agg] {failure}")
        return False, None, None, [], agg_dim, failure

    collate_fn = lambda batch: _collate_target_sets(batch, use_positional_encoding)
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    has_validation = (
        validation_anchor_vecs is not None
        and validation_target_stack is not None
        and validation_metadata is not None
        and len(validation_anchor_vecs) > 0
    )
    if has_validation:
        val_dataset = TargetSetDataset(validation_anchor_vecs, validation_target_stack, validation_metadata)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        val_loader = None

    anchor_dim = anchor_vecs[0].numel()
    target_dim = target_stack_per_anchor[0].size(1)
    print(f"[DEBUG] anchor_dim={anchor_dim}, target_dim={target_dim}, agg_dim={agg_dim}, num_layers={num_layers}")
    
    aggregator = CrossAttentionAggregator(
        anchor_dim,
        target_dim,
        agg_dim,
        num_layers=max(1, num_layers),
        num_heads=max(1, num_heads),
        dropout=dropout,
        use_positional_encoding=use_positional_encoding,
    ).to(device)
    proj_anchor = ProjectionHead(anchor_dim, agg_dim, num_layers=num_layers).to(device)
    proj_target = ProjectionHead(agg_dim, agg_dim, num_layers=num_layers).to(device)
    
    optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(proj_anchor.parameters()) + list(proj_target.parameters()),
        lr=lr,
    )
    
    if steps <= 0:
        aggregator.eval()
        proj_anchor.eval()
        proj_target.eval()
        target_head = AggregatorTargetHead(aggregator, proj_target, use_positional_encoding=use_positional_encoding)
        target_head.eval()
        run_logger.log("[agg] No training steps requested; returning aggregator in eval mode.")
        return True, proj_anchor, target_head, [], agg_dim, None

    def _evaluate(loader: Optional[DataLoader]) -> Optional[Dict[str, object]]:
        if loader is None:
            return None
        
        aggregator.eval()
        proj_anchor.eval()
        proj_target.eval()
        losses: List[float] = []
        batches = 0
        singular_store: List[torch.Tensor] = []
        
        with torch.no_grad():
            for anchor_batch, target_batch, mask_batch, pos_batch in loader:
                anchor_batch = anchor_batch.to(device)
                target_batch = target_batch.to(device)
                mask_batch = mask_batch.to(device)
                if pos_batch is not None:
                    pos_batch = pos_batch.to(device)
                fused = aggregator(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
                u = proj_anchor(anchor_batch)
                v = proj_target(fused)
                
                # Normalize projections to prevent numerical instability
                u = torch.nn.functional.normalize(u, p=2, dim=1)
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                
                loss, singulars, loss_info = dcca_loss(u, v, eps=dcca_eps, drop_ratio=drop_ratio)
                losses.append(loss.item())
                batches += 1
                if singulars.numel() > 0:
                    singular_store.append(singulars.detach())
                    
        if batches == 0:
            return None
        mean_loss = float(sum(losses) / batches)
        if singular_store:
            compiled = torch.cat(singular_store)
            mean_corr = float(compiled.mean().item())
            tcc_sum = float(compiled.sum().item())
            tcc_mean = float(compiled.mean().item())
            k_val = int(compiled.numel())
        else:
            mean_corr = None
            tcc_sum = None
            tcc_mean = None
            k_val = 0
        aggregator.train()
        proj_anchor.train()
        return {
            "loss": mean_loss,
            "mean_corr": mean_corr,
            "tcc_sum": tcc_sum,
            "tcc_mean": tcc_mean,
            "k": k_val,
            "batches": batches,
        }

    def _format(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "None"
        return f"{value:.6f}"

    epoch_history: List[Dict[str, object]] = []
    failure_reason: Optional[str] = None

    for epoch_idx in range(steps):
        aggregator.train()
        proj_anchor.train()
        proj_target.train()
        epoch_loss = 0.0
        epoch_batches = 0
        had_nan_grad = False
        ihad_nan_grad = 0
        
        for anchor_batch, target_batch, mask_batch, pos_batch in train_loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            if pos_batch is not None:
                pos_batch = pos_batch.to(device)
            fused = aggregator(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
            u = proj_anchor(anchor_batch)
            v = proj_target(fused)
            
            # Normalize projections to prevent numerical instability
            u = torch.nn.functional.normalize(u, p=2, dim=1)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            
            loss, _, _ = dcca_loss(u, v, eps=dcca_eps, drop_ratio=drop_ratio)
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients and skip update if found
            has_nan_grad = False
            for param in itertools.chain(
                aggregator.parameters(),
                proj_anchor.parameters(),
                proj_target.parameters()
            ):
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    ihad_nan_grad += 1
                    break
            
            if has_nan_grad:
                had_nan_grad = True
                continue
            
            torch.nn.utils.clip_grad_norm_(
                list(aggregator.parameters()) + list(proj_anchor.parameters()) + list(proj_target.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1

        if had_nan_grad:
            run_logger.log(f"[Trans-AGG-DCCA] INFO: NaN gradients detected in epoch {epoch_idx+1}. Corresponding {ihad_nan_grad} batches skipped among total {epoch_batches} batches.")

        train_metrics = _evaluate(train_eval_loader)
        val_metrics = _evaluate(val_loader) if has_validation else None

        train_loss_log = train_metrics.get("loss") if train_metrics else epoch_loss / max(1, epoch_batches)
        train_corr_log = train_metrics.get("mean_corr") if train_metrics else None
        train_tcc_log = train_metrics.get("tcc_sum") if train_metrics else None
        train_tcc_mean_log = train_metrics.get("tcc_mean") if train_metrics else None
        train_k_log = train_metrics.get("k") if train_metrics else None
        val_loss_log = val_metrics.get("loss") if val_metrics else None
        val_corr_log = val_metrics.get("mean_corr") if val_metrics else None
        val_tcc_log = val_metrics.get("tcc_sum") if val_metrics else None
        val_tcc_mean_log = val_metrics.get("tcc_mean") if val_metrics else None
        val_k_log = val_metrics.get("k") if val_metrics else None
        train_batches_count = train_metrics.get("batches") if train_metrics else epoch_batches
        val_batches_count = val_metrics.get("batches") if val_metrics else None
        
        run_logger.log(
            "[Trans-AGG-DCCA] epoch {epoch}: train_loss={train_loss}, train_mean_corr={train_corr}, "
            "[Trans-AGG-DCCA] train_TCC = {train_tcc}, val_loss={val_loss}, val_mean_corr={val_corr}, "
            "[Trans-AGG-DCCA] val_TCC = {val_tcc}, batches={batches}".format(
                epoch=epoch_idx + 1,
                train_loss=_format(train_loss_log), train_corr=_format(train_corr_log), train_tcc=_format(train_tcc_log),
                val_loss=_format(val_loss_log),     val_corr=_format(val_corr_log),     val_tcc=_format(val_tcc_log),
                batches=train_batches_count,
            )
        )

        epoch_history.append(
            {
                "epoch": epoch_idx + 1,
                "loss": float(train_loss_log) if train_loss_log is not None else None,
                "mean_correlation": float(train_corr_log) if train_corr_log is not None else None,
                "train_eval_loss": float(train_loss_log) if train_loss_log is not None else None,
                "train_eval_mean_correlation": float(train_corr_log) if train_corr_log is not None else None,
                "train_eval_tcc": float(train_tcc_log) if train_tcc_log is not None else None,
                "train_eval_tcc_mean": float(train_tcc_mean_log) if train_tcc_mean_log is not None else None,
                "train_eval_k": float(train_k_log) if train_k_log is not None else None,
                "val_eval_loss": float(val_loss_log) if val_loss_log is not None else None,
                "val_eval_mean_correlation": float(val_corr_log) if val_corr_log is not None else None,
                "val_eval_tcc": float(val_tcc_log) if val_tcc_log is not None else None,
                "val_eval_tcc_mean": float(val_tcc_mean_log) if val_tcc_mean_log is not None else None,
                "val_eval_k": float(val_k_log) if val_k_log is not None else None,
                "batches": int(train_batches_count),
                "val_batches": int(val_batches_count) if val_batches_count is not None else None,
                "projection_dim": int(agg_dim),
            }
        )
        
    aggregator.eval()
    proj_anchor.eval()
    proj_target.eval()
    target_head = AggregatorTargetHead(aggregator, proj_target, use_positional_encoding=use_positional_encoding)
    target_head.eval()
    return True, proj_anchor, target_head, epoch_history, agg_dim, failure_reason


def nnpu_basic_loss_from_logits(logits_p, logits_u, pi_p: float):
    """
    Classic non-negative PU risk (Kiryo et al., 2017) on logits.
    """
    pos_loss = F.softplus(-logits_p)   # ell^+(z)
    neg_on_p = F.softplus( logits_p)   # ell^-(z) on P
    neg_on_u = F.softplus( logits_u)   # ell^-(z) on U

    R_p = pi_p * pos_loss.mean()
    R_n = torch.clamp(neg_on_u.mean() - pi_p * neg_on_p.mean(), min=0.0)
    return R_p + R_n


def nnpu_weighted_loss_from_logits(
    logits_p, logits_u, pi_p: float,
    w_p: float = 10.0, w_n: float = 1.0,
    focal_gamma: float | None = 1.5,
    prior_penalty: float | None = 5.0
):
    """
    Weighted nnPU risk with optional focalization (positives) and prior matching on U.
    """
    pos_loss = F.softplus(-logits_p)   # ell^+(z)
    neg_on_p = F.softplus( logits_p)   # ell^-(z)
    neg_on_u = F.softplus( logits_u)   # ell^-(z)

    # focalize only the positive term (helps rare positives)
    if focal_gamma is not None:
        with torch.no_grad():
            p_pos = torch.sigmoid(logits_p).clamp_(1e-6, 1-1e-6)
        pos_loss = ((1.0 - p_pos) ** float(focal_gamma)) * pos_loss

    R_p = pi_p * pos_loss.mean()
    R_n = torch.clamp(neg_on_u.mean() - pi_p * neg_on_p.mean(), min=0.0)
    loss = w_p * R_p + w_n * R_n

    # prior-matching penalty to avoid all-negative collapse
    if prior_penalty is not None and prior_penalty > 0:
        p_u = torch.sigmoid(logits_u).mean()
        loss = loss + prior_penalty * (p_u - float(pi_p))**2
    return loss