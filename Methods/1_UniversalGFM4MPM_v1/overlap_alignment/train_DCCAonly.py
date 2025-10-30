from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:  # optional plotting for debug mode
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from .config import AlignmentConfig, load_config
from .workspace import OverlapAlignmentPair, OverlapAlignmentWorkspace
from .datasets import auto_coord_error
from Common.overlap_debug_plot import save_overlap_debug_plot

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer (positive-only)")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    parser.add_argument("--dcca-eps", type=float, default=1e-5, help="Epsilon value for DCCA covariance regularization (default: 1e-5).")
    parser.add_argument("--singular-value-drop-ratio", type=float, default=0.01, help="Ratio threshold for dropping small singular values in DCCA (default: 0.01).")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of aligned pairs reserved for validation evaluation (set to 0 to disable).",
    )
    parser.add_argument(
        "--tcc-ratio",
        type=float,
        default=1.0,
        help="Fraction of canonical correlations to include when computing TCC (0 < ratio <= 1).",
    )
    parser.add_argument(
        "--dcca-mlp-layers",
        type=int,
        default=4,
        help="Number of linear layers used in each DCCA projection head MLP (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if torch is None or nn is None or DataLoader is None:
        raise ImportError("Overlap alignment training requires PyTorch; install torch before running the trainer.")

    if args.objective is not None:
        cfg.alignment_objective = args.objective.lower()
    if args.aggregator is not None:
        cfg.aggregator = args.aggregator.lower()
    if args.use_positive_only:
        cfg.use_positive_only = True
    if args.use_positive_augmentation:
        cfg.use_positive_augmentation = True
    debug_mode = bool(args.debug)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    workspace = OverlapAlignmentWorkspace(cfg)
    anchor_name, target_name = _dataset_pair(cfg)
    max_coord_error = auto_coord_error(workspace, anchor_name, target_name)

    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    debug_overlap_stats = _compute_overlap_debug(pairs, anchor_name, target_name) if debug_mode else None
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

    augmentation_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    augmentation_loaded_counts: Dict[str, int] = {}
    if cfg.use_positive_augmentation:
        for dataset_cfg in cfg.datasets:
            if dataset_cfg.pos_aug_path is None:
                continue
            aug_map, aug_count = _load_augmented_embeddings(dataset_cfg.pos_aug_path, dataset_cfg.region_filter)
            if aug_map:
                augmentation_by_dataset[dataset_cfg.name] = aug_map
                augmentation_loaded_counts[dataset_cfg.name] = aug_count
                print(
                    f"[info] Loaded {aug_count} augmented embeddings for dataset {dataset_cfg.name} "
                    f"from {dataset_cfg.pos_aug_path}"
                )
            else:
                print(
                    f"[warn] No augmented embeddings retained for dataset {dataset_cfg.name}; "
                    "region filter or bundle contents may have excluded all entries."
                )

    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
    (
        anchor_vecs,
        target_vecs,
        label_hist,
        debug_data,
        augmentation_stats,
        pair_metadata,
    ) = _build_positive_aligned_pairs(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=cfg.use_positive_only,
        aggregator=cfg.aggregator,
        gaussian_sigma=cfg.gaussian_sigma,
        debug=debug_mode,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=augmentation_by_dataset.get(anchor_name),
        target_augment_map=augmentation_by_dataset.get(target_name),
    )
    print(f"[dev] label_histogram={dict(label_hist)}")

    
    if not anchor_vecs:
        print("[warn] No qualifying overlap groups were found for training.")
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
        return
    if cfg.alignment_objective.lower() == "dcca" and len(anchor_vecs) < 2:
        print(
            "[warn] Not enough positive overlap pairs to optimise DCCA "
            f"(need at least 2, found {len(anchor_vecs)}). Aborting training."
        )
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
            _persist_metrics(cfg, {"objective": cfg.alignment_objective.lower(), "max_coord_error": max_coord_error}, [], debug_data)
        return

    dataset_positive_summary = _dataset_positive_counts(cfg, workspace)
    if cfg.use_positive_augmentation:
        _print_augmentation_usage(
            anchor_name,
            target_name,
            augmentation_loaded_counts,
            augmentation_stats,
            total_pairs=len(anchor_vecs),
        )

    if debug_overlap_stats is not None:
        print(
            "[debug] overlap positives: "
            f"{anchor_name}={debug_overlap_stats['anchor_positive']} / {debug_overlap_stats['anchor_overlap']}, "
            f"{target_name}={debug_overlap_stats['target_positive']} / {debug_overlap_stats['target_overlap']}, "
            f"positive-overlap-pairs={debug_overlap_stats['positive_pairs']}"
        )

    anchor_tensor = torch.stack(anchor_vecs)
    target_tensor = torch.stack(target_vecs)
    total_pairs = anchor_tensor.size(0)

    validation_split = float(args.validation_split)
    if not math.isfinite(validation_split):
        validation_split = 0.0
    validation_split = max(0.0, min(validation_split, 0.9))
    val_count = int(total_pairs * validation_split)
    minimum_train = 2
    if total_pairs - val_count < minimum_train and total_pairs >= minimum_train:
        val_count = max(0, total_pairs - minimum_train)
    if val_count < 2:
        val_count = 0

    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    indices = torch.randperm(total_pairs, generator=generator)
    val_indices = indices[:val_count] if val_count > 0 else torch.empty(0, dtype=torch.long)
    train_indices = indices[val_count:]
    if train_indices.numel() < minimum_train and val_indices.numel() > 0:
        needed = minimum_train - train_indices.numel()
        recover = val_indices[:needed]
        train_indices = torch.cat([train_indices, recover]) if train_indices.numel() else recover
        val_indices = val_indices[needed:]
    if train_indices.numel() < minimum_train:
        raise RuntimeError("Unable to allocate at least two samples for DCCA training after validation split.")

    train_anchor = anchor_tensor[train_indices]
    train_target = target_tensor[train_indices]
    train_dataset = TensorDataset(train_anchor, train_target)

    train_indices_list = train_indices.tolist()
    val_indices_list = val_indices.tolist()
    for idx in train_indices_list:
        pair_metadata[idx]["split"] = "train"
    for idx in val_indices_list:
        pair_metadata[idx]["split"] = "val"

    validation_dataset: Optional[TensorDataset]
    if val_indices.numel() >= 2:
        val_anchor = anchor_tensor[val_indices]
        val_target = target_tensor[val_indices]
        validation_dataset = TensorDataset(val_anchor, val_target)
        actual_validation_fraction = val_indices.numel() / total_pairs
    else:
        validation_dataset = None
        actual_validation_fraction = 0.0

    train_batch_size = min(cfg.training.batch_size, len(train_dataset))

    objective = cfg.alignment_objective.lower()
    if objective not in {"dcca", "barlow"}:
        raise ValueError(f"Unsupported alignment objective: {objective}")

    drop_ratio = float(args.singular_value_drop_ratio)
    if not math.isfinite(drop_ratio) or drop_ratio <= 0.0:
        drop_ratio = SINGULAR_VALUE_DROP_RATIO
    drop_ratio = min(drop_ratio, 1.0)

    tcc_ratio = float(args.tcc_ratio)
    if not math.isfinite(tcc_ratio) or tcc_ratio <= 0.0:
        tcc_ratio = 1.0
    tcc_ratio = min(tcc_ratio, 1.0)

    try:
        mlp_layers = int(args.dcca_mlp_layers)
    except Exception:
        mlp_layers = 4
    if mlp_layers < 1:
        mlp_layers = 1

    run_logger = _RunLogger(cfg)
    run_logger.log(
        "Starting stage-1 overlap alignment with initial projection_dim={proj}; "
        "train_pairs={train}, val_pairs={val}, validation_fraction={vf:.3f}, "
        "drop_ratio={drop:.4f}, tcc_ratio={tcc:.4f}, mlp_layers={layers}".format(
            proj=cfg.projection_dim,
            train=len(train_dataset),
            val=len(validation_dataset) if validation_dataset is not None else 0,
            vf=actual_validation_fraction,
            drop=drop_ratio,
            tcc=tcc_ratio,
            layers=mlp_layers,
        )
    )

    if objective != "dcca":
        raise NotImplementedError("Only DCCA objective supports adaptive projection control.")

    train_success, projector_a, projector_b, epoch_history, final_proj_dim, failure_reason = _train_with_adaptive_projection(
        cfg=cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=train_batch_size,
        device=device,
        anchor_dim=train_anchor.size(1),
        target_dim=train_target.size(1),
        dcca_eps=args.dcca_eps,
        run_logger=run_logger,
        drop_ratio=drop_ratio,
        tcc_ratio=tcc_ratio,
        mlp_layers=mlp_layers,
    )

    if not train_success or projector_a is None or projector_b is None:
        raise RuntimeError(f"DCCA training failed: {failure_reason or 'unknown error'}")

    last_history = epoch_history[-1] if epoch_history else {}

    summary = {
        "objective": objective,
        "aggregator": cfg.aggregator,
        "use_positive_only": cfg.use_positive_only,
        "use_positive_augmentation": cfg.use_positive_augmentation,
        "projection_dim": final_proj_dim,
        "projection_layers": mlp_layers,
        "label_histogram": dict(label_hist),
        "num_pairs": len(anchor_vecs),
        "epochs": cfg.training.epochs,
        "batch_size": train_batch_size,
        "lr": cfg.training.lr,
        "max_coord_error": max_coord_error,
        "final_loss": last_history.get("train_eval_loss", last_history.get("loss")),
        "final_mean_correlation": last_history.get("train_eval_mean_correlation", last_history.get("mean_correlation")),
        "final_train_tcc": last_history.get("train_eval_tcc"),
        "final_train_tcc_mean": last_history.get("train_eval_tcc_mean"),
        "final_train_tcc_k": int(last_history["train_eval_k"]) if last_history.get("train_eval_k") is not None else None,
        "final_val_loss": last_history.get("val_eval_loss"),
        "final_val_mean_correlation": last_history.get("val_eval_mean_correlation"),
        "final_val_tcc": last_history.get("val_eval_tcc"),
        "final_val_tcc_mean": last_history.get("val_eval_tcc_mean"),
        "final_val_tcc_k": int(last_history["val_eval_k"]) if last_history.get("val_eval_k") is not None else None,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset) if validation_dataset is not None else 0,
        "validation_fraction": actual_validation_fraction,
        "tcc_ratio": tcc_ratio,
        "singular_value_drop_ratio": drop_ratio,
    }
    summary["selected_pairs_augmented"] = int(augmentation_stats.get("selected_pairs_augmented", 0))
    summary["anchor_augmented_pairs"] = int(augmentation_stats.get("anchor_augmented_pairs", 0))
    summary["target_augmented_pairs"] = int(augmentation_stats.get("target_augmented_pairs", 0))
    summary.update(dataset_positive_summary)
    print("[info] training summary:", summary)

    _persist_state(cfg, projector_a, projector_b, summary)
    _persist_metrics(cfg, summary, epoch_history, debug_data if debug_mode else None)
    if debug_mode:
        _maybe_save_debug_figures(cfg, debug_data)
        _create_debug_alignment_figures(
            cfg=cfg,
            projector_a=projector_a,
            projector_b=projector_b,
            anchor_tensor=anchor_tensor,
            target_tensor=target_tensor,
            pair_metadata=pair_metadata,
            run_logger=run_logger,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
            dcca_eps=args.dcca_eps,
            device=device,
            sample_seed=int(cfg.seed),
        )


def _filter_singular_values(singular_values: torch.Tensor, drop_ratio: float) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "total": int(singular_values.numel()),
        "dropped_count": 0,
        "dropped_indices": [],
        "dropped_values": [],
        "threshold": None,
        "all_dropped": False,
    }
    if singular_values.numel() == 0:
        info["all_dropped"] = True
        return singular_values, info

    max_val = float(singular_values.max().item())
    if not math.isfinite(max_val):
        max_val = 0.0
    ratio_threshold = max_val * drop_ratio if max_val > 0 else 0.0
    info["threshold"] = float(ratio_threshold)

    keep_mask = singular_values >= ratio_threshold
    if keep_mask.all():
        drop_count = max(1, int(math.floor(info["total"] * drop_ratio)))
        if drop_count >= info["total"]:
            drop_count = info["total"] - 1
        if drop_count > 0:
            sorted_vals, sorted_idx = torch.sort(singular_values)
            drop_indices = sorted_idx[:drop_count]
            keep_mask[drop_indices] = False

    drop_indices = (~keep_mask).nonzero(as_tuple=False).flatten()
    if drop_indices.numel() > 0:
        info["dropped_count"] = int(drop_indices.numel())
        info["dropped_indices"] = drop_indices.cpu().tolist()
        info["dropped_values"] = singular_values[drop_indices].detach().cpu().tolist()

    if not keep_mask.any():
        info["all_dropped"] = True
        return singular_values.new_empty(0), info

    filtered = singular_values[keep_mask]
    info["kept_count"] = int(filtered.numel())
    if filtered.numel() > 0:
        info["kept_min"] = float(filtered.min().item())
        info["kept_max"] = float(filtered.max().item())
    else:
        info["kept_min"] = None
        info["kept_max"] = None
    return filtered, info


def dcca_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    eps: float = 1e-5,
    drop_ratio: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        dummy = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        info = {"total": 0, "dropped_count": 0, "all_dropped": True}
        return dummy, torch.zeros(0, device=z_a.device), info

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    n = z_a.size(0)
    cov_aa = (z_a.T @ z_a) / (n - 1) + eps * torch.eye(z_a.size(1), device=z_a.device)
    cov_bb = (z_b.T @ z_b) / (n - 1) + eps * torch.eye(z_b.size(1), device=z_b.device)
    cov_ab = (z_a.T @ z_b) / (n - 1)

    inv_sqrt_aa = _matrix_inverse_sqrt(cov_aa, eps)
    inv_sqrt_bb = _matrix_inverse_sqrt(cov_bb, eps)
    t_mat = inv_sqrt_aa @ cov_ab @ inv_sqrt_bb
    jitter = _adaptive_tmat_jitter(t_mat)
    if jitter > 0:
        eye = torch.eye(t_mat.size(-1), device=t_mat.device, dtype=t_mat.dtype)
        t_mat = t_mat + jitter * eye
    try:
        singular_values = torch.linalg.svdvals(t_mat)
    except RuntimeError:
        zero = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        info = {
            "total": 0,
            "dropped_count": 0,
            "all_dropped": True,
            "error": "svd_failed",
        }
        return zero, torch.zeros(0, device=z_a.device), info

    filtered, filter_info = _filter_singular_values(singular_values, drop_ratio)
    filter_info["kept_count"] = int(filtered.numel())
    filter_info["total"] = int(singular_values.numel())
    filter_info["original_max"] = float(singular_values.max().item()) if singular_values.numel() else None
    filter_info["original_min"] = float(singular_values.min().item()) if singular_values.numel() else None

    if filter_info.get("all_dropped"):
        zero = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        return zero, filtered, filter_info

    loss = -filtered.sum()
    return loss, filtered, filter_info


def _format_optional_scalar(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6f}"


def _train_with_adaptive_projection(
    *,
    cfg: AlignmentConfig,
    train_dataset: TensorDataset,
    validation_dataset: Optional[TensorDataset],
    batch_size: int,
    device: torch.device,
    anchor_dim: int,
    target_dim: int,
    dcca_eps: float,
    run_logger: "_RunLogger",
    drop_ratio: float,
    tcc_ratio: float,
    mlp_layers: int,
) -> Tuple[bool, Optional[nn.Module], Optional[nn.Module], List[Dict[str, object]], int, Optional[str]]:
    projection_dim = int(cfg.projection_dim)
    attempt = 1
    failure_reason: Optional[str] = None
    max_attempts = 10

    while projection_dim > 0 and attempt <= max_attempts:
        current_dim = max(projection_dim, 1)
        run_logger.log(f"Attempt {attempt}: projection_dim={current_dim}, mlp_layers={mlp_layers}")
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > 1,
        )
        projector_a = ProjectionHead(anchor_dim, current_dim, num_layers=mlp_layers).to(device)
        projector_b = ProjectionHead(target_dim, current_dim, num_layers=mlp_layers).to(device)

        optimizer = torch.optim.Adam(
            list(projector_a.parameters()) + list(projector_b.parameters()),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        epoch_history: List[Dict[str, object]] = []
        attempt_success = True

        for epoch in range(cfg.training.epochs):
            running_loss = 0.0
            running_corr = 0.0
            batches = 0
            iterator = loader
            if tqdm is not None:
                iterator = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.training.epochs} (proj={current_dim})", leave=False)

            for batch_index, (anchor_batch, target_batch) in enumerate(iterator):
                anchor_batch = anchor_batch.to(device)
                target_batch = target_batch.to(device)
                if anchor_batch.size(0) < 2:
                    if tqdm is not None:
                        iterator.set_postfix({"skip": "batch<2"})
                    continue

                optimizer.zero_grad()
                z_a = projector_a(anchor_batch)
                z_b = projector_b(target_batch)

                loss, singular_vals, filter_info = dcca_loss(z_a, z_b, eps=dcca_eps, drop_ratio=drop_ratio)

#                if filter_info.get("dropped_count", 0):
#                    run_logger.log(
#                        "Attempt {attempt} epoch {epoch} batch {batch}: dropped {count} singular values "
#                        "indices={indices};".format(
#                            attempt=attempt,
#                            epoch=epoch + 1,
#                            batch=batch_index + 1,
#                            count=filter_info["dropped_count"],
#                            indices=filter_info.get("dropped_indices"),
#                        )
#                    )

                if filter_info.get("all_dropped"):
                    failure_reason = (
                        f"All singular values filtered out at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                if not torch.isfinite(loss):
                    failure_reason = (
                        f"Non-finite loss detected at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                loss.backward()

                if _has_nonfinite_gradients((projector_a, projector_b)):
                    failure_reason = (
                        f"Non-finite gradients detected at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                optimizer.step()

                running_loss += float(loss.item())
                running_corr += float(singular_vals.mean().item()) if singular_vals.numel() else 0.0
                batches += 1

                if tqdm is not None:
                    iterator.set_postfix({"loss": loss.item(), "corr": running_corr / max(batches, 1)})

            if not attempt_success:
                break

            if batches == 0:
                failure_reason = f"No batches processed at attempt {attempt}, epoch {epoch + 1}"
                run_logger.log(failure_reason)
                attempt_success = False
                break

            avg_loss = running_loss / batches
            avg_corr = running_corr / batches if batches else float("nan")
            train_eval = _evaluate_dcca(
                projector_a,
                projector_b,
                train_dataset,
                device=device,
                eps=dcca_eps,
                drop_ratio=drop_ratio,
                tcc_ratio=tcc_ratio,
                batch_size=batch_size,
            )
            val_eval = (
                _evaluate_dcca(
                    projector_a,
                    projector_b,
                    validation_dataset,
                    device=device,
                    eps=dcca_eps,
                    drop_ratio=drop_ratio,
                    tcc_ratio=tcc_ratio,
                    batch_size=batch_size,
                )
                if validation_dataset is not None and len(validation_dataset) >= 2
                else None
            )
            train_loss_log = float(train_eval["loss"]) if train_eval and "loss" in train_eval else avg_loss
            train_corr_log = float(train_eval["mean_corr"]) if train_eval and "mean_corr" in train_eval else avg_corr
            train_tcc_log = train_eval.get("tcc_sum") if train_eval else None
            train_tcc_mean_log = train_eval.get("tcc_mean") if train_eval else None
            train_k_log = train_eval.get("k") if train_eval else None
            val_loss_log = val_eval.get("loss") if val_eval else None
            val_corr_log = val_eval.get("mean_corr") if val_eval else None
            val_tcc_log = val_eval.get("tcc_sum") if val_eval else None
            val_tcc_mean_log = val_eval.get("tcc_mean") if val_eval else None
            val_k_log = val_eval.get("k") if val_eval else None

            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(avg_loss),
                    "mean_correlation": float(avg_corr),
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
                    "batches": batches,
                    "projection_dim": current_dim,
                }
            )
            run_logger.log(
                "Attempt {attempt} epoch {epoch}: "
                "train_loss={train_loss:.6f}, train_mean_corr={train_corr:.6f}, train_TCC = {train_tcc}, "
                "val_loss={val_loss}, val_mean_corr={val_corr}, val_TCC = {val_tcc}, batches={batches}".format(
                    attempt=attempt,
                    epoch=epoch + 1,
                    train_loss=train_loss_log,
                    train_corr=train_corr_log,
                    train_tcc=_format_optional_scalar(train_tcc_log),
                    val_loss=_format_optional_scalar(val_loss_log),
                    val_corr=_format_optional_scalar(val_corr_log),
                    val_tcc=_format_optional_scalar(val_tcc_log),
                    batches=batches,
                )
            )

        if attempt_success:
            run_logger.log(f"Attempt {attempt} succeeded with projection_dim={current_dim}")
            return True, projector_a, projector_b, epoch_history, current_dim, None

        run_logger.log(
            f"Attempt {attempt} failed with projection_dim={current_dim}; reason={failure_reason or 'unspecified'}"
        )
        prev_dim = current_dim
        projection_dim = max(int(math.floor(prev_dim * 0.8)), 1)
        if projection_dim == prev_dim:
            projection_dim = max(prev_dim - 1, 1)
            if projection_dim == prev_dim:
                failure_reason = failure_reason or "Reached minimum projection dimension without success."
                break
        run_logger.log(f"Reducing projection_dim from {prev_dim} to {projection_dim}")
        attempt += 1

    if failure_reason is None:
        failure_reason = "Exceeded maximum adaptive attempts."
    return False, None, None, [], max(projection_dim, 1), failure_reason


def _has_nonfinite_gradients(modules: Sequence[nn.Module]) -> bool:
    for module in modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
    return False


def _evaluate_dcca(
    projector_a: nn.Module,
    projector_b: nn.Module,
    dataset: Optional[TensorDataset],
    *,
    device: torch.device,
    eps: float,
    drop_ratio: float,
    tcc_ratio: float,
    batch_size: int,
) -> Optional[Dict[str, object]]:
    if dataset is None or len(dataset) < 2:
        return None

    eval_batch = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=eval_batch, shuffle=False, drop_last=False)

    was_training_a = projector_a.training
    was_training_b = projector_b.training
    projector_a.eval()
    projector_b.eval()

    projections_a: List[torch.Tensor] = []
    projections_b: List[torch.Tensor] = []
    with torch.no_grad():
        for anchor_batch, target_batch in loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            proj_a = projector_a(anchor_batch).detach()
            proj_b = projector_b(target_batch).detach()
            projections_a.append(proj_a)
            projections_b.append(proj_b)

    projector_a.train(was_training_a)
    projector_b.train(was_training_b)

    if not projections_a or not projections_b:
        return None

    z_a = torch.cat(projections_a, dim=0)
    z_b = torch.cat(projections_b, dim=0)
    metrics = _canonical_metrics(z_a, z_b, eps=eps, drop_ratio=drop_ratio, tcc_ratio=tcc_ratio)
    if metrics is None:
        return None
    normalised: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalised[key] = float(value)
        else:
            normalised[key] = value
    return normalised


def _canonical_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    eps: float,
    drop_ratio: float,
    tcc_ratio: float,
) -> Optional[Dict[str, object]]:
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return None

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    n = z_a.size(0)
    cov_aa = (z_a.T @ z_a) / (n - 1) + eps * torch.eye(z_a.size(1), device=z_a.device, dtype=z_a.dtype)
    cov_bb = (z_b.T @ z_b) / (n - 1) + eps * torch.eye(z_b.size(1), device=z_b.device, dtype=z_b.dtype)
    cov_ab = (z_a.T @ z_b) / (n - 1)

    inv_sqrt_aa = _matrix_inverse_sqrt(cov_aa, eps)
    inv_sqrt_bb = _matrix_inverse_sqrt(cov_bb, eps)
    t_mat = inv_sqrt_aa @ cov_ab @ inv_sqrt_bb
    jitter = _adaptive_tmat_jitter(t_mat)
    if jitter > 0:
        t_mat = t_mat + jitter * torch.eye(t_mat.size(-1), device=t_mat.device, dtype=t_mat.dtype)

    try:
        singular_values = torch.linalg.svdvals(t_mat)
    except RuntimeError:
        return None

    filtered, filter_info = _filter_singular_values(singular_values, drop_ratio)
    if filter_info.get("all_dropped") or filtered.numel() == 0:
        return None

    sorted_vals, _ = torch.sort(filtered, descending=True)
    total = int(sorted_vals.numel())
    if total == 0:
        return None

    ratio = float(max(min(tcc_ratio, 1.0), 1e-6))
    k = max(1, min(total, int(math.ceil(total * ratio))))
    topk = sorted_vals[:k]

    metrics: Dict[str, object] = {
        "loss": -float(filtered.sum().item()),
        "mean_corr": float(filtered.mean().item()),
        "tcc_sum": float(topk.sum().item()),
        "tcc_mean": float(topk.mean().item()),
        "k": int(k),
        "canonical_correlations": int(total),
    }
    metrics["drop_info"] = filter_info
    metrics["canonical_correlations_sorted"] = [float(val) for val in sorted_vals.detach().cpu().tolist()]
    return metrics


def _project_in_batches(tensor: torch.Tensor, model: nn.Module, device: torch.device, batch_size: int) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty_like(tensor)
    outputs: List[torch.Tensor] = []
    batch_size = max(1, batch_size)
    with torch.no_grad():
        for start in range(0, tensor.size(0), batch_size):
            end = min(start + batch_size, tensor.size(0))
            chunk = tensor[start:end].to(device)
            projected = model(chunk).detach().cpu()
            outputs.append(projected)
    return torch.cat(outputs, dim=0)


def _select_subset_indices(total: int, max_points: int, seed: int) -> torch.Tensor:
    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    if total <= max_points:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator)
    return perm[:max_points]


def _reduce_dimensionality(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE  # type: ignore

        perplexity = max(5, min(30, features.shape[0] // 3))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca")
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    try:
        from sklearn.decomposition import PCA  # type: ignore

        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    if features.shape[1] >= 2:
        return features[:, :2]
    pad = np.zeros((features.shape[0], 2), dtype=np.float32)
    pad[:, 0] = features[:, 0]
    return pad


def _create_debug_alignment_figures(
    *,
    cfg: AlignmentConfig,
    projector_a: nn.Module,
    projector_b: nn.Module,
    anchor_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pair_metadata: Sequence[Dict[str, object]],
    run_logger: "_RunLogger",
    drop_ratio: float,
    tcc_ratio: float,
    dcca_eps: float,
    device: torch.device,
    sample_seed: int,
) -> None:
    if plt is None:
        run_logger.log("matplotlib is unavailable; skipping advanced debug figures.")
        return

    analysis_primary = None
    if cfg.output_dir is not None:
        analysis_primary = cfg.output_dir / "overlap_visualizations" / "alignment_analysis"
    elif cfg.log_dir is not None:
        analysis_primary = cfg.log_dir / "overlap_visualizations" / "alignment_analysis"
    analysis_dir = _prepare_output_dir(analysis_primary, "overlap_alignment_outputs/alignment_analysis")
    if analysis_dir is None:
        run_logger.log("Unable to prepare alignment analysis directory; skipping figures.")
        return

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        run_logger.log(f"Failed to create alignment analysis directory {analysis_dir}: {exc}")
        return

    if anchor_tensor.size(0) == 0 or target_tensor.size(0) == 0:
        run_logger.log("No samples available for debug plotting.")
        return

    analysis_batch = max(32, min(2048, anchor_tensor.size(0)))
    projected_anchor = _project_in_batches(anchor_tensor, projector_a, device, analysis_batch)
    projected_target = _project_in_batches(target_tensor, projector_b, device, analysis_batch)

    # t-SNE / UMAP figure
    try:
        subset_indices = _select_subset_indices(anchor_tensor.size(0), max_points=1000, seed=sample_seed)
        subset_list = subset_indices.tolist()
        if subset_indices.numel() >= 2:
            anchor_raw = anchor_tensor.detach().cpu()
            target_raw = target_tensor.detach().cpu()
            proj_anchor_raw = projected_anchor.detach().cpu()
            proj_target_raw = projected_target.detach().cpu()

            before_features = torch.cat(
                [anchor_raw[subset_indices], target_raw[subset_indices]], dim=0
            ).numpy()
            after_features = torch.cat(
                [proj_anchor_raw[subset_indices], proj_target_raw[subset_indices]], dim=0
            ).numpy()

            embedding_before = _reduce_dimensionality(before_features, random_state=sample_seed)
            embedding_after = _reduce_dimensionality(after_features, random_state=sample_seed + 1)

            num_subset = subset_indices.numel()
            coords_before_anchor = embedding_before[:num_subset]
            coords_before_target = embedding_before[num_subset:]
            coords_after_anchor = embedding_after[:num_subset]
            coords_after_target = embedding_after[num_subset:]

            fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            colors = {"anchor": "#1f77b4", "target": "#ff7f0e"}
            for ax, coords_anchor, coords_target, title in (
                (axes[0], coords_before_anchor, coords_before_target, "Embeddings (pre-DCCA)"),
                (axes[1], coords_after_anchor, coords_after_target, "Embeddings (post-DCCA)"),
            ):
                ax.scatter(
                    coords_anchor[:, 0],
                    coords_anchor[:, 1],
                    c=colors["anchor"],
                    label="Anchor",
                    alpha=0.6,
                    s=20,
                )
                ax.scatter(
                    coords_target[:, 0],
                    coords_target[:, 1],
                    c=colors["target"],
                    label="Target",
                    alpha=0.6,
                    s=20,
                    marker="s",
                )
                val_anchor_x = []
                val_anchor_y = []
                val_target_x = []
                val_target_y = []
                for i, idx in enumerate(subset_list):
                    if i >= coords_anchor.shape[0]:
                        break
                    split = pair_metadata[idx].get("split") if idx < len(pair_metadata) else None
                    if split == "val":
                        val_anchor_x.append(coords_anchor[i, 0])
                        val_anchor_y.append(coords_anchor[i, 1])
                        val_target_x.append(coords_target[i, 0])
                        val_target_y.append(coords_target[i, 1])
                    ax.plot(
                        [coords_anchor[i, 0], coords_target[i, 0]],
                        [coords_anchor[i, 1], coords_target[i, 1]],
                        color="gray",
                        alpha=0.2,
                        linewidth=0.5,
                    )
                if val_anchor_x:
                    ax.scatter(
                        val_anchor_x,
                        val_anchor_y,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        s=40,
                        label="Anchor (val)" if ax is axes[0] else None,
                    )
                if val_target_x:
                    ax.scatter(
                        val_target_x,
                        val_target_y,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        s=40,
                        marker="s",
                        label="Target (val)" if ax is axes[0] else None,
                    )
                ax.set_title(title)
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
            axes[1].legend(loc="upper right")
            tsne_path = analysis_dir / "alignment_tsne_umap.png"
            fig.suptitle("Paired Embeddings Before vs After DCCA")
            fig.savefig(tsne_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            run_logger.log(f"Saved t-SNE/UMAP alignment figure to {tsne_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create t-SNE/UMAP debug figure: {exc}")

    # Canonical correlation spectrum
    try:
        metrics_before = _canonical_metrics(
            anchor_tensor.detach().to(projected_anchor.dtype),
            target_tensor.detach().to(projected_target.dtype),
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
        )
        metrics_after = _canonical_metrics(
            projected_anchor,
            projected_target,
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
        )
        if metrics_before and metrics_after:
            vals_before = metrics_before.get("canonical_correlations_sorted") or []
            vals_after = metrics_after.get("canonical_correlations_sorted") or []
            if vals_before and vals_after:
                vals_before = np.asarray(vals_before, dtype=float)
                vals_after = np.asarray(vals_after, dtype=float)
                k_plot = int(min(len(vals_before), len(vals_after), 128))
                if k_plot > 0:
                    idx = np.arange(1, k_plot + 1)
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                    for ax, values, title in (
                        (axes[0], vals_before[:k_plot], "Pre-DCCA"),
                        (axes[1], vals_after[:k_plot], "Post-DCCA"),
                    ):
                        ax.bar(idx, values, color="#4c72b0", alpha=0.7)
                        cumulative = np.cumsum(values)
                        ax.plot(idx, cumulative, color="#dd8452", linewidth=1.5, label="Cumulative sum")
                        ax.set_title(f"{title} Canonical Spectrum (Top {k_plot})")
                        ax.set_xlabel("Canonical component")
                        ax.set_ylabel("Correlation")
                        ax.set_ylim(0, 1.05)
                        ax.legend(loc="upper right")
                    spectrum_path = analysis_dir / "canonical_correlation_spectrum.png"
                    fig.savefig(spectrum_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    run_logger.log(f"Saved canonical correlation spectrum to {spectrum_path}")
    except Exception as exc:
        run_logger.log(f"Failed to generate canonical correlation spectrum: {exc}")

    # Geospatial distance heatmap
    try:
        geo_indices: List[int] = []
        geo_coords: List[Tuple[float, float]] = []
        geo_splits: List[str] = []
        for idx, meta in enumerate(pair_metadata):
            coord = meta.get("anchor_coord")
            if coord is None:
                coord = meta.get("target_weighted_coord")
            if coord is None:
                continue
            try:
                x, y = float(coord[0]), float(coord[1])
            except Exception:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            geo_indices.append(idx)
            geo_coords.append((x, y))
            geo_splits.append(str(meta.get("split") or "train"))
        if geo_indices:
            pre_dist = torch.linalg.norm(anchor_tensor - target_tensor, dim=1).detach().cpu().numpy()
            post_dist = torch.linalg.norm(projected_anchor - projected_target, dim=1).detach().cpu().numpy()
            pre_geo = pre_dist[geo_indices]
            post_geo = post_dist[geo_indices]
            geo_x = np.asarray([coord[0] for coord in geo_coords], dtype=float)
            geo_y = np.asarray([coord[1] for coord in geo_coords], dtype=float)
            vmin = float(min(pre_geo.min(), post_geo.min()))
            vmax = float(max(pre_geo.max(), post_geo.max()))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharex=True, sharey=True)
            for ax, values, title in (
                (axes[0], pre_geo, "Pre-DCCA"),
                (axes[1], post_geo, "Post-DCCA"),
            ):
                val_mask = np.array([split == "val" for split in geo_splits], dtype=bool)
                sc = ax.scatter(
                    geo_x,
                    geo_y,
                    c=values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    s=40,
                    alpha=0.9,
                )
                if val_mask.any():
                    ax.scatter(
                        geo_x[val_mask],
                        geo_y[val_mask],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.8,
                        s=60,
                        label="Validation",
                    )
                    ax.legend(loc="upper right")
                ax.set_title(f"{title} Pairwise Distance")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                color_label = (
                    "||Latent_base(A) - Latent_base(B)||" if title == "Pre-DCCA" else "||Latent_DCCA(A) - Latent_DCCA(B)||"
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)
            geo_path = analysis_dir / "geospatial_distance_map.png"
            fig.suptitle("Pairwise Distance Heatmap (Anchor vs Target)")
            fig.savefig(geo_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            run_logger.log(f"Saved geospatial distance heatmap to {geo_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create geospatial distance map: {exc}")


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
                else:
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

def _dataset_pair(cfg: AlignmentConfig) -> Tuple[str, str]:
    if len(cfg.datasets) < 2:
        raise ValueError("Overlap alignment requires at least two datasets.")
    return cfg.datasets[0].name, cfg.datasets[1].name


def _compute_overlap_debug(
    pairs: Sequence[OverlapAlignmentPair],
    anchor_name: str,
    target_name: str,
) -> Dict[str, int]:
    overlap_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
    positive_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
    overlapping_positive_pairs = 0
    for pair in pairs:
        for dataset_name, record in (
            (pair.anchor_dataset, pair.anchor_record),
            (pair.target_dataset, pair.target_record),
        ):
            marker = record.tile_id if record.tile_id is not None else record.index
            overlap_tiles.setdefault(dataset_name, set()).add(marker)
            if int(record.label) == 1:
                positive_tiles.setdefault(dataset_name, set()).add(marker)
        if int(pair.anchor_record.label) == 1 and int(pair.target_record.label) == 1:
            overlapping_positive_pairs += 1
    return {
        "anchor_positive": len(positive_tiles.get(anchor_name, set())),
        "anchor_overlap": len(overlap_tiles.get(anchor_name, set())),
        "target_positive": len(positive_tiles.get(target_name, set())),
        "target_overlap": len(overlap_tiles.get(target_name, set())),
        "positive_pairs": overlapping_positive_pairs,
    }


def _dataset_positive_counts(cfg: AlignmentConfig, workspace: OverlapAlignmentWorkspace) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dataset_cfg in cfg.datasets:
        region_tokens = dataset_cfg.region_filter or [dataset_cfg.name]
        region_key = "_".join(str(token) for token in region_tokens) if region_tokens else dataset_cfg.name
        region_key = region_key.replace(" ", "_")
        bundle = workspace.datasets.get(dataset_cfg.name)
        if bundle is None:
            continue
        summary = getattr(bundle, "summary", {}) or {}
        label_info = summary.get("labels") if isinstance(summary, dict) else None
        pos_val = None
        if isinstance(label_info, dict):
            pos_val = label_info.get(1)
            if pos_val is None:
                try:
                    pos_val = label_info.get("1")
                except Exception:
                    pos_val = None
        if pos_val is None:
            try:
                pos_val = sum(1 for record in getattr(bundle, "records", []) if int(getattr(record, "label", 0)) == 1)
            except Exception:
                pos_val = 0
        counts[f"data_{region_key}_positives"] = int(pos_val or 0)
    return counts


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

def _build_positive_aligned_pairs(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
    Dict[str, int],
    List[Dict[str, object]],
]:
    anchor_augment_map = anchor_augment_map or {}
    target_augment_map = target_augment_map or {}
    grouped: Dict[str, Dict[str, object]] = {}
    anchor_aug_added = 0
    target_aug_added = 0
    label_hist: Counter = Counter()
    debug_data: Optional[Dict[str, object]] = None
    if debug:
        debug_data = {
            "anchor_positive": {},
            "target_positive": {},
            "selected_pairs": [],
            "anchor_name": anchor_name,
            "target_name": target_name,
        }

    for pair in pairs:
        anchor_record, target_record = pair.anchor_record, pair.target_record
        anchor_ds, target_ds = pair.anchor_dataset, pair.target_dataset

        if anchor_ds != anchor_name or target_ds != target_name:
            if pair.target_dataset == anchor_name and pair.anchor_dataset == target_name:
                anchor_record, target_record = pair.target_record, pair.anchor_record
                anchor_ds, target_ds = pair.target_dataset, pair.anchor_dataset
            else:
                continue

        anchor_label = int(anchor_record.label)
        target_label = int(target_record.label)

        if debug_data is not None:
            if anchor_label == 1:
                _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
            if target_label == 1:
                _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))

        if use_positive_only and anchor_label != 1:
            continue
        if use_positive_only and target_label != 1:
            continue

        label_key = _label_key(anchor_label, target_label, anchor_name, target_name)
        label_hist[label_key] += 1

        if debug_data is not None:
            if anchor_label == 1 and target_label == 1:
                anchor_sample = _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
                target_sample = _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))
                if anchor_sample is not None and target_sample is not None:
                    debug_data["selected_pairs"].append({"anchor": anchor_sample, "target": target_sample})

        tile_id = anchor_record.tile_id
        grouped.setdefault(tile_id, {
            "anchor": anchor_record,
            "targets": [],
            "weights": [],
        })
        grouped[tile_id]["targets"].append(target_record)
        grouped[tile_id]["weights"].append(_pair_weight(anchor_record, target_record, gaussian_sigma))

    anchor_vecs: List[torch.Tensor] = []
    target_vecs: List[torch.Tensor] = []
    pair_metadata: List[Dict[str, object]] = []

    for entry in grouped.values():
        targets: List = entry["targets"]
        if not targets:
            continue
        aggregated, normalized_weights, target_stack = _aggregate_targets(entry['anchor'], targets, entry['weights'], aggregator)
        if aggregated is None:
            continue
        anchor_tensor = torch.from_numpy(entry['anchor'].embedding).float()
        anchor_vecs.append(anchor_tensor)
        target_vecs.append(aggregated.clone())
        pair_metadata.append(
            _make_pair_meta(
                anchor_record=entry["anchor"],
                targets=targets,
                weights=normalized_weights,
                is_augmented=False,
            )
        )
        anchor_aug_list = anchor_augment_map.get(entry['anchor'].tile_id, [])
        for aug_emb in anchor_aug_list:
            aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
            if aug_tensor.shape != anchor_tensor.shape:
                continue
            anchor_vecs.append(aug_tensor)
            target_vecs.append(aggregated.clone())
            label_hist[label_key] += 1
            anchor_aug_added += 1
            pair_metadata.append(
                _make_pair_meta(
                    anchor_record=entry["anchor"],
                    targets=targets,
                    weights=normalized_weights,
                    is_augmented=True,
                    augmentation_role="anchor",
                )
            )

        for idx_target, target_record in enumerate(targets):
            aug_list = target_augment_map.get(target_record.tile_id, [])
            if not aug_list:
                continue
            for aug_emb in aug_list:
                aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
                if aug_tensor.shape != target_stack[idx_target].shape:
                    continue
                modified_stack = target_stack.clone()
                modified_stack[idx_target] = aug_tensor
                aggregated_aug = torch.matmul(normalized_weights.unsqueeze(0), modified_stack).squeeze(0)
                anchor_vecs.append(anchor_tensor.clone())
                target_vecs.append(aggregated_aug)
                label_hist[label_key] += 1
                target_aug_added += 1
                pair_metadata.append(
                    _make_pair_meta(
                        anchor_record=entry["anchor"],
                        targets=targets,
                        weights=normalized_weights,
                        is_augmented=True,
                        augmentation_role="target",
                    )
                )


    aug_stats = {
        "anchor_augmented_pairs": int(anchor_aug_added),
        "target_augmented_pairs": int(target_aug_added),
        "selected_pairs_augmented": int(anchor_aug_added + target_aug_added),
    }

    if debug_data is not None:
        anchor_serialised = [_serialise_sample(sample) for sample in debug_data["anchor_positive"].values()]
        target_serialised = [_serialise_sample(sample) for sample in debug_data["target_positive"].values()]
        pair_serialised = [
            {
                "anchor": _serialise_sample(pair["anchor"]),
                "target": _serialise_sample(pair["target"]),
            }
            for pair in debug_data["selected_pairs"]
        ]
        debug_payload = {
            "anchor_positive": anchor_serialised,
            "target_positive": target_serialised,
            "selected_pairs": pair_serialised,
            "anchor_name": anchor_name,
            "target_name": target_name,
            'anchor_augmented_pairs': int(anchor_aug_added),
            'target_augmented_pairs': int(target_aug_added),
        }
    else:
        debug_payload = None

    return anchor_vecs, target_vecs, label_hist, debug_payload, aug_stats, pair_metadata

def _add_debug_sample(
    store: Dict[str, Dict[str, object]],
    record,
    dataset_name: str,
    dataset_meta: Optional[Dict[str, object]],
    *,
    is_augmented: bool = False,
) -> Optional[Dict[str, object]]:
    if record.coord is None or any(math.isnan(c) for c in record.coord[:2]):
        return None
    coord = (float(record.coord[0]), float(record.coord[1]))
    sample = store.get(record.tile_id)
    if sample is None:
        sample = {
            "tile_id": record.tile_id,
            "coord": coord,
            "dataset": dataset_name,
        }
        window = record.window_size
        pixel_res = record.pixel_resolution
        if window is None and dataset_meta:
            spacing = dataset_meta.get("window_spacing") or dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
            if spacing is not None:
                pixel_res = float(spacing)
                window = (1, 1)
        if window is not None:
            sample["window_size"] = [int(window[0]), int(window[1])]
        if pixel_res is None and dataset_meta:
            pixel_res = dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
        if pixel_res is not None:
            sample["pixel_resolution"] = float(pixel_res)
        if window is not None and pixel_res is not None:
            width = window[1] * float(pixel_res)
            height = window[0] * float(pixel_res)
            sample["footprint"] = [width, height]
        sample["is_augmented"] = bool(is_augmented)
        store[record.tile_id] = sample
    return sample

def _serialise_sample(sample: Dict[str, object]) -> Dict[str, object]:
    coord = sample.get("coord")
    coord_list = [float(coord[0]), float(coord[1])] if coord is not None else None
    data: Dict[str, object] = {
        "tile_id": sample.get("tile_id"),
    }
    if coord_list is not None:
        data["coord"] = coord_list
    if sample.get("dataset") is not None:
        data["dataset"] = sample["dataset"]
    if "window_size" in sample and sample["window_size"] is not None:
        data["window_size"] = [int(sample["window_size"][0]), int(sample["window_size"][1])]
    if "pixel_resolution" in sample and sample["pixel_resolution"] is not None:
        data["pixel_resolution"] = float(sample["pixel_resolution"])
    if sample.get("footprint") is not None:
        data["footprint"] = [float(sample["footprint"][0]), float(sample["footprint"][1])]
    if sample.get("is_augmented"):
        data["is_augmented"] = True
    return data

def _label_key(anchor_label: int, target_label: int, anchor_name: str, target_name: str) -> str:
    if anchor_label == 1 and target_label == 1:
        return "positive_common"
    if anchor_label == 1 and target_label != 1:
        return f"positive_{anchor_name}"
    if anchor_label != 1 and target_label == 1:
        return f"positive_{target_name}"
    return "unlabelled"


def _pair_weight(anchor_record, target_record, gaussian_sigma: Optional[float]) -> float:
    if gaussian_sigma is None:
        return 1.0
    if anchor_record.coord is None or target_record.coord is None:
        return 1.0
    anchor_xy = np.asarray(anchor_record.coord, dtype=float)
    target_xy = np.asarray(target_record.coord, dtype=float)
    if anchor_xy.size < 2 or target_xy.size < 2:
        return 1.0
    dist = float(np.linalg.norm(anchor_xy[:2] - target_xy[:2]))
    if dist <= 0.0:
        return 1.0
    weight = math.exp(- (dist ** 2) / (2.0 * (gaussian_sigma ** 2)))
    return float(weight + 1e-8)


def _aggregate_targets(anchor_record, targets: Sequence, weights: Sequence[float], aggregator: str) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if aggregator != "weighted_pool":
        raise NotImplementedError(f"Aggregator '{aggregator}' is not implemented yet.")

    embeddings = [torch.from_numpy(target.embedding).float() for target in targets]
    if not embeddings:
        return None, torch.empty(0), torch.empty(0)
    stacked = torch.stack(embeddings)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.isnan(weight_tensor).any() or float(weight_tensor.sum()) <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    aggregated = torch.matmul(weight_tensor.unsqueeze(0), stacked).squeeze(0)
    return aggregated, weight_tensor, stacked


def _normalise_coord(coord: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if coord is None:
        return None
    try:
        x = float(coord[0])
        y = float(coord[1])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _weighted_average_coord(
    coords: Sequence[Optional[Tuple[float, float]]],
    weights: Sequence[float],
) -> Optional[Tuple[float, float]]:
    total_weight = 0.0
    accum_x = 0.0
    accum_y = 0.0
    for coord, weight in zip(coords, weights):
        if coord is None:
            continue
        try:
            w = float(weight)
        except Exception:
            continue
        if not math.isfinite(w) or w <= 0:
            continue
        accum_x += coord[0] * w
        accum_y += coord[1] * w
        total_weight += w
    if total_weight <= 0:
        return None
    return (accum_x / total_weight, accum_y / total_weight)


def _make_pair_meta(
    anchor_record,
    targets: Sequence,
    weights: Optional[torch.Tensor],
    is_augmented: bool,
    augmentation_role: Optional[str] = None,
) -> Dict[str, object]:
    target_coords = [_normalise_coord(getattr(target, "coord", None)) for target in targets]
    target_tile_ids = [getattr(target, "tile_id", None) for target in targets]
    target_labels = [int(getattr(target, "label", 0)) for target in targets]
    weights_list: Optional[List[float]] = None
    weighted_coord: Optional[Tuple[float, float]] = None
    if isinstance(weights, torch.Tensor) and weights.numel() == len(targets):
        weights_cpu = weights.detach().cpu()
        weights_list = [float(w) for w in weights_cpu.tolist()]
        weighted_coord = _weighted_average_coord(target_coords, weights_list)

    meta: Dict[str, object] = {
        "anchor_tile_id": getattr(anchor_record, "tile_id", None),
        "anchor_coord": _normalise_coord(getattr(anchor_record, "coord", None)),
        "anchor_region": getattr(anchor_record, "region", None),
        "anchor_label": int(getattr(anchor_record, "label", 0)),
        "target_tile_ids": target_tile_ids,
        "target_coords": target_coords,
        "target_labels": target_labels,
        "target_weights": weights_list,
        "target_weighted_coord": weighted_coord,
        "is_augmented": bool(is_augmented),
    }
    if augmentation_role is not None:
        meta["augmentation_role"] = augmentation_role
    return meta


if torch is not None:

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, *, num_layers: int = 2):
            super().__init__()
            if num_layers < 1:
                raise ValueError("ProjectionHead requires at least one layer.")
            hidden_dim = max(out_dim, min(in_dim, 512))
            layers: List[nn.Module] = []
            prev_dim = in_dim
            if num_layers == 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                for layer_idx in range(num_layers - 1):
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU(inplace=True))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class ProjectionHead:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ProjectionHead requires PyTorch. Install torch before training.")


def _matrix_inverse_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        eigvals, eigvecs = torch.linalg.eigh(mat)
    except RuntimeError:
        dim = mat.size(0)
        return torch.eye(dim, device=mat.device, dtype=mat.dtype)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return inv_sqrt


def _adaptive_tmat_jitter(mat: torch.Tensor, base: float = 1e-3, scale_ratio: float = 1e-2) -> float:
    if mat.numel() == 0:
        return 0.0
    try:
        fro_norm = torch.linalg.norm(mat, ord="fro")
    except RuntimeError:
        fro_norm = torch.sqrt(torch.sum(mat * mat))
    dim = float(mat.size(-1))
    if dim <= 0:
        return float(base)
    scaled = float(fro_norm.detach().cpu()) / math.sqrt(mat.numel())
    jitter = max(base, scaled * scale_ratio)
    return float(jitter)


def _prepare_output_dir(primary: Optional[Path], fallback_relative: str) -> Optional[Path]:
    if primary is not None:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except Exception as exc:
            print(f"[warn] Unable to create directory {primary}: {exc}. Falling back to local path.")
    fallback = Path.cwd() / fallback_relative
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception as exc:
        print(f"[warn] Unable to create fallback directory {fallback}: {exc}")
        return None


class _RunLogger:
    def __init__(self, cfg: AlignmentConfig):
        base = cfg.output_dir if cfg.output_dir is not None else cfg.log_dir
        target_dir = _prepare_output_dir(base, "overlap_alignment_outputs")
        self.path: Optional[Path] = None
        if target_dir is not None:
            self.path = target_dir / "overlap_alignment_stage1_run.log"

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception as exc:
                print(f"[warn] Unable to append to run log {self.path}: {exc}")
        print(f"[run-log] {message}")


def _persist_state(cfg: AlignmentConfig, proj_a: nn.Module, proj_b: nn.Module, summary: Dict[str, object]) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return

    state_path = target_dir / "overlap_alignment_stage1.pt"
    payload = {
        "projection_head_a": proj_a.state_dict(),
        "projection_head_b": proj_b.state_dict(),
        "summary": summary,
    }
    try:
        torch.save(payload, state_path)
        print(f"[info] saved stage-1 weights to {state_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist training state: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            alt_path = fallback_dir / state_path.name
            try:
                torch.save(payload, alt_path)
                print(f"[info] saved stage-1 weights to fallback location {alt_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist training state to fallback location: {exc_fallback}")


def _persist_metrics(
    cfg: AlignmentConfig,
    summary: Dict[str, object],
    epoch_history: Sequence[Dict[str, float]],
    debug_data: Optional[Dict[str, List[Dict[str, object]]]] = None,
) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return
    metrics_path = target_dir / "overlap_alignment_stage1_metrics.json"
    payload = {
        "summary": summary,
        "epoch_history": list(epoch_history),
    }
    if debug_data:
        filtered_debug = {key: value for key, value in debug_data.items() if value}
        if filtered_debug:
            payload["debug"] = filtered_debug
    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] saved metrics to {metrics_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist metrics: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            fallback_path = fallback_dir / metrics_path.name
            try:
                with fallback_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                print(f"[info] saved metrics to fallback location {fallback_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist metrics to fallback location: {exc_fallback}")


def _maybe_save_debug_figures(cfg: AlignmentConfig, debug_data: Optional[Dict[str, List[Dict[str, object]]]]) -> None:
    if debug_data is None:
        print("[warn] Debug data unavailable; skipping figure generation.")
        return
    viz_primary = None
    if cfg.output_dir is not None:
        viz_primary = cfg.output_dir / "overlap_visualizations" / "overlap"
    elif cfg.log_dir is not None:
        viz_primary = cfg.log_dir / "overlap_visualizations" / "overlap"
    viz_dir = _prepare_output_dir(viz_primary, "overlap_alignment_debug/overlap_visualizations/overlap")
    if viz_dir is None:
        return

    geometry = None
    overlap_json = cfg.overlap_pairs_path or cfg.overlap_pairs_augmented_path
    if overlap_json is not None:
        try:
            with Path(overlap_json).open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
            geometry = doc.get("overlap", {}).get("geometry")
        except Exception:
            geometry = None

    anchor_samples = debug_data.get("anchor_positive", []) or []
    target_samples = debug_data.get("target_positive", []) or []
    anchor_label = (
        anchor_samples[0].get("dataset")
        if anchor_samples and anchor_samples[0].get("dataset")
        else debug_data.get("anchor_name", "anchor")
    )
    target_label = (
        target_samples[0].get("dataset")
        if target_samples and target_samples[0].get("dataset")
        else debug_data.get("target_name", "target")
    )

    anchor_serialised = [_serialise_sample(sample) for sample in anchor_samples]
    target_serialised = [_serialise_sample(sample) for sample in target_samples]
    plot_path = viz_dir / "debug_positive_overlap.png"
    save_overlap_debug_plot(
        plot_path,
        geometry,
        anchor_serialised,
        target_serialised,
        title="Positive overlap",
        anchor_label=str(anchor_label),
        target_label=str(target_label),
        centroid_points=None,
    )
    print(f"[info] saved debug figure to {plot_path}")

    if cfg.use_positive_augmentation:
        aug_anchor = [sample for sample in anchor_serialised if sample.get("is_augmented")]
        aug_target = [sample for sample in target_serialised if sample.get("is_augmented")]
        if aug_anchor or aug_target:
            aug_path = viz_dir / "debug_positive_aug_overlap.png"
            save_overlap_debug_plot(
                aug_path,
                geometry,
                aug_anchor or anchor_serialised,
                aug_target or target_serialised,
                title="Positive overlap (augmented)",
                anchor_label=str(anchor_label),
                target_label=str(target_label),
                centroid_points=None,
            )
            print(f"[info] saved debug figure to {aug_path}")


if __name__ == "__main__":
    main()
