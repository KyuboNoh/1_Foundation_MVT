from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

try:  # optional raster IO
    import rasterio
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore[assignment]

from Common.Unifying.Labels_TwoDatasets.fusion_utils.workspace import (
    DatasetBundle,
    OverlapAlignmentWorkspace,
)
from Common.Unifying.Labels_TwoDatasets import (
    _normalise_coord,
    _normalise_row_col,
)

DEFAULT_INFERENCE_BATCH_SIZE = 2048

if torch is not None and nn is not None:

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
                for _ in range(num_layers - 1):
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


def dcca_loss(
    z_a: "torch.Tensor",
    z_b: "torch.Tensor",
    eps: float = 1e-5,
    drop_ratio: float = 0.01,
) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, object]]:
    if torch is None:
        raise RuntimeError("dcca_loss requires PyTorch. Install torch before calling this function.")

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


def _filter_singular_values(singular_values: "torch.Tensor", drop_ratio: float) -> Tuple["torch.Tensor", Dict[str, object]]:
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
        info["kept_min"] = None
        info["kept_max"] = None
    return filtered, info


def _matrix_inverse_sqrt(mat: "torch.Tensor", eps: float) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("matrix_inverse_sqrt requires PyTorch. Install torch before calling this function.")

    try:
        eigvals, eigvecs = torch.linalg.eigh(mat)
    except RuntimeError:
        dim = mat.size(0)
        return torch.eye(dim, device=mat.device, dtype=mat.dtype)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return inv_sqrt


def _adaptive_tmat_jitter(mat: "torch.Tensor", base: float = 1e-3, scale_ratio: float = 1e-2) -> float:
    if torch is None:
        raise RuntimeError("adaptive_tmat_jitter requires PyTorch. Install torch before calling this function.")

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


def _resolve_dcca_weights_path(cfg: Any, override: Optional[str]) -> Optional[Path]:
    if override:
        try:
            candidate = Path(override).expanduser().resolve()
        except Exception:
            candidate = Path(override)
        if candidate.exists():
            return candidate
        return candidate

    candidates: List[Path] = []
    output_dir = getattr(cfg, "output_dir", None)
    if output_dir is not None:
        try:
            candidates.append((Path(output_dir) / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
        except Exception:
            pass
    log_dir = getattr(cfg, "log_dir", None)
    if log_dir is not None:
        try:
            candidates.append((Path(log_dir) / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
        except Exception:
            pass
    candidates.append((Path.cwd() / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _load_pretrained_dcca_state(path: Path) -> Tuple[Tuple[Dict[str, "torch.Tensor"], Dict[str, "torch.Tensor"]], Optional[Dict[str, object]]]:
    if torch is None:
        raise RuntimeError("Loading DCCA state requires PyTorch. Install torch before calling this function.")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} does not contain a valid dictionary payload.")
    if "projection_head_a" not in checkpoint or "projection_head_b" not in checkpoint:
        raise KeyError(f"Checkpoint {path} is missing DCCA projection head weights.")
    state_a = checkpoint["projection_head_a"]
    state_b = checkpoint["projection_head_b"]
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        raise ValueError(f"Checkpoint {path} does not contain valid state dicts for DCCA projections.")
    summary = checkpoint.get("summary")
    if summary is not None and not isinstance(summary, dict):
        summary = None
    return (state_a, state_b), summary


def _projection_head_from_state(state_dict: Dict[str, "torch.Tensor"]) -> "nn.Module":
    if torch is None or nn is None:
        raise RuntimeError("Restoring DCCA projection heads requires PyTorch. Install torch before calling this function.")
    weight_keys = [key for key in state_dict.keys() if key.endswith(".weight")]
    if not weight_keys:
        raise ValueError("DCCA state dict does not contain linear layer weights.")

    def _layer_index(key: str) -> int:
        parts = key.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    sorted_keys = sorted(weight_keys, key=_layer_index)
    first_weight = state_dict[sorted_keys[0]]
    last_weight = state_dict[sorted_keys[-1]]
    if first_weight.ndim != 2 or last_weight.ndim != 2:
        raise ValueError("Unexpected weight tensor shape in DCCA state dict.")
    in_dim = int(first_weight.size(1))
    out_dim = int(last_weight.size(0))
    num_layers = len(sorted_keys)
    head = ProjectionHead(in_dim, out_dim, num_layers=num_layers)
    head.load_state_dict(state_dict, strict=True)
    return head


def _has_nonfinite_gradients(modules: Sequence[nn.Module]) -> bool:
    for module in modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
    return False

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

def _project_in_batches(
    tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    progress_desc: Optional[str] = None,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty_like(tensor)
    outputs: List[torch.Tensor] = []
    batch_size = max(1, batch_size)
    with torch.no_grad():
        iterator = range(0, tensor.size(0), batch_size)
        if progress_desc and tqdm is not None:
            total = math.ceil(tensor.size(0) / batch_size)
            iterator = tqdm(iterator, desc=progress_desc, leave=False, total=total)
        for start in iterator:
            end = min(start + batch_size, tensor.size(0))
            chunk = tensor[start:end].to(device)
            projected = model(chunk).detach().cpu()
            outputs.append(projected)
    return torch.cat(outputs, dim=0)

def _get_bundle_embeddings(bundle: DatasetBundle) -> torch.Tensor:
    if bundle.embeddings is not None and bundle.embeddings.numel() > 0:
        return bundle.embeddings.detach().cpu().float()
    tensors: List[torch.Tensor] = []
    for record in bundle.records:
        emb_arr = np.asarray(record.embedding, dtype=np.float32)
        tensors.append(torch.from_numpy(emb_arr))
    if not tensors:
        return torch.empty(0, 0, dtype=torch.float32)
    return torch.stack(tensors, dim=0)

def _collect_full_dataset_projection(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    projector: Optional[nn.Module],
    device: torch.device,
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    overlap_mask: Optional[Dict[str, object]] = None,
    mask_only: bool = True,
) -> Optional[Tuple[DatasetBundle, torch.Tensor, List[int], List[Optional[Tuple[int, int]]], List[bool]]]:
    if projector is None:
        return None
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None or not bundle.records:
        return None
    embeddings = _get_bundle_embeddings(bundle)
    if embeddings.numel() == 0:
        return None
    selected_indices: List[int] = []
    selected_rowcols: List[Optional[Tuple[int, int]]] = []
    mask_flags: List[bool] = []
    mask_array = overlap_mask.get("array") if overlap_mask else None
    mask_transform = overlap_mask.get("transform") if overlap_mask else None
    mask_shape = overlap_mask.get("shape") if overlap_mask else None
    mask_nodata = overlap_mask.get("nodata") if overlap_mask else None
    use_mask = mask_array is not None and mask_transform is not None and mask_shape is not None and rasterio is not None
    for idx, record in enumerate(bundle.records):
        coord = _normalise_coord(getattr(record, "coord", None))
        if coord is None:
            continue
        mask_inside = False
        mask_rowcol: Optional[Tuple[int, int]] = None
        include = True
        if use_mask:
            try:
                row, col = rasterio.transform.rowcol(mask_transform, coord[0], coord[1])
            except Exception:
                include = not mask_only
            else:
                include = 0 <= row < int(mask_shape[0]) and 0 <= col < int(mask_shape[1])
                if include:
                    value = mask_array[row, col]
                    if mask_nodata is not None and np.isfinite(mask_nodata):
                        if np.isclose(value, mask_nodata) or not np.isfinite(value):
                            value = 0
                    mask_inside = bool(value != 0)
                    mask_rowcol = (int(row), int(col))
                    if mask_only and not mask_inside:
                        include = False
                else:
                    include = not mask_only
        else:
            include = True

        if not include:
            continue

        selected_indices.append(idx)
        if use_mask and mask_rowcol is not None:
            selected_rowcols.append(mask_rowcol if mask_inside else None)
        else:
            rc = getattr(record, "row_col", None)
            selected_rowcols.append(_normalise_row_col(rc))
        mask_flags.append(mask_inside)
    if not selected_indices:
        return None
    index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
    if index_tensor.numel() == 0:
        return None
    embeddings = embeddings.index_select(0, index_tensor)
    if embeddings.numel() == 0:
        return None
    projected = _project_in_batches(
        embeddings,
        projector,
        device,
        batch_size,
        progress_desc=f"Project {dataset_name}",
    )
    return bundle, projected, selected_indices, selected_rowcols, mask_flags

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
            normalised[key] = value
    return normalised

def _train_DCCA(
    *,
    cfg: "AlignmentConfig",
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
    train_dcca: bool,
    pretrained_state: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    pretrained_summary: Optional[Dict[str, object]],
    pretrained_path: Optional[Path],
) -> Tuple[bool, Optional[nn.Module], Optional[nn.Module], List[Dict[str, object]], int, Optional[str]]:
    projection_dim = int(cfg.projection_dim)
    attempt = 1
    failure_reason: Optional[str] = None
    max_attempts = 10

    pretrained_proj_dim: Optional[int] = None
    pretrained_layers: Optional[int] = None
    if isinstance(pretrained_summary, dict):
        proj_value = pretrained_summary.get("projection_dim")
        layer_value = pretrained_summary.get("projection_layers")
        try:
            pretrained_proj_dim = int(proj_value) if proj_value is not None else None
        except (TypeError, ValueError):
            pretrained_proj_dim = None
        try:
            pretrained_layers = int(layer_value) if layer_value is not None else None
        except (TypeError, ValueError):
            pretrained_layers = None

    warned_dim_mismatch = False
    warned_layer_mismatch = False

    if not train_dcca:
        if pretrained_state is None:
            failure_reason = "Requested --read-dcca without providing pretrained weights."
            run_logger.log(failure_reason)
            return False, None, None, [], projection_dim, failure_reason
        if pretrained_proj_dim is not None:
            projection_dim = int(pretrained_proj_dim)
        current_dim = max(int(projection_dim), 1)
        if pretrained_layers is not None and pretrained_layers != mlp_layers:
            failure_reason = (
                f"Pretrained DCCA weights expect {pretrained_layers} layers but {mlp_layers} were requested."
            )
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
        projector_a = ProjectionHead(anchor_dim, current_dim, num_layers=mlp_layers).to(device)
        projector_b = ProjectionHead(target_dim, current_dim, num_layers=mlp_layers).to(device)
        try:
            projector_a.load_state_dict(pretrained_state[0], strict=True)
            projector_b.load_state_dict(pretrained_state[1], strict=True)
        except RuntimeError as exc:
            failure_reason = f"Failed to load pretrained DCCA weights: {exc}"
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
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
        if train_eval is None:
            failure_reason = "Unable to evaluate pretrained DCCA weights on training data."
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
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
        train_loss_val = train_eval.get("loss")
        train_corr_val = train_eval.get("mean_corr")
        train_tcc_val = train_eval.get("tcc_sum")
        train_tcc_mean_val = train_eval.get("tcc_mean")
        train_k_val = train_eval.get("k")
        val_loss_val = val_eval.get("loss") if val_eval else None
        val_corr_val = val_eval.get("mean_corr") if val_eval else None
        val_tcc_val = val_eval.get("tcc_sum") if val_eval else None
        val_tcc_mean_val = val_eval.get("tcc_mean") if val_eval else None
        val_k_val = val_eval.get("k") if val_eval else None
        epoch_history: List[Dict[str, object]] = [
            {
                "epoch": 0,
                "loss": float(train_loss_val) if train_loss_val is not None else None,
                "mean_correlation": float(train_corr_val) if train_corr_val is not None else None,
                "train_eval_loss": float(train_loss_val) if train_loss_val is not None else None,
                "train_eval_mean_correlation": float(train_corr_val) if train_corr_val is not None else None,
                "train_eval_tcc": float(train_tcc_val) if train_tcc_val is not None else None,
                "train_eval_tcc_mean": float(train_tcc_mean_val) if train_tcc_mean_val is not None else None,
                "train_eval_k": float(train_k_val) if train_k_val is not None else None,
                "val_eval_loss": float(val_loss_val) if val_loss_val is not None else None,
                "val_eval_mean_correlation": float(val_corr_val) if val_corr_val is not None else None,
                "val_eval_tcc": float(val_tcc_val) if val_tcc_val is not None else None,
                "val_eval_tcc_mean": float(val_tcc_mean_val) if val_tcc_mean_val is not None else None,
                "val_eval_k": float(val_k_val) if val_k_val is not None else None,
                "batches": 0,
                "projection_dim": current_dim,
            }
        ]
        source_desc = f" from {pretrained_path}" if pretrained_path is not None else ""
        run_logger.log(f"Loaded pretrained DCCA weights{source_desc} without additional training.")
        return True, projector_a, projector_b, epoch_history, current_dim, None

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

        state_for_attempt = None
        if pretrained_state is not None:
            if pretrained_proj_dim is not None and int(pretrained_proj_dim) != current_dim:
                if not warned_dim_mismatch:
                    run_logger.log(
                        f"Pretrained DCCA projection_dim {pretrained_proj_dim} does not match attempt dimension {current_dim}; skipping weight initialisation."
                    )
                    warned_dim_mismatch = True
            elif pretrained_layers is not None and pretrained_layers != mlp_layers:
                if not warned_layer_mismatch:
                    run_logger.log(
                        f"Pretrained DCCA projection layers {pretrained_layers} do not match configured {mlp_layers}; skipping weight initialisation."
                    )
                    warned_layer_mismatch = True
                state_for_attempt = pretrained_state

        if state_for_attempt is not None:
            try:
                projector_a.load_state_dict(state_for_attempt[0], strict=True)
                projector_b.load_state_dict(state_for_attempt[1], strict=True)
                source_desc = f" from {pretrained_path}" if pretrained_path is not None else ""
                run_logger.log(f"Loaded pretrained DCCA weights{source_desc} for attempt {attempt}.")
            except RuntimeError as exc:
                run_logger.log(f"Failed to load pretrained DCCA weights at attempt {attempt}: {exc}")

        optimizer = torch.optim.Adam(
            list(projector_a.parameters()) + list(projector_b.parameters()),
            lr=cfg.dcca_training.lr,
        )

        epoch_history: List[Dict[str, object]] = []
        attempt_success = True

        for epoch in range(cfg.dcca_training.epochs):
            running_loss = 0.0
            running_corr = 0.0
            batches = 0
            iterator = loader
            if tqdm is not None:
                iterator = tqdm(
                    loader,
                    desc=f"epoch {epoch+1}/{cfg.dcca_training.epochs} (proj={current_dim})",
                    leave=False,
                )

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

def reembedding_DCCA(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    projector: Optional[nn.Module],
    device: torch.device,
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    mask_only: bool = True,
) -> Optional[Dict[str, object]]:
    collected = _collect_full_dataset_projection(
        workspace,
        dataset_name,
        projector,
        device,
        batch_size=batch_size,
        overlap_mask=overlap_mask,
        mask_only=mask_only,
    )
    if collected is None:
        return None
    bundle, projected, record_indices, record_rowcols, mask_flags = collected
    projected_cpu = projected.detach().cpu()
    metadata_entries: List[Dict[str, object]] = []
    labels: List[int] = []
    coords: List[Optional[Tuple[float, float]]] = []
    for pos, (record_idx, row_col, mask_flag) in enumerate(zip(record_indices, record_rowcols, mask_flags)):
        record = bundle.records[record_idx]
        coord = _normalise_coord(getattr(record, "coord", None))
        label_val = int(getattr(record, "label", 0))
        coords.append(coord)
        metadata_entries.append(
            {
                "dataset": dataset_name,
                "index": int(getattr(record, "index", record_idx)),
                "record_index": int(record_idx),
                "coord": coord,
                "region": getattr(record, "region", None),
                "row_col": getattr(record, "row_col", None),
                "row_col_mask": [int(row_col[0]), int(row_col[1])] if row_col is not None else None,
                "tile_id": getattr(record, "tile_id", None),
                "position": int(pos),
                "label": label_val,
                "overlap_flag": bool(mask_flag),
            }
        )
        labels.append(label_val)
    return {
        "dataset": dataset_name,
        "features": projected_cpu,
        "indices": [int(idx) for idx in record_indices],
        "coords": coords,
        "metadata": metadata_entries,
        "labels": np.asarray(labels, dtype=np.int16),
        "row_cols": [tuple(rc) if rc is not None else None for rc in record_rowcols],
        "mask_flags": [bool(flag) for flag in mask_flags],
        "row_cols_mask": [tuple(rc) if rc is not None else None for rc in record_rowcols],
    }

__all__ = [
    "ProjectionHead",
    "dcca_loss",
    "_filter_singular_values",
    "_matrix_inverse_sqrt",
    "_adaptive_tmat_jitter",
    "_format_optional_scalar",
    "_resolve_dcca_weights_path",
    "_load_pretrained_dcca_state",
    "_projection_head_from_state",
    "_has_nonfinite_gradients",
    "_canonical_metrics",
    "_project_in_batches",
    "_get_bundle_embeddings",
    "_collect_full_dataset_projection",
    "_evaluate_dcca",
    "_train_DCCA",
    "reembedding_DCCA",
]
