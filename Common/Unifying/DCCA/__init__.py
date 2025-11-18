from __future__ import annotations

import json
import math
import hashlib
import importlib.util
import sys
import gzip
from collections import Counter, defaultdict
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

try:  # optional plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

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
    _serialise_sample,
)
from Common.overlap_debug_plot import save_overlap_debug_plot

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

    class CrossAttentionAggregator(nn.Module):
        """Cross-attention aggregator for combining multiple target embeddings into a single representation."""
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
        """Wraps a projection head with an aggregator for set-to-set pairing."""
        def __init__(
            self, 
            aggregator: CrossAttentionAggregator, 
            proj_target: nn.Module, 
            *, 
            use_positional_encoding: bool
        ) -> None:
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

else:

    class ProjectionHead:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ProjectionHead requires PyTorch. Install torch before training.")

    class CrossAttentionAggregator:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CrossAttentionAggregator requires PyTorch. Install torch before training.")

    class AggregatorTargetHead:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("AggregatorTargetHead requires PyTorch. Install torch before training.")


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
    # Support loading either a ProjectionHead state_dict (keys like "net.0.weight")
    # or a wrapped AggregatorTargetHead state_dict where the projection is saved
    # under the prefix "proj_target." (e.g. "proj_target.net.0.weight").
    # If the latter, extract the proj_target sub-dictionary and continue.
    if any(key.startswith("proj_target.") for key in state_dict.keys()):
        sub: Dict[str, "torch.Tensor"] = {}
        for k, v in state_dict.items():
            if k.startswith("proj_target."):
                sub[k[len("proj_target.") :]] = v
        # replace local reference with the extracted projection state
        state_dict = sub

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


def _load_DCCA_projectors(
    dcca_path: Path,
    anchor_in_dim: int,
    target_in_dim: int,
    aggregator_config: Optional[Dict[str, object]] = None,
    device: Optional["torch.device"] = None,
) -> Tuple["nn.Module", "nn.Module"]:
    """
    Load DCCA projection heads from a checkpoint file.
    
    Args:
        dcca_path: Path to the DCCA checkpoint file
        anchor_in_dim: Expected input dimension for anchor projection head
        target_in_dim: Expected input dimension for target projection head
        aggregator_config: Optional dict with aggregator parameters if checkpoint contains aggregator.
            Required keys: 'projection_dim', 'num_layers', 'num_heads', 'use_positional_encoding'
            Optional keys: 'dropout' (default: 0.1)
        device: Device to load the projectors onto. If None, uses CPU.
    
    Returns:
        Tuple of (projector_anchor, projector_target) modules.
        If checkpoint contains aggregator, projector_target will be an AggregatorTargetHead,
        otherwise a ProjectionHead.
    
    Raises:
        FileNotFoundError: If dcca_path does not exist
        RuntimeError: If PyTorch is not available
        ValueError: If checkpoint format is invalid or dimensions don't match
    """
    if torch is None or nn is None:
        raise RuntimeError("Loading DCCA projectors requires PyTorch. Install torch before calling this function.")
    
    if not dcca_path.exists():
        raise FileNotFoundError(f"DCCA checkpoint not found: {dcca_path}")
    
    if device is None:
        device = torch.device("cpu")
    
    # Load checkpoint and extract state dictionaries
    checkpoint = torch.load(dcca_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {dcca_path} does not contain a valid dictionary payload.")
    
    state_a = checkpoint.get("projection_head_a")
    state_b = checkpoint.get("projection_head_b")
    
    if state_a is None or state_b is None:
        raise KeyError(f"Checkpoint {dcca_path} is missing DCCA projection head weights.")
    
    # Build anchor projection head
    projector_anchor = _projection_head_from_state(state_a)
    
    # Validate anchor input dimension
    actual_anchor_in = projector_anchor.net[0].in_features
    if actual_anchor_in != anchor_in_dim:
        raise ValueError(
            f"Anchor projection head expects input dimension {actual_anchor_in}, "
            f"but got {anchor_in_dim}"
        )
    
    # Check if checkpoint contains aggregator
    has_aggregator = any(k.startswith("aggregator.") for k in state_b.keys())
    
    if not has_aggregator:
        # No aggregator: load as ProjectionHead
        projector_target = _projection_head_from_state(state_b)
        
        # Validate target input dimension
        actual_target_in = projector_target.net[0].in_features
        if actual_target_in != target_in_dim:
            raise ValueError(
                f"Target projection head expects input dimension {actual_target_in}, "
                f"but got {target_in_dim}"
            )
    else:
        # Checkpoint contains aggregator: need to reconstruct AggregatorTargetHead
        if aggregator_config is None:
            raise ValueError(
                "Checkpoint contains aggregator but aggregator_config was not provided. "
                "Please provide aggregator configuration parameters."
            )
        
        # Extract configuration
        projection_dim = aggregator_config.get('projection_dim')
        num_layers = aggregator_config.get('num_layers', 4)
        num_heads = aggregator_config.get('num_heads', 4)
        use_pos_enc = aggregator_config.get('use_positional_encoding', False)
        dropout = aggregator_config.get('dropout', 0.1)
        
        if projection_dim is None:
            raise ValueError("aggregator_config must include 'projection_dim'")
        
        # CrossAttentionAggregator and AggregatorTargetHead are now defined in this module
        # Construct aggregator
        try:
            aggregator = CrossAttentionAggregator(
                anchor_dim=anchor_in_dim,
                target_dim=target_in_dim,
                agg_dim=projection_dim,
                num_layers=max(1, num_layers),
                num_heads=max(1, num_heads),
                dropout=float(dropout),
                use_positional_encoding=bool(use_pos_enc),
            )
        except Exception:
            # Fallback conservative construction
            aggregator = CrossAttentionAggregator(
                anchor_in_dim,
                target_in_dim,
                projection_dim,
                num_layers=max(1, num_layers),
                num_heads=max(1, num_heads),
                use_positional_encoding=bool(use_pos_enc),
            )
        
        # Extract proj_target and wrap in AggregatorTargetHead
        proj_target_inner = _projection_head_from_state(state_b)
        
        # Validate target projection head input dimension
        # When aggregator is present, the projection head takes aggregator output (projection_dim) as input
        actual_target_in = proj_target_inner.net[0].in_features
        if actual_target_in != projection_dim:
            raise ValueError(
                f"Target projection head expects input dimension {actual_target_in}, "
                f"but aggregator_config specifies projection_dim={projection_dim}. "
                f"These must match since the projection head receives aggregator output."
            )
        
        projector_target = AggregatorTargetHead(
            aggregator,
            proj_target_inner,
            use_positional_encoding=bool(use_pos_enc)
        )
        
        # Load the full state dict into the wrapped head
        try:
            projector_target.load_state_dict(state_b, strict=True)
        except Exception:
            # Try non-strict load if strict fails
            projector_target.load_state_dict(state_b, strict=False)
    
    # Move to device
    projector_anchor = projector_anchor.to(device)
    projector_target = projector_target.to(device)
    
    return projector_anchor, projector_target


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


def _persist_state(
    cfg: Any,
    proj_a: nn.Module,
    proj_b: nn.Module,
    *,
    filename: str = "overlap_alignment_stage1.pt",
) -> None:
    primary = None
    if getattr(cfg, "output_dir", None) is not None:
        primary = cfg.output_dir
    elif getattr(cfg, "log_dir", None) is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return

    state_path = target_dir / filename
    payload = {
        "projection_head_a": proj_a.state_dict(),
        "projection_head_b": proj_b.state_dict(),
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
    cfg: Any,
    summary: Dict[str, object],
    epoch_history: Sequence[Dict[str, float]],
    *,
    filename: str = "overlap_alignment_stage1_metrics.json",
) -> None:
    primary = None
    if getattr(cfg, "output_dir", None) is not None:
        primary = cfg.output_dir
    elif getattr(cfg, "log_dir", None) is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return
    metrics_path = target_dir / filename
    payload = {
        "summary": summary,
        "epoch_history": list(epoch_history),
    }

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


def _maybe_save_debug_figures(cfg: Any, debug_data: Optional[Dict[str, List[Dict[str, object]]]]) -> None:
    if debug_data is None:
        print("[warn] Debug data unavailable; skipping figure generation.")
        return
    viz_primary = None
    if getattr(cfg, "output_dir", None) is not None:
        viz_primary = cfg.output_dir / "overlap_visualizations" / "overlap"
    elif getattr(cfg, "log_dir", None) is not None:
        viz_primary = cfg.log_dir / "overlap_visualizations" / "overlap"
    viz_dir = _prepare_output_dir(viz_primary, "overlap_alignment_debug/overlap_visualizations/overlap")
    if viz_dir is None:
        return

    geometry = None
    overlap_json = getattr(cfg, "overlap_pairs_path", None) or getattr(cfg, "overlap_pairs_augmented_path", None)
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

    if getattr(cfg, "use_positive_augmentation", False):
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
    cfg: Any,
    projector_a: nn.Module,
    projector_b: nn.Module,
    anchor_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    anchor_tensor_post: Optional[torch.Tensor] = None,
    target_tensor_post: Optional[torch.Tensor] = None,
    pair_metadata: Sequence[Dict[str, object]],
    run_logger: Any,
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
    if getattr(cfg, "output_dir", None) is not None:
        analysis_primary = cfg.output_dir / "overlap_visualizations" / "alignment_analysis"
    elif getattr(cfg, "log_dir", None) is not None:
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
    if anchor_tensor_post is not None:
        projected_anchor = anchor_tensor_post.detach().cpu()
    else:
        projected_anchor = _project_in_batches(anchor_tensor, projector_a, device, analysis_batch)

    if target_tensor_post is not None:
        projected_target = target_tensor_post.detach().cpu()
    else:
        projected_target = _project_in_batches(target_tensor, projector_b, device, analysis_batch)

    try:
        subset_indices = _select_subset_indices(anchor_tensor.size(0), max_points=1000, seed=sample_seed)
        subset_list = subset_indices.tolist()
        if subset_indices.numel() >= 2:
            anchor_raw = anchor_tensor.detach().cpu()
            target_raw = target_tensor.detach().cpu()
            proj_anchor_raw = projected_anchor
            proj_target_raw = projected_target

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
            fig.savefig(tsne_path, dpi=200)
            plt.close(fig)
            run_logger.log(f"Saved t-SNE/UMAP alignment figure to {tsne_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create t-SNE/UMAP figure: {exc}")

    try:
        latent_metrics = _canonical_metrics(
            anchor_tensor.detach().to(projected_anchor.dtype),
            target_tensor.detach().to(projected_target.dtype),
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
        )
        metrics_path = analysis_dir / "alignment_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(latent_metrics, handle, indent=2)
        run_logger.log(f"Saved alignment metrics to {metrics_path}")
    except Exception as exc:
        run_logger.log(f"Failed to compute alignment metrics: {exc}")

    # Canonical correlation spectrum (updated to work with existing tensors). K.N. Nov 2025
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
                        (axes[0], vals_before[:k_plot], "Pre-Aggregation"),
                        (axes[1], vals_after[:k_plot], "Post-Aggregation"),
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

    # Geospatial distance heatmap (adapted to aggregated embeddings). K.N. Nov 2025
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
                (axes[0], pre_geo, "Pre-Aggregation"),
                (axes[1], post_geo, "Post-Aggregation"),
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
                # if val_mask.any():
                #     ax.scatter(
                #         geo_x[val_mask],
                #         geo_y[val_mask],
                #         facecolors="none",
                #         edgecolors="black",
                #         linewidths=0.8,
                #         s=60,
                #         label="Validation",
                #     )
                #     ax.legend(loc="upper right")
                ax.set_title(f"{title} Pairwise Distance")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                color_label = (
                    "||Latent_base(A) - Latent_base(B)||" if title == "Pre-Aggregation" else "||Latent_fused(A) - Latent_fused(B)||"
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)
            geo_path = analysis_dir / "geospatial_distance_map.png"
            fig.suptitle("Pairwise Distance Heatmap (Anchor vs Target)")
            fig.savefig(geo_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            run_logger.log(f"Saved geospatial distance heatmap to {geo_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create geospatial distance map: {exc}")

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


def _compute_dcca_checkpoint_hash(dcca_path: Path) -> str:
    """Compute SHA256 hash of DCCA checkpoint file for cache invalidation."""
    if not dcca_path.exists():
        return "no_checkpoint"
    
    hasher = hashlib.sha256()
    with open(dcca_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity


def _save_dcca_projections(
    output_dir: Path,
    dataset_name: str,
    sample_data: Dict[str, object],
    dcca_checkpoint_hash: str,
    aggregator_config: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save DCCA projection results to disk with gzip compression.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset (e.g., 'anchor', 'target', 'anchor_overlap')
        sample_data: Dictionary containing 'features', 'labels', 'indices', 'coords', etc.
        dcca_checkpoint_hash: Hash of DCCA checkpoint for cache validation
        aggregator_config: Optional aggregator configuration dict
    """
    cache_dir = output_dir / "dcca_projections"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert embeddings to float16 to save space (50% reduction)
    features = sample_data.get("features")
    if isinstance(features, torch.Tensor):
        features = features.half()  # float32 → float16
    
    # Prepare metadata for validation
    metadata = {
        "dcca_checkpoint_hash": dcca_checkpoint_hash,
        "dataset_name": dataset_name,
        "num_samples": len(sample_data.get("indices", [])),
        "feature_shape": list(features.shape) if hasattr(features, "shape") else None,
    }
    
    if aggregator_config is not None:
        metadata["aggregator_config"] = aggregator_config
    
    # Prepare save data with compressed features
    save_data = {
        "metadata": metadata,
        "sample_data": {
            "features": features,
            "labels": sample_data.get("labels"),
            "indices": sample_data.get("indices"),
            "coords": sample_data.get("coords"),
            "dataset": sample_data.get("dataset"),
            "metadata": sample_data.get("metadata"),
            "row_cols": sample_data.get("row_cols"),
            "mask_flags": sample_data.get("mask_flags"),
            "row_cols_mask": sample_data.get("row_cols_mask"),
        },
    }
    
    # Save with gzip compression (60-70% additional reduction)
    cache_file = cache_dir / f"{dataset_name}.pt.gz"
    
    with gzip.open(cache_file, 'wb', compresslevel=6) as f:
        torch.save(save_data, f)


def _load_dcca_projections(
    output_dir: Path,
    dataset_name: str,
    dcca_checkpoint_hash: str,
    aggregator_config: Optional[Dict[str, object]] = None,
    force_recompute: bool = False,
) -> Optional[Dict[str, object]]:
    """
    Load cached DCCA projection results from disk.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        dcca_checkpoint_hash: Expected hash of DCCA checkpoint
        aggregator_config: Expected aggregator configuration
        force_recompute: If True, ignore cache and return None
    
    Returns:
        Cached sample data dict if valid cache exists, None otherwise
    """
    if force_recompute:
        return None
    
    cache_file = output_dir / "dcca_projections" / f"{dataset_name}.pt.gz"
    
    if not cache_file.exists():
        return None
    
    try:
        # Load compressed file
        with gzip.open(cache_file, 'rb') as f:
            saved_data = torch.load(f, map_location='cpu')
        
        metadata = saved_data.get("metadata", {})
        
        # Validate checkpoint hash
        if metadata.get("dcca_checkpoint_hash") != dcca_checkpoint_hash:
            return None
        
        # Validate aggregator config if provided
        if aggregator_config is not None:
            saved_config = metadata.get("aggregator_config")
            if saved_config != aggregator_config:
                return None
        
        # Get sample data and convert features back to float32
        sample_data = saved_data.get("sample_data")
        if sample_data and "features" in sample_data:
            features = sample_data["features"]
            if isinstance(features, torch.Tensor) and features.dtype == torch.float16:
                sample_data["features"] = features.float()  # float16 → float32
        
        # Return the cached sample data
        return sample_data
    
    except Exception as e:
        # If loading fails for any reason, return None to trigger recomputation
        print(f"[warn] Failed to load cache for {dataset_name}: {e}")
        return None


def _check_complete_dcca_cache(
    output_dir: Path,
    anchor_name: str,
    target_name: str,
    dcca_checkpoint_hash: str,
    aggregator_config: Optional[Dict[str, object]] = None,
    force_recompute: bool = False,
) -> Tuple[
    Optional[Dict[str, Dict[str, object]]],  # dcca_sets
    Optional[Dict[str, object]],             # anchor_overlap_samples
    Optional[Dict[str, object]],             # anchor_non_overlap_samples  
    Optional[Dict[str, object]],             # target_overlap_samples
]:
    """
    Check if ALL required DCCA embeddings exist in cache.
    Returns complete set or all None if any are missing (all-or-nothing).
    """
    # Import here to avoid circular imports
    from Common.Unifying.Labels_TwoDatasets import _subset_classifier_sample
    if force_recompute:
        return None, None, None, None
    
    # Required cache files
    required_datasets = [
        anchor_name,
        target_name,
        f"{anchor_name}_overlap",
        f"{target_name}_overlap",
    ]
    
    cached_data = {}
    
    # Check all required caches exist and are valid
    for dataset_name in required_datasets:
        cached = _load_dcca_projections(
            output_dir=output_dir,
            dataset_name=dataset_name,
            dcca_checkpoint_hash=dcca_checkpoint_hash,
            aggregator_config=aggregator_config,
            force_recompute=False,
        )
        
        if cached is None:
            # Missing or invalid cache - return all None
            return None, None, None, None
        
        cached_data[dataset_name] = cached
    
    # All caches valid - reconstruct the expected data structure
    dcca_sets = {
        anchor_name: cached_data[anchor_name],
        target_name: cached_data[target_name],
    }
    
    anchor_overlap_samples = cached_data[f"{anchor_name}_overlap"]
    target_overlap_samples = cached_data[f"{target_name}_overlap"]
    
    # Reconstruct anchor_non_overlap_samples from overlap indices
    anchor_all = cached_data[anchor_name]
    anchor_overlap = cached_data[f"{anchor_name}_overlap"]
    
    overlap_indices = set(anchor_overlap.get("indices", []))
    if overlap_indices and anchor_all:
        keep_mask = [idx not in overlap_indices for idx in anchor_all.get("indices", [])]
        anchor_non_overlap_samples = _subset_classifier_sample(anchor_all, keep_mask, subset_tag="non_overlap")
    else:
        anchor_non_overlap_samples = None
    
    return dcca_sets, anchor_overlap_samples, anchor_non_overlap_samples, target_overlap_samples


def resolve_dcca_embeddings_and_projectors(
    cfg,  # AlignmentConfig
    args,  # argparse.Namespace
    workspace,  # OverlapAlignmentWorkspace
    anchor_name: str,
    target_name: str,
    anchor_vecs: List[torch.Tensor],
    target_stack_per_anchor: List[torch.Tensor],
    pair_metadata: List[Dict[str, object]],
    device: torch.device,
    run_logger,  # "_RunLogger"
    use_transformer_agg: bool,
    pn_label_maps: Dict[str, Optional[Dict[str, set]]],
    overlap_mask_info: Dict[str, object],
) -> Tuple[
    torch.nn.Module,  # projector_a
    torch.nn.Module,  # projector_b
    Dict[str, Dict[str, object]],  # dcca_sets
    Optional[Dict[str, object]],   # anchor_overlap_samples
    Optional[Dict[str, object]],   # anchor_non_overlap_samples
    Optional[Dict[str, object]],   # target_overlap_samples
]:
    """
    Unified DCCA resolution with three-stage fallback:
    1. Load cached embeddings (skip everything)
    2. Load weights + compute embeddings (skip training)
    3. Train new DCCA + compute embeddings (full pipeline)
    
    Returns projectors and all required embedding sets for classifier training.
    """
    
    # Prepare cache parameters
    weights_candidate = _resolve_dcca_weights_path(cfg, args.dcca_weights_path)

    print(args.dcca_weights_path, weights_candidate)
    dcca_checkpoint_hash = _compute_dcca_checkpoint_hash(weights_candidate) if weights_candidate else "no_checkpoint"
    
    aggregator_config = {
        'projection_dim': cfg.projection_dim,
        'num_layers': args.agg_trans_num_layers,
        'num_heads': args.agg_trans_num_heads,
        'dropout': float(getattr(cfg.dcca_training, "agg_dropout", 0.1)),
        'use_positional_encoding': bool(args.agg_trans_pos_enc),
    } if use_transformer_agg else None
    
    force_recompute = bool(getattr(args, 'force_recompute_dcca', False))
    
    # Stage 1: Check for complete cached embeddings
    run_logger.log("[DCCA] Stage 1: Checking for cached embeddings...")
    dcca_sets, anchor_overlap, anchor_non_overlap, target_overlap = _check_complete_dcca_cache(
        output_dir=Path(cfg.output_dir),
        anchor_name=anchor_name,
        target_name=target_name,
        dcca_checkpoint_hash=dcca_checkpoint_hash,
        aggregator_config=aggregator_config,
        force_recompute=force_recompute,
    )
    
    if dcca_sets is not None:
        run_logger.log("[DCCA] ✅ Found complete cached embeddings, loading projectors from weights...")
        
        # Load projectors from weights
        anchor_dim = int(anchor_vecs[0].numel()) if anchor_vecs else cfg.projection_dim
        try:
            target_dim = int(target_stack_per_anchor[0].size(1)) if target_stack_per_anchor else anchor_dim
        except Exception:
            target_dim = anchor_dim
        
        projector_a, projector_b = _load_DCCA_projectors(
            dcca_path=weights_candidate,
            anchor_in_dim=anchor_dim,
            target_in_dim=target_dim,
            aggregator_config=aggregator_config,
            device=device,
        )
        
        return projector_a, projector_b, dcca_sets, anchor_overlap, anchor_non_overlap, target_overlap
    
    # Stage 2: Check for weights, compute embeddings
    run_logger.log("[DCCA] Stage 2: Checking for saved weights...")
    
    if weights_candidate and weights_candidate.exists():
        run_logger.log(f"[DCCA] ✅ Found weights at {weights_candidate}, loading and computing embeddings...")
        
        # Load projectors from weights
        anchor_dim = int(anchor_vecs[0].numel()) if anchor_vecs else cfg.projection_dim
        try:
            target_dim = int(target_stack_per_anchor[0].size(1)) if target_stack_per_anchor else anchor_dim
        except Exception:
            target_dim = anchor_dim
        
        projector_a, projector_b = _load_DCCA_projectors(
            dcca_path=weights_candidate,
            anchor_in_dim=anchor_dim,
            target_in_dim=target_dim,
            aggregator_config=aggregator_config,
            device=device,
        )
        
        # Compute and cache embeddings
        dcca_sets, anchor_overlap, anchor_non_overlap, target_overlap = _compute_and_cache_dcca_embeddings(
            cfg=cfg,
            workspace=workspace,
            projector_a=projector_a,
            projector_b=projector_b,
            anchor_name=anchor_name,
            target_name=target_name,
            pn_label_maps=pn_label_maps,
            overlap_mask_info=overlap_mask_info,
            dcca_checkpoint_hash=dcca_checkpoint_hash,
            aggregator_config=aggregator_config,
            device=device,
            run_logger=run_logger,
            batch_size=cfg.dcca_training.batch_size,
        )
        
        return projector_a, projector_b, dcca_sets, anchor_overlap, anchor_non_overlap, target_overlap
    
    # Stage 3: Train from scratch
    run_logger.log("[DCCA] Stage 3: No cached embeddings or weights found, training DCCA from scratch...")
    
    # Check if training is enabled
    if not args.train_dcca:
        raise RuntimeError(
            "DCCA training is disabled (--no-train-dcca) but no cached embeddings or weights found. "
            "Either enable training with --train-dcca or provide valid cached data."
        )
    
    # This would need the full DCCA training implementation moved here
    # For now, raise an error indicating this needs to be implemented
    raise NotImplementedError(
        "Stage 3 (train from scratch) not yet implemented. "
        "Please ensure --read-dcca is enabled with valid weights, or implement the full training pipeline."
    )


def _compute_and_cache_dcca_embeddings(
    cfg,  # AlignmentConfig
    workspace,  # OverlapAlignmentWorkspace
    projector_a: torch.nn.Module,
    projector_b: torch.nn.Module,
    anchor_name: str,
    target_name: str,
    pn_label_maps: Dict[str, Optional[Dict[str, set]]],
    overlap_mask_info: Dict[str, object],
    dcca_checkpoint_hash: str,
    aggregator_config: Optional[Dict[str, object]],
    device: torch.device,
    run_logger,  # "_RunLogger"
    batch_size: int = 256,
) -> Tuple[
    Dict[str, Dict[str, object]],  # dcca_sets
    Optional[Dict[str, object]],   # anchor_overlap_samples
    Optional[Dict[str, object]],   # anchor_non_overlap_samples
    Optional[Dict[str, object]],   # target_overlap_samples
]:
    """
    Compute DCCA projections for all datasets and cache them.
    Handles both global and overlap-specific projections.
    """
    
    # Import here to avoid circular imports
    from Common.Unifying.Labels_TwoDatasets import _apply_projector_based_PUNlabels, _subset_classifier_sample
    import gc
    
    projector_a.eval()
    projector_b.eval()
    projector_map = {anchor_name: projector_a, target_name: projector_b}
    
    dcca_sets = {}
    anchor_overlap_samples = None
    anchor_non_overlap_samples = None
    target_overlap_samples = None
    
    # Use batch size from configuration
    memory_efficient_batch_size = batch_size  # Use the full config batch size
    run_logger.log(f"[DCCA] Using batch size: {memory_efficient_batch_size}")
    
    # Compute projections for each dataset
    for dataset_name, projector in projector_map.items():
        if projector is None:
            run_logger.log(f"[DCCA] Projector unavailable for dataset {dataset_name}; skipping.")
            continue
        
        # Try to load from cache first
        cached_data = _load_dcca_projections(
            output_dir=Path(cfg.output_dir),
            dataset_name=dataset_name,
            dcca_checkpoint_hash=dcca_checkpoint_hash,
            aggregator_config=aggregator_config,
            force_recompute=False,
        )
        
        if cached_data is not None:
            run_logger.log(f"[DCCA] Loaded cached projections for dataset {dataset_name}")
            sample_set = cached_data
        else:
            run_logger.log(f"[DCCA] Computing projections for dataset {dataset_name}...")
            sample_set = _apply_projector_based_PUNlabels(
                workspace=workspace,
                dataset_name=dataset_name,
                pn_lookup=pn_label_maps.get(dataset_name),
                projector=projector,
                batch_size=memory_efficient_batch_size,
                device=device,
                run_logger=run_logger,
                overlap_mask=overlap_mask_info,
                apply_overlap_filter=False,
            )
            
            # Save to cache for future runs
            if sample_set:
                _save_dcca_projections(
                    output_dir=Path(cfg.output_dir),
                    dataset_name=dataset_name,
                    sample_data=sample_set,
                    dcca_checkpoint_hash=dcca_checkpoint_hash,
                    aggregator_config=aggregator_config,
                )
                run_logger.log(f"[DCCA] Saved projections to cache for dataset {dataset_name}")
        
        if sample_set:
            dcca_sets[dataset_name] = sample_set
        
        # Compute overlap-specific samples
        if dataset_name == anchor_name and sample_set:
            # Try cache for overlap samples
            cached_overlap = _load_dcca_projections(
                output_dir=Path(cfg.output_dir),
                dataset_name=f"{dataset_name}_overlap",
                dcca_checkpoint_hash=dcca_checkpoint_hash,
                aggregator_config=aggregator_config,
                force_recompute=False,
            )
            
            if cached_overlap is not None:
                run_logger.log(f"[DCCA] Loaded cached overlap samples for {dataset_name}")
                anchor_overlap_samples = cached_overlap
            else:
                anchor_overlap_samples = _apply_projector_based_PUNlabels(
                    workspace=workspace, dataset_name=dataset_name, pn_lookup=pn_label_maps.get(dataset_name),
                    projector=projector, batch_size=memory_efficient_batch_size, device=device, run_logger=run_logger,
                    overlap_mask=overlap_mask_info, apply_overlap_filter=True,
                )
                if anchor_overlap_samples:
                    _save_dcca_projections(
                        output_dir=Path(cfg.output_dir),
                        dataset_name=f"{dataset_name}_overlap",
                        sample_data=anchor_overlap_samples,
                        dcca_checkpoint_hash=dcca_checkpoint_hash,
                        aggregator_config=aggregator_config,
                    )
            
            # Compute non-overlap samples
            overlap_indices = set(anchor_overlap_samples.get("indices", [])) if anchor_overlap_samples else set()
            if overlap_indices:
                keep_mask = [idx not in overlap_indices for idx in sample_set.get("indices", [])]
                anchor_non_overlap_samples = _subset_classifier_sample(sample_set, keep_mask, subset_tag="non_overlap")
        
        # Target overlap samples
        if dataset_name == target_name and sample_set:
            # Try cache for target overlap samples
            cached_target_overlap = _load_dcca_projections(
                output_dir=Path(cfg.output_dir),
                dataset_name=f"{dataset_name}_overlap",
                dcca_checkpoint_hash=dcca_checkpoint_hash,
                aggregator_config=aggregator_config,
                force_recompute=False,
            )
            
            if cached_target_overlap is not None:
                run_logger.log(f"[DCCA] Loaded cached overlap samples for {dataset_name}")
                target_overlap_samples = cached_target_overlap
            else:
                target_overlap_samples = _apply_projector_based_PUNlabels(
                    workspace=workspace, dataset_name=dataset_name, pn_lookup=pn_label_maps.get(dataset_name),
                    projector=projector, batch_size=memory_efficient_batch_size, device=device, run_logger=run_logger,
                    overlap_mask=None, apply_overlap_filter=False,
                )
                if target_overlap_samples:
                    _save_dcca_projections(
                        output_dir=Path(cfg.output_dir),
                        dataset_name=f"{dataset_name}_overlap",
                        sample_data=target_overlap_samples,
                        dcca_checkpoint_hash=dcca_checkpoint_hash,
                        aggregator_config=aggregator_config,
                    )
        
        # Force garbage collection after each dataset to free memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            run_logger.log(f"[DCCA] Cleared memory cache after processing {dataset_name}")
    
    # Final cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    run_logger.log("[DCCA] ✅ Completed computing and caching all DCCA projections with memory-efficient processing")
    return dcca_sets, anchor_overlap_samples, anchor_non_overlap_samples, target_overlap_samples


__all__ = [
    "ProjectionHead",
    "CrossAttentionAggregator",
    "AggregatorTargetHead",
    "dcca_loss",
    "_filter_singular_values",
    "_matrix_inverse_sqrt",
    "_adaptive_tmat_jitter",
    "_format_optional_scalar",
    "_resolve_dcca_weights_path",
    "_load_pretrained_dcca_state",
    "_projection_head_from_state",
    "_load_DCCA_projectors",
    "_has_nonfinite_gradients",
    "_canonical_metrics",
    "_project_in_batches",
    "_prepare_output_dir",
    "_persist_state",
    "_persist_metrics",
    "_maybe_save_debug_figures",
    "_select_subset_indices",
    "_reduce_dimensionality",
    "_create_debug_alignment_figures",
    "_get_bundle_embeddings",
    "_collect_full_dataset_projection",
    "_evaluate_dcca",
    "_train_DCCA",
    "reembedding_DCCA",
    "_compute_dcca_checkpoint_hash",
    "_save_dcca_projections",
    "_load_dcca_projections",
    "_check_complete_dcca_cache",
    "resolve_dcca_embeddings_and_projectors",
    "_compute_and_cache_dcca_embeddings",
]
