from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


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
]
