# src/gfm4mpm/training/train_cls.py
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics


def _collect_outputs(encoder, mlp, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run encoder+MLP over a loader and return stacked labels and probabilities on CPU."""
    if loader is None:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )

    labels: List[torch.Tensor] = []
    probs: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            z = encoder.encode(x)
            p = mlp(z)
        labels.append(y.detach().view(-1).cpu().long())
        probs.append(p.detach().view(-1).cpu().float())

    if not labels:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )

    return torch.cat(labels), torch.cat(probs)


def _compute_metrics(targets: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics given labels and predicted probabilities."""
    metrics: Dict[str, float] = {
        "f1": float("nan"),
        "mcc": float("nan"),
        "auprc": float("nan"),
        "auroc": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
    }

    if targets.numel() == 0:
        return metrics

    targets_np = targets.numpy()
    probs_np = probs.numpy()
    preds_np = (probs_np >= 0.5).astype(int)

    # Basic rate calculations
    total = targets_np.size
    correct = (preds_np == targets_np).sum()
    tp = ((preds_np == 1) & (targets_np == 1)).sum()
    tn = ((preds_np == 0) & (targets_np == 0)).sum()
    fp = ((preds_np == 1) & (targets_np == 0)).sum()
    fn = ((preds_np == 0) & (targets_np == 1)).sum()

    metrics["accuracy"] = correct / total if total else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    metrics["balanced_accuracy"] = 0.5 * (tpr + tnr)

    metrics["f1"] = f1_score(targets_np, preds_np, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(targets_np, preds_np)

    unique_classes = np.unique(targets_np)
    if unique_classes.size > 1:
        preds_tensor = probs.float()
        targets_tensor = targets.int()
        metrics["auroc"] = float(BinaryAUROC()(preds_tensor, targets_tensor))
        metrics["auprc"] = float(BinaryAveragePrecision()(preds_tensor, targets_tensor))

    return metrics


def train_classifier(
    encoder,
    mlp,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
    return_history: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.eval().to(device)       # IMPORTANT: freeze encoder
    mlp.to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    if loss_weights is None:
        loss_weights = {"bce": 1.0}
    best = {"f1": -1, "state_dict": None}
    history = []
    for ep in range(1, epochs+1):
        mlp.train()
        running_loss = 0.0
        sample_count = 0
        for x, y in tqdm(train_loader, desc=f"CLS epoch {ep}"):
            x, y = x.to(device), y.float().to(device)
            with torch.no_grad():
                z = encoder.encode(x)
            p = mlp(z)
            raw_loss = bce(p, y)
            weighted_loss = loss_weights.get("bce", 1.0) * raw_loss
            opt.zero_grad()
            weighted_loss.backward()
            opt.step()

            running_loss += raw_loss.item() * y.size(0)
            sample_count += y.size(0)

        avg_loss = running_loss / sample_count if sample_count else float('nan')

        mlp.eval()
        train_targets, train_probs = _collect_outputs(encoder, mlp, train_loader, device)
        train_metrics = _compute_metrics(train_targets, train_probs)
        val_targets, val_probs = _collect_outputs(encoder, mlp, val_loader, device)
        val_metrics = _compute_metrics(val_targets, val_probs)
        if val_targets.numel():
            val_loss = F.binary_cross_entropy(val_probs, val_targets.float()).item()
        else:
            val_loss = float("nan")

        val_f1 = val_metrics.get("f1", float("nan"))
        comparable_f1 = val_f1 if not math.isnan(val_f1) else -1.0
        if comparable_f1 > best["f1"]:
            best = {"f1": comparable_f1, "state_dict": mlp.state_dict()}

        train_log = {"loss": float(avg_loss), **train_metrics}
        val_log = {"loss": float(val_loss), **val_metrics}

        log_metrics("train", train_log, order=DEFAULT_METRIC_ORDER)
        log_metrics("val", val_log, order=DEFAULT_METRIC_ORDER)

        history.append(
            {
                "epoch": int(ep),
                "train": normalize_metrics(train_log),
                "val": normalize_metrics(val_log),
                "train_weighted_loss": float(weighted_loss.item()),
                "val_weighted_loss": float(loss_weights.get("bce", 1.0) * val_loss) if not math.isnan(val_loss) else float("nan"),
            }
        )

    if best["state_dict"] is not None:
        mlp.load_state_dict(best["state_dict"])
    if return_history:
        return mlp, history
    return mlp

def eval_classifier(encoder, mlp, loader, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.eval().to(device)
    mlp.eval().to(device)
    targets, probs = _collect_outputs(encoder, mlp, loader, device)
    metrics = _compute_metrics(targets, probs)
    if targets.numel():
        raw_loss = torch.nn.functional.binary_cross_entropy(probs, targets.float())
        metrics["loss"] = float(raw_loss.item())
        metrics["weighted_loss"] = float(raw_loss.item())
    else:
        metrics["loss"] = float("nan")
        metrics["weighted_loss"] = float("nan")
    return metrics
