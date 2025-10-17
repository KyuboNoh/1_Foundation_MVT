# src/gfm4mpm/training/train_cls.py
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm


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


def train_classifier(encoder, mlp, train_loader, val_loader, epochs=50, lr=1e-3, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.eval().to(device)       # IMPORTANT: freeze encoder
    mlp.to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    best = {"f1": -1, "state_dict": None}
    for ep in range(1, epochs+1):
        mlp.train()
        running_loss = 0.0
        sample_count = 0
        for x, y in tqdm(train_loader, desc=f"CLS epoch {ep}"):
            x, y = x.to(device), y.float().to(device)
            with torch.no_grad():
                z = encoder.encode(x)
            p = mlp(z)
            loss = bce(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
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

        print(
            "[TRAIN] "
            f"loss={avg_loss:.4f} "
            f"f1={train_metrics['f1']:.3f} "
            f"mcc={train_metrics['mcc']:.3f} "
            f"auprc={train_metrics['auprc']:.3f} "
            f"auroc={train_metrics['auroc']:.3f} "
            f"acc={train_metrics['accuracy']:.3f} "
            f"bacc={train_metrics['balanced_accuracy']:.3f}"
        )
        print(
            "[VAL] "
            f"loss={val_loss:.4f} "
            f"f1={val_metrics['f1']:.3f} "
            f"mcc={val_metrics['mcc']:.3f} "
            f"auprc={val_metrics['auprc']:.3f} "
            f"auroc={val_metrics['auroc']:.3f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"bacc={val_metrics['balanced_accuracy']:.3f}"
        )

    if best["state_dict"] is not None:
        mlp.load_state_dict(best["state_dict"])
    return mlp

def eval_classifier(encoder, mlp, loader, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.eval().to(device)
    mlp.eval().to(device)
    targets, probs = _collect_outputs(encoder, mlp, loader, device)
    metrics = _compute_metrics(targets, probs)
    return (
        metrics["f1"],
        metrics["mcc"],
        metrics["auprc"],
        metrics["auroc"],
    )
