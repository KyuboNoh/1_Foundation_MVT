# src/gfm4mpm/training/train_cls.py
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics
from Common.data_utils import read_stack_patch
from torch.utils.data import DataLoader, Dataset

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

def dataloader_metric_inputORembedding(Xtr, Xval, ytr, yval, batch_size, positive_augmentation=False,
                      augmented_patches_all: Optional[np.ndarray]=None, augmented_sources_all: Optional[np.ndarray]=None,
                      pos_coord_to_index: Optional[Dict[Tuple[int, ...], int]]=None,
                      window_size=None, stack=None,
                      embedding: Optional[np.ndarray]=None,
                      epochs=50) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    # Prepare dataloaders and compute dataset metrics for training
    # Robust to have input in input domain (data domain) or embedding

    train_pos = int(sum(ytr))
    val_pos = int(sum(yval))

    metrics_summary: Dict[str, Any] = {}

    metrics_summary["train_samples"] = len(Xtr)
    metrics_summary["val_samples"] = len(Xval)
    metrics_summary["train_pos"] = train_pos
    metrics_summary["train_neg"] = len(Xtr) - train_pos
    metrics_summary["val_pos"] = val_pos
    metrics_summary["val_neg"] = len(Xval) - val_pos

    extra_train_samples: List[Tuple[np.ndarray, int]] = []
    train_aug_sources = np.array([], dtype=int)
    if positive_augmentation and augmented_patches_all is not None and augmented_sources_all is not None:
        train_pos_indices: set[int] = set()
        for coord, label in zip(Xtr, ytr):
            if label == 1:
                idx = pos_coord_to_index.get(tuple(coord))
                if idx is not None:
                    train_pos_indices.add(idx)
        if train_pos_indices:
            mask_train_aug = np.isin(augmented_sources_all, list(train_pos_indices))
            if mask_train_aug.any():
                train_aug_patches = augmented_patches_all[mask_train_aug]
                train_aug_sources = augmented_sources_all[mask_train_aug]
                extra_train_samples = [(patch, 1) for patch in train_aug_patches]
                metrics_summary['train_augmented'] = int(len(train_aug_sources))
        else:
            train_aug_sources = np.array([], dtype=int)
    else:
        train_aug_sources = np.array([], dtype=int)
    if 'train_augmented' not in metrics_summary:
        metrics_summary['train_augmented'] = int(len(extra_train_samples))
    metrics_summary['train_samples_with_aug'] = len(Xtr) + len(extra_train_samples)
    if window_size is not None:
        ds_tr = LabeledPatches(stack, Xtr, ytr, window=window_size, extra_samples=extra_train_samples)
        ds_va = LabeledPatches(stack, Xval, yval, window=window_size)
        worker_count = 0 if getattr(stack, 'kind', None) == 'raster' else 8
    elif embedding is not None:
        ds_tr = torch.utils.data.TensorDataset(
            torch.from_numpy(embedding[np.array(Xtr)]).float(),
            torch.from_numpy(np.array(ytr)).long()
        )
        ds_va = torch.utils.data.TensorDataset(
            torch.from_numpy(embedding[np.array(Xval)]).float(),
            torch.from_numpy(np.array(yval)).long()
        )
        worker_count = 0
    else:
        raise NotImplementedError("Non-windowed data loading is not implemented.")


    if extra_train_samples:
        print(f'    [info] Added {len(extra_train_samples)} augmented positive samples to training loader.')

#    if worker_count == 0:
#        print('[info] Using single-process data loading for raster stack')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=worker_count)
    metrics_summary["batch_size"] = int(batch_size)
    metrics_summary["epochs"] = int(epochs)

    return dl_tr, dl_va, metrics_summary


class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, window=32, extra_samples: Optional[Sequence[Tuple[np.ndarray, int]]] = None):
        self.stack = stack
        self.coords = list(coords)
        self.labels = list(labels)
        self.window = window
        self.extra: List[Tuple[np.ndarray, int]] = []
        if extra_samples:
            for patch, label in extra_samples:
                arr = np.asarray(patch, dtype=np.float32)
                self.extra.append((arr, int(label)))

    def __len__(self):
        return len(self.coords) + len(self.extra)

    def __getitem__(self, idx):
        if idx < len(self.coords):
            coord = self.coords[idx]
            x = read_stack_patch(self.stack, coord, self.window)
            y = self.labels[idx]
        else:
            patch, label = self.extra[idx - len(self.coords)]
            x = patch
            y = label
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)
