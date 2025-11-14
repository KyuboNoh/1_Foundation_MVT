from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from Common.cls.models.mlp_dropout import MLPDropout


#####################################################################################   
def run_inference_base(
    samples: Dict,
    cls: nn.Module,
    device: torch.device,
    output_dir: Path,
    run_logger: Any,  # _RunLogger type
    passes: int = 10,
    pos_crd: Optional[List[Tuple[float, float]]] = None,
    tag: Dict[str, Any] = None,
) -> Dict[str, object]:
    """Run inference on overlap data using only the cls."""
    
   
    # Get corresponding metadata
    matched_coords = samples["coords"]
    features = samples["features"].float().to(device)
    run_logger.log(f"[inference-cls] Processing {len(matched_coords)} anchor samples")
    
    # Prepare output directory
    if tag is not None:
        model_dir = output_dir / f"{tag}"
    else:
        model_dir = output_dir / "Base"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # MC Dropout inference
    cls.train()  # Enable dropout
    predictions_list = []
    
    # ✅ Check if model outputs probabilities or logits (To handle both PN and PU cases)
    outputs_probs = isinstance(cls, MLPDropout)  # MLPDropout outputs probabilities directly

    with torch.no_grad():
        for _ in range(passes):
            output = cls(features)
            
            if outputs_probs:
                # ✅ Model already outputs probabilities, use directly
                pred = output
            else:
                # ✅ Model outputs logits, apply sigmoid
                pred = torch.sigmoid(output)
            
            predictions_list.append(pred.cpu().numpy())
    
    cls.eval()  # Disable dropout
    
    # Compute statistics
    predictions_array = np.stack(predictions_list, axis=0)  # (passes, N)
    mean_pred = predictions_array.mean(axis=0)
    std_pred = predictions_array.std(axis=0)
    
    # Save raw predictions
    np.save(model_dir / "predictions_mean.npy", mean_pred)
    np.save(model_dir / "predictions_std.npy", std_pred)
    np.save(model_dir / "coordinates.npy", np.array(matched_coords))
    
    # Create scatter plots
    _create_inference_plots(
        model_dir=model_dir,
        model_name="base_cls",
        coords=matched_coords,
        mean_pred=mean_pred,
        std_pred=std_pred,
        pos_crd=pos_crd
    )

    # Compute summary statistics
    summary = _compute_inference_summary(
        model_name="base_cls",
        mean_pred=mean_pred,
        std_pred=std_pred,
        labels=None, 
    )
    
    # Save summary
    with open(model_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    run_logger.log(f"[inference-cls] Saved results to {model_dir}")
    
    return {
        "predictions_mean": mean_pred,
        "predictions_std": std_pred,
        "coordinates": matched_coords,
        "summary": summary
    }


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
