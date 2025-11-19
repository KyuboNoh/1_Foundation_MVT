"""
Base training functions for PN (Positive-Negative) classifiers with meta-evaluation support.

This module provides:
1. Base classifier training functions
2. Positive dropout training with rotation schedule
3. Modular meta-evaluation metric functions
4. Cached prediction loading and evaluation
"""

from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Common.cls.training.train_cls import (
    dataloader_metric_inputORembedding,
    train_classifier,
)
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.sampling.likely_negatives import (
    pu_select_negatives, 
    pu_select_negatives_optimized_for_rotation, 
    precompute_distances_for_rotation_drops,
    precompute_distances_per_positive,
    pu_select_negatives_from_individual_distances,
)

# All metrics are now treated equally - no default registry
POSITIVE_ITERATION_METRICS = {"PosDrop_Acc", "Focus"}
NEGATIVE_META_EVALUATION = {"pu_fpr", "background_rejection", "pu_npv", "pu_fdr", "pu_tpr"}
EXTENDED_META_METRICS = {"pauc", "topk"} | NEGATIVE_META_EVALUATION


@dataclass
class MetaEvaluationConfig:
    metrics: Set[str] = field(default_factory=lambda: {"PosDrop_Acc", "Focus", "pauc", "topk"})
    pauc_prior_variants: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3])
    topk_k_ratio: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])
    topk_k_values: List[int] = field(default_factory=list)
    topk_area_percentages: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    pu_prior_range: List[float] = field(default_factory=lambda: [0.05, 0.25])
    pu_metric_thresholds: List[float] = field(default_factory=list)
    pu_metric_threshold_count: int = 25

    @classmethod
    def from_args(cls, args: Any) -> "MetaEvaluationConfig":
        metrics = set(getattr(args, "meta_evaluation", [])) if getattr(args, "meta_evaluation", None) else {"PosDrop_Acc", "Focus", "pauc", "topk"}
        return cls(
            metrics=metrics,
            pauc_prior_variants=list(getattr(args, "pauc_prior_variants", [0.01, 0.05, 0.1, 0.2, 0.3])),
            topk_k_ratio=list(getattr(args, "topk_k_ratio", [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])),
            topk_k_values=list(getattr(args, "topk_k_values", [])),
            topk_area_percentages=list(getattr(args, "topk_area_percentages", [0.1, 0.5, 1.0, 2.0, 5.0])),
            pu_prior_range=list(getattr(args, "pu_prior_range", [0.05, 0.25])),
            pu_metric_thresholds=list(getattr(args, "pu_metric_thresholds", [])),
            pu_metric_threshold_count=int(getattr(args, "pu_metric_threshold_count", 25)),
        )



# Meta-evaluation functions are now called directly - no registry needed

# ============================================================================
# Meta-Evaluation Metric Functions
# ============================================================================

def compute_posdrop_acc(
    *,
    predictions_mean: np.ndarray,
    all_pos_idx: List[int],
    pos_indices_this_iter: set,
) -> float:
    """
    Compute PosDrop_Acc: Average prediction accuracy on positives dropped in this iteration.
    
    Args:
        predictions_mean: Mean predictions array (all samples)
        all_pos_idx: List of all positive sample indices
        pos_indices_this_iter: Set of positive indices used in training (not dropped)
    
    Returns:
        Accuracy score (mean of dropped positive predictions)
    """
    dropped_pos_idx = [idx for idx in all_pos_idx if idx not in pos_indices_this_iter]
    
    if len(dropped_pos_idx) > 0:
        dropped_pos_probs = predictions_mean[dropped_pos_idx]
        accuracy = float(np.mean(dropped_pos_probs))
        return accuracy
    else:
        return 0.0


def compute_focus(
    *,
    predictions_mean: np.ndarray,
) -> float:
    """
    Compute Focus: Measure of model selectivity/sparsity.
    
    Focus = 1.0 - (sum of predictions / total predictions)
    Higher focus means the model is more selective (predicts fewer areas as positive).
    
    Args:
        predictions_mean: Mean predictions array (all samples)
    
    Returns:
        Focus score
    """
    predictions_sum = predictions_mean.sum()
    total_predictions = len(predictions_mean)
    focus_score = 1.0 - (predictions_sum / total_predictions)
    return float(focus_score)



def compute_meta_evaluation_metric(
    *,
    metric_name: str,
    predictions_mean: np.ndarray,
    all_pos_idx: Optional[List[int]] = None,
    pos_indices_this_iter: Optional[set] = None,
    predictions_std: Optional[np.ndarray] = None,
) -> float:
    """
    Compute a single meta-evaluation metric.
    
    Args:
        metric_name: Name of the metric to compute
        predictions_mean: Mean predictions array
        all_pos_idx: List of all positive indices (required for PosDrop_Acc)
        pos_indices_this_iter: Set of positives used in training (required for PosDrop_Acc)
        predictions_std: Standard deviation predictions (for future metrics)
    
    Returns:
        Computed metric value
    
    Raises:
        ValueError: If metric_name is not recognized
    """
    if metric_name == "PosDrop_Acc":
        if all_pos_idx is None or pos_indices_this_iter is None:
            raise ValueError(f"Metric '{metric_name}' requires all_pos_idx and pos_indices_this_iter")
        return compute_posdrop_acc(
            predictions_mean=predictions_mean,
            all_pos_idx=all_pos_idx,
            pos_indices_this_iter=pos_indices_this_iter
        )
    elif metric_name == "Focus":
        return compute_focus(predictions_mean=predictions_mean)
    else:
        raise ValueError(f"Unknown meta-evaluation metric: {metric_name}. Available: ['PosDrop_Acc', 'Focus']")


# ============================================================================
# Base Classifier Training
# ============================================================================

def train_base_classifier(
    *,
    Xtr: torch.Tensor,
    Xval: torch.Tensor, 
    ytr: torch.Tensor,
    yval: torch.Tensor,
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    inference_fn: Optional[callable] = None,
    pos_crd_plot: Optional[List] = None,
    tag: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Train a base classifier using prepared training/validation data.
    
    Args:
        Xtr: Training features tensor
        Xval: Validation features tensor  
        ytr: Training labels tensor
        yval: Validation labels tensor
        common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
        data_use: Dictionary containing original data for inference
        inference_fn: Optional inference function to run after training
        pos_crd_plot: Optional positive coordinates for plotting
        tag: Tag for naming outputs
    
    Returns:
        Dictionary containing trained model, history, and inference results
    """
    # Extract common parameters
    cfg = common['cfg']
    device = common['device']
    mlp_hidden_dims = common['mlp_hidden_dims']
    mlp_dropout = common['mlp_dropout']
    debug_mode = common.get('debug_mode', False)
    run_logger = common.get('run_logger', None)
    
    # Phase 1: Create DataLoaders
    dl_tr, dl_va, metrics_summary_append = dataloader_metric_inputORembedding(
        Xtr=Xtr,
        Xval=Xval,
        ytr=ytr,
        yval=yval,
        batch_size=cfg.cls_training.batch_size,
        positive_augmentation=False,
        augmented_patches_all=None,
        pos_coord_to_index=None,
        window_size=None,
        stack=None,
        embedding=True,
        epochs=cfg.cls_training.epochs
    )
    
    # Phase 2: Build Model & Train
    in_dim = Xtr.size(1)
    cls = MLPDropout(in_dim=in_dim, hidden_dims=mlp_hidden_dims, p=float(mlp_dropout)).to(device)
    
    # Identity encoder since we're already in projected space
    encA = nn.Identity().to(device)
    
    if run_logger is not None:
        run_logger.log("[train_base_classifier] Training classifier...")
    
    # Train classifier - use verbose setting from common with early stopping
    verbose = common.get('verbose', True)
    cls, epoch_history, summary = train_classifier(
        encA,
        cls,
        dl_tr,
        dl_va,
        epochs=cfg.cls_training.epochs,
        return_history=True,
        loss_weights={'bce': 1.0},
        verbose=verbose,
        early_stopping=True,
        patience=15,
        min_delta=1e-3,
        restore_best_weights=True
    )
    
    # Phase 3: Run Inference (optional - only if inference function provided)
    inference_result = None
    if inference_fn is not None:
        # ✅ MODIFIED: Only save coordinates for final training (detected by "_final" in tag)
        is_final_training = tag is not None and "_final" in str(tag)
        
        inference_result = inference_fn(
            samples=data_use,
            cls=cls,
            device=device,
            output_dir=Path(common['cfg'].output_dir) / "cls_1_training_results" / "All",
            run_logger=common.get('run_logger'),
            passes=common.get('mlp_dropout_passes', 5),
            pos_crd=pos_crd_plot,
            tag=tag,
            save_coordinates=is_final_training  # ✅ NEW: Only save coordinates for final training
        )
    
    # Return results
    results = {
        'cls': cls,
        'epoch_history': epoch_history,
        'inference_result': inference_result,
        'metrics_summary': metrics_summary_append,
        'early_stopping_summary': summary,
    }

    return results


# ============================================================================
# Simple PN Training (without positive dropout)
# ============================================================================

def train_cls_1_PN(
    *,
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    filter_top_pct: float = 0.10,
    negs_per_pos: int = 10,
    action: Optional[Dict[str, Any]] = None,
    inference_fn: Optional[callable] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a PN (Positive-Negative) classifier using negative selection from unlabeled data.
    
    Args:
        common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
        data_use: Dictionary containing 'features', 'labels', 'coords' keys
        filter_top_pct: Percentage of top similar samples to filter during negative selection
        negs_per_pos: Number of negative samples to select per positive sample
        action: Optional dictionary for future extensibility (learning rates, etc.)
        inference_fn: Optional inference function to run after training
        tag: Tag for naming outputs
    
    Returns:
        Dictionary containing trained model, history, and inference results
    """
    
    # Extract common parameters
    cfg = common['cfg']
    device = common['device']
    mlp_hidden_dims = common['mlp_hidden_dims']
    mlp_dropout = common['mlp_dropout']
    debug_mode = common.get('debug_mode', False)
    run_logger = common.get('run_logger', None)
    
    if data_use is None:
        raise RuntimeError("Data unavailable for classifier training; aborting.")
    
    # Extract features and labels
    features = data_use["features"].float().to(device)
    yA_pu = torch.where(data_use["labels"] >= 0.9, 1, -1).long().to(device)
    
    # Get coordinates for visualization
    temp_crd = data_use.get("coords")
    coords_array = np.array(temp_crd) if temp_crd is not None else None
    pos_mask = (yA_pu == 1).cpu().numpy()
    pos_crd_plot = None
    if coords_array is not None:
        pos_crd_plot = [coords_array[i] for i in range(len(coords_array)) if pos_mask[i]]
    
    # Phase 1: Negative Selection & Dataset Preparation
    pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
    unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
    pos_idx_arr = np.asarray(pos_idx, dtype=int)
    unk_idx_arr = np.asarray(unl_idx, dtype=int)
    
    if debug_mode and run_logger is not None:
        run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")
    
    if run_logger is not None:
        run_logger.log(f"[cls-1-PN] Converting features to numpy...")
        run_logger.log(f"[cls-1-PN] Starting negative selection...")
    
    # Perform negative selection
    neg_idx_region = pu_select_negatives(
        Z_all=features.cpu().numpy(),
        pos_idx=pos_idx_arr,
        unk_idx=unk_idx_arr,
        filter_top_pct=filter_top_pct,
        negatives_per_pos=negs_per_pos,
        tag=tag
    )
    
    if debug_mode and run_logger is not None:
        run_logger.log(f"[cls-1-PN] Finished negative selection: selected {len(neg_idx_region)} negatives")
    
    neg_idx_arr = np.asarray(neg_idx_region, dtype=int)
    
    # Phase 2: Create Balanced P+N Dataset
    pn_indices = np.concatenate([pos_idx_arr, neg_idx_arr], axis=0)
    inf_indices = np.setdiff1d(np.arange(len(features)), pn_indices)
    
    # Extract features and labels for P+N
    features_pn = features[pn_indices]
    yA_pn = torch.where(
        torch.tensor([i in pos_idx for i in pn_indices], dtype=torch.bool),
        1,  # Positive
        0   # Negative (0 for BCE loss)
    ).long()
    
    # Phase 3: Train/Val Split on P+N
    val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
    total_pn = len(pn_indices)
    val_count = int(total_pn * val_frac)
    
    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed))
    indices_pn = torch.randperm(total_pn, generator=gen)
    
    val_indices_pn = indices_pn[:val_count].tolist()
    train_indices_pn = indices_pn[val_count:].tolist()
    
    # Create train/val data
    Xtr = features_pn[train_indices_pn]
    ytr = yA_pn[train_indices_pn]
    Xval = features_pn[val_indices_pn]
    yval = yA_pn[val_indices_pn]
    
    # Phase 4: Train the classifier using the base training function
    results = train_base_classifier(
        Xtr=Xtr,
        Xval=Xval,
        ytr=ytr,
        yval=yval,
        common=common,
        data_use=data_use,
        inference_fn=inference_fn,
        pos_crd_plot=pos_crd_plot
    )
    
    # Add PN-specific information to results
    results.update({
        'pos_crd_plot': pos_crd_plot,
        'pn_indices': pn_indices,
        'inf_indices': inf_indices,
    })
    
    return results


# ============================================================================
# Positive Drop Schedule Creation
# ============================================================================

def create_pos_drop_schedule(
    all_pos_idx: List[int], 
    drop_rate: float, 
    seed: int,
    min_training_size: int = 1
) -> List[List[int]]:
    """
    Create a rotation drop schedule where each positive sample is dropped exactly once across iterations.
    
    Args:
        all_pos_idx: List of all positive sample indices
        drop_rate: Fraction of positives to drop per iteration (determines number of iterations)
        seed: Random seed for reproducibility
        min_training_size: Minimum number of training samples required per iteration
    
    Returns:
        List of lists, where each inner list contains the positive indices to USE (not drop) in that iteration
    """
    if not all_pos_idx:
        return [[]]
    
    # Calculate number of iterations based on drop rate
    num_iterations = int(1.0 / drop_rate)
    total_positives = len(all_pos_idx)
    
    # Ensure minimum training size constraint
    max_drop_per_iter = total_positives - min_training_size
    if max_drop_per_iter <= 0:
        # If we can't drop any samples while maintaining min_training_size, use all samples in all iterations
        return [all_pos_idx.copy() for _ in range(num_iterations)]
    
    # Shuffle positives for random distribution across iterations
    rng = np.random.default_rng(seed)
    shuffled_pos_idx = rng.permutation(all_pos_idx).tolist()
    
    # Calculate how many samples to drop per iteration
    samples_per_drop = total_positives // num_iterations
    remaining_samples = total_positives % num_iterations
    
    # Create drop schedule - each sample is dropped exactly once
    drop_schedule = []
    start_idx = 0
    for i in range(num_iterations):
        # Add one extra sample to first 'remaining_samples' iterations
        extra = 1 if i < remaining_samples else 0
        end_idx = start_idx + samples_per_drop + extra
        samples_to_drop = shuffled_pos_idx[start_idx:end_idx]
        drop_schedule.append(samples_to_drop)
        start_idx = end_idx
    
    # Convert drop schedule to "use schedule" (samples to keep for training)
    use_schedule = []
    for iteration_drops in drop_schedule:
        samples_to_use = [idx for idx in all_pos_idx if idx not in iteration_drops]
        
        # Ensure minimum training size
        if len(samples_to_use) < min_training_size:
            # If dropping would result in too few samples, use all samples
            samples_to_use = all_pos_idx.copy()
        
        use_schedule.append(samples_to_use)
    
    return use_schedule


def create_pos_drop_schedule_kmeans(
    all_pos_idx: List[int],
    features: torch.Tensor,
    meta_evaluation_n_clusters: int,
    seed: int = 42,
    min_training_size: int = 1
) -> List[List[int]]:
    """
    Create K-means clustering-based drop schedule.
    Each iteration drops samples from different clusters to ensure diverse cross-validation.
    """
    from sklearn.cluster import KMeans
    
    if not all_pos_idx:
        return [[]]
    
    num_iterations = meta_evaluation_n_clusters
    
    # Extract positive features for clustering
    pos_features = features[all_pos_idx].cpu().numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_iterations, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(pos_features)
    
    # Group indices by cluster
    clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(all_pos_idx[i])
    
    # Create drop schedule: each iteration drops one cluster
    use_schedule = []
    
    for iteration in range(num_iterations):
        # Drop cluster 'iteration', keep all others
        kept_indices = []
        
        for cluster_id, cluster_indices in clusters.items():
            if cluster_id != iteration:
                # Keep this entire cluster
                kept_indices.extend(cluster_indices)
            # else: drop this cluster (don't add to kept_indices)
        
        # Ensure minimum training size
        if len(kept_indices) < min_training_size:
            kept_indices = all_pos_idx.copy()
        
        use_schedule.append(kept_indices)
    
    return use_schedule


def create_pos_drop_schedule_hierarchical(
    all_pos_idx: List[int],
    features: torch.Tensor,
    meta_evaluation_n_clusters: int,
    linkage: str = "ward",
    seed: int = 42,
    min_training_size: int = 1
) -> List[List[int]]:
    """
    Create hierarchical clustering-based drop schedule.
    Uses agglomerative clustering to create meaningful positive groups.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    if not all_pos_idx:
        return [[]]
    
    num_iterations = meta_evaluation_n_clusters
    
    # Extract positive features for clustering
    pos_features = features[all_pos_idx].cpu().numpy()
    
    # Perform hierarchical clustering with reproducible results
    import numpy as np
    np.random.seed(seed)
    
    clustering = AgglomerativeClustering(
        n_clusters=num_iterations,
        linkage=linkage
    )
    cluster_labels = clustering.fit_predict(pos_features)
    
    # Group indices by cluster
    clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(all_pos_idx[i])
    
    # ✅ BALANCED DROPPING: Redistribute cluster members to ensure balanced drop sizes
    balanced_drop_groups = _rebalance_clusters_for_equal_dropping(
        clusters, num_iterations, len(all_pos_idx), seed
    )
    
    # Create keep schedules from balanced drop groups
    use_schedule = []
    for drop_group in balanced_drop_groups:
        kept_indices = [idx for idx in all_pos_idx if idx not in drop_group]
        
        # Ensure minimum training size
        if len(kept_indices) < min_training_size:
            kept_indices = all_pos_idx.copy()
        
        use_schedule.append(kept_indices)
    
    return use_schedule


def create_pos_drop_schedule_random(
    all_pos_idx: List[int],
    features: torch.Tensor,
    meta_evaluation_n_clusters: int,
    seed: int = 42,
    min_training_size: int = 1
) -> List[List[int]]:
    """
    Create random drop schedule with balanced distribution.
    Uses cluster-based approach for consistency with other clustering methods.
    """
    if not all_pos_idx:
        return [[]]
    
    num_iterations = meta_evaluation_n_clusters
    total_positives = len(all_pos_idx)
    
    # Ensure minimum training size constraint
    max_drop_per_iter = total_positives - min_training_size
    if max_drop_per_iter <= 0:
        return [all_pos_idx.copy() for _ in range(num_iterations)]
    
    # Shuffle positives for random distribution
    rng = np.random.default_rng(seed)
    shuffled_pos_idx = rng.permutation(all_pos_idx).tolist()
    
    # Create pseudo-clusters by assigning samples randomly to clusters
    clusters = {i: [] for i in range(num_iterations)}
    for idx, sample_idx in enumerate(shuffled_pos_idx):
        cluster_id = idx % num_iterations
        clusters[cluster_id].append(sample_idx)
    
    # Apply balanced redistribution to ensure similar drop sizes
    balanced_clusters = _rebalance_clusters_for_equal_dropping(
        clusters, 
        num_iterations, 
        total_positives,
        seed
    )
    
    # Create drop schedule from balanced clusters
    drop_schedule = []
    for i in range(num_iterations):
        samples_to_drop = balanced_clusters[i]
        drop_schedule.append(samples_to_drop)
    
    # Convert to use schedule
    use_schedule = []
    for iteration_drops in drop_schedule:
        samples_to_use = [idx for idx in all_pos_idx if idx not in iteration_drops]
        
        if len(samples_to_use) < min_training_size:
            samples_to_use = all_pos_idx.copy()
        
        use_schedule.append(samples_to_use)
    
    return use_schedule


def _rebalance_clusters_for_equal_dropping(
    clusters: Dict[int, List[int]],
    num_iterations: int,
    total_positives: int,
    seed: int = 42
) -> List[List[int]]:
    """
    Rebalance cluster assignments to ensure each iteration drops similar numbers.
    
    Args:
        clusters: Dictionary mapping cluster_id -> list of sample indices
        num_iterations: Number of iterations (target number of balanced groups)
        total_positives: Total number of positive samples
        seed: Random seed for reproducibility
        
    Returns:
        List of balanced drop groups, each containing similar numbers of indices
    """
    import numpy as np
    
    np.random.seed(seed)
    
    # Calculate target group sizes
    base_size = total_positives // num_iterations
    remainder = total_positives % num_iterations
    
    # Target sizes: some groups get +1 sample if remainder exists
    target_sizes = [
        base_size + (1 if i < remainder else 0) 
        for i in range(num_iterations)
    ]
    
    # Collect all samples, shuffling within each cluster to maintain some clustering structure
    all_samples = []
    for cluster_id in sorted(clusters.keys()):
        cluster_samples = clusters[cluster_id].copy()
        np.random.shuffle(cluster_samples)  # Shuffle within cluster
        all_samples.extend(cluster_samples)
    
    # Additional shuffle for better balance
    np.random.shuffle(all_samples)
    
    # Distribute samples into balanced groups
    balanced_groups = []
    start_idx = 0
    
    for i in range(num_iterations):
        target_size = target_sizes[i]
        end_idx = start_idx + target_size
        
        if end_idx <= len(all_samples):
            balanced_groups.append(all_samples[start_idx:end_idx])
        else:
            balanced_groups.append(all_samples[start_idx:])
        
        start_idx = end_idx
    
    return balanced_groups


def create_pos_drop_schedule_unified(
    all_pos_idx: List[int],
    features: torch.Tensor,
    meta_evaluation_n_clusters: int,
    seed: int,
    method: str = "random",
    linkage: str = "ward",
    min_training_size: int = 1
) -> List[List[int]]:
    """
    Unified interface for creating positive drop schedules using different clustering methods.
    """
    if method == "random":
        return create_pos_drop_schedule_random(
            all_pos_idx=all_pos_idx,
            features=features,
            meta_evaluation_n_clusters=meta_evaluation_n_clusters,
            seed=seed,
            min_training_size=min_training_size
        )
    
    elif method == "kmeans":
        return create_pos_drop_schedule_kmeans(
            all_pos_idx=all_pos_idx,
            features=features,
            meta_evaluation_n_clusters=meta_evaluation_n_clusters,
            seed=seed,
            min_training_size=min_training_size
        )
    
    elif method == "hierarchical":
        return create_pos_drop_schedule_hierarchical(
            all_pos_idx=all_pos_idx,
            features=features,
            meta_evaluation_n_clusters=meta_evaluation_n_clusters,
            seed=seed,
            linkage=linkage,
            min_training_size=min_training_size
        )
    
    else:
        raise ValueError(f"Unknown clustering method: {method}. Available: ['random', 'kmeans', 'hierarchical']")


# ============================================================================
# PN Training with Positive Dropout
# ============================================================================

def train_cls_1_PN_PosDrop(
    *,
    meta_evaluation_n_clusters: int = 10,  # ✅ CHANGED: from drop_rate to n_clusters
    clustering_method: str = "random",     # ✅ NEW: clustering method selection
    linkage: str = "ward",                 # ✅ NEW: for hierarchical clustering
    seed: int = 42,                        # ✅ NEW: random seed for reproducibility
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    filter_top_pct: float = 0.10,
    negs_per_pos: int = 10,
    action: Optional[Dict[str, Any]] = None,
    inference_fn: Optional[callable] = None,
    tag_main: Optional[str] = None,
    use_individual_distances: bool = True,
    run_meta_evaluation: bool = True,
    meta_eval_config: Optional[MetaEvaluationConfig] = None,
) -> Dict[str, Any]:
    """
    Train a PN classifier with positive dropout using clustering and compute meta-evaluation metrics.
    
    Args:
        meta_evaluation_n_clusters: Number of clusters for cross-validation (2, 3, 4, 6, 8, 10)
        clustering_method: Method for clustering ('random', 'kmeans', 'hierarchical')
        linkage: Linkage criterion for hierarchical clustering ('ward', 'complete', 'average', 'single')
        common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
        data_use: Dictionary containing 'features', 'labels', 'coords' keys
        filter_top_pct: Percentage of top similar samples to filter during negative selection
        negs_per_pos: Number of negative samples to select per positive sample
        action: Optional dictionary for future extensibility
        inference_fn: Optional inference function to run after training
        tag_main: Main tag for caching and naming
        use_individual_distances: Use individual distance files instead of full matrix (memory-safe)
        run_meta_evaluation: Whether to run meta-evaluation loops
        meta_eval_config: Meta-evaluation configuration (metrics + parameters)

    Returns:
        Dictionary containing meta-evaluation results
    """
    
    # Extract common parameters
    cfg = common['cfg']
    device = common['device']
    mlp_hidden_dims = common['mlp_hidden_dims']
    mlp_dropout = common['mlp_dropout']
    debug_mode = common.get('debug_mode', False)
    run_logger = common.get('run_logger', None)
    verbose = common.get('verbose', True)
    
    if meta_eval_config is None:
        meta_eval_config = MetaEvaluationConfig()
    meta_evaluation_metrics = set(meta_eval_config.metrics)
    
    if run_logger is not None:
        run_logger.log(f"[cls-1-PN-PosDrop] Meta_evaluation_n_clusters: {meta_evaluation_n_clusters}")
        run_logger.log(f"[cls-1-PN-PosDrop] Clustering method: {clustering_method}")
        run_logger.log(f"[cls-1-PN-PosDrop] Seed: {seed}")
        run_logger.log(f"[cls-1-PN-PosDrop] Meta_Evaluation metrics: {meta_evaluation_metrics}")
    
    if data_use is None:
        raise RuntimeError("Data unavailable for classifier training; aborting.")
    
    # Extract features and labels
    features = data_use["features"]
    
    # Convert to tensor if needed
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    elif isinstance(features, torch.Tensor):
        features = features.float()
    else:
        features = torch.tensor(features, dtype=torch.float32)
    features = features.to(device)

    labels_tensor = torch.from_numpy(data_use["labels"]) if isinstance(data_use["labels"], np.ndarray) else data_use["labels"]
    yA_pu = torch.where(labels_tensor >= 0.9, 1, -1).long().to(device)
    
    # Get coordinates for visualization
    temp_crd = data_use.get("coords") or data_use.get("coordinates")
    coords_array = np.array(temp_crd) if temp_crd is not None else None

    # Prepare positive dropping
    all_pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
    all_unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
    
    num_iterations = meta_evaluation_n_clusters
    
    # ✅ MODIFIED: Create drop schedule using unified clustering interface
    pos_splits = create_pos_drop_schedule_unified(
        all_pos_idx=all_pos_idx,
        features=features,
        meta_evaluation_n_clusters=meta_evaluation_n_clusters,
        method=clustering_method,
        linkage=linkage,
        seed=seed,
        min_training_size=1
    )
    
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN-PosDrop] Created {clustering_method} drop schedule: {num_iterations} iterations, {len(all_pos_idx)} total positives")
        
        # Log cluster information for debugging
        if clustering_method in ["kmeans", "hierarchical"]:
            for i, pos_indices_iter in enumerate(pos_splits):
                dropped_count = len(all_pos_idx) - len(pos_indices_iter)
                run_logger.log(f"  Iteration {i+1}: using {len(pos_indices_iter)} positives, dropping {dropped_count} positives")
    
    # Memory estimation
    n_unknowns = len(all_unl_idx)
    n_positives = len(all_pos_idx)
    estimated_memory_gb = (n_unknowns * n_positives * 4) / (1024**3)
    
    if run_logger is not None:
        run_logger.log(f"[cls-1-PN-PosDrop] Dataset size: {n_unknowns} unknowns, {n_positives} positives")
        run_logger.log(f"[cls-1-PN-PosDrop] Estimated memory for full matrix: {estimated_memory_gb:.2f} GB")
    
    # Precompute distances if using individual files
    individual_distances_dir = None
    if use_individual_distances:
        if run_logger is not None and verbose:
            run_logger.log(f"[cls-1-PN-PosDrop] Using individual distance files (memory-safe approach)")
        
        try:
            rotation_tag = f"{tag_main}_posdrop_{clustering_method}_individual" if tag_main else None
            individual_distances_dir = precompute_distances_per_positive(
                Z_all=features,
                all_pos_idx=all_pos_idx,
                unk_idx=all_unl_idx,
                tag=rotation_tag,
                use_float16=True
            )
            
            if run_logger is not None and verbose:
                run_logger.log(f"[cls-1-PN-PosDrop] Pre-computed individual distance files in: {individual_distances_dir}")
        except Exception as e:
            if run_logger is not None:
                run_logger.log(f"[cls-1-PN-PosDrop] WARNING: Failed to pre-compute individual distances ({e}), falling back to standard method")
            individual_distances_dir = None
            use_individual_distances = False
    
    # Training iterations
    all_results = []
    for iteration in range(num_iterations):
        if run_logger is not None and verbose:
            run_logger.log(f"[cls-1-PN-PosDrop] Starting training for iteration {iteration+1}/{num_iterations}...")
        
        # ✅ MODIFIED: Updated tag to include clustering method and n_clusters
        tag = f"{tag_main}/{clustering_method}{meta_evaluation_n_clusters}_iter{iteration+1}" if tag_main is not None else None

        pos_idx = pos_splits[iteration]
        unl_idx = all_unl_idx.copy()
        
        pos_crd_plot = None
        if coords_array is not None:
            pos_crd_plot = [coords_array[i] for i in pos_idx]

            # Spatial analysis for debugging
            dropped_idx = [idx for idx in all_pos_idx if idx not in pos_idx]
            if len(dropped_idx) > 0:
                run_logger.log(f"[PosDrop-Spatial-{clustering_method}] Iter {iteration+1}: Kept positives: {len(pos_idx)}  Dropped positives: {len(dropped_idx)}")

        pos_idx_arr = np.asarray(pos_idx, dtype=int)
        unk_idx_arr = np.asarray(unl_idx, dtype=int)
        
        # Negative selection
        if use_individual_distances and individual_distances_dir is not None:
            neg_idx_region = pu_select_negatives_from_individual_distances(
                individual_distances_dir=individual_distances_dir,
                all_pos_idx=all_pos_idx,
                unk_idx=unk_idx_arr,
                active_pos_idx=pos_idx_arr,
                filter_top_pct=filter_top_pct,
                negatives_per_pos=negs_per_pos
            )
        else:
            neg_idx_region = pu_select_negatives(
                Z_all=features,
                pos_idx=pos_idx_arr,
                unk_idx=unk_idx_arr,
                filter_top_pct=filter_top_pct,
                negatives_per_pos=negs_per_pos,
                tag=tag
            )
        
        neg_idx_arr = np.asarray(neg_idx_region, dtype=int)
        
        # Create balanced dataset
        pn_indices = np.concatenate([pos_idx_arr, neg_idx_arr], axis=0)
        features_pn = features[pn_indices]
        yA_pn = torch.where(
            torch.tensor([i in pos_idx for i in pn_indices], dtype=torch.bool),
            1, 0
        ).long()
        
        # Train/Val split
        val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
        total_pn = len(pn_indices)
        val_count = int(total_pn * val_frac)
        
        gen = torch.Generator()
        gen.manual_seed(int(cfg.seed))
        indices_pn = torch.randperm(total_pn, generator=gen)
        
        val_indices_pn = indices_pn[:val_count].tolist()
        train_indices_pn = indices_pn[val_count:].tolist()
        
        Xtr = features_pn[train_indices_pn]
        ytr = yA_pn[train_indices_pn]
        Xval = features_pn[val_indices_pn]
        yval = yA_pn[val_indices_pn]
        
        # Train classifier
        iteration_results = train_base_classifier(
            Xtr=Xtr,
            Xval=Xval,
            ytr=ytr,
            yval=yval,
            common=common,
            data_use=data_use,
            inference_fn=inference_fn,
            pos_crd_plot=pos_crd_plot,
            tag=tag
        )

        iteration_results['pos_indices_this_iter'] = pos_idx
        all_results.append(iteration_results)
    
    # Final training with ALL positives
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN-PosDrop] Starting final 'no positive drop' training with all {len(all_pos_idx)} positives")
    
    # ✅ MODIFIED: Updated final tag to include clustering method
    tag_all = f"{tag_main}/{clustering_method}{meta_evaluation_n_clusters}_final" if tag_main is not None else None
    pos_idx = all_pos_idx
    pos_crd_plot_all = None
    if coords_array is not None:
        pos_crd_plot_all = [coords_array[i] for i in pos_idx]

    pos_idx_arr = np.asarray(pos_idx, dtype=int)
    unk_idx_arr = np.asarray(all_unl_idx, dtype=int)
    
    if use_individual_distances and individual_distances_dir is not None:
        neg_idx_region_all = pu_select_negatives_from_individual_distances(
            individual_distances_dir=individual_distances_dir,
            all_pos_idx=all_pos_idx,
            unk_idx=unk_idx_arr,
            active_pos_idx=pos_idx_arr,
            filter_top_pct=filter_top_pct,
            negatives_per_pos=negs_per_pos
        )
    else:
        neg_idx_region_all = pu_select_negatives(
            Z_all=features.cpu().numpy(),
            pos_idx=pos_idx_arr,
            unk_idx=unk_idx_arr,
            filter_top_pct=filter_top_pct,
            negatives_per_pos=negs_per_pos,
            tag=tag_all
        )
    
    neg_idx_arr_all = np.asarray(neg_idx_region_all, dtype=int)
    pn_indices_all = np.concatenate([pos_idx_arr, neg_idx_arr_all], axis=0)
    features_pn_all = features[pn_indices_all]
    yA_pn_all = torch.where(
        torch.tensor([i in pos_idx for i in pn_indices_all], dtype=torch.bool),
        1, 0
    ).long()
    
    val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
    total_pn_all = len(pn_indices_all)
    val_count_all = int(total_pn_all * val_frac)
    
    gen_all = torch.Generator()
    gen_all.manual_seed(int(cfg.seed))
    indices_pn_all = torch.randperm(total_pn_all, generator=gen_all)
    
    val_indices_pn_all = indices_pn_all[:val_count_all].tolist()
    train_indices_pn_all = indices_pn_all[val_count_all:].tolist()
    
    Xtr_all = features_pn_all[train_indices_pn_all]
    ytr_all = yA_pn_all[train_indices_pn_all]
    Xval_all = features_pn_all[val_indices_pn_all]
    yval_all = yA_pn_all[val_indices_pn_all]
    
    all_results_final = train_base_classifier(
        Xtr=Xtr_all,
        Xval=Xval_all,
        ytr=ytr_all,
        yval=yval_all,
        common=common,
        data_use=data_use,
        inference_fn=inference_fn,
        pos_crd_plot=pos_crd_plot_all,
        tag=tag_all
    )
    
    all_results_final['pos_indices_this_iter'] = pos_idx
    all_results.append(all_results_final)
    
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN-PosDrop] Completed final 'no positive drop' training")
    
    if not run_meta_evaluation:
        if run_logger is not None:
            run_logger.log("[cls-1-PN-PosDrop] Meta evaluation disabled; returning empty summary")
        return {}

    if not run_meta_evaluation:
        if run_logger:
            run_logger.log("[load_cached_predictions] Meta evaluation disabled; returning empty summary")
        return {}

    # Compute meta-evaluation metrics (all metrics treated uniformly)
    meta_evaluation = {}
    
    # Handle per-iteration metrics (PosDrop_Acc, Focus)
    per_iteration_metrics = POSITIVE_ITERATION_METRICS
    iteration_based = [m for m in meta_evaluation_metrics if m in per_iteration_metrics]
    for metric in iteration_based:
        metric_scores = []

        for iteration in range(num_iterations):
            iter_result = all_results[iteration]
            pos_indices_this_iter = set(iter_result['pos_indices_this_iter'])
            
            predictions_mean = iter_result['inference_result']['predictions_mean']
            if isinstance(predictions_mean, torch.Tensor):
                predictions_mean = predictions_mean.cpu().numpy()
            
            predictions_std = iter_result['inference_result'].get('predictions_std')
            if predictions_std is not None and isinstance(predictions_std, torch.Tensor):
                predictions_std = predictions_std.cpu().numpy()
            
            try:
                score = compute_meta_evaluation_metric(
                    metric_name=metric,
                    predictions_mean=predictions_mean,
                    all_pos_idx=all_pos_idx,
                    pos_indices_this_iter=pos_indices_this_iter,
                    predictions_std=predictions_std
                )
                metric_scores.append(score)
            except ValueError as e:
                if run_logger is not None:
                    run_logger.log(f"[cls-1-PN-PosDrop] WARNING: Failed to compute {metric}: {e}")
                metric_scores.append(0.0)
        
        meta_evaluation[metric] = {
            'scores': metric_scores,
            'mean': np.mean(metric_scores) if metric_scores else 0.0,
            'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
        }

        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
    # Handle cross-iteration metrics (PAUC, TopK)
    cross_iteration_metrics = EXTENDED_META_METRICS
    cross_based = [m for m in meta_evaluation_metrics if m in cross_iteration_metrics]
    if cross_based:
        # Prepare data for extended metrics computation
        all_probabilities = []
        all_labels = []
        
        for iteration in range(num_iterations):
            iter_result = all_results[iteration]
            pos_indices_this_iter = set(iter_result['pos_indices_this_iter'])
            
            predictions_mean = iter_result['inference_result']['predictions_mean']
            if isinstance(predictions_mean, torch.Tensor):
                predictions_mean = predictions_mean.cpu().numpy()
            
            # Create binary labels for this iteration (1 for known positives, 0 for unlabeled)
            labels_this_iter = np.zeros(len(predictions_mean))
            for idx in pos_indices_this_iter:
                if idx < len(labels_this_iter):
                    labels_this_iter[idx] = 1
            
            all_probabilities.append(predictions_mean)
            all_labels.append(labels_this_iter)
        
        extended_results = compute_extended_meta_evaluation(
            meta_eval_config=meta_eval_config,
            requested_metrics=cross_based,
            all_probabilities=all_probabilities,
            all_labels=all_labels,
            run_logger=run_logger,
        )
        
        # Merge cross-iteration results into meta_evaluation
        meta_evaluation.update(extended_results)
        
        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] Computed cross-iteration metrics: {list(extended_results.keys())}")
    
    return meta_evaluation


# ============================================================================
# Cached Prediction Loading and Evaluation
# ============================================================================

def load_and_evaluate_existing_predictions(
    *,
    tag_main: str,
    meta_evaluation_n_clusters: Optional[int] = None,  # ✅ CHANGED: from drop_rate to n_clusters
    clustering_method: str = "random", # ✅ NEW: clustering method
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    run_logger: Optional[Any] = None,
    run_meta_evaluation: bool = True,
    meta_eval_config: Optional[MetaEvaluationConfig] = None,
    drop_rate: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load existing predictions from disk and compute Meta_Evaluation metrics.
    
    Args:
        tag_main: Tag identifier for the experiment
        meta_evaluation_n_clusters: Number of clusters/iterations (2, 3, 4, 6, 8, 10)
        drop_rate: Legacy positive-drop rate (used if n_clusters omitted)
        clustering_method: Clustering method used ('random', 'kmeans', 'hierarchical')
        common: Dictionary containing cfg, device, run_logger, etc.
        data_use: Dictionary containing 'features', 'labels', 'coords' keys
        run_logger: Optional logger for status messages
        run_meta_evaluation: Whether to compute meta-evaluation metrics
        meta_eval_config: Meta-evaluation configuration bundle
    
    Returns:
        Dictionary containing meta_evaluation metrics, or None if predictions not found
    """
    cfg = common['cfg']
    
    if meta_evaluation_n_clusters is None:
        if drop_rate is not None and drop_rate > 0:
            meta_evaluation_n_clusters = int(round(1.0 / drop_rate))
        else:
            raise ValueError("meta_evaluation_n_clusters must be provided when drop_rate is absent.")
    expected_num_iterations = meta_evaluation_n_clusters
    
    base_output_dir = Path(cfg.output_dir) / "cls_1_training_results" / "All" / tag_main
    
    if meta_eval_config is None:
        meta_eval_config = MetaEvaluationConfig()
    meta_evaluation_metrics = set(meta_eval_config.metrics)
    
    if run_logger:
        run_logger.log(f"[load_cached_predictions] Searching in: {base_output_dir}")
        run_logger.log(f"[load_cached_predictions] Clustering method: {clustering_method}")
        run_logger.log(f"[load_cached_predictions] Required n_clusters: {meta_evaluation_n_clusters}")
        run_logger.log(f"[load_cached_predictions] Expected iterations: {expected_num_iterations}")
        run_logger.log(f"[load_cached_predictions] Meta_Evaluation metrics: {meta_evaluation_metrics}")
    
    if not base_output_dir.exists():
        if run_logger:
            run_logger.log(f"[load_cached_predictions] ❌ Directory not found: {base_output_dir}")
        return None
    
    # ✅ MODIFIED: Search for SPECIFIC pattern matching clustering method and n_clusters
    import glob
    specific_pattern = str(base_output_dir / f"{clustering_method}{meta_evaluation_n_clusters}_iter*")
    iter_dirs = sorted(glob.glob(specific_pattern))
    
    if run_logger:
        run_logger.log(f"[load_cached_predictions] Pattern: {clustering_method}{meta_evaluation_n_clusters}_iter*")
        run_logger.log(f"[load_cached_predictions] Found {len(iter_dirs)} matching directories")
    
    # Validate: Must find exactly expected number of iterations
    if len(iter_dirs) != expected_num_iterations:
        if run_logger:
            run_logger.log(f"[load_cached_predictions] ❌ Expected {expected_num_iterations} iterations, found {len(iter_dirs)}")
            
            # Check if different drop_rate directories exist
            all_patterns = list(base_output_dir.glob("all*_iter*"))
            if all_patterns:
                found_patterns = set()
                for p in all_patterns:
                    match = p.name.split('_iter')[0] if '_iter' in p.name else None
                    if match:
                        found_patterns.add(match)
                run_logger.log(f"[load_cached_predictions] Found predictions for: {sorted(found_patterns)}")
                run_logger.log(f"[load_cached_predictions] But required: all{expected_num_iterations}")
        return None
    
    num_iterations = expected_num_iterations
    
    # Extract positive indices
    labels = data_use['labels']
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    all_pos_idx = np.where(labels == 1)[0].tolist()
    
    if not all_pos_idx:
        if run_logger:
            run_logger.log("[load_cached_predictions] ❌ ERROR: No positive samples found in data_use")
        return None
    
    # ✅ MODIFIED: Reconstruct drop schedule using clustering method
    features = data_use["features"]
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    elif isinstance(features, torch.Tensor):
        features = features.float()
    else:
        features = torch.tensor(features, dtype=torch.float32)
    
    drop_schedule = create_pos_drop_schedule_unified(
        all_pos_idx=all_pos_idx,
        features=features,
        meta_evaluation_n_clusters=meta_evaluation_n_clusters,
        seed=int(cfg.seed),
        method=clustering_method,
        min_training_size=1
    )
    
    if run_logger:
        run_logger.log(f"[load_cached_predictions] Reconstructed {clustering_method} drop schedule with {len(drop_schedule)} iterations")
    
    if run_logger:
        run_logger.log(f"[load_cached_predictions] Reconstructed drop schedule with {len(drop_schedule)} iterations")
    
    # Load predictions
    all_predictions_mean = []
    all_predictions_std = []
    all_pos_indices_per_iter = []
    
    for iter_idx, iter_dir_path in enumerate(iter_dirs):
        iter_dir = Path(iter_dir_path)
        pred_mean_path = iter_dir / "predictions_mean.npy"
        pred_std_path = iter_dir / "predictions_std.npy"
        
        if not pred_mean_path.exists():
            if run_logger:
                run_logger.log(f"[load_cached_predictions] ❌ Missing predictions_mean.npy in {iter_dir}")
            return None
        
        if not pred_std_path.exists():
            if run_logger:
                run_logger.log(f"[load_cached_predictions] ❌ Missing predictions_std.npy in {iter_dir}")
            return None
        
        try:
            pred_mean = np.load(pred_mean_path)
            pred_std = np.load(pred_std_path)
            
            all_predictions_mean.append(pred_mean)
            all_predictions_std.append(pred_std)
            all_pos_indices_per_iter.append(drop_schedule[iter_idx])
            
            if run_logger:
                run_logger.log(f"[load_cached_predictions] ✅ Loaded iteration {iter_idx + 1}/{num_iterations}: mean shape={pred_mean.shape}, std shape={pred_std.shape}")
        except Exception as e:
            if run_logger:
                run_logger.log(f"[load_cached_predictions] ❌ Error loading {iter_dir}: {e}")
            return None
    
    # Compute meta-evaluation metrics (all metrics treated uniformly)
    meta_evaluation = {}
    
    # Handle per-iteration metrics (PosDrop_Acc, Focus)
    per_iteration_metrics = POSITIVE_ITERATION_METRICS
    iteration_based = [m for m in meta_evaluation_metrics if m in per_iteration_metrics]
    for metric in iteration_based:
        metric_scores = []
        
        for iteration in range(num_iterations):
            pos_indices_this_iter = set(all_pos_indices_per_iter[iteration])
            predictions_mean = all_predictions_mean[iteration]
            predictions_std = all_predictions_std[iteration]
            
            try:
                score = compute_meta_evaluation_metric(
                    metric_name=metric,
                    predictions_mean=predictions_mean,
                    all_pos_idx=all_pos_idx,
                    pos_indices_this_iter=pos_indices_this_iter,
                    predictions_std=predictions_std
                )
                metric_scores.append(score)
            except ValueError as e:
                if run_logger:
                    run_logger.log(f"[load_cached_predictions] WARNING: Failed to compute {metric}: {e}")
                metric_scores.append(0.0)
        
        meta_evaluation[metric] = {
            'scores': metric_scores,
            'mean': np.mean(metric_scores) if metric_scores else 0.0,
            'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
        }
        
        if run_logger:
            run_logger.log(f"[load_cached_predictions] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
    # Handle cross-iteration metrics (PAUC, TopK)
    cross_iteration_metrics = EXTENDED_META_METRICS
    cross_based = [m for m in meta_evaluation_metrics if m in cross_iteration_metrics]
    if cross_based:
        # Prepare data for extended metrics computation
        all_probabilities = []
        all_labels = []
        
        for iteration in range(num_iterations):
            pos_indices_this_iter = set(all_pos_indices_per_iter[iteration])
            predictions_mean = all_predictions_mean[iteration]
            
            # Create binary labels for this iteration (1 for known positives, 0 for unlabeled)
            labels_this_iter = np.zeros(len(predictions_mean))
            for idx in pos_indices_this_iter:
                if idx < len(labels_this_iter):
                    labels_this_iter[idx] = 1
            
            all_probabilities.append(predictions_mean)
            all_labels.append(labels_this_iter)
        
        extended_results = compute_extended_meta_evaluation(
            meta_eval_config=meta_eval_config,
            requested_metrics=cross_based,
            all_probabilities=all_probabilities,
            all_labels=all_labels,
            run_logger=run_logger,
        )
        
        # Merge cross-iteration results into meta_evaluation
        meta_evaluation.update(extended_results)
        
        if run_logger:
            run_logger.log(f"[load_cached_predictions] Computed cross-iteration metrics: {list(extended_results.keys())}")
    
    return meta_evaluation


# ============================================================================
# Multi-Clustering Training Functions
# ============================================================================

def train_cls_1_PN_PosDrop_MultiClustering(
    *,
    meta_evaluation_n_clusters_list: List[int] = [10],
    clustering_methods: List[str] = ["random"],
    linkage: str = "ward",
    seed: int = 42,
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    filter_top_pct: float = 0.10,
    negs_per_pos: int = 10,
    action: Optional[Dict[str, Any]] = None,
    inference_fn: Optional[callable] = None,
    tag_main: Optional[str] = None,
    use_individual_distances: bool = True,
    meta_evaluation_metrics: Optional[set] = None,
    # Extended metrics parameters
    pauc_prior_variants: Optional[List[float]] = None,
    topk_k_ratio: Optional[List[float]] = None,
    topk_k_values: Optional[List[int]] = None,
    topk_area_percentages: Optional[List[float]] = None,
    run_meta_evaluation: bool = True,
    meta_eval_config: Optional[MetaEvaluationConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Train PN classifiers with multiple clustering methods and n_clusters configurations.
    
    Args:
        meta_evaluation_n_clusters_list: List of cluster counts to test (e.g., [2, 4, 6, 8, 10])
        clustering_methods: List of clustering methods to test (e.g., ["random", "kmeans", "hierarchical"])
        linkage: Linkage criterion for hierarchical clustering
        common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
        data_use: Dictionary containing 'features', 'labels', 'coords' keys
        filter_top_pct: Percentage of top similar samples to filter during negative selection
        negs_per_pos: Number of negative samples to select per positive sample
        action: Optional dictionary for future extensibility
        inference_fn: Optional inference function to run after training
        tag_main: Main tag for caching and naming
        use_individual_distances: Use individual distance files instead of full matrix (memory-safe)
        meta_evaluation_metrics: Set of metric names to compute (default: PosDrop_Acc, Focus)
        run_meta_evaluation / meta_eval_config: Control and configure meta evaluation
    
    Returns:
        Dictionary mapping config names to meta-evaluation results
    """
    
    run_logger = common.get('run_logger', None)
    verbose = common.get('verbose', True)
    
    if meta_evaluation_metrics is None:
        meta_evaluation_metrics = {"PosDrop_Acc", "Focus", "pauc", "topk"}
    
    # Generate all configurations to test
    all_configs = []
    for method in clustering_methods:
        for n_clusters in meta_evaluation_n_clusters_list:
            config_name = f"{method}{n_clusters}"
            all_configs.append({
                'name': config_name,
                'method': method,
                'n_clusters': n_clusters
            })
    
    if run_logger and verbose:
        run_logger.log(f"[Multi-Clustering] Testing {len(all_configs)} configurations:")
        for config in all_configs:
            run_logger.log(f"  - {config['name']}: {config['method']} with {config['n_clusters']} clusters")
    
    # Train with each configuration
    all_results = {}
    
    for config in all_configs:
        config_name = config['name']
        method = config['method']
        n_clusters = config['n_clusters']
        
        if run_logger and verbose:
            run_logger.log(f"[Multi-Clustering] Starting training with {config_name}...")
        
        try:
            result = train_cls_1_PN_PosDrop(
                meta_evaluation_n_clusters=n_clusters,
                clustering_method=method,
                linkage=linkage,
                seed=seed,
                common={**common, 'verbose': False},  # Reduce verbosity for multi-config
                data_use=data_use,
                filter_top_pct=filter_top_pct,
                negs_per_pos=negs_per_pos,
                action=action,
                inference_fn=inference_fn,
                tag_main=tag_main,
                use_individual_distances=use_individual_distances,
                run_meta_evaluation=run_meta_evaluation,
                meta_eval_config=meta_eval_config,
            )
            
            all_results[config_name] = result
            
            if run_logger and verbose:
                run_logger.log(f"[Multi-Clustering] Completed {config_name}")
                # Log summary metrics
                for metric, data in result.items():
                    if isinstance(data, dict) and 'mean' in data:
                        run_logger.log(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f}")
        
        except Exception as e:
            if run_logger:
                run_logger.log(f"[Multi-Clustering] ERROR training {config_name}: {e}")
            all_results[config_name] = {"error": str(e)}
    
    # Compare results
    if run_logger and verbose:
        run_logger.log(f"[Multi-Clustering] COMPARISON SUMMARY:")
        run_logger.log(f"{'Config':<15} {'PosDrop_Acc':<12} {'Focus':<12}")
        run_logger.log("-" * 45)
        
        for config_name, result in all_results.items():
            if "error" not in result:
                posdrop = result.get('PosDrop_Acc', {}).get('mean', 0.0)
                focus = result.get('Focus', {}).get('mean', 0.0)
                run_logger.log(f"{config_name:<15} {posdrop:<12.4f} {focus:<12.4f}")
            else:
                run_logger.log(f"{config_name:<15} {'ERROR':<12} {'ERROR':<12}")
    
    return all_results

def _convert_to_serializable(obj):
    """Recursively convert numpy types and other non-serializable types to JSON-serializable types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        if obj.ndim == 0:  # scalar
            return float(obj)
        else:
            return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif obj is None:
        return None
    else:
        return obj


def save_meta_evaluation_results(
    *,
    meta_evaluation: Dict[str, Any],
    tag_main: str,
    common: Dict[str, Any],
    run_logger: Optional[Any] = None
) -> None:
    """
    Save Meta_Evaluation results to JSON file.
    
    Args:
        meta_evaluation: Dictionary containing Meta_Evaluation metrics
        tag_main: Tag identifier for the experiment
        common: Dictionary containing cfg
        run_logger: Optional logger for status messages
    """
    import json
    from pathlib import Path
    import numpy as np
    
    cfg = common['cfg']
    output_dir = Path(cfg.output_dir) / "cls_1_training_results/meta_evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{tag_main}_meta_eval.json"
    
    # Convert numpy types to native Python types for JSON serialization
    serializable_meta = {}
    for metric_name, metric_data in meta_evaluation.items():
        if isinstance(metric_data, dict):
            # Handle simple metric structure (PosDrop_Acc, Focus)
            if 'scores' in metric_data and 'mean' in metric_data and 'std' in metric_data:
                serializable_meta[metric_name] = {
                    'scores': [float(s) for s in metric_data['scores']],
                    'mean': float(metric_data['mean']),
                    'std': float(metric_data['std']),
                    'scores_info': {
                        'count': len(metric_data.get('scores', [])),
                        'description': "Per-iteration metric values (one entry per positive-drop fold)."
                    }
                }
            else:
                # Handle complex nested structures (pauc, topk) - serialize as-is with type conversion
                serializable_meta[metric_name] = _convert_to_serializable(metric_data)
        else:
            # Handle scalar values
            serializable_meta[metric_name] = float(metric_data) if metric_data is not None else None
    
    with open(output_path, 'w') as f:
        json.dump(serializable_meta, f, indent=2)
    
    if run_logger:
        run_logger.log(f"[save_meta_evaluation] Saved Meta_Evaluation results to {output_path}")


def compute_proxy_auc_variants(probs_all, y_true, prior_variants, run_logger=None):
    """
    Compute Proxy AUC (PAUC) with multiple prior assumptions for PU learning sensitivity analysis.
    
    Args:
        probs_all: Array of all probabilities/predictions
        y_true: True binary labels (1 for positives, 0 for unlabeled)
        prior_variants: List of prior probabilities (π values) to test
        run_logger: Optional logger for tracking progress
    
    Returns:
        dict: Results for each prior variant with AUC scores
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
   
    results = {}
    positive_mask = y_true == 1
    n_positives = positive_mask.sum()
    n_total = len(y_true)
    
    if n_positives == 0:
        if run_logger:
            run_logger.log("[compute_proxy_auc_variants] Warning: No positive samples found")
        return {f"prior_{prior:.3f}": 0.5 for prior in prior_variants}
    
    for prior in prior_variants:
        try:
            # Proxy AUC: Correct predictions using estimated class prior
            # For PU learning: P(s=1|y=1) = 1, P(s=1|y=0) = π/(1-π) * P(y=1|x)
            corrected_probs = probs_all.copy()
            
            # Apply prior correction for unlabeled samples
            unlabeled_mask = ~positive_mask
            if unlabeled_mask.sum() > 0:
                # Estimate true positive probability given prior assumption
                corrected_probs[unlabeled_mask] = corrected_probs[unlabeled_mask] * prior / (
                    corrected_probs[unlabeled_mask] * prior + (1 - corrected_probs[unlabeled_mask]) * (1 - prior)
                )
            
            # Compute AUC with corrected probabilities
            if len(np.unique(y_true)) > 1:
                auc_score = roc_auc_score(y_true, corrected_probs)
            else:
                auc_score = 0.5
                
            results[f"prior_{prior:.3f}"] = float(auc_score)
            
        except Exception as e:
            if run_logger:
                run_logger.log(f"[compute_proxy_auc_variants] Error with prior π={prior:.3f}: {e}")
            results[f"prior_{prior:.3f}"] = 0.5
    
    if run_logger:
        run_logger.log(f"[compute_proxy_auc_variants] Completed PAUC for {len(prior_variants)} priors")
    return results


def compute_topk_positive_capture_with_ratios(probs_all, y_true, k_ratios, k_values, area_percentages, run_logger=None):
    """
    Compute Top-K Positive Capture with ratio-based K values and area percentages.
    
    Args:
        probs_all: Array of all probabilities/predictions
        y_true: True binary labels (1 for positives, 0 for unlabeled)
        k_ratios: List of K ratios relative to number of known positives
        k_values: List of absolute K values to test
        area_percentages: List of area percentages for analysis
        run_logger: Optional logger for tracking progress
    
    Returns:
        dict: Results for different K specifications and area analyses
    """
    import numpy as np
    
   
    results = {}
    positive_mask = y_true == 1
    n_positives = positive_mask.sum()
    n_total = len(y_true)
    
    if n_positives == 0:
        if run_logger:
            run_logger.log("[compute_topk_positive_capture] Warning: No positive samples found")
        return {"no_positives": True}
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(probs_all)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_probs = probs_all[sorted_indices]
    
    ratio_summaries = []
    ratio_results = {}
    for ratio in k_ratios:
        if ratio <= 0:
            continue
        k_effective = max(1, int(ratio * n_positives))  # At least 1
        k_effective = min(k_effective, n_total)  # At most total samples
        
        # Count positives in top K
        positives_in_topk = sorted_labels[:k_effective].sum()
        capture_rate = positives_in_topk / n_positives if n_positives > 0 else 0.0
        precision = positives_in_topk / k_effective if k_effective > 0 else 0.0
        
        ratio_results[f"ratio_{ratio:.2f}"] = {
            'k_effective': int(k_effective),
            'positives_captured': int(positives_in_topk),
            'capture_rate': float(capture_rate),
            'precision': float(precision),
            'area_fraction': k_effective / n_total
        }
        ratio_summaries.append({'ratio': ratio, 'capture': capture_rate, 'precision': precision})
    
    results['ratios'] = ratio_results
    # Keep verbose summary suppressed to avoid log spam
    
    # Absolute K values
    absolute_summaries = []
    absolute_results = {}
    for k in k_values:
        k_effective = min(k, n_total)
        
        positives_in_topk = sorted_labels[:k_effective].sum()
        capture_rate = positives_in_topk / n_positives if n_positives > 0 else 0.0
        precision = positives_in_topk / k_effective if k_effective > 0 else 0.0
        
        absolute_results[f"k_{k}"] = {
            'k_effective': int(k_effective),
            'positives_captured': int(positives_in_topk),
            'capture_rate': float(capture_rate),
            'precision': float(precision),
            'area_fraction': k_effective / n_total
        }
        absolute_summaries.append({'k': k, 'capture': capture_rate})
    
    results['absolute'] = absolute_results
    # Logging muted for absolute sweep
    
    # Area percentage analysis
    area_summaries = []
    area_results = {}
    for area_pct in area_percentages:
        k_area = max(1, int(area_pct * n_total / 100))  # Convert percentage to count
        k_area = min(k_area, n_total)
        
        positives_in_area = sorted_labels[:k_area].sum()
        capture_rate = positives_in_area / n_positives if n_positives > 0 else 0.0
        precision = positives_in_area / k_area if k_area > 0 else 0.0
        
        area_results[f"area_{area_pct:.1f}pct"] = {
            'k_effective': int(k_area),
            'positives_captured': int(positives_in_area),
            'capture_rate': float(capture_rate),
            'precision': float(precision),
            'area_fraction': k_area / n_total
        }
        area_summaries.append({'area': area_pct, 'capture': capture_rate})
    
    results['area_percentages'] = area_results
    # Logging muted for area sweep
    
    # Summary statistics
    results['summary'] = {
        'total_positives': int(n_positives),
        'total_samples': int(n_total),
        'positive_rate': float(n_positives / n_total),
        'max_prob': float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0,
        'min_prob': float(sorted_probs[-1]) if len(sorted_probs) > 0 else 0.0
    }
    
    return results


def compute_pu_negative_metrics(
    probs_all: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    pi_values: List[float],
    run_logger=None,
) -> Optional[Dict[str, Any]]:
    """Compute PU-based negative metrics (FPR, NPV, specificity, FDR, background rejection)."""
    probs = np.asarray(probs_all, dtype=float)
    labels = np.asarray(labels)
    pos_mask = labels == 1
    unl_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_unl = int(unl_mask.sum())
    if n_pos == 0 or n_unl == 0:
        if run_logger:
            run_logger.log("[compute_pu_negative_metrics] Skipping due to insufficient positives/unlabeled")
        return None
    
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=float)
    thresholds = np.unique(thresholds)
    
    pos_sorted = np.sort(probs[pos_mask])
    unl_sorted = np.sort(probs[unl_mask])
    
    pos_left = np.searchsorted(pos_sorted, thresholds, side="left")
    unl_left = np.searchsorted(unl_sorted, thresholds, side="left")
    
    tpr = (n_pos - pos_left) / max(n_pos, 1)
    mix_mass = (n_unl - unl_left) / max(n_unl, 1)
    br = unl_left / max(n_unl, 1)
    pos_less = pos_left / max(n_pos, 1)
    
    cleaned_pi = [pi for pi in pi_values if 0.0 < pi < 1.0]
    if not cleaned_pi:
        cleaned_pi = [0.05, 0.25]
    
    eps = 1e-8
    fpr_bounds = {}
    npv_bounds = {}
    spec_bounds = {}
    fdr_bounds = {}
    for pi in cleaned_pi:
        denom = max(1.0 - pi, eps)
        fpr = np.clip((mix_mass - pi * tpr) / denom, 0.0, 1.0)
        specificity = np.clip(1.0 - fpr, 0.0, 1.0)
        npv = np.clip((br - pi * pos_less) / denom, 0.0, 1.0)
        fdr = np.clip(((1 - pi) * fpr) / (pi * tpr + (1 - pi) * fpr + eps), 0.0, 1.0)
        
        fpr_bounds[pi] = fpr
        npv_bounds[pi] = npv
        spec_bounds[pi] = specificity
        fdr_bounds[pi] = fdr
    
    return {
        "thresholds": thresholds.tolist(),
        "tpr": tpr.tolist(),
        "background_rejection": br.tolist(),
        "fpr_bounds": fpr_bounds,
        "npv_bounds": npv_bounds,
        "specificity_bounds": spec_bounds,
        "fdr_bounds": fdr_bounds,
    }


def _aggregate_pi_metric(results: List[Dict[str, Any]], key: str, pi_values: List[float]) -> Dict[str, Any]:
    """Aggregate per-threshold metrics that depend on class prior."""
    if not results:
        return {}
    thresholds = results[0]["thresholds"]
    aggregated = []
    for pi in pi_values:
        stacks = [res[key][pi] for res in results if pi in res[key]]
        if not stacks:
            continue
        arr = np.stack(stacks)
        aggregated.append({
            "pi": float(pi),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        })
    return {"thresholds": thresholds, "pi_stats": aggregated}


def _aggregate_simple_metric(results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Aggregate per-threshold metrics without prior dependence."""
    if not results:
        return {}
    thresholds = results[0]["thresholds"]
    arr = np.stack([np.asarray(res[key], dtype=float) for res in results])
    return {
        "thresholds": thresholds,
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def compute_extended_meta_evaluation(
    *,
    meta_eval_config: MetaEvaluationConfig,
    requested_metrics: Optional[List[str]],
    all_probabilities,
    all_labels,
    run_logger=None,
):
    """
    Compute extended meta-evaluation metrics based on a config bundle.
    
    Args:
        meta_eval_config: MetaEvaluationConfig containing parameter defaults
        requested_metrics: Subset of metrics to compute
        all_probabilities: List of probability arrays from different clustering configurations
        all_labels: List of label arrays from different clustering configurations
        run_logger: Optional logger for tracking progress
    
    Returns:
        dict: Extended meta-evaluation results including PAUC, TopK, and PU metrics
    """
    import numpy as np
    
    extended_results = {}
    
    # Check if extended metrics are requested
    meta_eval_metrics = set(requested_metrics or [])
    
    # Compute Proxy AUC variants if requested
    if 'pauc' in meta_eval_metrics:
        prior_variants = meta_eval_config.pauc_prior_variants or [0.01, 0.05, 0.1, 0.2, 0.3]
        
        pauc_results = []
        for i, (probs, labels) in enumerate(zip(all_probabilities, all_labels)):
            config_pauc = compute_proxy_auc_variants(probs, labels, prior_variants, run_logger)
            pauc_results.append(config_pauc)
        
        # Aggregate PAUC results across configurations
        pauc_aggregated = {}
        for prior_key in pauc_results[0].keys():
            prior_scores = [result[prior_key] for result in pauc_results]
            pauc_aggregated[prior_key] = {
                'scores': prior_scores,
                'mean': float(np.mean(prior_scores)),
                'std': float(np.std(prior_scores))
            }
        
        extended_results['pauc'] = pauc_aggregated
        
        if run_logger:
            run_logger.log(f"[compute_extended_meta_evaluation] PAUC aggregated across {len(pauc_results)} configurations")
    
    # Compute Top-K Positive Capture if requested
    if 'topk' in meta_eval_metrics:
        k_ratios = meta_eval_config.topk_k_ratio or [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
        k_values = meta_eval_config.topk_k_values or [10, 50, 100, 500, 1000]
        area_percentages = meta_eval_config.topk_area_percentages or [0.1, 0.5, 1.0, 2.0, 5.0]
        
        topk_results = []
        for i, (probs, labels) in enumerate(zip(all_probabilities, all_labels)):
            config_topk = compute_topk_positive_capture_with_ratios(
                probs, labels, k_ratios, k_values, area_percentages, run_logger
            )
            topk_results.append(config_topk)
        
        # Aggregate TopK results across configurations
        topk_aggregated = {'ratios': {}, 'absolute': {}, 'area_percentages': {}, 'summary': {}}
        
        # Aggregate ratio-based results
        if 'ratios' in topk_results[0] and topk_results[0]['ratios']:
            for ratio_key in topk_results[0]['ratios'].keys():
                ratio_data = [result['ratios'][ratio_key] for result in topk_results if 'ratios' in result and ratio_key in result['ratios']]
                if ratio_data:
                    topk_aggregated['ratios'][ratio_key] = {
                        'capture_rates': [d['capture_rate'] for d in ratio_data],
                        'precisions': [d['precision'] for d in ratio_data],
                        'mean_capture_rate': float(np.mean([d['capture_rate'] for d in ratio_data])),
                        'std_capture_rate': float(np.std([d['capture_rate'] for d in ratio_data])),
                        'mean_precision': float(np.mean([d['precision'] for d in ratio_data])),
                        'std_precision': float(np.std([d['precision'] for d in ratio_data]))
                    }
        
        # Aggregate absolute K results
        if 'absolute' in topk_results[0] and topk_results[0]['absolute']:
            for k_key in topk_results[0]['absolute'].keys():
                k_data = [result['absolute'][k_key] for result in topk_results if 'absolute' in result and k_key in result['absolute']]
                if k_data:
                    topk_aggregated['absolute'][k_key] = {
                        'capture_rates': [d['capture_rate'] for d in k_data],
                        'precisions': [d['precision'] for d in k_data],
                        'mean_capture_rate': float(np.mean([d['capture_rate'] for d in k_data])),
                        'std_capture_rate': float(np.std([d['capture_rate'] for d in k_data])),
                        'mean_precision': float(np.mean([d['precision'] for d in k_data])),
                        'std_precision': float(np.std([d['precision'] for d in k_data]))
                    }
        
        # Aggregate area percentage results
        if 'area_percentages' in topk_results[0] and topk_results[0]['area_percentages']:
            for area_key in topk_results[0]['area_percentages'].keys():
                area_data = [result['area_percentages'][area_key] for result in topk_results if 'area_percentages' in result and area_key in result['area_percentages']]
                if area_data:
                    topk_aggregated['area_percentages'][area_key] = {
                        'capture_rates': [d['capture_rate'] for d in area_data],
                        'precisions': [d['precision'] for d in area_data],
                        'mean_capture_rate': float(np.mean([d['capture_rate'] for d in area_data])),
                        'std_capture_rate': float(np.std([d['capture_rate'] for d in area_data])),
                        'mean_precision': float(np.mean([d['precision'] for d in area_data])),
                        'std_precision': float(np.std([d['precision'] for d in area_data]))
                    }
        
        extended_results['topk'] = topk_aggregated
        
        if run_logger:
            run_logger.log(f"[compute_extended_meta_evaluation] TopK aggregated across {len(topk_results)} configurations")
    
    # PU-based negative metrics (FPR, background rejection, NPV, FDR)
    requested_neg_metrics = [m for m in meta_eval_metrics if m in NEGATIVE_META_EVALUATION]
    if requested_neg_metrics:
        pi_range = meta_eval_config.pu_prior_range or [0.05, 0.25]
        pi_values = sorted({float(pi) for pi in pi_range})
        
        thresholds = meta_eval_config.pu_metric_thresholds
        if thresholds:
            thresholds = np.array(sorted({float(t) for t in thresholds}))
        else:
            threshold_count = max(5, int(meta_eval_config.pu_metric_threshold_count))
            combined_probs = np.concatenate([np.asarray(p) for p in all_probabilities]) if all_probabilities else np.array([0.0])
            quantiles = np.linspace(0.0, 1.0, threshold_count)
            thresholds = np.quantile(combined_probs, quantiles)
        
        neg_results = []
        for probs, labels in zip(all_probabilities, all_labels):
            res = compute_pu_negative_metrics(probs, labels, thresholds, pi_values, run_logger)
            if res is not None:
                neg_results.append(res)
        
        if neg_results:
            if 'pu_fpr' in requested_neg_metrics:
                extended_results['pu_fpr'] = _aggregate_pi_metric(neg_results, 'fpr_bounds', pi_values)
            if 'background_rejection' in requested_neg_metrics:
                extended_results['background_rejection'] = _aggregate_simple_metric(neg_results, 'background_rejection')
            if 'pu_npv' in requested_neg_metrics:
                extended_results['pu_npv'] = {
                    'npv': _aggregate_pi_metric(neg_results, 'npv_bounds', pi_values),
                    'specificity': _aggregate_pi_metric(neg_results, 'specificity_bounds', pi_values),
                }
            if 'pu_fdr' in requested_neg_metrics:
                extended_results['pu_fdr'] = _aggregate_pi_metric(neg_results, 'fdr_bounds', pi_values)
            if 'pu_tpr' in requested_neg_metrics:
                extended_results['pu_tpr'] = _aggregate_simple_metric(neg_results, 'tpr')
            
            if run_logger:
                run_logger.log(f"[compute_extended_meta_evaluation] PU negative metrics aggregated over {len(neg_results)} configurations")
        elif run_logger:
            run_logger.log("[compute_extended_meta_evaluation] Skipped PU negative metrics (insufficient data)")

    return extended_results


# ============================================================================
# Export Functions for External Use
# ============================================================================

__all__ = [
    'MetaEvaluationConfig',
    'train_cls_1_PN_PosDrop',
    'train_cls_1_PN_PosDrop_MultiClustering',
    'load_and_evaluate_existing_predictions',
    'save_meta_evaluation_results',
    'create_pos_drop_schedule_unified',
    'compute_meta_evaluation_metric',
    'compute_proxy_auc_variants',
    'compute_topk_positive_capture_with_ratios',
    'compute_extended_meta_evaluation'
]
