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
from typing import Dict, List, Optional, Tuple, Any, Callable

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

# Default meta-evaluation metrics (can be overridden via command-line argument)
DEFAULT_META_EVALUATION = {"PosDrop_Acc", "Focus"}


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


# Registry of meta-evaluation functions
META_EVALUATION_FUNCTIONS: Dict[str, Callable] = {
    "PosDrop_Acc": compute_posdrop_acc,
    "Focus": compute_focus,
}


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
    if metric_name not in META_EVALUATION_FUNCTIONS:
        raise ValueError(f"Unknown meta-evaluation metric: {metric_name}. Available: {list(META_EVALUATION_FUNCTIONS.keys())}")
    
    metric_fn = META_EVALUATION_FUNCTIONS[metric_name]
    
    # Build kwargs based on metric requirements
    kwargs = {"predictions_mean": predictions_mean}
    
    if metric_name == "PosDrop_Acc":
        if all_pos_idx is None or pos_indices_this_iter is None:
            raise ValueError(f"Metric '{metric_name}' requires all_pos_idx and pos_indices_this_iter")
        kwargs["all_pos_idx"] = all_pos_idx
        kwargs["pos_indices_this_iter"] = pos_indices_this_iter
    
    # Add predictions_std if provided and metric supports it
    if predictions_std is not None and "predictions_std" in metric_fn.__code__.co_varnames:
        kwargs["predictions_std"] = predictions_std
    
    return metric_fn(**kwargs)


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
    meta_evaluation_metrics: Optional[set] = None,
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
        meta_evaluation_metrics: Set of metric names to compute (default: PosDrop_Acc, Focus)
    
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
    
    # Use provided metrics or default
    if meta_evaluation_metrics is None:
        meta_evaluation_metrics = DEFAULT_META_EVALUATION
    
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
    
    # Compute meta-evaluation metrics
    meta_evaluation = {}
    
    for metric in meta_evaluation_metrics:
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
    
    return meta_evaluation


# ============================================================================
# Cached Prediction Loading and Evaluation
# ============================================================================

def load_and_evaluate_existing_predictions(
    *,
    tag_main: str,
    meta_evaluation_n_clusters: int,  # ✅ CHANGED: from drop_rate to n_clusters
    clustering_method: str = "random", # ✅ NEW: clustering method
    common: Dict[str, Any],
    data_use: Dict[str, Any],
    run_logger: Optional[Any] = None,
    meta_evaluation_metrics: Optional[set] = None
) -> Optional[Dict[str, Any]]:
    """
    Load existing predictions from disk and compute Meta_Evaluation metrics.
    
    Args:
        tag_main: Tag identifier for the experiment
        meta_evaluation_n_clusters: Number of clusters/iterations (2, 3, 4, 6, 8, 10)
        clustering_method: Clustering method used ('random', 'kmeans', 'hierarchical')
        common: Dictionary containing cfg, device, run_logger, etc.
        data_use: Dictionary containing 'features', 'labels', 'coords' keys
        run_logger: Optional logger for status messages
        meta_evaluation_metrics: Set of metric names to compute (default: PosDrop_Acc, Focus)
    
    Returns:
        Dictionary containing meta_evaluation metrics, or None if predictions not found
    """
    cfg = common['cfg']
    
    expected_num_iterations = meta_evaluation_n_clusters
    
    base_output_dir = Path(cfg.output_dir) / "cls_1_training_results" / "All" / tag_main
    
    # Use provided metrics or default
    if meta_evaluation_metrics is None:
        meta_evaluation_metrics = DEFAULT_META_EVALUATION
    
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
    
    # Compute meta-evaluation metrics
    meta_evaluation = {}
    
    for metric in meta_evaluation_metrics:
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
    
    Returns:
        Dictionary mapping config names to meta-evaluation results
    """
    
    run_logger = common.get('run_logger', None)
    verbose = common.get('verbose', True)
    
    if meta_evaluation_metrics is None:
        meta_evaluation_metrics = DEFAULT_META_EVALUATION
    
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
                meta_evaluation_metrics=meta_evaluation_metrics
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
    cfg = common['cfg']
    output_dir = Path(cfg.output_dir) / "cls_1_training_results/meta_evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{tag_main}_meta_evaluation.json"
    
    # Convert numpy types to native Python types for JSON serialization
    serializable_meta = {}
    for metric_name, metric_data in meta_evaluation.items():
        serializable_meta[metric_name] = {
            'scores': [float(s) for s in metric_data['scores']],
            'mean': float(metric_data['mean']),
            'std': float(metric_data['std'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_meta, f, indent=2)
    
    if run_logger:
        run_logger.log(f"[save_meta_evaluation] Saved Meta_Evaluation results to {output_path}")


# ============================================================================
# Export Functions for External Use
# ============================================================================

__all__ = [
    'train_cls_1_PN_PosDrop',
    'train_cls_1_PN_PosDrop_MultiClustering',
    'load_and_evaluate_existing_predictions',
    'save_meta_evaluation_results',
    'create_pos_drop_schedule_unified',
    'compute_meta_evaluation_metric',
    'DEFAULT_META_EVALUATION'
]
