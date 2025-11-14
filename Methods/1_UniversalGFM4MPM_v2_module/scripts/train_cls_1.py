from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
from Common.cls.sampling.likely_negatives import pu_select_negatives

Meta_Evaluation = {"PosDrop_Acc", "Focus"}

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
    
    Args:s
        Xtr: Training features tensor
        Xval: Validation features tensor  
        ytr: Training labels tensor
        yval: Validation labels tensor
        common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
        data_use: Dictionary containing original data for inference
        inference_fn: Optional inference function to run after training
        pos_crd_plot: Optional positive coordinates for plotting
    
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
        inference_result = inference_fn(
            samples=data_use,
            cls=cls,
            device=device,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "All",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes,
            pos_crd=pos_crd_plot,
            tag=tag
        )
    
    # Return results
    results = {
        'cls': cls,
        'epoch_history': epoch_history,
        'inference_result': inference_result,
        'metrics_summary': metrics_summary_append,
        'early_stopping_summary': summary,  # ✅ Add early stopping summary
    }

    return results

def train_cls_1_PN(
    *,
    common: Dict[str, Any],  # Contains cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
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
    
    Returns:
        Dictionary containing:
            - 'gA': Trained classifier model
            - 'epoch_history': Training history
            - 'inference_result': Inference results on training data
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
        tag = tag
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
    
    # Extract coordinates for dataloaders
    coords_pn = [data_use["coords"][i] for i in pn_indices] if "coords" in data_use else None
    
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



def train_cls_1_PN_PosDrop(
    *,
    drop_rate: float = 0.1,
    common: Dict[str, Any],  # Contains cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
    data_use: Dict[str, Any],
    filter_top_pct: float = 0.10,
    negs_per_pos: int = 10,
    action: Optional[Dict[str, Any]] = None,
    inference_fn: Optional[callable] = None,
    tag_main: Optional[str] = None,
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
    
    Returns:
        Dictionary containing:
            - 'gA': Trained classifier model
            - 'epoch_history': Training history
            - 'inference_result': Inference results on training data
    """
    
    # Extract common parameters
    cfg = common['cfg']
    device = common['device']
    mlp_hidden_dims = common['mlp_hidden_dims']
    mlp_dropout = common['mlp_dropout']
    debug_mode = common.get('debug_mode', False)
    run_logger = common.get('run_logger', None)
    verbose = common.get('verbose', True)  # ✅ Add verbose control
    
    if data_use is None:
        raise RuntimeError("Data unavailable for classifier training; aborting.")
    
    # Extract features and labels
    features = data_use["features"].float().to(device)
    yA_pu = torch.where(data_use["labels"] >= 0.9, 1, -1).long().to(device)
    
    # Get coordinates for visualization
    temp_crd = data_use.get("coords")
    coords_array = np.array(temp_crd) if temp_crd is not None else None

    # Prepare positive dropping: split positives across iterations
    all_pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
    all_unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
    
    num_iterations = int(1.0 / drop_rate)
    pos_per_iteration = len(all_pos_idx) // num_iterations
    remaining_pos = len(all_pos_idx) % num_iterations
    
    # Shuffle positives for random distribution across iterations
    rng = np.random.default_rng(int(cfg.seed))
    shuffled_pos_idx = rng.permutation(all_pos_idx).tolist()
    
    # Split positives across iterations
    pos_splits = []
    start_idx = 0
    for i in range(num_iterations):
        # Add one extra positive to first 'remaining_pos' iterations
        extra = 1 if i < remaining_pos else 0
        end_idx = start_idx + pos_per_iteration + extra
        pos_splits.append(shuffled_pos_idx[start_idx:end_idx])
        start_idx = end_idx
    
   
    all_results = []
    meta_evaluation = []
    for iteration in range(num_iterations):
        tag = f"{tag_main}_iter{iteration+1}_drop{drop_rate}" if tag_main is not None else None


        # Use only the positives assigned to this iteration
        pos_idx = pos_splits[iteration]
        unl_idx = all_unl_idx.copy()  # All unlabeled samples available for each iteration
        
        # Create pos_mask for this iteration's positives only
        pos_mask_iteration = np.zeros(len(yA_pu), dtype=bool)
        pos_mask_iteration[pos_idx] = True
        
        # Create coordinate plot for this iteration's positives
        pos_crd_plot = None
        if coords_array is not None:
            pos_crd_plot = [coords_array[i] for i in pos_idx]

        pos_idx_arr = np.asarray(pos_idx, dtype=int)
        unk_idx_arr = np.asarray(unl_idx, dtype=int)
        
        if debug_mode and run_logger is not None and verbose:
            run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")
        
        # Reduce logging frequency - only log at first iteration or if verbose
        if run_logger is not None and (iteration == 0 or verbose):
            run_logger.log(f"[cls-1-PN] Converting features to numpy...")
            run_logger.log(f"[cls-1-PN] Starting negative selection...")
        
        # Perform negative selection
        neg_idx_region = pu_select_negatives(
            Z_all=features.cpu().numpy(),
            pos_idx=pos_idx_arr,
            unk_idx=unk_idx_arr,
            filter_top_pct=filter_top_pct,
            negatives_per_pos=negs_per_pos,
            tag = tag
        )
        
        if debug_mode and run_logger is not None and verbose:
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
        
        # Extract coordinates for dataloaders
        coords_pn = [data_use["coords"][i] for i in pn_indices] if "coords" in data_use else None
        
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
        iteration_results = train_base_classifier(
            Xtr=Xtr,
            Xval=Xval,
            ytr=ytr,
            yval=yval,
            common=common,
            data_use=data_use,
            inference_fn=inference_fn,
            pos_crd_plot=pos_crd_plot,
            tag = tag
        )

        # ✅ ADD: Include iteration-specific metadata
        iteration_results['pos_indices_this_iter'] = pos_idx  # The positives used in THIS iteration

        all_results.append(iteration_results)
        
   
    # ✅ ADD: "No Positive Drop" - Train with ALL positives
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN-PosDrop] Starting final 'no positive drop' training with all {len(all_pos_idx)} positives")
    
    tag_all = f"{tag_main}_all" if tag_main is not None else None
    
    # Use ALL positives (no dropping)
    pos_idx = all_pos_idx
    unl_idx = all_unl_idx.copy()
    
    # Create coordinate plot for ALL positives
    pos_crd_plot_all = None
    if coords_array is not None:
        pos_crd_plot_all = [coords_array[i] for i in pos_idx]

    pos_idx_arr = np.asarray(pos_idx, dtype=int)
    unk_idx_arr = np.asarray(unl_idx, dtype=int)
    
    if debug_mode and run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")
    
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN] Converting features to numpy...")
        run_logger.log(f"[cls-1-PN] Starting negative selection...")
    
    # Perform negative selection using ALL positives
    neg_idx_region_all = pu_select_negatives(
        Z_all=features.cpu().numpy(),
        pos_idx=pos_idx_arr,
        unk_idx=unk_idx_arr,
        filter_top_pct=filter_top_pct,
        negatives_per_pos=negs_per_pos,
        tag=tag_all
    )
    
    if debug_mode and run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN] Finished negative selection: selected {len(neg_idx_region_all)} negatives")
    
    neg_idx_arr_all = np.asarray(neg_idx_region_all, dtype=int)
    
    # Create Balanced P+N Dataset with ALL positives
    pn_indices_all = np.concatenate([pos_idx_arr, neg_idx_arr_all], axis=0)
    inf_indices_all = np.setdiff1d(np.arange(len(features)), pn_indices_all)
    
    # Extract features and labels for P+N
    features_pn_all = features[pn_indices_all]
    yA_pn_all = torch.where(
        torch.tensor([i in pos_idx for i in pn_indices_all], dtype=torch.bool),
        1,  # Positive
        0   # Negative (0 for BCE loss)
    ).long()
    
    # Train/Val Split on P+N
    val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
    total_pn_all = len(pn_indices_all)
    val_count_all = int(total_pn_all * val_frac)
    
    gen_all = torch.Generator()
    gen_all.manual_seed(int(cfg.seed))
    indices_pn_all = torch.randperm(total_pn_all, generator=gen_all)
    
    val_indices_pn_all = indices_pn_all[:val_count_all].tolist()
    train_indices_pn_all = indices_pn_all[val_count_all:].tolist()
    
    # Create train/val data
    Xtr_all = features_pn_all[train_indices_pn_all]
    ytr_all = yA_pn_all[train_indices_pn_all]
    Xval_all = features_pn_all[val_indices_pn_all]
    yval_all = yA_pn_all[val_indices_pn_all]
    
    # Train the classifier using ALL positives
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
    
    # Include metadata for the "all positives" run
    all_results_final['pos_indices_this_iter'] = pos_idx  # ALL positives used
    all_results.append(all_results_final)
    
    if run_logger is not None and verbose:
        run_logger.log(f"[cls-1-PN-PosDrop] Completed final 'no positive drop' training")
    
    # get meta-evaluation
    meta_evaluation = {}
    
    for metric in Meta_Evaluation:
        metric_scores = []

        for iteration in range(num_iterations):
            # ✅ Safely get iteration results
            iter_result = all_results[iteration]

            pos_indices_this_iter = set(iter_result['pos_indices_this_iter'])

            if metric == "PosDrop_Acc":
                # PosDrop_Acc: average prediction accuracy on positives dropped in each iteration
                # Find positives that were NOT used in this iteration (dropped positives)
                dropped_pos_idx = [idx for idx in all_pos_idx if idx not in pos_indices_this_iter]
                
                if len(dropped_pos_idx) > 0:
                    # Get predictions for dropped positives
                    predictions_mean = iter_result['inference_result']['predictions_mean']
                    
                    # ✅ Convert to numpy if tensor
                    if isinstance(predictions_mean, torch.Tensor):
                        predictions_mean = predictions_mean.cpu().numpy()
                    
                    # Index into predictions to get dropped positive predictions
                    dropped_pos_probs = predictions_mean[dropped_pos_idx]
                    accuracy = float(np.mean(dropped_pos_probs))
                    metric_scores.append(accuracy)
                else:
                    # No dropped positives in this iteration (shouldn't happen with drop_rate < 1.0)
                    metric_scores.append(0.0)
                        
            elif metric == "Focus":
                # Get predictions for dropped positives
                predictions_mean = iter_result['inference_result']['predictions_mean']
                
                # ✅ Convert to numpy if tensor
                if isinstance(predictions_mean, torch.Tensor):
                    predictions_mean = predictions_mean.cpu().numpy()
                
                # Calculate Focus score (proportion of predictions <= 0.5)
                predictions_sum = predictions_mean.sum()
                total_predictions = len(predictions_mean)
                focus_score = 1.0 - (predictions_sum / total_predictions)
                metric_scores.append(float(focus_score))
        
        meta_evaluation[metric] = {
            'scores': metric_scores,
            'mean': np.mean(metric_scores) if metric_scores else 0.0,
            'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
        }

        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
    return meta_evaluation    