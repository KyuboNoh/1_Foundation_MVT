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
    gA = MLPDropout(in_dim=in_dim, hidden_dims=mlp_hidden_dims, p=float(mlp_dropout)).to(device)
    
    # Identity encoder since we're already in projected space
    encA = nn.Identity().to(device)
    
    if run_logger is not None:
        run_logger.log("[train_base_classifier] Training classifier...")
    
    # Train classifier
    gA, epoch_history = train_classifier(
        encA,
        gA,
        dl_tr,
        dl_va,
        epochs=cfg.cls_training.epochs,
        return_history=True,
        loss_weights={'bce': 1.0},
    )
    
    # Phase 3: Run Inference (optional - only if inference function provided)
    inference_result = None
    if inference_fn is not None:
        inference_result = inference_fn(
            samples=data_use,
            gA=gA,
            device=device,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "All",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes,
            pos_crd=pos_crd_plot
        )
    
    # Return results
    results = {
        'gA': gA,
        'epoch_history': epoch_history,
        'inference_result': inference_result,
        'metrics_summary': metrics_summary_append,
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
    
    if run_logger is not None:
        run_logger.log(f"[cls-1-PN-PosDrop] Splitting {len(all_pos_idx)} positives across {num_iterations} iterations")
        run_logger.log(f"[cls-1-PN-PosDrop] Positives per iteration: {[len(split) for split in pos_splits]}")
    
    all_results = []
    meta_evaluation = []
    for iteration in range(num_iterations):
        tag = f"{tag}_iter{iteration+1}_drop{drop_rate}" if tag is not None else None

        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] Starting iteration {iteration+1}/{num_iterations}")

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
        iteration_results = train_base_classifier(
            Xtr=Xtr,
            Xval=Xval,
            ytr=ytr,
            yval=yval,
            common=common,
            data_use=data_use,
            inference_fn=inference_fn if iteration == num_iterations - 1 else None,  # Only run inference on last iteration
            pos_crd_plot=pos_crd_plot
        )
        
        # Add iteration-specific information
        iteration_results.update({
            'iteration': iteration,
            'pos_indices_this_iter': pos_idx,
            'pn_indices': pn_indices,
            'inf_indices': inf_indices,
        })
        
        all_results.append(iteration_results)
        
        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] Completed iteration {iteration+1}/{num_iterations}")
    
    # get meta-evaluation
    meta_evaluation = {}
    
    for metric in Meta_Evaluation:
        metric_scores = []
        
        if metric == "PosDrop_Acc":
            # PosDrop_Acc: average prediction accuracy on positives dropped in each iteration
            for iteration in range(num_iterations):
                gA = all_results[iteration]['gA']
                pos_indices_this_iter = set(all_results[iteration]['pos_indices_this_iter'])
                
                # Find positives that were NOT used in this iteration (dropped positives)
                dropped_pos_idx = [idx for idx in all_pos_idx if idx not in pos_indices_this_iter]
                
                if len(dropped_pos_idx) > 0:
                    # Get features for dropped positives
                    dropped_pos_features = features[dropped_pos_idx]
                    
                    # Run inference on dropped positives
                    gA.eval()
                    with torch.no_grad():
                        logits = gA(dropped_pos_features)
                        probs = torch.sigmoid(logits).squeeze()
                        
                        # Calculate accuracy (predictions > 0.5 for positive samples)
                        predictions = (probs > 0.5).float()
                        accuracy = predictions.mean().item()
                        metric_scores.append(accuracy)
                else:
                    # No dropped positives in this iteration (shouldn't happen with drop_rate < 1.0)
                    metric_scores.append(1.0)
                    
        elif metric == "Focus":
            # Focus: how many predictions are above 0.5 on all predictions
            # Focus score is 1 - (num_predictions_above_0.5) / (total_num_predictions)
            for iteration in range(num_iterations):
                gA = all_results[iteration]['gA']
                
                # Run inference on all features
                gA.eval()
                with torch.no_grad():
                    # Process in batches to avoid memory issues
                    batch_size = 1000
                    all_probs = []
                    
                    for i in range(0, len(features), batch_size):
                        batch_features = features[i:i+batch_size]
                        batch_logits = gA(batch_features)
                        batch_probs = torch.sigmoid(batch_logits).squeeze()
                        all_probs.append(batch_probs)
                    
                    # Concatenate all probabilities
                    all_probs = torch.cat(all_probs, dim=0)
                    
                    # Calculate Focus score
                    num_above_threshold = (all_probs > 0.5).sum().item()
                    total_predictions = len(all_probs)
                    focus_score = 1.0 - (num_above_threshold / total_predictions)
                    metric_scores.append(focus_score)
        
        meta_evaluation[metric] = {
            'scores': metric_scores,
            'mean': np.mean(metric_scores) if metric_scores else 0.0,
            'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
        }
        
        if run_logger is not None:
            run_logger.log(f"[cls-1-PN-PosDrop] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
    return meta_evaluation    