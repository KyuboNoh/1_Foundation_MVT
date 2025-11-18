from __future__ import annotations
import json
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
from Common.cls.sampling.likely_negatives import (
    pu_select_negatives, 
    pu_select_negatives_optimized_for_rotation, 
    precompute_distances_for_rotation_drops,
    precompute_distances_per_positive,
    pu_select_negatives_from_individual_distances,
)

# Default meta-evaluation metrics (can be overridden via command-line argument)
DEFAULT_META_EVALUATION = {"PosDrop_Acc", "Focus"}

# def train_base_classifier(
#     *,
#     Xtr: torch.Tensor,
#     Xval: torch.Tensor, 
#     ytr: torch.Tensor,
#     yval: torch.Tensor,
#     common: Dict[str, Any],
#     data_use: Dict[str, Any],
#     inference_fn: Optional[callable] = None,
#     pos_crd_plot: Optional[List] = None,
#     tag: Dict[str, Any] = None,
# ) -> Dict[str, Any]:
#     """
#     Train a base classifier using prepared training/validation data.
    
#     Args:s
#         Xtr: Training features tensor
#         Xval: Validation features tensor  
#         ytr: Training labels tensor
#         yval: Validation labels tensor
#         common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
#         data_use: Dictionary containing original data for inference
#         inference_fn: Optional inference function to run after training
#         pos_crd_plot: Optional positive coordinates for plotting
    
#     Returns:
#         Dictionary containing trained model, history, and inference results
#     """
#     # Extract common parameters
#     cfg = common['cfg']
#     device = common['device']
#     mlp_hidden_dims = common['mlp_hidden_dims']
#     mlp_dropout = common['mlp_dropout']
#     debug_mode = common.get('debug_mode', False)
#     run_logger = common.get('run_logger', None)
    
#     # Phase 1: Create DataLoaders
#     dl_tr, dl_va, metrics_summary_append = dataloader_metric_inputORembedding(
#         Xtr=Xtr,
#         Xval=Xval,
#         ytr=ytr,
#         yval=yval,
#         batch_size=cfg.cls_training.batch_size,
#         positive_augmentation=False,
#         augmented_patches_all=None,
#         pos_coord_to_index=None,
#         window_size=None,
#         stack=None,
#         embedding=True,
#         epochs=cfg.cls_training.epochs
#     )
    
#     # Phase 2: Build Model & Train
#     in_dim = Xtr.size(1)
#     cls = MLPDropout(in_dim=in_dim, hidden_dims=mlp_hidden_dims, p=float(mlp_dropout)).to(device)
    
#     # Identity encoder since we're already in projected space
#     encA = nn.Identity().to(device)
    
#     if run_logger is not None:
#         run_logger.log("[train_base_classifier] Training classifier...")
    
#     # Train classifier - use verbose setting from common with early stopping
#     verbose = common.get('verbose', True)
#     cls, epoch_history, summary = train_classifier(
#         encA,
#         cls,
#         dl_tr,
#         dl_va,
#         epochs=cfg.cls_training.epochs,
#         return_history=True,
#         loss_weights={'bce': 1.0},
#         verbose=verbose,
#         early_stopping=True,
#         patience=15,
#         min_delta=1e-3,
#         restore_best_weights=True
#     )
    
#     # Phase 3: Run Inference (optional - only if inference function provided)
#     inference_result = None
#     if inference_fn is not None:
#         inference_result = inference_fn(
#             samples=data_use,
#             cls=cls,
#             device=device,
#             output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "All",
#             run_logger=run_logger,
#             passes=cfg.cls_training.mc_dropout_passes,
#             pos_crd=pos_crd_plot,
#             tag=tag
#         )
    
#     # Return results
#     results = {
#         'cls': cls,
#         'epoch_history': epoch_history,
#         'inference_result': inference_result,
#         'metrics_summary': metrics_summary_append,
#         'early_stopping_summary': summary,  # ✅ Add early stopping summary
#     }

#     return results

# def train_cls_1_PN(
#     *,
#     common: Dict[str, Any],  # Contains cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
#     data_use: Dict[str, Any],
#     filter_top_pct: float = 0.10,
#     negs_per_pos: int = 10,
#     action: Optional[Dict[str, Any]] = None,
#     inference_fn: Optional[callable] = None,
#     tag: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     Train a PN (Positive-Negative) classifier using negative selection from unlabeled data.
    
#     Args:
#         common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
#         data_use: Dictionary containing 'features', 'labels', 'coords' keys
#         filter_top_pct: Percentage of top similar samples to filter during negative selection
#         negs_per_pos: Number of negative samples to select per positive sample
#         action: Optional dictionary for future extensibility (learning rates, etc.)
#         inference_fn: Optional inference function to run after training
    
#     Returns:
#         Dictionary containing:
#             - 'gA': Trained classifier model
#             - 'epoch_history': Training history
#             - 'inference_result': Inference results on training data
#     """
    
#     # Extract common parameters
#     cfg = common['cfg']
#     device = common['device']
#     mlp_hidden_dims = common['mlp_hidden_dims']
#     mlp_dropout = common['mlp_dropout']
#     debug_mode = common.get('debug_mode', False)
#     run_logger = common.get('run_logger', None)
    
#     if data_use is None:
#         raise RuntimeError("Data unavailable for classifier training; aborting.")
    
#     # Extract features and labels
#     features = data_use["features"].float().to(device)
#     yA_pu = torch.where(data_use["labels"] >= 0.9, 1, -1).long().to(device)
    
#     # Get coordinates for visualization
#     temp_crd = data_use.get("coords")
#     coords_array = np.array(temp_crd) if temp_crd is not None else None
#     pos_mask = (yA_pu == 1).cpu().numpy()
#     pos_crd_plot = None
#     if coords_array is not None:
#         pos_crd_plot = [coords_array[i] for i in range(len(coords_array)) if pos_mask[i]]
    
#     # Phase 1: Negative Selection & Dataset Preparation
#     pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
#     unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
#     pos_idx_arr = np.asarray(pos_idx, dtype=int)
#     unk_idx_arr = np.asarray(unl_idx, dtype=int)
    
#     if debug_mode and run_logger is not None:
#         run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")
    
#     if run_logger is not None:
#         run_logger.log(f"[cls-1-PN] Converting features to numpy...")
#         run_logger.log(f"[cls-1-PN] Starting negative selection...")
    
#     # Perform negative selection
#     neg_idx_region = pu_select_negatives(
#         Z_all=features.cpu().numpy(),
#         pos_idx=pos_idx_arr,
#         unk_idx=unk_idx_arr,
#         filter_top_pct=filter_top_pct,
#         negatives_per_pos=negs_per_pos,
#         tag = tag
#     )
    
#     if debug_mode and run_logger is not None:
#         run_logger.log(f"[cls-1-PN] Finished negative selection: selected {len(neg_idx_region)} negatives")
    
#     neg_idx_arr = np.asarray(neg_idx_region, dtype=int)
    
#     # Phase 2: Create Balanced P+N Dataset
#     pn_indices = np.concatenate([pos_idx_arr, neg_idx_arr], axis=0)
#     inf_indices = np.setdiff1d(np.arange(len(features)), pn_indices)
    
#     # Extract features and labels for P+N
#     features_pn = features[pn_indices]
#     yA_pn = torch.where(
#         torch.tensor([i in pos_idx for i in pn_indices], dtype=torch.bool),
#         1,  # Positive
#         0   # Negative (0 for BCE loss)
#     ).long()
    
#     # Extract coordinates for dataloaders
#     coords_pn = [data_use["coords"][i] for i in pn_indices] if "coords" in data_use else None
    
#     # Phase 3: Train/Val Split on P+N
#     val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
#     total_pn = len(pn_indices)
#     val_count = int(total_pn * val_frac)
    
#     gen = torch.Generator()
#     gen.manual_seed(int(cfg.seed))
#     indices_pn = torch.randperm(total_pn, generator=gen)
    
#     val_indices_pn = indices_pn[:val_count].tolist()
#     train_indices_pn = indices_pn[val_count:].tolist()
    
#     # Create train/val data
#     Xtr = features_pn[train_indices_pn]
#     ytr = yA_pn[train_indices_pn]
#     Xval = features_pn[val_indices_pn]
#     yval = yA_pn[val_indices_pn]
    
#     # Phase 4: Train the classifier using the base training function
#     results = train_base_classifier(
#         Xtr=Xtr,
#         Xval=Xval,
#         ytr=ytr,
#         yval=yval,
#         common=common,
#         data_use=data_use,
#         inference_fn=inference_fn,
#         pos_crd_plot=pos_crd_plot
#     )
    
#     # Add PN-specific information to results
#     results.update({
#         'pos_crd_plot': pos_crd_plot,
#         'pn_indices': pn_indices,
#         'inf_indices': inf_indices,
#     })
    
#     return results



# def create_pos_drop_schedule(
#     all_pos_idx: List[int], 
#     drop_rate: float, 
#     seed: int,
#     min_training_size: int = 1
# ) -> List[List[int]]:
#     """
#     Create a rotation drop schedule where each positive sample is dropped exactly once across iterations.
    
#     Args:
#         all_pos_idx: List of all positive sample indices
#         drop_rate: Fraction of positives to drop per iteration (determines number of iterations)
#         seed: Random seed for reproducibility
#         min_training_size: Minimum number of training samples required per iteration
    
#     Returns:
#         List of lists, where each inner list contains the positive indices to USE (not drop) in that iteration
#     """
#     if not all_pos_idx:
#         return [[]]
    
#     # Calculate number of iterations based on drop rate
#     num_iterations = int(1.0 / drop_rate)
#     total_positives = len(all_pos_idx)
    
#     # Ensure minimum training size constraint
#     max_drop_per_iter = total_positives - min_training_size
#     if max_drop_per_iter <= 0:
#         # If we can't drop any samples while maintaining min_training_size, use all samples in all iterations
#         return [all_pos_idx.copy() for _ in range(num_iterations)]
    
#     # Shuffle positives for random distribution across iterations
#     rng = np.random.default_rng(seed)
#     shuffled_pos_idx = rng.permutation(all_pos_idx).tolist()
    
#     # Calculate how many samples to drop per iteration
#     samples_per_drop = total_positives // num_iterations
#     remaining_samples = total_positives % num_iterations
    
#     # Create drop schedule - each sample is dropped exactly once
#     drop_schedule = []
#     start_idx = 0
#     for i in range(num_iterations):
#         # Add one extra sample to first 'remaining_samples' iterations
#         extra = 1 if i < remaining_samples else 0
#         end_idx = start_idx + samples_per_drop + extra
#         samples_to_drop = shuffled_pos_idx[start_idx:end_idx]
#         drop_schedule.append(samples_to_drop)
#         start_idx = end_idx
    
#     # Convert drop schedule to "use schedule" (samples to keep for training)
#     use_schedule = []
#     for iteration_drops in drop_schedule:
#         samples_to_use = [idx for idx in all_pos_idx if idx not in iteration_drops]
        
#         # Ensure minimum training size
#         if len(samples_to_use) < min_training_size:
#             # If dropping would result in too few samples, use all samples
#             samples_to_use = all_pos_idx.copy()
        
#         use_schedule.append(samples_to_use)
    
#     return use_schedule

# def train_cls_1_PN_PosDrop(
#     *,
#     drop_rate: float = 0.1,
#     common: Dict[str, Any],  # Contains cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
#     data_use: Dict[str, Any],
#     filter_top_pct: float = 0.10,
#     negs_per_pos: int = 10,
#     action: Optional[Dict[str, Any]] = None,
#     inference_fn: Optional[callable] = None,
#     tag_main: Optional[str] = None,
#     use_individual_distances: bool = True,      # ✅ Memory-safe distance computation
#     meta_evaluation_metrics: Optional[set] = None,  # ✅ Configurable meta-evaluation metrics
# ) -> Dict[str, Any]:
#     """
#     Train a PN (Positive-Negative) classifier using negative selection from unlabeled data with positive dropout.
    
#     Args:
#         drop_rate: Fraction of positives to drop per iteration
#         common: Dictionary containing cfg, device, mlp_hidden_dims, mlp_dropout, debug_mode, run_logger
#         data_use: Dictionary containing 'features', 'labels', 'coords' keys
#         filter_top_pct: Percentage of top similar samples to filter during negative selection
#         negs_per_pos: Number of negative samples to select per positive sample
#         action: Optional dictionary for future extensibility
#         inference_fn: Optional inference function to run after training
#         tag_main: Main tag for caching and naming
#         use_individual_distances: Use individual distance files instead of full matrix (memory-safe)
#         meta_evaluation_metrics: Set of metric names to compute (default: PosDrop_Acc, Focus)
    
#     Returns:
#         Dictionary containing meta-evaluation results and training outcomes
#     """
    
#     # Extract common parameters
#     cfg = common['cfg']
#     device = common['device']
#     mlp_hidden_dims = common['mlp_hidden_dims']
#     mlp_dropout = common['mlp_dropout']
#     debug_mode = common.get('debug_mode', False)
#     run_logger = common.get('run_logger', None)
#     verbose = common.get('verbose', True)
    
#     # Use provided metrics or default
#     if meta_evaluation_metrics is None:
#         meta_evaluation_metrics = DEFAULT_META_EVALUATION
    
#     if run_logger is not None:
#         run_logger.log(f"[cls-1-PN-PosDrop] Meta_Evaluation metrics: {meta_evaluation_metrics}")
    
#     if data_use is None:
#         raise RuntimeError("Data unavailable for classifier training; aborting.")
    
#     # Extract features and labels
#     features = data_use["features"]
#     # ✅ CHECK 1: Check the raw data type and location
#     print(f"[Data Location Check] features type: {type(features)}")
#     features = torch.from_numpy(features).float().to(device) if isinstance(features, np.ndarray) else features.float().to(device)

#     # Convert to tensor if needed
#     if isinstance(features, np.ndarray):
#         features = torch.from_numpy(features).float()
#     elif isinstance(features, torch.Tensor):
#         features = features.float()
#     else:
#         features = torch.tensor(features, dtype=torch.float32)
#     features = features.to(device)
#     print(f"[Data Location Check] features is now a Torch tensor (on {features.device})")

#     labels_tensor = torch.from_numpy(data_use["labels"]) if isinstance(data_use["labels"], np.ndarray) else data_use["labels"]
#     yA_pu = torch.where(labels_tensor >= 0.9, 1, -1).long().to(device)
    
#     # Get coordinates for visualization (handle both 'coords' and 'coordinates' keys)
#     temp_crd = data_use.get("coords") or data_use.get("coordinates")
#     coords_array = np.array(temp_crd) if temp_crd is not None else None

#     # Prepare positive dropping: create drop schedule where each positive is dropped exactly once
#     all_pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
#     all_unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
    
#     num_iterations = int(1.0 / drop_rate)
    
#     # Create drop schedule with minimum training size of 1
#     pos_splits = create_pos_drop_schedule(
#         all_pos_idx=all_pos_idx,
#         drop_rate=drop_rate,
#         seed=int(cfg.seed),
#         min_training_size=1
#     )
    
#     if run_logger is not None and verbose:
#         run_logger.log(f"[cls-1-PN-PosDrop] Created positive drop schedule: {num_iterations} iterations, {len(all_pos_idx)} total positives")
#         for i, pos_list in enumerate(pos_splits):
#             run_logger.log(f"[cls-1-PN-PosDrop] Iteration {i+1}: using {len(pos_list)} positives (dropping {len(all_pos_idx) - len(pos_list)})")
    
#     # ✅ MEMORY ESTIMATION: Calculate memory requirements for optimization choice
#     n_unknowns = len(all_unl_idx)
#     n_positives = len(all_pos_idx)
#     estimated_memory_gb = (n_unknowns * n_positives * 4) / (1024**3)  # float32 estimate
    
#     if run_logger is not None:
#         run_logger.log(f"[cls-1-PN-PosDrop] Dataset size: {n_unknowns} unknowns, {n_positives} positives")
#         run_logger.log(f"[cls-1-PN-PosDrop] Estimated memory for full matrix: {estimated_memory_gb:.2f} GB")
    
#     # ✅ OPTIMIZATION STRATEGY: Choose method based on user preference and memory requirements
#     individual_distances_dir = None
    
#     if use_individual_distances:
#         # ✅ Use individual distance files (memory-safe approach)
#         if run_logger is not None and verbose:
#             run_logger.log(f"[cls-1-PN-PosDrop] Using individual distance files (memory-safe approach)")
        
#         try:
#             rotation_tag = f"{tag_main}_posdrop_individual" if tag_main else None
#             individual_distances_dir = precompute_distances_per_positive(
#                 Z_all=features,
#                 all_pos_idx=all_pos_idx,
#                 unk_idx=all_unl_idx,
#                 tag=rotation_tag,
#                 use_float16=True  # Use float16 for additional memory savings
#             )
            
#             if run_logger is not None and verbose:
#                 run_logger.log(f"[cls-1-PN-PosDrop] Pre-computed individual distance files in: {individual_distances_dir}")
#                 run_logger.log(f"[cls-1-PN-PosDrop] This will enable memory-safe negative selection across {num_iterations} iterations")
#         except Exception as e:
#             if run_logger is not None:
#                 run_logger.log(f"[cls-1-PN-PosDrop] WARNING: Failed to pre-compute individual distances ({e}), falling back to standard method")
#             individual_distances_dir = None
#             use_individual_distances = False
#     else:
#         # Standard method - no optimization
#         if run_logger is not None:
#             run_logger.log(f"[cls-1-PN-PosDrop] Using standard negative selection method (no pre-computation)")
    
#     # ✅ Training iterations with optimized negative selection
#     all_results = []
#     for iteration in range(num_iterations):
#         if run_logger is not None and verbose:
#             run_logger.log(f"[cls-1-PN-PosDrop] Starting training for iteration {iteration+1}/{num_iterations}...")
        
#         tag = f"{tag_main}/all{num_iterations}_iter{iteration+1}" if tag_main is not None else None

#         # Use only the positives assigned to this iteration
#         pos_idx = pos_splits[iteration]
#         unl_idx = all_unl_idx.copy()
        
#         # Create coordinate plot for this iteration's positives
#         pos_crd_plot = None
#         if coords_array is not None:
#             pos_crd_plot = [coords_array[i] for i in pos_idx]

#         pos_idx_arr = np.asarray(pos_idx, dtype=int)
#         unk_idx_arr = np.asarray(unl_idx, dtype=int)
        
#         if debug_mode and run_logger is not None and verbose:
#             run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")
        
#         # ✅ OPTIMIZED NEGATIVE SELECTION: Choose method based on pre-computation
#         if use_individual_distances and individual_distances_dir is not None:
#             # Use individual distance files approach
#             neg_idx_region = pu_select_negatives_from_individual_distances(
#                 individual_distances_dir=individual_distances_dir,
#                 all_pos_idx=all_pos_idx,
#                 unk_idx=unk_idx_arr,
#                 active_pos_idx=pos_idx_arr,
#                 filter_top_pct=filter_top_pct,
#                 negatives_per_pos=negs_per_pos
#             )
#         else:
#             # Fallback to standard method
#             neg_idx_region = pu_select_negatives(
#                 Z_all=features,
#                 pos_idx=pos_idx_arr,
#                 unk_idx=unk_idx_arr,
#                 filter_top_pct=filter_top_pct,
#                 negatives_per_pos=negs_per_pos,
#                 tag=tag
#             )
        
#         if debug_mode and run_logger is not None and verbose:
#             run_logger.log(f"[cls-1-PN] Finished negative selection: selected {len(neg_idx_region)} negatives")
        
#         neg_idx_arr = np.asarray(neg_idx_region, dtype=int)
        
#         # Phase 2: Create Balanced P+N Dataset
#         pn_indices = np.concatenate([pos_idx_arr, neg_idx_arr], axis=0)
        
#         # Extract features and labels for P+N
#         features_pn = features[pn_indices]
#         yA_pn = torch.where(
#             torch.tensor([i in pos_idx for i in pn_indices], dtype=torch.bool),
#             1,  # Positive
#             0   # Negative (0 for BCE loss)
#         ).long()
        
#         # Phase 3: Train/Val Split on P+N
#         val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
#         total_pn = len(pn_indices)
#         val_count = int(total_pn * val_frac)
        
#         gen = torch.Generator()
#         gen.manual_seed(int(cfg.seed))
#         indices_pn = torch.randperm(total_pn, generator=gen)
        
#         val_indices_pn = indices_pn[:val_count].tolist()
#         train_indices_pn = indices_pn[val_count:].tolist()
        
#         # Create train/val data
#         Xtr = features_pn[train_indices_pn]
#         ytr = yA_pn[train_indices_pn]
#         Xval = features_pn[val_indices_pn]
#         yval = yA_pn[val_indices_pn]
        
#         # Phase 4: Train the classifier
#         iteration_results = train_base_classifier(
#             Xtr=Xtr,
#             Xval=Xval,
#             ytr=ytr,
#             yval=yval,
#             common=common,
#             data_use=data_use,
#             inference_fn=inference_fn,
#             pos_crd_plot=pos_crd_plot,
#             tag=tag
#         )

#         # Include iteration-specific metadata
#         iteration_results['pos_indices_this_iter'] = pos_idx
#         all_results.append(iteration_results)
    
#     # ✅ Final training with ALL positives (no dropping)
#     if run_logger is not None and verbose:
#         run_logger.log(f"[cls-1-PN-PosDrop] Starting final 'no positive drop' training with all {len(all_pos_idx)} positives")
    
#     tag_all = f"{tag_main}" if tag_main is not None else None
    
#     # Use ALL positives
#     pos_idx = all_pos_idx
#     pos_crd_plot_all = None
#     if coords_array is not None:
#         pos_crd_plot_all = [coords_array[i] for i in pos_idx]

#     pos_idx_arr = np.asarray(pos_idx, dtype=int)
#     unk_idx_arr = np.asarray(all_unl_idx, dtype=int)
    
#     # ✅ Final negative selection with same optimization
#     if use_individual_distances and individual_distances_dir is not None:
#         neg_idx_region_all = pu_select_negatives_from_individual_distances(
#             individual_distances_dir=individual_distances_dir,
#             all_pos_idx=all_pos_idx,
#             unk_idx=unk_idx_arr,
#             active_pos_idx=pos_idx_arr,  # All positives are active
#             filter_top_pct=filter_top_pct,
#             negatives_per_pos=negs_per_pos
#         )
#     else:
#         neg_idx_region_all = pu_select_negatives(
#             Z_all=features.cpu().numpy(),
#             pos_idx=pos_idx_arr,
#             unk_idx=unk_idx_arr,
#             filter_top_pct=filter_top_pct,
#             negatives_per_pos=negs_per_pos,
#             tag=tag_all
#         )
    
#     neg_idx_arr_all = np.asarray(neg_idx_region_all, dtype=int)
    
#     # Create final balanced dataset and train
#     pn_indices_all = np.concatenate([pos_idx_arr, neg_idx_arr_all], axis=0)
#     features_pn_all = features[pn_indices_all]
#     yA_pn_all = torch.where(
#         torch.tensor([i in pos_idx for i in pn_indices_all], dtype=torch.bool),
#         1, 0
#     ).long()
    
#     # Train/Val split for final run
#     val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))
#     total_pn_all = len(pn_indices_all)
#     val_count_all = int(total_pn_all * val_frac)
    
#     gen_all = torch.Generator()
#     gen_all.manual_seed(int(cfg.seed))
#     indices_pn_all = torch.randperm(total_pn_all, generator=gen_all)
    
#     val_indices_pn_all = indices_pn_all[:val_count_all].tolist()
#     train_indices_pn_all = indices_pn_all[val_count_all:].tolist()
    
#     Xtr_all = features_pn_all[train_indices_pn_all]
#     ytr_all = yA_pn_all[train_indices_pn_all]
#     Xval_all = features_pn_all[val_indices_pn_all]
#     yval_all = yA_pn_all[val_indices_pn_all]
    
#     # Final training
#     all_results_final = train_base_classifier(
#         Xtr=Xtr_all,
#         Xval=Xval_all,
#         ytr=ytr_all,
#         yval=yval_all,
#         common=common,
#         data_use=data_use,
#         inference_fn=inference_fn,
#         pos_crd_plot=pos_crd_plot_all,
#         tag=tag_all
#     )
    
#     all_results_final['pos_indices_this_iter'] = pos_idx
#     all_results.append(all_results_final)
    
#     if run_logger is not None and verbose:
#         run_logger.log(f"[cls-1-PN-PosDrop] Completed final 'no positive drop' training")
    
#     # ✅ Meta-evaluation calculation
#     meta_evaluation = {}
    
#     for metric in meta_evaluation_metrics:
#         metric_scores = []

#         for iteration in range(num_iterations):
#             iter_result = all_results[iteration]
#             pos_indices_this_iter = set(iter_result['pos_indices_this_iter'])

#             if metric == "PosDrop_Acc":
#                 # Average prediction accuracy on positives dropped in each iteration
#                 dropped_pos_idx = [idx for idx in all_pos_idx if idx not in pos_indices_this_iter]
                
#                 if len(dropped_pos_idx) > 0:
#                     predictions_mean = iter_result['inference_result']['predictions_mean']
                    
#                     if isinstance(predictions_mean, torch.Tensor):
#                         predictions_mean = predictions_mean.cpu().numpy()
                    
#                     dropped_pos_probs = predictions_mean[dropped_pos_idx]
#                     accuracy = float(np.mean(dropped_pos_probs))
#                     metric_scores.append(accuracy)
#                 else:
#                     metric_scores.append(0.0)
                        
#             elif metric == "Focus":
#                 predictions_mean = iter_result['inference_result']['predictions_mean']
                
#                 if isinstance(predictions_mean, torch.Tensor):
#                     predictions_mean = predictions_mean.cpu().numpy()
                
#                 predictions_sum = predictions_mean.sum()
#                 total_predictions = len(predictions_mean)
#                 focus_score = 1.0 - (predictions_sum / total_predictions)
#                 metric_scores.append(float(focus_score))
        
#         meta_evaluation[metric] = {
#             'scores': metric_scores,
#             'mean': np.mean(metric_scores) if metric_scores else 0.0,
#             'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
#         }

#         if run_logger is not None:
#             run_logger.log(f"[cls-1-PN-PosDrop] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
#     return meta_evaluation


# def load_and_evaluate_existing_predictions(
#     *,
#     tag_main: str,
#     common: Dict[str, Any],
#     data_use: Dict[str, Any],
#     run_logger: Optional[Any] = None,
#     meta_evaluation_metrics: Optional[set] = None
# ) -> Optional[Dict[str, Any]]:
#     """
#     Load existing predictions from disk and compute Meta_Evaluation metrics.
    
#     Auto-detects iteration directories, infers drop_rate from number of iterations,
#     and computes requested meta-evaluation metrics from cached predictions.
    
#     Args:
#         tag_main: Tag identifier for the experiment (e.g., "Unifying_OverlapOnly")
#         common: Dictionary containing cfg, device, run_logger, etc.
#         data_use: Dictionary containing 'features', 'labels', 'coords' keys
#         run_logger: Optional logger for status messages
#         meta_evaluation_metrics: Set of metric names to compute (default: PosDrop_Acc, Focus)
    
#     Returns:
#         Dictionary containing meta_evaluation metrics, or None if predictions not found
#     """
#     cfg = common['cfg']
#     base_output_dir = Path(cfg.output_dir) / "cls_1_inference_results" / "All" / tag_main
    
#     # Use provided metrics or default
#     if meta_evaluation_metrics is None:
#         meta_evaluation_metrics = DEFAULT_META_EVALUATION
    
#     if run_logger:
#         run_logger.log(f"[load_cached_predictions] Meta_Evaluation metrics: {meta_evaluation_metrics}")
    
#     if not base_output_dir.exists():
#         if run_logger:
#             run_logger.log(f"[load_cached_predictions] Directory not found: {base_output_dir}")
#         return None
    
#     # Auto-detect iteration directories using glob pattern: all*_iter*
#     import glob
#     iter_pattern = str(base_output_dir / "all*_iter*")
#     iter_dirs = sorted(glob.glob(iter_pattern))
    
#     if not iter_dirs:
#         if run_logger:
#             run_logger.log(f"[load_cached_predictions] No iteration directories found matching pattern: {iter_pattern}")
#         return None
    
#     num_iterations = len(iter_dirs)
#     if run_logger:
#         run_logger.log(f"[load_cached_predictions] Found {num_iterations} iteration directories")
    
#     # Infer drop_rate from number of iterations
#     drop_rate = 1.0 / num_iterations
#     if run_logger:
#         run_logger.log(f"[load_cached_predictions] Inferred drop_rate: {drop_rate:.4f}")
    
#     # Extract positive indices
#     labels = data_use['labels']
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu().numpy()
#     all_pos_idx = np.where(labels == 1)[0].tolist()
    
#     if not all_pos_idx:
#         if run_logger:
#             run_logger.log("[load_cached_predictions] ERROR: No positive samples found in data_use")
#         return None
    
#     # Reconstruct drop schedule
#     drop_schedule = create_pos_drop_schedule(
#         all_pos_idx=all_pos_idx,
#         drop_rate=drop_rate,
#         seed=int(cfg.seed),
#         min_training_size=1
#     )
    
#     # Load predictions for each iteration
#     all_predictions_mean = []
#     all_predictions_std = []
#     all_pos_indices_per_iter = []
    
#     for iter_idx, iter_dir_path in enumerate(iter_dirs):
#         iter_dir = Path(iter_dir_path)
#         pred_mean_path = iter_dir / "predictions_mean.npy"
#         pred_std_path = iter_dir / "predictions_std.npy"
        
#         # Check if both files exist
#         if not pred_mean_path.exists():
#             if run_logger:
#                 run_logger.log(f"[load_cached_predictions] Missing predictions_mean.npy in {iter_dir}")
#             return None
        
#         if not pred_std_path.exists():
#             if run_logger:
#                 run_logger.log(f"[load_cached_predictions] Missing predictions_std.npy in {iter_dir}")
#             return None
        
#         # Load predictions
#         pred_mean = np.load(pred_mean_path)
#         pred_std = np.load(pred_std_path)
        
#         all_predictions_mean.append(pred_mean)
#         all_predictions_std.append(pred_std)
#         all_pos_indices_per_iter.append(drop_schedule[iter_idx])
        
#         if run_logger:
#             run_logger.log(f"[load_cached_predictions] Loaded iteration {iter_idx}: mean shape={pred_mean.shape}, std shape={pred_std.shape}")
    
#     # Compute Meta_Evaluation metrics
#     meta_evaluation = {}
    
#     for metric in meta_evaluation_metrics:
#         metric_scores = []
        
#         for iteration in range(num_iterations):
#             pos_indices_this_iter = set(all_pos_indices_per_iter[iteration])
#             predictions_mean = all_predictions_mean[iteration]
            
#             if metric == "PosDrop_Acc":
#                 # Average prediction accuracy on positives dropped in each iteration
#                 dropped_pos_idx = [idx for idx in all_pos_idx if idx not in pos_indices_this_iter]
                
#                 if len(dropped_pos_idx) > 0:
#                     dropped_pos_probs = predictions_mean[dropped_pos_idx]
#                     accuracy = float(np.mean(dropped_pos_probs))
#                     metric_scores.append(accuracy)
#                 else:
#                     metric_scores.append(0.0)
            
#             elif metric == "Focus":
#                 predictions_sum = predictions_mean.sum()
#                 total_predictions = len(predictions_mean)
#                 focus_score = 1.0 - (predictions_sum / total_predictions)
#                 metric_scores.append(float(focus_score))
        
#         meta_evaluation[metric] = {
#             'scores': metric_scores,
#             'mean': np.mean(metric_scores) if metric_scores else 0.0,
#             'std': np.std(metric_scores) if len(metric_scores) > 1 else 0.0
#         }
        
#         if run_logger:
#             run_logger.log(f"[load_cached_predictions] {metric}: mean={meta_evaluation[metric]['mean']:.4f}, std={meta_evaluation[metric]['std']:.4f}")
    
#     return meta_evaluation


# def save_meta_evaluation_results(
#     *,
#     meta_evaluation: Dict[str, Any],
#     tag_main: str,
#     common: Dict[str, Any],
#     run_logger: Optional[Any] = None
# ) -> None:
#     """
#     Save Meta_Evaluation results to JSON file.
    
#     Args:
#         meta_evaluation: Dictionary containing Meta_Evaluation metrics
#         tag_main: Tag identifier for the experiment
#         common: Dictionary containing cfg
#         run_logger: Optional logger for status messages
#     """
#     cfg = common['cfg']
#     output_dir = Path(cfg.output_dir) / "meta_evaluation_results"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     output_path = output_dir / f"{tag_main}_meta_evaluation.json"
    
#     # Convert numpy types to native Python types for JSON serialization
#     serializable_meta = {}
#     for metric_name, metric_data in meta_evaluation.items():
#         serializable_meta[metric_name] = {
#             'scores': [float(s) for s in metric_data['scores']],
#             'mean': float(metric_data['mean']),
#             'std': float(metric_data['std'])
#         }
    
#     with open(output_path, 'w') as f:
#         json.dump(serializable_meta, f, indent=2)
    
#     if run_logger:
#         run_logger.log(f"[save_meta_evaluation] Saved Meta_Evaluation results to {output_path}")
