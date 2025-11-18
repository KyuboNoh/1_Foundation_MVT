# Note: The terminology of "anchor" and "target" datasets is used throughout this module to refer to the two datasets. 
#       (Dataset 1 is the anchor and 2 is the target. No semantic meaning beyond that.) by K.N. 30Oct2025
#       [CRITICAL] Currently, the code assumes that target dataset falls within the anchor dataset spatially for cls-2 training. by K.N. 30Oct2025

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np
import random
from affine import Affine
import torch, torch.nn as nn, torch.nn.functional as F

from .config import AlignmentConfig, load_config

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = Dataset = None  # type: ignore[assignment]

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:  # optional plotting for debug mode
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency
    CRS = Transformer = None  # type: ignore[assignment]
from Common.Unifying.Labels_TwoDatasets.datasets import (
    auto_coord_error,
    _load_pn_lookup,
    _count_pn_lookup, 
    _MaskStack,
    load_embedding_records,
    summarise_records,
)
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from Common.Unifying.Labels_TwoDatasets import (
    _build_aligned_pairs_OneToOne,
    _build_aligned_pairs_SetToSet,
    _apply_projector_based_PUNlabels,
    _subset_classifier_sample,
    _normalise_cross_matches,
    _prepare_classifier_labels,
    _normalise_coord,
    _normalise_row_col,
    _apply_projector_based_PNlabels,
)

from Common.Unifying.Labels_TwoDatasets.fusion_utils import (
    align_overlap_embeddings_for_pn_one_to_one as _align_overlap_embeddings_for_pn_OneToOne,
    prepare_fusion_overlap_dataset_one_to_one as _prepare_fusion_overlap_dataset_OneToOne,
    prepare_fusion_overlap_dataset_for_inference as _prepare_fusion_overlap_dataset_for_inference,
    fusion_export_results as _fusion_export_results,
)

from Common.Unifying.Labels_TwoDatasets.fusion_utils.workspace import (
    DatasetBundle,
    OverlapAlignmentLabels,
    OverlapAlignmentPair,
    OverlapAlignmentWorkspace,
)
from Common.Unifying.Labels_TwoDatasets.splits import _overlap_split_indices
from Common.Augmentation import (
    _load_augmented_embeddings,
    _print_augmentation_usage,
)
from Common.Unifying.DCCA import (
    _format_optional_scalar,
    _project_in_batches,
    _load_pretrained_dcca_state,
    _projection_head_from_state,
    _load_DCCA_projectors,
    _resolve_dcca_weights_path,
    _prepare_output_dir,
    _persist_state,
    _persist_metrics,
    _maybe_save_debug_figures,
    _create_debug_alignment_figures,
    dcca_loss,
    reembedding_DCCA,
    _train_DCCA,
    ProjectionHead,
    _compute_dcca_checkpoint_hash,
    _save_dcca_projections,
    _load_dcca_projections,
    resolve_dcca_embeddings_and_projectors,
)

# Import transformer-aggregated DCCA components from the new module
from Common.Unifying.DCCA.method1_phi_TransAgg import (
    PNHeadUnified,
    CrossAttentionAggregator, 
    AggregatorTargetHead,
    build_phi,
    cosine_sim,
    fit_unified_head_OVERLAP_from_uv,
    _train_transformer_aggregator,
    run_overlap_inference_gAB_from_pairs,
    build_cls2_dataset_from_dcca_pairs,
    TargetSetDataset,
    PUBatchSampler,
    _collate_target_sets,
)

from Common.cls.infer.infer_maps import (group_coords, mc_predict_map_from_embeddings, write_prediction_outputs,)
from Common.cls.sampling.likely_negatives import pu_select_negatives
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.training.train_cls import (
    dataloader_metric_inputORembedding,
    eval_classifier,
    train_classifier,
    _collect_outputs,
    _compute_metrics,
)
from Common.cls.miscellaneous import _prepare_classifier_dir
from Common.Unifying.Labels_TwoDatasets.overlaps import _load_overlap_mask_data
from Common.DataReader import get_original_labels_from_cfg, get_original_embeddings_from_cfg, extract_positive_samples_from_original, load_anchor_labels_for_substitution, get_original_dataset_for_training
from Common.overlap_utils import extract_overlap_only_using_masks
from Common.cls.training import train_cls_PN_base

from . inference_module import run_inference_base

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    # parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    # parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    # parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    # parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    # parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of aligned pairs reserved for validation evaluation (set to 0 to disable).", )
    
    parser.add_argument("--train-dcca", action=argparse.BooleanOptionalAction, default=False, help="Train the DCCA projection heads (default: true). Use --no-train-dcca to disable.",)
    parser.add_argument("--read-dcca",  action=argparse.BooleanOptionalAction, default=True, help="Load existing DCCA projection head weights before training (default: false).",)
    parser.add_argument("--dcca-weights-path", type=str, default=None, help="Optional path to a saved DCCA checkpoint used when --read-dcca is enabled.",)
    parser.add_argument("--use-transformer-aggregator", action=argparse.BooleanOptionalAction, default=True, help="Enable transformer-based aggregation before DCCA (set-to-set pairing only).",)
    parser.add_argument("--agg-trans-num-layers", type=int, default=4, help="Number of cross-attention layers in the aggregator.")
    parser.add_argument("--agg-trans-num-heads", type=int, default=4, help="Number of attention heads in the aggregator.")
    parser.add_argument("--agg-trans-dropout", type=float, default=0.1, help="Dropout used inside the transformer aggregator.")
    parser.add_argument("--agg-trans-pos-enc", action=argparse.BooleanOptionalAction, default=False, help="Use positional encoding based on anchor/target coordinate differences.",)

    # parser.add_argument("--train-cls-1", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Non-Overlapping part.",)
    # parser.add_argument("--train-cls-1-Method", choices=["PN", "PU"], default="PU", help="Classifier training method (default: PU). ")
    parser.add_argument('--filter-top-pct', type=float, default=0.10)
    parser.add_argument('--negs-per-pos', type=int, default=10)

    # parser.add_argument("--train-cls-2", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Overlapping part.",)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128], help="Hidden layer sizes for classifier MLP heads (space-separated).",)
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout probability applied to classifier MLP layers.", )
    parser.add_argument("--mlp-dropout-passes", type=int, default=5, help="Number of Monte Carlo dropout passes for uncertainty estimation in classifier inference.", )

    parser.add_argument("--read-inference", action=argparse.BooleanOptionalAction, default=False, help="Read inference on aligned datasets after training (default: false).",)
    
    parser.add_argument("--skip-training-if-cached", action="store_true", help="Skip training if cached predictions are found and compute Meta_Evaluation from cached results.")
    
    # Legacy drop-rate parameter for backward compatibility
    parser.add_argument("--drop-rate", type=float, default=None, help="Legacy parameter - use --meta-evaluation-n-clusters instead")
    
    # Multi-clustering arguments
    parser.add_argument("--clustering-methods", type=str, nargs="+", default=["random"], 
                       choices=["random", "kmeans", "hierarchical"],
                       help="Clustering methods for positive dropping (space-separated). Available: random, kmeans, hierarchical")
    
    parser.add_argument("--meta-evaluation-n-clusters", type=int, nargs="+", default=[10],
                       help="Number of clusters/iterations for meta-evaluation (space-separated). Line search over these values.")
    
    parser.add_argument("--hierarchical-linkage", type=str, default="ward",
                       choices=["ward", "complete", "average", "single"],
                       help="Linkage criterion for hierarchical clustering")

    parser.add_argument("--meta-evaluation", type=str, nargs="+", default=["PosDrop_Acc", "Focus"], help="Meta-evaluation metrics to compute (space-separated). Available: PosDrop_Acc, Focus")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

def create_clean_tag(
    classifier_stage: str = "cls1",
    encoder_type: str = "dcca", 
    data_type: str = "overlap",
) -> str:
    """Create clean, readable tags for caching and results."""
    return f"{classifier_stage}_{encoder_type}_{data_type}"

def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    run_logger = _RunLogger(cfg)

    if torch is None or nn is None or DataLoader is None:
        raise ImportError("Overlap alignment training requires PyTorch; install torch before running the trainer.")

    cfg_hidden_dims = getattr(cfg.cls_training, "mlp_hidden_dims", tuple()) if cfg.cls_training is not None else tuple()
    if not cfg_hidden_dims:
        cfg_hidden_dims = getattr(cfg.dcca_training, "mlp_hidden_dims", tuple())
    if cfg_hidden_dims:
        mlp_hidden_dims = tuple(int(h) for h in cfg_hidden_dims if int(h) > 0)
    else:
        mlp_hidden_dims = tuple(int(h) for h in args.mlp_hidden_dims if int(h) > 0)
    if not mlp_hidden_dims:
        raise ValueError("At least one positive hidden dimension must be provided for classifier MLP heads.")
    mlp_dropout = float(args.mlp_dropout)
    if not (0.0 <= mlp_dropout < 1.0):
        raise ValueError("Classifier MLP dropout must be in the range [0.0, 1.0).")
    mlp_dropout_passes = int(args.mlp_dropout_passes)
    if not (1 <= mlp_dropout_passes):
        raise ValueError("Classifier MLP dropout passes must be larger than 1.")

    read_dcca = bool(args.read_dcca)
    if read_dcca:
        args.train_dcca = False
    train_dcca = bool(args.train_dcca)
    if not train_dcca and not read_dcca:
        raise ValueError("Cannot disable DCCA training without enabling --read-dcca to load existing weights.")

    weights_path: Optional[Path] = None
    pretrained_state: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None
    pretrained_summary: Optional[Dict[str, object]] = None
    if read_dcca:
        weights_path = _resolve_dcca_weights_path(cfg, args.dcca_weights_path)
        if weights_path is None or not weights_path.exists():
            hint = args.dcca_weights_path or "default locations"
            raise FileNotFoundError(f"Unable to locate DCCA weights to load (checked {hint}).")
        pretrained_state, pretrained_summary = _load_pretrained_dcca_state(weights_path)

        if weights_path is not None:
            run_logger.log(f"Reading DCCA weights from {weights_path}")
    else:
            run_logger.log(
                "Starting stage-1 overlap alignment with initial projection_dim={proj}; "
                "mlp_layers={layers}, train_dcca={train_flag}, read_dcca={read_flag}".format(
                    proj=cfg.projection_dim,
                    layers=cfg.dcca_training.mlp_hidden_dims,
                    train_flag=train_dcca,
                    read_flag=read_dcca,
                )
            )

    cfg.aggregator = "weighted_pool"
    cfg.use_positive_only = False
    cfg.use_positive_augmentation = False

                # Prepare aggregator config if needed
    aggregator_config = {
        'projection_dim': cfg.projection_dim,
        'num_layers': args.agg_trans_num_layers,
        'num_heads': args.agg_trans_num_heads,
        'dropout': float(getattr(cfg.dcca_training, "agg_dropout", 0.1)),
        'use_positional_encoding': bool(args.agg_trans_pos_enc),
    }

    debug_mode = bool(args.debug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle legacy drop_rate parameter and new multi-clustering parameters
    if args.drop_rate is not None:
        run_logger.log(f"[main] Using legacy drop_rate parameter: {args.drop_rate}")
        drop_rate = args.drop_rate
        # Convert to new parameter format for backward compatibility
        if args.clustering_methods == ["random"] and args.meta_evaluation_n_clusters == [10]:
            # Only override if using defaults
            args.meta_evaluation_n_clusters = [int(1 / args.drop_rate)]
            run_logger.log(f"[main] Converted drop_rate={args.drop_rate} to meta_evaluation_n_clusters={args.meta_evaluation_n_clusters}")
    else:
        # Use first cluster count for legacy compatibility
        if args.meta_evaluation_n_clusters:
            drop_rate = 1.0 / args.meta_evaluation_n_clusters[0]
        else:
            drop_rate = 0.1  # Default fallback
    
    # Log execution mode for debugging
    is_multi_clustering = (
        len(args.clustering_methods) > 1 or 
        len(args.meta_evaluation_n_clusters) > 1 or
        args.clustering_methods != ["random"]
    )
    
    if is_multi_clustering:
        run_logger.log(f"[main] MULTI-CLUSTERING mode detected:")
        run_logger.log(f"[main] - Methods: {args.clustering_methods}")
        run_logger.log(f"[main] - Cluster counts: {args.meta_evaluation_n_clusters}")
        run_logger.log(f"[main] - Hierarchical linkage: {args.hierarchical_linkage}")
    else:
        drop_rate = 1.0 / args.meta_evaluation_n_clusters[0] if args.meta_evaluation_n_clusters else 0.1
        run_logger.log(f"[main] LEGACY mode detected with drop_rate={drop_rate}")
    
    # Set global random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    run_logger.log(f"[main] Set global random seed to {seed}")
    
    common = {
        'cfg': cfg,
        'device': device,
        'mlp_hidden_dims': mlp_hidden_dims,
        'mlp_dropout': mlp_dropout,
        'debug_mode': debug_mode,
        'run_logger': run_logger,
        'seed': seed,
    }


    ####################################### Get DCCA embeddings and overlap info  #######################################
    anchor_name, target_name = _dataset_pair(cfg)
    dcca_sets, overlap_info = get_dcca_embeddings_with_overlaps(cfg, args, device, run_logger)
    # unified_overlap_data = Unifying_METHOD1_PHI_OverlapOnly(overlap_info, device, run_logger, batch_size=1000)
    # overlap_info_pair_metadata_only = get_overlap_info_pair_metadata_only(cfg, args, device, run_logger)


    ####################################### Train Classifier 1 on target_data (Original)  #######################################
    method_name = "base"
    data_name = target_name
    encoder_name = "MAEViT"
    data_use = get_original_dataset_for_training(cfg=cfg, dataset_name=data_name, run_logger=run_logger)

    # Substitute labels
    label_name = anchor_name
    label_use = get_original_dataset_for_training(cfg=cfg, dataset_name=label_name, run_logger=run_logger)
    data_use = _substitute_label_overlap_constrained(data=data_use,  label_source=label_use, overlap_info=overlap_info, run_logger=run_logger)
    
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, data_use, action, common, args, cfg, run_logger)

    exit()

    ####################################### Train on target [z_b | u_b] #######################################
    method_name = "Method2_concat"
    data_name = target_name
    encoder_name = "MAEViT_plus_TransDCCA"
    # Get original embeddings (z_b)
    data_original = get_original_dataset_for_training(cfg=cfg, dataset_name=data_name, run_logger=run_logger)
    # Get DCCA embeddings (u_b)
    data_dcca = dcca_sets[data_name]
    # Combine them
    data_use = Unifying_METHOD2_concat_two_embeddings(original_data=data_original, dcca_data=data_dcca, overlap_info=overlap_info, device=device, run_logger=run_logger, mode="concat")

    # Substitute labels
    label_name = anchor_name
    label_use = get_original_dataset_for_training(cfg=cfg, dataset_name=label_name, run_logger=run_logger)
    data_use = _substitute_label_overlap_constrained(data=data_use,  label_source=label_use, overlap_info=overlap_info, run_logger=run_logger)

    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, data_use, action, common, args, cfg, run_logger)

    exit()


    ####################################### Train Classifier 1 on anchor_name (Original)  #######################################
    method_name = "base"
    data_name = anchor_name 
    encoder_name = "MAEViT"
    data_use = get_original_dataset_for_training(cfg=cfg, dataset_name=anchor_name, run_logger=run_logger)
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)

    ####################################### Train on anchor [z_a | u_a] #######################################
    method_name = "Method2_concat"
    data_name = anchor_name
    encoder_name = "MAEViT_plus_TransDCCA"
    # Get original embeddings (z_b)
    data_original = get_original_dataset_for_training(cfg=cfg, dataset_name=data_name, run_logger=run_logger)
    data_original = extract_overlap_only_using_masks(data=data_original, overlap_mask=overlap_info["mask"], run_logger=run_logger)
    # Get DCCA embeddings (u_b)
    data_dcca = dcca_sets[data_name]
    data_dcca = extract_overlap_only_using_masks(data=data_dcca, overlap_mask=overlap_info["mask"], run_logger=run_logger)
    # Combine them
    data_use = Unifying_METHOD2_concat_two_embeddings(original_data=data_original, dcca_data=data_dcca,
                                                      overlap_info=overlap_info, device=device, 
                                                      run_logger=run_logger, mode="concat")
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)

    ####################################### Train Classifier 1 on anchor_name (Original; extracted)  #######################################
    method_name = "base"
    data_name = anchor_name + "_only_overlap"
    encoder_name = "TransDCCA_MAEViT"
    data_use = dcca_sets[anchor_name]  # Already processed DCCA embeddings
    data_use = extract_overlap_only_using_masks(data=data_use, overlap_mask=overlap_info["mask"], run_logger=run_logger)
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)

    ####################################### Train Classifier 1 on target_data (DCCA)  #######################################
    method_name = "base"
    data_name = target_name
    label_name = anchor_name
    encoder_name = "TransDCCA_MAEViT"
    data_use = dcca_sets[data_name]  # Already processed DCCA embeddings
    data_use = _substitute_label_overlap_constrained(data=data_use,  label_source=dcca_sets[label_name], overlap_info=overlap_info, run_logger=run_logger)
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)

    ####################################### Train Classifier 1 on anchor_name (Original; extracted)  #######################################
    method_name = "base"
    data_name = anchor_name + "_only_overlap"
    encoder_name = "MAEViT"
    data_use = get_original_dataset_for_training(cfg=cfg, dataset_name=anchor_name, run_logger=run_logger)
    data_use = extract_overlap_only_using_masks(data=data_use, overlap_mask=overlap_info["mask"], run_logger=run_logger)
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)

    ####################################### Train Classifier 1 on target_data (DCCA)  #######################################
    method_name = "TransDCCA_PN_Unified"
    data_name = target_name
    encoder_name = "TransDCCA_MAEViT"
    data_use = unified_overlap_data
    action = None

    tag = create_clean_tag(method_name, encoder_name, data_name)
    train_and_save_result(tag, drop_rate, data_use, action, common, args, cfg, run_logger)


    return



#####################################################################################   #####################################################################################   #####################################################################################   

def Unifying_METHOD1_PHI_OverlapOnly(
    overlap_info: Dict[str, object],
    device: torch.device,
    run_logger: "_RunLogger",
    batch_size: int = 1000,
) -> Dict[str, object]:
    """
    MEMORY-EFFICIENT OVERLAP-ONLY: Create unified phi dataset from overlap pairs only.
    Processes ~10K overlap samples instead of 1.4M full dataset samples.
    
    Args:
        overlap_info: Dictionary containing anchor_vecs, target_stack_per_anchor, 
                     pair_metadata, projector_a, projector_b
    """
    run_logger.log("[phi-unify-overlap] Creating unified dataset from overlap pairs only")
    
    # Extract overlap components
    anchor_vecs = overlap_info['anchor_vecs']
    target_stack_per_anchor = overlap_info['target_stack_per_anchor']
    pair_metadata = overlap_info['pair_metadata']
    projector_a = overlap_info['projector_a']
    projector_b = overlap_info['projector_b']
    
    n_pairs = len(anchor_vecs)
    
    if n_pairs == 0:
        raise ValueError("No overlap pairs found for unified processing")
    

    # Process overlap pairs in batches to manage memory
    phi_batches = []
    coordinates = []
    indices = []
    labels = []  # Will be set based on pair metadata or default to positive
    
    for i in range(0, n_pairs, batch_size):
        batch_end = min(i + batch_size, n_pairs)
        batch_anchor_vecs = anchor_vecs[i:batch_end]
        batch_target_stacks = target_stack_per_anchor[i:batch_end]
        batch_metadata = pair_metadata[i:batch_end] if pair_metadata else []
        
        with torch.no_grad():
            # Use build_cls2_dataset_from_dcca_pairs to get projected embeddings
            U_batch, V_batch, Bmiss_batch = build_cls2_dataset_from_dcca_pairs(
                batch_anchor_vecs,
                batch_target_stacks,
                projector_a,
                projector_b,
                device
            )
            
            # Build phi features from projected embeddings
            phi_batch = build_phi(U_batch, V_batch, Bmiss_batch)
            
            # Move to CPU immediately to save GPU memory
            phi_batches.append(phi_batch.cpu().numpy())
            
            # ✅ FIXED: Extract coordinates from DCCA pair metadata using correct keys
            for j, meta in enumerate(batch_metadata):
                if meta:
                    # ✅ Try multiple possible coordinate keys from DCCA metadata
                    anchor_coord = meta.get('anchor_coord', meta.get('anchor_coordinate'))
                    target_coord = meta.get('target_coord', meta.get('target_coordinate', meta.get('target_weighted_coord')))
                    
                    # ✅ Use anchor coordinates as primary (more reliable for overlap region)
                    if anchor_coord and anchor_coord != (0, 0):
                        coord = anchor_coord
                    elif target_coord and target_coord != (0, 0):
                        coord = target_coord
                    else:
                        coord = (0, 0)  # Fallback
                    
                    idx = meta.get('anchor_index', meta.get('index', f"overlap_{i+j}"))
                    label = meta.get('anchor_label', meta.get('label', 0))
                    
                else:
                    coord = (0, 0)
                    idx = f"overlap_{i+j}"
                    label = 0
                
                coordinates.append(coord)
                indices.append(idx)
                labels.append(label)
        
        # Clear GPU memory after each batch
        del U_batch, V_batch, Bmiss_batch, phi_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    
    # Combine all batches
    run_logger.log(f"[phi-unify-overlap] Combining {len(phi_batches)} phi feature batches...")
    all_phi = np.concatenate(phi_batches, axis=0)
    
    # Convert labels to numpy array
    all_labels = np.array(labels)
    
    # ✅ CRITICAL: Add coordinate diagnostics for overlap data
    if coordinates:
        coords_array = np.array(coordinates)
        x_coords = coords_array[:, 0]
        y_coords = coords_array[:, 1]
        
        # ✅ WARNING if still all zeros
        if x_coords.min() == x_coords.max() == 0 and y_coords.min() == y_coords.max() == 0:
            run_logger.log("[phi-unify-overlap] ❌ WARNING: All OVERLAP coordinates are still (0,0)!")
        elif x_coords.min() == x_coords.max() or y_coords.min() == y_coords.max():
            run_logger.log("[phi-unify-overlap] ❌ WARNING: OVERLAP coordinates have no spatial spread!")
        else:
            run_logger.log("[phi-unify-overlap] ✅ OVERLAP coordinates have proper spatial spread")
    
    # ✅ DEBUG: Show label distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    run_logger.log(f"[DEBUG] OVERLAP Label distribution: {label_dist}")
    
    # Clear intermediate data
    del phi_batches
    
    run_logger.log(f"[phi-unify-overlap] Created {len(all_phi)} unified overlap samples with {all_phi.shape[1]} phi features")
    
    return {
        'features': all_phi,
        'labels': all_labels,
        'coords': coordinates,  # ✅ Now contains real coordinates from DCCA metadata
        'coordinates': coordinates,  # ✅ Also provide as 'coordinates' key
        'indices': indices,
        'phi_unified': True,
        'source_type': 'overlap_pairs',
        'n_overlap_pairs': n_pairs,
    }

def Unifying_METHOD2_concat_two_embeddings(
    original_data: Dict[str, object],
    dcca_data: Dict[str, object],
    overlap_info: Dict[str, object],
    device: torch.device,
    run_logger: "_RunLogger",
    batch_size: int = 1000,
    mode: str = "concat",  # "concat", "original_only", "dcca_only"
) -> Dict[str, object]:
    """
    Create unified dataset by combining original embeddings (z) with DCCA embeddings (u).
    
    Supports three modes:
    - "concat": [z_a | u_a] or [z_b | u_b] - concatenate original and DCCA
    - "original_only": z_a or z_b - original embeddings only
    - "dcca_only": u_a or u_b - DCCA embeddings only
    
    Args:
        original_data: Dict with 'features' (z_a or z_b), 'labels', 'coords'
        dcca_data: Dict with 'features' (u_a or u_b), 'labels', 'coords'
        overlap_info: Dict with projector_a, projector_b for processing
        device: Torch device
        run_logger: Logger
        batch_size: Batch size for processing
        mode: Feature combination mode
        
    Returns:
        Dict with combined features, labels, coordinates
    """
    run_logger.log(f"[unify-original-dcca] Creating unified dataset with mode={mode}")
    
    # Extract features
    z_features = original_data.get('features')  # Original embeddings (N, d_original)
    u_features = dcca_data.get('features')      # DCCA embeddings (N, d_dcca)
    
    # Extract labels and coordinates (should match between both datasets)
    labels = original_data.get('labels')
    coords = original_data.get('coords', original_data.get('coordinates', []))
    
    # Convert to tensors if needed
    if isinstance(z_features, np.ndarray):
        z_features = torch.from_numpy(z_features).float()
    if isinstance(u_features, np.ndarray):
        u_features = torch.from_numpy(u_features).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).float()
    
    # Ensure same number of samples
    n_samples = len(z_features)
    assert len(u_features) == n_samples, f"Mismatch: z has {n_samples} samples, u has {len(u_features)}"
    assert len(labels) == n_samples, f"Mismatch: labels has {len(labels)} samples"
    
    run_logger.log(f"[unify-original-dcca] Processing {n_samples} samples")
    run_logger.log(f"[unify-original-dcca] z_features shape: {z_features.shape}")
    run_logger.log(f"[unify-original-dcca] u_features shape: {u_features.shape}")
    
    # ✅ Create combined features based on mode
    if mode == "concat":
        # Concatenate [z | u]
        combined_features = torch.cat([z_features, u_features], dim=1)
        run_logger.log(f"[unify-original-dcca] Concatenated features: {z_features.shape} + {u_features.shape} = {combined_features.shape}")
        feature_type = f"z_u_concat_dim{combined_features.shape[1]}"
        
    elif mode == "original_only":
        # Use only original embeddings z
        combined_features = z_features
        run_logger.log(f"[unify-original-dcca] Using original embeddings only: {combined_features.shape}")
        feature_type = f"z_only_dim{combined_features.shape[1]}"
        
    elif mode == "dcca_only":
        # Use only DCCA embeddings u
        combined_features = u_features
        run_logger.log(f"[unify-original-dcca] Using DCCA embeddings only: {combined_features.shape}")
        feature_type = f"u_only_dim{combined_features.shape[1]}"
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'concat', 'original_only', or 'dcca_only'")
    
    # Convert back to numpy for compatibility with training pipeline
    combined_features_np = combined_features.cpu().numpy()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # ✅ Coordinate diagnostics
    if coords:
        coords_array = np.array(coords)
        x_coords = coords_array[:, 0]
        y_coords = coords_array[:, 1]
        
        if x_coords.min() == x_coords.max() == 0 and y_coords.min() == y_coords.max() == 0:
            run_logger.log("[unify-original-dcca] ⚠️  WARNING: All coordinates are (0,0)")
        else:
            run_logger.log(f"[unify-original-dcca] ✅ Coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    
    # ✅ Label distribution
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    run_logger.log(f"[unify-original-dcca] Label distribution: {label_dist}")
    
    return {
        'features': combined_features_np,
        'labels': labels_np,
        'coords': coords,
        'coordinates': coords,
        'indices': original_data.get('indices', list(range(n_samples))),
        'feature_type': feature_type,
        'feature_mode': mode,
        'original_dim': z_features.shape[1],
        'dcca_dim': u_features.shape[1],
        'combined_dim': combined_features_np.shape[1],
        'source_type': 'original_plus_dcca',
    }   

def _substitute_label_overlap_constrained(
    data: Dict[str, object],
    label_source: Dict[str, object],
    overlap_info: Optional[Dict[str, object]],
    run_logger: "_RunLogger"
) -> Dict[str, object]:
    """
    Overlap-constrained label substitution using spatial coordinate matching.
    Uses overlap_info to ensure only overlap region samples are considered.
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    
    if overlap_info is None:
        run_logger.log("[substitute_label] ERROR: overlap_info required for overlap-constrained matching")
        raise ValueError("overlap_info must be provided for overlap-constrained label substitution")
    
    # ✅ Extract overlap region positive coordinates from overlap_info
    pair_metadata = overlap_info.get('pair_metadata', [])
    if not pair_metadata:
        run_logger.log("[substitute_label] ERROR: No pair_metadata found in overlap_info")
        raise ValueError("pair_metadata required for overlap-constrained matching")
    
    # ✅ Get positive coordinates from overlap region (from pair_metadata)
    overlap_positive_coords = []
    overlap_positive_indices = []
    
    for i, meta in enumerate(pair_metadata):
        anchor_label = meta.get('anchor_label', 0)
        if anchor_label > 0:  # This is a positive in the overlap region
            anchor_coord = meta.get('anchor_coord')
            if anchor_coord and anchor_coord != (0, 0):
                overlap_positive_coords.append(anchor_coord)
                overlap_positive_indices.append(meta.get('anchor_index', i))
    
    run_logger.log(f"[substitute_label] Found {len(overlap_positive_coords)} overlap positive coordinates")
    
    if len(overlap_positive_coords) == 0:
        run_logger.log("[substitute_label] WARNING: No positive coordinates found in overlap region")
        return data
    
    # ✅ Get target dataset information
    target_labels = data.get('labels')
    target_coords = data.get('coordinates', data.get('coords', []))
    
    if target_labels is None:
        run_logger.log("[substitute_label] WARNING: No labels found in target data")
        return data
    
    if len(target_coords) == 0:
        run_logger.log("[substitute_label] ERROR: No coordinates found in target data for matching")
        raise ValueError("Target coordinates required for overlap-constrained matching")
    
    # Remember original tensor format
    labels_was_tensor = isinstance(target_labels, torch.Tensor)
    labels_device = target_labels.device if labels_was_tensor else None
    labels_dtype = target_labels.dtype if labels_was_tensor else None
    
    # Convert to numpy for processing
    if isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.cpu().numpy()
    
    # ✅ CRITICAL: Keep original dataset size, only modify labels
    # new_labels = target_labels.copy()
    new_labels = np.zeros_like(target_labels)
    matches_found = 0
    
    # ✅ Convert coordinate lists to numpy arrays for efficient distance calculation
    overlap_coords_array = np.array(overlap_positive_coords)  # Shape: (n_overlap_positives, 2)
    target_coords_array = np.array(target_coords)            # Shape: (n_target_samples, 2)
    
    run_logger.log(f"[substitute_label] Matching {len(overlap_coords_array)} overlap positives against {len(target_coords_array)} target coordinates")
    
    # ✅ Handle potential coordinate format issues
    # Remove any None or invalid coordinates
    valid_target_mask = np.array([
        coord is not None and len(coord) == 2 and 
        not (coord[0] == 0 and coord[1] == 0) and
        all(isinstance(c, (int, float)) for c in coord)
        for coord in target_coords
    ])
    
    if not valid_target_mask.any():
        run_logger.log("[substitute_label] ERROR: No valid coordinates found in target dataset")
        raise ValueError("No valid target coordinates for matching")
    
    valid_indices = np.where(valid_target_mask)[0]
    valid_target_coords = target_coords_array[valid_indices]
    
    run_logger.log(f"[substitute_label] Using {len(valid_target_coords)} valid target coordinates out of {len(target_coords_array)}")
    
    
    # ✅ Compute pairwise distances between overlap positives and valid target coordinates
    distances = cdist(overlap_coords_array, valid_target_coords, metric='euclidean')
    
    # ✅ Find matches based on closest distance
    for i, overlap_coord in enumerate(overlap_coords_array):
        # Find closest target coordinate for this overlap positive
        distances_to_targets = distances[i, :]
        closest_target_idx = np.argmin(distances_to_targets)
        closest_distance = distances_to_targets[closest_target_idx]
        
        original_target_idx = valid_indices[closest_target_idx]
        new_labels[original_target_idx] = 1
        matches_found += 1
            
        run_logger.log(f"[substitute_label] Match {matches_found}: overlap_coord={overlap_coord} -> target_idx={original_target_idx} (distance={closest_distance:.2f})")
    
    # ✅ Apply additional overlap mask constraint if available
    overlap_mask = data.get('overlap_mask')
    if overlap_mask is not None:
        if isinstance(overlap_mask, torch.Tensor):
            overlap_mask = overlap_mask.cpu().numpy()
        
        # Only keep positive labels for samples that are actually in overlap regions
        overlap_constrained_labels = new_labels.copy()
        for i in range(len(new_labels)):
            if new_labels[i] > 0 and not overlap_mask[i]:
                overlap_constrained_labels[i] = 0  # Remove positive if not in overlap
        
        overlap_removed = (new_labels > 0).sum() - (overlap_constrained_labels > 0).sum()
        if overlap_removed > 0:
            run_logger.log(f"[substitute_label] Removed {overlap_removed} positives outside overlap mask")
            new_labels = overlap_constrained_labels
            matches_found -= overlap_removed
    
    # ✅ Convert labels back to original format
    if labels_was_tensor:
        new_labels_tensor = torch.from_numpy(new_labels).to(device=labels_device, dtype=labels_dtype)
        modified_data = data.copy()
        modified_data['labels'] = new_labels_tensor
    else:
        modified_data = data.copy()
        modified_data['labels'] = new_labels
    
    # ✅ Add metadata about the substitution
    modified_data['positive_label_source'] = 'overlap_constrained_coordinate_matched'
    modified_data['n_substituted_positives'] = matches_found
    modified_data['n_original_positives'] = int((target_labels > 0).sum())
    
    # ✅ Log final results
    unique_labels, counts = np.unique(new_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    
    run_logger.log(f"[substitute_label] ✅ OVERLAP-CONSTRAINED MATCHING COMPLETED:")
    run_logger.log(f"[substitute_label] - Original dataset size: {len(new_labels)} samples")
    run_logger.log(f"[substitute_label] - Original positives: {int((target_labels > 0).sum())}")
    run_logger.log(f"[substitute_label] - Overlap region matches found: {matches_found}")
    run_logger.log(f"[substitute_label] - Final label distribution: {label_dist}")
    
    if matches_found == 0:
        run_logger.log("[substitute_label] ⚠️  WARNING: No coordinate matches found - check coordinate alignment ")
    
    return modified_data

def train_and_save_result(
    tag: str,
    data_use: Dict[str, object],
    action: object,
    common: Dict[str, object],
    args,
    cfg,
    run_logger: "_RunLogger",
) -> None:
    """
    Unified training function that handles both legacy and multi-clustering modes.
    Always checks for cached predictions first before training.
    
    Args:
        tag: Unique identifier for this training run
        data_use: Processed data for training
        action: Action parameter for training
        common: Common parameters dictionary
        args: Arguments with training parameters
        cfg: Configuration object
        run_logger: Logger for debugging
    """
    
    # Extract seed from common parameters
    seed = common.get('seed', 42)
    
    # Convert meta_evaluation argument to set
    meta_evaluation_metrics = set(args.meta_evaluation)
    
    # Determine execution mode based on parameters
    is_multi_clustering = (
        len(args.clustering_methods) > 1 or 
        len(args.meta_evaluation_n_clusters) > 1 or
        args.clustering_methods != ["random"]
    )
    
    if is_multi_clustering:
        run_logger.log(f"[train_and_save_result] MULTI-CLUSTERING MODE for '{tag}'")
        run_logger.log(f"[train_and_save_result] Methods: {args.clustering_methods}")
        run_logger.log(f"[train_and_save_result] Cluster counts: {args.meta_evaluation_n_clusters}")
        run_logger.log(f"[train_and_save_result] Hierarchical linkage: {args.hierarchical_linkage}")
        
        # ✅ CHECK FOR CACHED RESULTS FIRST (multi-clustering)
        if args.skip_training_if_cached:
            run_logger.log(f"[train_and_save_result] Checking for cached multi-clustering results...")
            
            # Check each method-cluster combination for cached predictions
            cached_results = {}
            missing_configs = []
            
            for method in args.clustering_methods:
                for n_clusters in args.meta_evaluation_n_clusters:
                    config_name = f"{method}{n_clusters}"
                    
                    cached_result = train_cls_PN_base.load_and_evaluate_existing_predictions(
                        tag_main=tag,
                        meta_evaluation_n_clusters=n_clusters,
                        clustering_method=method,
                        common=common,
                        data_use=data_use,
                        run_logger=run_logger,
                        meta_evaluation_metrics=meta_evaluation_metrics
                    )
                    
                    if cached_result is not None:
                        cached_results[config_name] = cached_result
                        run_logger.log(f"[train_and_save_result] ✅ Found cached predictions for {config_name}")
                    else:
                        missing_configs.append({'method': method, 'n_clusters': n_clusters, 'name': config_name})
                        run_logger.log(f"[train_and_save_result] ❌ No cached predictions for {config_name}")
            
            # Handle cached results individually - save them immediately  
            total_configs = len(args.clustering_methods) * len(args.meta_evaluation_n_clusters)
            for config_name, result in cached_results.items():
                method_tag = f"{tag}_{config_name}"
                train_cls_PN_base.save_meta_evaluation_results(
                    meta_evaluation=result,
                    tag_main=method_tag,
                    common=common,
                    run_logger=run_logger
                )
                run_logger.log(f"[train_and_save_result] ✅ CACHED: {config_name} - saved existing predictions")
            
            # If all configurations are cached, skip training entirely
            if len(cached_results) == total_configs:
                run_logger.log(f"[train_and_save_result] ✅ All {total_configs} configurations cached, no training needed")
                
                # Log summary of cached results
                run_logger.log(f"[train_and_save_result] ===== CACHED MULTI-CLUSTERING RESULTS =====")
                for config_name, result in cached_results.items():
                    for metric, data in result.items():
                        if isinstance(data, dict) and 'mean' in data:
                            run_logger.log(f"[train_and_save_result] {config_name} {metric}: {data['mean']:.4f} ± {data['std']:.4f}")
                run_logger.log(f"[train_and_save_result] ===== END CACHED RESULTS =====")
                return
            
            # Selective training: only train missing configurations
            run_logger.log(f"[train_and_save_result] CACHE SUMMARY: {len(cached_results)} cached, {len(missing_configs)} to train (total: {total_configs})")
            
            # Filter methods and clusters to only include missing ones
            methods_to_train = list(set(config['method'] for config in missing_configs))
            clusters_to_train = list(set(config['n_clusters'] for config in missing_configs))
            
            run_logger.log(f"[train_and_save_result] Methods to train: {methods_to_train}")
            run_logger.log(f"[train_and_save_result] Clusters to train: {clusters_to_train}")
            
        else:
            # No caching - train everything
            methods_to_train = args.clustering_methods
            clusters_to_train = args.meta_evaluation_n_clusters
            run_logger.log(f"[train_and_save_result] No caching - training all configurations")
        
        # Execute multi-clustering training (only for missing configurations if caching enabled)
        if args.skip_training_if_cached and 'methods_to_train' in locals() and len(missing_configs) > 0:
            training_results = train_cls_PN_base.train_cls_1_PN_PosDrop_MultiClustering(
                clustering_methods=methods_to_train,
                meta_evaluation_n_clusters_list=clusters_to_train,
                linkage=args.hierarchical_linkage,
                seed=seed,
                common={**common, 'verbose': False},
                data_use=data_use,
                filter_top_pct=args.filter_top_pct,
                negs_per_pos=args.negs_per_pos,
                action=action,
                inference_fn=run_inference_base,
                tag_main=tag,
                meta_evaluation_metrics=meta_evaluation_metrics
            )
        elif not args.skip_training_if_cached:
            # Train all configurations (no caching)
            training_results = train_cls_PN_base.train_cls_1_PN_PosDrop_MultiClustering(
                clustering_methods=args.clustering_methods,
                meta_evaluation_n_clusters_list=args.meta_evaluation_n_clusters,
                linkage=args.hierarchical_linkage,
                seed=seed,
                common={**common, 'verbose': False},
                data_use=data_use,
                filter_top_pct=args.filter_top_pct,
                negs_per_pos=args.negs_per_pos,
                action=action,
                inference_fn=run_inference_base,
                tag_main=tag,
                meta_evaluation_metrics=meta_evaluation_metrics
            )
        else:
            # All cached, no training needed
            training_results = {}
        
        # Combine cached and training results for final summary
        if args.skip_training_if_cached and 'cached_results' in locals():
            all_results = {**cached_results, **training_results}
            run_logger.log(f"[train_and_save_result] Combined {len(cached_results)} cached + {len(training_results)} trained = {len(all_results)} total results")
        else:
            all_results = training_results
        
        # Save newly trained results (cached results already saved above)
        for method_cluster_key, result in training_results.items():
            if "error" not in result:
                method_tag = f"{tag}_{method_cluster_key}"
                
                # Save meta evaluation results
                train_cls_PN_base.save_meta_evaluation_results(
                    meta_evaluation=result,
                    tag_main=method_tag,
                    common=common,
                    run_logger=run_logger
                )
                
                # Save full training results
                save_training_results(
                    result=result,
                    tag=method_tag,
                    output_dir=Path(cfg.output_dir) / method_tag,
                    run_logger=run_logger
                )
                run_logger.log(f"[train_and_save_result] ✅ TRAINED: {method_cluster_key}")
            else:
                run_logger.log(f"[train_and_save_result] ❌ ERROR in {method_cluster_key}: {result['error']}")
        
        # Final comprehensive summary showing both cached and trained results
        run_logger.log(f"[train_and_save_result] ===== COMPLETE MULTI-CLUSTERING SUMMARY for {tag} =====")
        for method_cluster_key, result in all_results.items():
            if "error" not in result:
                # Determine if this result was cached or trained
                status = "CACHED" if (args.skip_training_if_cached and 'cached_results' in locals() and method_cluster_key in cached_results) else "TRAINED"
                for metric, data in result.items():
                    if isinstance(data, dict) and 'mean' in data:
                        run_logger.log(f"[train_and_save_result] {method_cluster_key} ({status}): {metric}={data['mean']:.4f} ± {data['std']:.4f}")
        run_logger.log(f"[train_and_save_result] ===== END SUMMARY =====")
    
    else:
        # LEGACY MODE: Single method, single cluster count
        clustering_method = args.clustering_methods[0]  # Should be "random"
        n_clusters = args.meta_evaluation_n_clusters[0]  # Single value
        
        # Convert to legacy drop_rate for backward compatibility
        drop_rate = 1.0 / n_clusters
        
        run_logger.log(f"[train_and_save_result] LEGACY MODE for '{tag}'")
        run_logger.log(f"[train_and_save_result] Method: {clustering_method}, n_clusters: {n_clusters} (drop_rate: {drop_rate:.3f})")
        
        # ✅ CHECK FOR CACHED RESULTS FIRST (legacy mode)
        cached_meta_evaluation = None
        if args.skip_training_if_cached:
            run_logger.log(f"[train_and_save_result] Checking for cached predictions...")
            
            # Try new format first
            cached_meta_evaluation = train_cls_PN_base.load_and_evaluate_existing_predictions(
                tag_main=tag,
                meta_evaluation_n_clusters=n_clusters,
                clustering_method=clustering_method,
                common=common,
                data_use=data_use,
                run_logger=run_logger,
                meta_evaluation_metrics=meta_evaluation_metrics
            )
            
            # ✅ BACKWARD COMPATIBILITY: Try legacy format if new format fails
            if cached_meta_evaluation is None:
                try:
                    # Try with legacy drop_rate parameter if the function supports it
                    cached_meta_evaluation = train_cls_PN_base.load_and_evaluate_existing_predictions(
                        tag_main=tag,
                        drop_rate=drop_rate,
                        common=common,
                        data_use=data_use,
                        run_logger=run_logger,
                        meta_evaluation_metrics=meta_evaluation_metrics
                    )
                except TypeError:
                    # Function doesn't support legacy drop_rate parameter
                    pass
        
        if cached_meta_evaluation is not None:
            run_logger.log(f"[train_and_save_result] ✅ Found cached predictions for '{tag}'")
            
            # Save cached meta evaluation results
            train_cls_PN_base.save_meta_evaluation_results(
                meta_evaluation=cached_meta_evaluation,
                tag_main=tag,
                common=common,
                run_logger=run_logger
            )
            
            run_logger.log(f"[train_and_save_result] Skipping training (cached predictions used)")
            for metric, data in cached_meta_evaluation.items():
                if isinstance(data, dict) and 'mean' in data:
                    run_logger.log(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f}")
            return
        else:
            run_logger.log(f"[train_and_save_result] No cached predictions found, proceeding with training")
        
        # Execute single-configuration training
        result = train_cls_PN_base.train_cls_1_PN_PosDrop(
            meta_evaluation_n_clusters=n_clusters,
            clustering_method=clustering_method,
            linkage=args.hierarchical_linkage,
            seed=seed,
            common={**common, 'verbose': False},
            data_use=data_use,
            filter_top_pct=args.filter_top_pct,
            negs_per_pos=args.negs_per_pos,
            action=action,
            inference_fn=run_inference_base,
            tag_main=tag,
            meta_evaluation_metrics=meta_evaluation_metrics
        )
        
        # Save training results
        save_training_results(
            result=result,
            tag=tag,
            output_dir=Path(cfg.output_dir) / tag,
            run_logger=run_logger
        )
        
        # Save meta evaluation results
        train_cls_PN_base.save_meta_evaluation_results(
            meta_evaluation=result,
            tag_main=tag,
            common=common,
            run_logger=run_logger
        )
        
        run_logger.log(f"[train_and_save_result] Completed training for {tag}")
        for metric, data in result.items():
            if isinstance(data, dict) and 'mean' in data:
                run_logger.log(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f}")


def save_training_results(
    result: Dict[str, Any],
    tag: str,
    output_dir: Path,
    run_logger: "_RunLogger",
) -> None:
    """
    Save training results to JSON file with proper serialization handling.
    Enhanced to extract and save key metrics (accuracy, AUCPR, focus score, etc.)
    
    Args:
        result: Training result dictionary from train_cls_1_PN_PosDrop
        tag: Unique identifier for this training run
        output_dir: Base output directory for saving results
        run_logger: Logger for debugging information
    """
    # Create results directory
    results_dir = output_dir / "cls_1_training_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ EXTRACT KEY METRICS FIRST (before serialization)
    key_metrics = _extract_key_metrics(result, run_logger)
    
    # Prepare serializable result dictionary
    serializable_result = {}
    
    try:
        for key, value in result.items():
            if key == 'cls':
                # Save model info instead of the actual model
                if hasattr(value, '__class__'):
                    serializable_result[key] = {
                        'model_type': value.__class__.__name__,
                        'model_repr': str(value),
                        'num_parameters': sum(p.numel() for p in value.parameters()) if hasattr(value, 'parameters') else 'N/A'
                    }
                else:
                    serializable_result[key] = str(value)
            elif key == 'epoch_history':
                # Convert epoch history to serializable format
                if isinstance(value, list):
                    serializable_history = []
                    for epoch_data in value:
                        if isinstance(epoch_data, dict):
                            serializable_epoch = {}
                            for k, v in epoch_data.items():
                                if isinstance(v, (int, float, str, bool)):
                                    serializable_epoch[k] = v
                                elif isinstance(v, torch.Tensor):
                                    serializable_epoch[k] = v.item() if v.numel() == 1 else v.tolist()
                                elif isinstance(v, np.ndarray):
                                    serializable_epoch[k] = v.tolist()
                                else:
                                    serializable_epoch[k] = str(v)
                            serializable_history.append(serializable_epoch)
                        else:
                            serializable_history.append(str(epoch_data))
                    serializable_result[key] = serializable_history
                else:
                    serializable_result[key] = str(value)
            elif key == 'inference_result':
                # Handle inference results
                if isinstance(value, dict):
                    serializable_inference = {}
                    for k, v in value.items():
                        if isinstance(v, (int, float, str, bool)):
                            serializable_inference[k] = v
                        elif isinstance(v, torch.Tensor):
                            serializable_inference[k] = v.tolist() if v.numel() > 1 else v.item()
                        elif isinstance(v, np.ndarray):
                            serializable_inference[k] = v.tolist()
                        elif isinstance(v, list):
                            # Handle coordinate lists
                            try:
                                serializable_inference[k] = [list(item) if hasattr(item, '__iter__') and not isinstance(item, str) else item for item in v]
                            except:
                                serializable_inference[k] = str(v)
                        else:
                            serializable_inference[k] = str(v)
                    serializable_result[key] = serializable_inference
                else:
                    serializable_result[key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                serializable_result[key] = value
            elif isinstance(value, torch.Tensor):
                serializable_result[key] = value.tolist() if value.numel() > 1 else value.item()
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                nested_dict = {}
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool)):
                        nested_dict[k] = v
                    elif isinstance(v, torch.Tensor):
                        nested_dict[k] = v.tolist() if v.numel() > 1 else v.item()
                    elif isinstance(v, np.ndarray):
                        nested_dict[k] = v.tolist()
                    else:
                        nested_dict[k] = str(v)
                serializable_result[key] = nested_dict
            else:
                serializable_result[key] = str(value)
        
        # ✅ ADD KEY METRICS TO SERIALIZABLE RESULT
        serializable_result['key_metrics'] = key_metrics
        
        # Add metadata
        serializable_result['_metadata'] = {
            'tag': tag,
            'timestamp': datetime.now().isoformat(),
            'save_time': str(datetime.now()),
        }
        
        # Save to JSON file
        json_filename = f"{tag}_results.json"
        json_path = results_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        run_logger.log(f"[save_results] Saved training results to {json_path}")
        
        # ✅ SAVE KEY METRICS TO SEPARATE CSV FOR EASY COMPARISON
        _save_metrics_csv(key_metrics, tag, results_dir, run_logger)
        
        # Also save a summary with key metrics
        summary = {
            'tag': tag,
            'timestamp': datetime.now().isoformat(),
            'model_type': serializable_result.get('cls', {}).get('model_type', 'Unknown'),
            'has_inference': 'inference_result' in serializable_result and serializable_result['inference_result'] is not None,
            'num_epochs': len(serializable_result.get('epoch_history', [])),
            'key_metrics': key_metrics,  # ✅ Include in summary
        }
        
        # Extract key metrics if available
        if 'metrics_summary' in serializable_result and serializable_result['metrics_summary']:
            summary['final_metrics'] = serializable_result['metrics_summary']
        
        if 'early_stopping_summary' in serializable_result and serializable_result['early_stopping_summary']:
            summary['early_stopping'] = serializable_result['early_stopping_summary']
        
        # Save summary
        summary_path = results_dir / f"{tag}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        run_logger.log(f"[save_results] Saved training summary to {summary_path}")
        
        # ✅ LOG KEY METRICS TO CONSOLE
        _log_key_metrics(key_metrics, tag, run_logger)
        
    except Exception as e:
        run_logger.log(f"[save_results] ERROR saving results for {tag}: {e}")
        import traceback
        run_logger.log(traceback.format_exc())
        
        # Save error info
        error_result = {
            'tag': tag,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
            'result_keys': list(result.keys()) if isinstance(result, dict) else 'Not a dict'
        }
        error_path = results_dir / f"{tag}_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_result, f, indent=2)


def _extract_key_metrics(result: Dict[str, Any], run_logger: "_RunLogger") -> Dict[str, Any]:
    """Extract key metrics from training result for easy access and comparison."""
    metrics = {
        'tag': result.get('tag', 'unknown'),
        'training_completed': True,
        'extraction_timestamp': datetime.now().isoformat(),
    }
    
    try:
        # Extract from metrics summary
        if 'metrics_summary' in result and result['metrics_summary']:
            metrics_summary = result['metrics_summary']
            metrics['final_accuracy'] = metrics_summary.get('accuracy')
            metrics['final_aucpr'] = metrics_summary.get('aucpr')
            metrics['final_precision'] = metrics_summary.get('precision')
            metrics['final_recall'] = metrics_summary.get('recall')
            metrics['final_f1'] = metrics_summary.get('f1')
            metrics['final_loss'] = metrics_summary.get('loss')
        
        # Extract from epoch history
        if 'epoch_history' in result and result['epoch_history']:
            epoch_history = result['epoch_history']
            if isinstance(epoch_history, list) and len(epoch_history) > 0:
                last_epoch = epoch_history[-1]
                if isinstance(last_epoch, dict):
                    metrics['num_epochs_trained'] = len(epoch_history)
                    metrics['last_epoch_num'] = last_epoch.get('epoch', len(epoch_history))
                    metrics['last_epoch_train_loss'] = last_epoch.get('train_loss')
                    metrics['last_epoch_val_loss'] = last_epoch.get('val_loss')
                    metrics['last_epoch_train_acc'] = last_epoch.get('train_accuracy')
                    metrics['last_epoch_val_acc'] = last_epoch.get('val_accuracy')
                    # PosDrop-specific metrics
                    metrics['last_epoch_pos_acc'] = last_epoch.get('pos_accuracy')
                    metrics['last_epoch_neg_acc'] = last_epoch.get('neg_accuracy')
                    metrics['last_epoch_focus'] = last_epoch.get('focus_score')
                
                # Find best epoch
                best_metrics = _find_best_epoch_metrics(epoch_history)
                metrics.update(best_metrics)
        
        # Extract from early stopping
        if 'early_stopping_summary' in result and result['early_stopping_summary']:
            es_summary = result['early_stopping_summary']
            metrics['early_stopped'] = es_summary.get('early_stopped', False)
            metrics['best_epoch'] = es_summary.get('best_epoch')
            metrics['best_val_loss'] = es_summary.get('best_val_loss')
            metrics['patience_used'] = es_summary.get('patience_counter')
            metrics['patience_limit'] = es_summary.get('patience_limit')
        
        # Extract training configuration
        metrics['drop_rate'] = result.get('drop_rate')
        metrics['filter_top_pct'] = result.get('filter_top_pct')
        metrics['negs_per_pos'] = result.get('negs_per_pos')
        
        # Convert tensor/numpy to scalars
        for key, value in list(metrics.items()):
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, np.ndarray):
                metrics[key] = value.item() if value.size == 1 else value.tolist()
    
    except Exception as e:
        run_logger.log(f"[extract_metrics] ERROR: {e}")
        metrics['extraction_error'] = str(e)
    
    return metrics


def _find_best_epoch_metrics(epoch_history: List[Dict]) -> Dict[str, Any]:
    """Find best epoch based on validation loss."""
    best_metrics = {}
    try:
        epochs_with_val_loss = [
            (i, epoch) for i, epoch in enumerate(epoch_history)
            if isinstance(epoch, dict) and 'val_loss' in epoch and epoch['val_loss'] is not None
        ]
        
        if epochs_with_val_loss:
            best_idx, best_epoch = min(epochs_with_val_loss, key=lambda x: x[1]['val_loss'])
            best_metrics['best_epoch_num'] = best_epoch.get('epoch', best_idx)
            best_metrics['best_val_loss'] = best_epoch.get('val_loss')
            best_metrics['best_val_acc'] = best_epoch.get('val_accuracy')
            best_metrics['best_pos_acc'] = best_epoch.get('pos_accuracy')
            best_metrics['best_neg_acc'] = best_epoch.get('neg_accuracy')
            best_metrics['best_focus'] = best_epoch.get('focus_score')
    except Exception as e:
        best_metrics['best_epoch_error'] = str(e)
    return best_metrics


def _save_metrics_csv(metrics: Dict[str, Any], tag: str, results_dir: Path, run_logger: "_RunLogger") -> None:
    """Save metrics to CSV for easy comparison across experiments."""
    import csv
    csv_path = results_dir / "all_experiments_metrics.csv"
    try:
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as csvfile:
            # Define field order for CSV
            fieldnames = ['tag', 'extraction_timestamp'] + sorted([
                k for k in metrics.keys() 
                if k not in ['tag', 'extraction_timestamp', 'training_completed', 'extraction_error']
            ])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        run_logger.log(f"[save_metrics_csv] Appended to {csv_path}")
    except Exception as e:
        run_logger.log(f"[save_metrics_csv] ERROR: {e}")


def _log_key_metrics(metrics: Dict[str, Any], tag: str, run_logger: "_RunLogger") -> None:
    """Log key metrics to console in readable format."""
    run_logger.log("=" * 80)
    run_logger.log(f"KEY METRICS FOR: {tag}")
    run_logger.log("=" * 80)
    
    if metrics.get('training_completed'):
        run_logger.log(f"✅ Training completed")
    
    # Final metrics
    if metrics.get('final_accuracy') is not None:
        run_logger.log(f"Final Accuracy: {metrics['final_accuracy']:.4f}")
    if metrics.get('final_aucpr') is not None:
        run_logger.log(f"Final AUCPR: {metrics['final_aucpr']:.4f}")
    
    # PosDrop-specific metrics
    if metrics.get('last_epoch_pos_acc') is not None:
        run_logger.log(f"Pos Accuracy: {metrics['last_epoch_pos_acc']:.4f}")
    if metrics.get('last_epoch_neg_acc') is not None:
        run_logger.log(f"Neg Accuracy: {metrics['last_epoch_neg_acc']:.4f}")
    if metrics.get('last_epoch_focus') is not None:
        run_logger.log(f"Focus Score: {metrics['last_epoch_focus']:.4f}")
    
    # Best epoch info
    if metrics.get('best_epoch_num') is not None:
        best_val_loss = metrics.get('best_val_loss')
        loss_str = f"{best_val_loss:.4f}" if best_val_loss is not None else 'N/A'
        run_logger.log(f"Best Epoch: {metrics['best_epoch_num']} (Val Loss: {loss_str})")
    
    # Training progress
    if metrics.get('num_epochs_trained') is not None:
        run_logger.log(f"Epochs: {metrics['num_epochs_trained']}")
    
    # Configuration
    if metrics.get('drop_rate') is not None:
        run_logger.log(f"Drop Rate: {metrics['drop_rate']}")
    if metrics.get('negs_per_pos') is not None:
        run_logger.log(f"Negs per Pos: {metrics['negs_per_pos']}")
    
    run_logger.log("=" * 80)


def _progress_iter(iterable, desc: str, *, leave: bool = False, total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave, total=total)

def _check_memory_usage(run_logger: "_RunLogger", stage: str) -> None:
    """Check and log current memory usage."""
    try:
        import psutil
        import os
        
        # Get current process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory info  
        system_memory = psutil.virtual_memory()
        total_mb = system_memory.total / 1024 / 1024
        available_mb = system_memory.available / 1024 / 1024
        used_percent = system_memory.percent
        
        run_logger.log(f"[Memory-{stage}] Process: {memory_mb:.1f}MB | System: {used_percent:.1f}% used ({available_mb:.1f}MB available)")
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2
            run_logger.log(f"[GPU-{stage}] Allocated: {gpu_allocated:.1f}MB | Reserved: {gpu_reserved:.1f}MB")
            
    except ImportError:
        run_logger.log(f"[Memory-{stage}] psutil not available for memory monitoring")
    except Exception as e:
        run_logger.log(f"[Memory-{stage}] Error checking memory: {e}")

def get_dcca_embeddings_with_overlaps(cfg: AlignmentConfig, args, device: torch.device, run_logger: "_RunLogger") -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """
    Extract DCCA datasets preparation logic for early access to dcca_sets.
    Returns dcca_sets dictionary containing projected datasets and overlap information.
    
    Returns:
        Tuple of (dcca_sets, overlap_info) where overlap_info contains:
        - anchor_vecs: List[Tensor] overlap samples for anchor
        - target_stack_per_anchor: List[Tensor] overlap samples for target  
        - pair_metadata: List[Dict] metadata for each overlap pair
        - projector_a: anchor projector model
        - projector_b: target projector model
    """
    debug_mode = bool(args.debug)
    
    #################################### Prepare Overlap Workspace ####################################
    workspace = OverlapAlignmentWorkspace(cfg)
    anchor_name, target_name = _dataset_pair(cfg)
    max_coord_error = auto_coord_error(workspace, anchor_name, target_name)
    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    if workspace.overlap is None:
        raise RuntimeError("Overlap pairs are unavailable; ensure overlap_pairs_path points to *_overlap_pairs.json from integrate_stac.")
    if debug_mode:
        report = workspace.pair_diagnostics(max_coord_error=max_coord_error, pairs=pairs)
        print("[debug] resolved_pairs:", report["resolved_pairs"])
        print("[debug] total_overlap_pairs:", report["total_overlap_pairs"])
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

    #################################### Preprocess - Augmenting positive labels ####################################
    augmentation_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    if cfg.use_positive_augmentation:
        for dataset_cfg in cfg.datasets:
            if dataset_cfg.pos_aug_path is None:
                continue
            aug_map, aug_count = _load_augmented_embeddings(dataset_cfg.pos_aug_path, dataset_cfg.region_filter)
            if aug_map:
                augmentation_by_dataset[dataset_cfg.name] = aug_map
                print(f"[info] Loaded {aug_count} augmented embeddings for dataset {dataset_cfg.name}")

    #################################### Preprocess - Alignment data preparation ####################################
    pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]] = {
        dataset_cfg.name: _load_pn_lookup(dataset_cfg.pn_split_path) for dataset_cfg in cfg.datasets
    }


    pairing_mode = (cfg.pairing_mode or "set_to_set").lower()
    if pairing_mode == "one_to_one":
        pair_builder = _build_aligned_pairs_OneToOne
    elif pairing_mode == "set_to_set":
        pair_builder = _build_aligned_pairs_SetToSet
    else:
        raise ValueError(f"Unsupported pairing_mode '{cfg.pairing_mode}'. Expected 'one_to_one' or 'set_to_set'.")

    use_transformer_agg = bool(getattr(args, "use_transformer_aggregator", False))
    if pairing_mode != "set_to_set" and use_transformer_agg:
        run_logger.log("[agg] Transformer aggregator requires pairing_mode='set_to_set'; disabling aggregator.")
        use_transformer_agg = False

    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
    (anchor_vecs, target_vecs, label_hist, debug_data,
     augmentation_stats, pair_metadata, target_stack_per_anchor, pn_index_summary) = pair_builder(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=cfg.use_positive_only,
        aggregator=cfg.aggregator,
        gaussian_sigma=cfg.gaussian_sigma,
        debug=debug_mode,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=augmentation_by_dataset.get(anchor_name),
        target_augment_map=augmentation_by_dataset.get(target_name),
        pn_label_maps=pn_label_maps,
        overlap_set=workspace.overlap,
    )

    if not anchor_vecs:
        print("[warn] No qualifying overlap groups were found for training.")
        run_logger.log("[DEBUG] RETURNING EMPTY - No qualifying overlap groups found")
        return {}, {}


    #################################### DCCA Three-Stage Resolution ####################################
    projector_a, projector_b, dcca_sets, anchor_overlap_samples, anchor_non_overlap_samples, target_overlap_samples = resolve_dcca_embeddings_and_projectors(
        cfg=cfg,
        args=args,
        workspace=workspace,
        anchor_name=anchor_name,
        target_name=target_name,
        anchor_vecs=anchor_vecs,
        target_stack_per_anchor=target_stack_per_anchor,
        pair_metadata=pair_metadata,
        device=device,
        run_logger=run_logger,
        use_transformer_agg=use_transformer_agg,
        pn_label_maps=pn_label_maps,
        overlap_mask_info=_load_overlap_mask_data(cfg.overlap_mask_path)
    )
    
    # Load overlap mask for spatial filtering
    overlap_mask_data = _load_overlap_mask_data(cfg.overlap_mask_path)
    
    # Package overlap information
    overlap_info = {
        'anchor_vecs': anchor_vecs,
        'target_stack_per_anchor': target_stack_per_anchor,
        'pair_metadata': pair_metadata,
        'projector_a': projector_a,
        'projector_b': projector_b,
        'mask': overlap_mask_data,  # Add mask for spatial filtering
    }
    
    return dcca_sets, overlap_info

def get_overlap_info_pair_metadata_only(cfg: AlignmentConfig, args, device: torch.device, run_logger: "_RunLogger") -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """
    Extract DCCA datasets preparation logic for early access to dcca_sets.
    Returns dcca_sets dictionary containing projected datasets and overlap information.
    
    Returns:
        - pair_metadata: List[Dict] metadata for each overlap pair

    """
    debug_mode = bool(args.debug)
    
    #################################### Prepare Overlap Workspace ####################################
    workspace = OverlapAlignmentWorkspace(cfg)
    anchor_name, target_name = _dataset_pair(cfg)
    max_coord_error = auto_coord_error(workspace, anchor_name, target_name)
    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    if workspace.overlap is None:
        raise RuntimeError("Overlap pairs are unavailable; ensure overlap_pairs_path points to *_overlap_pairs.json from integrate_stac.")
    if debug_mode:
        report = workspace.pair_diagnostics(max_coord_error=max_coord_error, pairs=pairs)
        print("[debug] resolved_pairs:", report["resolved_pairs"])
        print("[debug] total_overlap_pairs:", report["total_overlap_pairs"])
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

    #################################### Preprocess - Augmenting positive labels ####################################
    augmentation_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    if cfg.use_positive_augmentation:
        for dataset_cfg in cfg.datasets:
            if dataset_cfg.pos_aug_path is None:
                continue
            aug_map, aug_count = _load_augmented_embeddings(dataset_cfg.pos_aug_path, dataset_cfg.region_filter)
            if aug_map:
                augmentation_by_dataset[dataset_cfg.name] = aug_map
                print(f"[info] Loaded {aug_count} augmented embeddings for dataset {dataset_cfg.name}")

    #################################### Preprocess - Alignment data preparation ####################################
    pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]] = {
        dataset_cfg.name: _load_pn_lookup(dataset_cfg.pn_split_path) for dataset_cfg in cfg.datasets
    }
    

    pairing_mode = (cfg.pairing_mode or "set_to_set").lower()
    if pairing_mode == "one_to_one":
        pair_builder = _build_aligned_pairs_OneToOne
    elif pairing_mode == "set_to_set":
        pair_builder = _build_aligned_pairs_SetToSet
    else:
        raise ValueError(f"Unsupported pairing_mode '{cfg.pairing_mode}'. Expected 'one_to_one' or 'set_to_set'.")

    use_transformer_agg = bool(getattr(args, "use_transformer_aggregator", False))
    if pairing_mode != "set_to_set" and use_transformer_agg:
        run_logger.log("[agg] Transformer aggregator requires pairing_mode='set_to_set'; disabling aggregator.")
        use_transformer_agg = False
    
    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
    (_, _, _, _,
     _, pair_metadata, _, _) = pair_builder(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=cfg.use_positive_only,
        aggregator=cfg.aggregator,
        gaussian_sigma=cfg.gaussian_sigma,
        debug=debug_mode,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=augmentation_by_dataset.get(anchor_name),
        target_augment_map=augmentation_by_dataset.get(target_name),
        pn_label_maps=pn_label_maps,
        overlap_set=workspace.overlap,
    )

   
    # Package overlap information
    pair_metadata = {
        'pair_metadata': pair_metadata,
    }
    
    return pair_metadata
#####################################################################################   


def _dataset_pair(cfg: AlignmentConfig) -> Tuple[str, str]:
    if len(cfg.datasets) < 2:
        raise ValueError("Overlap alignment requires at least two datasets.")
    return cfg.datasets[0].name, cfg.datasets[1].name

class _RunLogger:
    def __init__(self, cfg: AlignmentConfig):
        self.cfg = cfg  # ✅ Store cfg for easy access
        base = cfg.output_dir if cfg.output_dir is not None else cfg.log_dir
        target_dir = _prepare_output_dir(base, "overlap_alignment_outputs")
        self.path: Optional[Path] = None
        if target_dir is not None:
            self.path = target_dir / "overlap_alignment_stage1_run.log"

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception as exc:
                print(f"[warn] Unable to append to run log {self.path}: {exc}")
        print(f"[run-log] {message}")

if __name__ == "__main__":
    main()
