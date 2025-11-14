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

from . import train_cls_1
from . inference_module import run_inference_base

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    # parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of aligned pairs reserved for validation evaluation (set to 0 to disable).", )
    
    parser.add_argument("--train-dcca", action=argparse.BooleanOptionalAction, default=True, help="Train the DCCA projection heads (default: true). Use --no-train-dcca to disable.",)
    parser.add_argument("--read-dcca",  action=argparse.BooleanOptionalAction, default=False,help="Load existing DCCA projection head weights before training (default: false).",)
    parser.add_argument("--dcca-weights-path", type=str, default=None, help="Optional path to a saved DCCA checkpoint used when --read-dcca is enabled.",)
    parser.add_argument("--use-transformer-aggregator", action=argparse.BooleanOptionalAction, default=True, help="Enable transformer-based aggregation before DCCA (set-to-set pairing only).",)
    parser.add_argument("--agg-trans-num-layers", type=int, default=4, help="Number of cross-attention layers in the aggregator.")
    parser.add_argument("--agg-trans-num-heads", type=int, default=4, help="Number of attention heads in the aggregator.")
    parser.add_argument("--agg-trans-dropout", type=float, default=0.1, help="Dropout used inside the transformer aggregator.")
    parser.add_argument("--agg-trans-pos-enc", action=argparse.BooleanOptionalAction, default=False, help="Use positional encoding based on anchor/target coordinate differences.",)

    parser.add_argument("--train-cls-1", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Non-Overlapping part.",)
    parser.add_argument("--train-cls-1-Method", choices=["PN", "PU"], default="PU", help="Classifier training method (default: PU). ")
    parser.add_argument('--filter-top-pct', type=float, default=0.10)
    parser.add_argument('--negs-per-pos', type=int, default=10)

    parser.add_argument("--train-cls-2", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Overlapping part.",)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128], help="Hidden layer sizes for classifier MLP heads (space-separated).",)
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout probability applied to classifier MLP layers.", )
    parser.add_argument("--mlp-dropout-passes", type=int, default=5, help="Number of Monte Carlo dropout passes for uncertainty estimation in classifier inference.", )

    parser.add_argument("--read-inference", action=argparse.BooleanOptionalAction, default=False, help="Read inference on aligned datasets after training (default: false).",)
    parser.add_argument("--force-recompute-dcca", action="store_true", help="Force recomputation of DCCA projections even if cached results exist.",)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

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

    if args.objective is not None:
        cfg.alignment_objective = args.objective.lower()
    # if args.aggregator is not None:
    #     cfg.aggregator = args.aggregator.lower()
    if args.use_positive_only:
        cfg.use_positive_only = True
    if args.use_positive_augmentation:
        cfg.use_positive_augmentation = True
    debug_mode = bool(args.debug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger = _RunLogger(cfg)

    read_dcca = bool(args.read_dcca)
    if read_dcca:
        args.train_dcca = False
    train_dcca = bool(args.train_dcca)
    if not train_dcca and not read_dcca:
        raise ValueError("Cannot disable DCCA training without enabling --read-dcca to load existing weights.")


    # make class wrapper later.
    # data_use = {dcca_sets[anchor_name]}
    # encoders = {projector_a}
    # action_1 = {"learning_rates": [0.01, 0.001, 0.0001]}
    # action = {"data": [data_use], "encoder": [projector_a], "action_1": action_1}

    common = {
        'cfg': cfg,
        'device': device,
        'mlp_hidden_dims': mlp_hidden_dims,
        'mlp_dropout': mlp_dropout,
        'debug_mode': debug_mode,
        'run_logger': run_logger,
    }

    # Get DCCA datasets early for classifier training
    _check_memory_usage(run_logger, "start")
    dcca_sets = get_dcca_embeddings(cfg, args, device, run_logger)
    _check_memory_usage(run_logger, "after-dcca")
    
    # Get anchor name for early access
    anchor_name, target_name = _dataset_pair(cfg)


    # ✅ SIMPLE: Replace individual dataset with unified phi dataset
    data_use = Unifying(dcca_sets, anchor_name, target_name, device, run_logger)
    action = None
    encoder_name = "TransAggDCCA"
    tag = f"cls_1_unified_Method1Phi_encoder={encoder_name}"
    
    result = train_cls_1.train_cls_1_PN_PosDrop(drop_rate=0.1,
                                                common={**common, 'verbose': False}, data_use=data_use,     
                                                filter_top_pct=args.filter_top_pct, negs_per_pos=args.negs_per_pos, action=action, 
                                                inference_fn=run_inference_base,
                                                tag_main=tag)
    print(result)
    exit()


    # Train Classifier 1 on DCCA anchor data
    data_use = dcca_sets[anchor_name]
    action = None
    encoder_name = "DCCA_Anchor"
    tag = f"cls_1_{anchor_name}_encoder={encoder_name}"     # to save computation
    result = train_cls_1.train_cls_1_PN_PosDrop(drop_rate=0.1,
                                                common={**common, 'verbose': False}, data_use=data_use,     
                                                filter_top_pct=args.filter_top_pct, negs_per_pos=args.negs_per_pos, action=action, 
                                                inferenceb_fn=run_inference_base,
                                                tag_main=tag)
    print(result)

    # Train Classifier 1 on DCCA target data
    data_use = dcca_sets[target_name]
    action = None
    encoder_name = "DCCA_Target"
    tag = f"cls_1_{target_name}_encoder={encoder_name}"     # to save computation
    result = train_cls_1.train_cls_1_PN_PosDrop(drop_rate=0.1,
                                                common={**common, 'verbose': False}, data_use=data_use,     
                                                filter_top_pct=args.filter_top_pct, negs_per_pos=args.negs_per_pos, action=action, 
                                                inference_fn=run_inference_base,
                                                tag_main=tag)
    print(result)


    # Note: DCCA datasets and all required data are already computed by get_dcca_embeddings()
    # No additional processing needed here - dcca_sets contains all projected datasets

    return
#####################################################################################   #####################################################################################   #####################################################################################   


def Unifying(
    dcca_sets: Dict[str, Dict[str, object]], 
    anchor_name: str, 
    target_name: str,
    device: torch.device,
    run_logger: "_RunLogger",
) -> Dict[str, object]:
    """
    SIMPLIFIED: Create unified phi dataset using existing DCCA projections.
    Uses cfg from run_logger to access configuration and avoid parameter passing.
    """
    
    # ✅ Extract cfg from run_logger (much cleaner!)
    cfg = run_logger.cfg if hasattr(run_logger, 'cfg') else None
    if cfg is None:
        raise ValueError("Configuration not available in run_logger")
    
    anchor_data = dcca_sets.get(anchor_name)
    target_data = dcca_sets.get(target_name)
    
    if not anchor_data or not target_data:
        raise ValueError(f"Missing DCCA data for {anchor_name} or {target_name}")
    
    # ✅ Debug: Check what keys are actually available
    run_logger.log(f"[phi-unify] DEBUG: anchor_data keys: {list(anchor_data.keys()) if isinstance(anchor_data, dict) else type(anchor_data)}")
    run_logger.log(f"[phi-unify] DEBUG: target_data keys: {list(target_data.keys()) if isinstance(target_data, dict) else type(target_data)}")
    
    # ✅ Check for different possible key names
    anchor_embeddings_key = None
    target_embeddings_key = None
    
    for key in ['embeddings', 'projected_embeddings', 'vectors', 'features']:
        if key in anchor_data:
            anchor_embeddings_key = key
            break
    
    for key in ['embeddings', 'projected_embeddings', 'vectors', 'features']:
        if key in target_data:
            target_embeddings_key = key
            break
    
    if anchor_embeddings_key is None or target_embeddings_key is None:
        run_logger.log(f"[phi-unify] ERROR: Could not find embeddings data")
        run_logger.log(f"[phi-unify] anchor_data structure: {anchor_data}")
        run_logger.log(f"[phi-unify] target_data structure: {target_data}")
        raise KeyError(f"Could not find embeddings in anchor_data (keys: {list(anchor_data.keys())}) or target_data (keys: {list(target_data.keys())})")
    
    run_logger.log(f"[phi-unify] Creating unified dataset from {len(anchor_data[anchor_embeddings_key])} anchor + {len(target_data[target_embeddings_key])} target samples")
    
    # ✅ Get DCCA projected embeddings (already in common space)
    anchor_features = anchor_data[anchor_embeddings_key]
    target_features = target_data[target_embeddings_key]
    
    # ✅ Handle both numpy arrays and tensors
    if isinstance(anchor_features, np.ndarray):
        anchor_embeddings = torch.from_numpy(anchor_features).float().to(device)
    else:
        anchor_embeddings = anchor_features.float().to(device)
        
    if isinstance(target_features, np.ndarray):
        target_embeddings = torch.from_numpy(target_features).float().to(device)
    else:
        target_embeddings = target_features.float().to(device)
    
    # ✅ Create phi features using the imported build_phi function
    with torch.no_grad():
        # Anchor samples: u=anchor_proj, v=0 (no target), b_missing=1
        u_anchor = anchor_embeddings
        v_anchor = torch.zeros_like(u_anchor)
        b_missing_anchor = torch.ones(len(u_anchor), 1, device=device)
        phi_anchor = build_phi(u_anchor, v_anchor, b_missing_anchor)
        
        # Target samples: u=target_proj, v=0 (no anchor), b_missing=1  
        u_target = target_embeddings
        v_target = torch.zeros_like(u_target)
        b_missing_target = torch.ones(len(u_target), 1, device=device)
        phi_target = build_phi(u_target, v_target, b_missing_target)
        
        # Combine all phi features
        all_phi = torch.cat([phi_anchor, phi_target], dim=0)
    
    # ✅ Combine metadata using cfg information
    anchor_labels = anchor_data['labels']
    target_labels = target_data['labels']
    
    # ✅ Handle both numpy arrays and tensors for labels
    if isinstance(anchor_labels, torch.Tensor):
        anchor_labels = anchor_labels.cpu().numpy()
    if isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.cpu().numpy()
        
    all_labels = np.concatenate([anchor_labels, target_labels])
    
    # ✅ Use cfg to determine coordinate handling
    anchor_coords = anchor_data.get('coordinates', anchor_data.get('coords', []))
    target_coords = target_data.get('coordinates', target_data.get('coords', []))
    all_coords = anchor_coords + target_coords
    
    anchor_indices = anchor_data.get('indices', list(range(len(anchor_labels))))
    target_indices = target_data.get('indices', list(range(len(target_labels))))
    all_indices = (
        [f"{anchor_name}_{i}" for i in anchor_indices] + 
        [f"{target_name}_{i}" for i in target_indices]
    )
    
    run_logger.log(f"[phi-unify] Created {len(all_phi)} unified samples with {all_phi.shape[1]} phi features")
    
    # ✅ Return in standard format with cfg-based metadata
    return {
        'embeddings': all_phi.cpu().numpy(),
        'labels': all_labels,
        'coordinates': all_coords,
        'indices': all_indices,
        'phi_unified': True,
        'source_datasets': [anchor_name, target_name],
        'pairing_mode': cfg.pairing_mode,  # ✅ Add cfg info
        'alignment_objective': cfg.alignment_objective,  # ✅ Add cfg info
    }


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

def get_dcca_embeddings(cfg: AlignmentConfig, args, device: torch.device, run_logger: "_RunLogger") -> Dict[str, Dict[str, object]]:
    """
    Extract DCCA datasets preparation logic for early access to dcca_sets.
    Returns dcca_sets dictionary containing projected datasets.
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
    
    for dataset_cfg in cfg.datasets:
        lookup = pn_label_maps.get(dataset_cfg.name)
        if not lookup:  
            print(f"[info] PN labels unavailable for dataset {dataset_cfg.name}")
            continue
        counts_filtered = _count_pn_lookup(lookup, dataset_cfg.region_filter)
        region_desc = dataset_cfg.region_filter if dataset_cfg.region_filter else ["ALL"]
        print(
            "[info] PN label counts for {name} (regions={regions}): pos={pos}, neg={neg}".format(
                name=dataset_cfg.name,
                regions=",".join(region_desc),
                pos=counts_filtered["positive"],
                neg=counts_filtered["negative"],
            )
        )

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
        return {}
    if cfg.alignment_objective.lower() == "dcca" and len(anchor_vecs) < 2:
        print(
            "[warn] Not enough positive overlap pairs to optimise DCCA "
            f"(need at least 2, found {len(anchor_vecs)}). Aborting training."
        )
        return {}

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
    
    return dcca_sets


@torch.no_grad()
def build_cls2_dataset_from_dcca_pairs(
    anchor_vecs,                 # List[Tensor], length = num_pairs
    target_stack_per_anchor,     # List[Tensor], each [M_i, d_B]
    projector_a,                 # nn.Module, A head
    projector_b,                 # nn.Module, B head (AggregatorTargetHead if used)
    device: torch.device,
):
    u_list, v_list, bmiss_list = [], [], []
    for a_emb, b_stack in zip(anchor_vecs, target_stack_per_anchor):
        a = a_emb.to(device).unsqueeze(0)          # [1, dA]
        if hasattr(projector_b, "forward") and b_stack is not None and b_stack.numel() > 0:
            # aggregator uses (anchor, target_stack)
            v = projector_b(a, b_stack.to(device).unsqueeze(0)).squeeze(0)  # [d]
            b_missing = 0.0
        else:
            v = torch.zeros_like(a_emb, device=device)  # fallback
            b_missing = 1.0
        u = projector_a(a).squeeze(0)                   # [d]
        u_list.append(u)
        v_list.append(v)
        bmiss_list.append(torch.tensor([b_missing], device=device, dtype=torch.float32))
    U = torch.stack(u_list)                # [N_A, d]
    V = torch.stack(v_list)                # [N_A, d]
    Bmiss = torch.stack(bmiss_list)        # [N_A, 1]
    return U, V, Bmiss

#####################################################################################   
# run_inference has been moved to inference_module.py
# Use: from . inference_module import run_inference_base

def run_overlap_inference_gAB_from_pairs(
    *,
    anchor_vecs: List[torch.Tensor],
    target_stack_per_anchor: List[torch.Tensor],
    pair_metadata: List[Dict[str, object]],
    projector_a: nn.Module,
    projector_b: nn.Module,      # AggregatorTargetHead if you trained with the transformer aggregator; ProjectionHead otherwise
    gAB: nn.Module,
    device: torch.device,
    cfg: AlignmentConfig,
    output_dir: Path,
    run_logger: "_RunLogger",
    passes: int = 10,
    target_vecs: Optional[List[torch.Tensor]] = None,  # optional fast-path if you already have fused B vectors
    batch_size: int = 256,
    pos_crd: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """
    Inference on overlap using the SAME pairing/aggregation as DCCA:
      U = projector_a(anchor_vec)
      V = projector_b(anchor_vec, target_stack)  # if aggregator-wrapped
        or projector_b(target_vec)               # if plain ProjectionHead and target_vecs provided
    """

    # ----- 1) Build U (A-projected) in batches -----
    run_logger.log(f"[inference-gAB] Preparing A/B projections for {len(anchor_vecs)} overlap pairs")
    A_all = torch.stack([a.detach().clone() for a in anchor_vecs]).to(device)  # [N, dA]

    def _proj_in_batches(X, head):
        outs = []
        for s in range(0, X.size(0), batch_size):
            outs.append(head(X[s:s+batch_size]).detach())
        return torch.cat(outs, dim=0)

    with torch.no_grad():
        projector_a.eval()
        U = _proj_in_batches(A_all, projector_a)  # [N, d]

    # ----- 2) Build V (B-projected) using the SAME path as training -----
    with torch.no_grad():
        projector_b.eval()
        if hasattr(projector_b, "aggregator"):  # AggregatorTargetHead path
            # Use the same helper you used during training to handle variable-length stacks
            V_list = _apply_transformer_target_head(
                projector_b,
                anchor_vecs,
                target_stack_per_anchor,
                pair_metadata,
                batch_size=batch_size,
                device=device,
            )
            if not V_list or len(V_list) != len(anchor_vecs):
                raise RuntimeError("[inference-gAB] Aggregator output mismatch; got "
                                   f"{0 if not V_list else len(V_list)} for N={len(anchor_vecs)}.")
            V = torch.stack([v.detach().to(device) for v in V_list])  # [N, d]
        else:
            # Plain ProjectionHead: expect pre-fused target vectors (same as DCCA input)
            if target_vecs is None or len(target_vecs) != len(anchor_vecs):
                raise RuntimeError("[inference-gAB] target_vecs must be provided (and match N) "
                                   "when projector_b is a plain ProjectionHead.")
            T_all = torch.stack([t.detach().clone() for t in target_vecs]).to(device)  # [N, dB]
            V = _proj_in_batches(T_all, projector_b)  # [N, d]

    # ----- 3) Coordinates & (optional) labels pulled from the DCCA pairing metadata -----
    coords = []
    labels = None
    maybe_labels = []
    for meta in pair_metadata:
        c = meta.get("anchor_coord")
        coords.append(c if c is not None else (None, None))
        # Try to recover labels if present; otherwise keep None
        lb = meta.get("anchor_label", None)
        maybe_labels.append(int(lb) if isinstance(lb, (int, np.integer)) else None)
    if any(l is not None for l in maybe_labels):
        labels = np.array([(-1 if l is None else l) for l in maybe_labels], dtype=np.int32)  # -1 means unknown

    run_logger.log(f"[inference-gAB] Projected shapes: U={tuple(U.shape)}, V={tuple(V.shape)}")

    # ----- 4) MC-Dropout inference on the unified head -----
    model_dir = output_dir / "gAB"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Check if unified model outputs probabilities or logits
    outputs_probs = isinstance(gAB, PNHeadUnified)  # PNHeadUnified outputs probabilities
    
    gAB.train()  # enable dropout
    preds_mc = []

    with torch.no_grad():
        N = U.size(0)
        zeros = torch.zeros(N, 1, device=device)  # B is present → b_missing=0
        for _ in range(int(max(1, passes))):
            batch_out = []
            for s in range(0, N, batch_size):
                u_b = U[s:s+batch_size]
                v_b = V[s:s+batch_size]
                bm_b = zeros[s:s+batch_size]
                
                output = gAB(u_b, v_b, bm_b)  # [B]
                
                if outputs_probs:
                    # ✅ Model already outputs probabilities
                    probs = output
                else:
                    # ✅ Model outputs logits, apply sigmoid
                    probs = torch.sigmoid(output)
                
                batch_out.append(probs.detach().cpu())
            preds_mc.append(torch.cat(batch_out, dim=0).numpy())

    gAB.eval()

    preds_mc = np.stack(preds_mc, axis=0)          # (passes, N)
    mean_pred = preds_mc.mean(axis=0)              # (N,)
    std_pred  = preds_mc.std(axis=0)               # (N,)

    # ----- 5) Save results -----
    np.save(model_dir / "predictions_mean.npy", mean_pred)
    np.save(model_dir / "predictions_std.npy",  std_pred)
    np.save(model_dir / "coordinates.npy",      np.asarray(coords, dtype=object))

    _create_inference_plots(
        model_dir=model_dir,
        model_name="gAB",
        coords=coords,
        mean_pred=mean_pred,
        std_pred=std_pred,
        pos_crd=pos_crd
    )

    summary = _compute_inference_summary(
        model_name="gAB",
        mean_pred=mean_pred,
        std_pred=std_pred,
        labels=labels
    )
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    run_logger.log(f"[inference-gAB] Saved results to {model_dir}")

    return {
        "predictions_mean": mean_pred,
        "predictions_std": std_pred,
        "coordinates": coords,
        "labels": labels,
        "summary": summary,
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
    # sc1 = axes[0].scatter(coords_array[:, 0], coords_array[:, 1], c=mean_pred, cmap='RdYlGn', s=5, marker='H', alpha=1.0, vmin=0, vmax=1)
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


def _read_inference(
    cfg: AlignmentConfig,
    output_dir: Path,
) -> Dict[str, object]:
    """Read saved inference results from output directory."""
    
    # Check for both possible model directories
    model_dirs = [output_dir / "gA", output_dir / "gAB"]
    model_dir = None
    
    for candidate_dir in model_dirs:
        if candidate_dir.exists():
            model_dir = candidate_dir
            break
    
    if model_dir is None:
        raise FileNotFoundError(f"No inference results found in {output_dir}. Expected gA/ or gAB/ subdirectory.")
    
    # Load saved arrays
    try:
        mean_pred = np.load(model_dir / "predictions_mean.npy")
        std_pred = np.load(model_dir / "predictions_std.npy")
        coordinates = np.load(model_dir / "coordinates.npy", allow_pickle=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing inference files in {model_dir}: {e}")
    
    # Load summary if available
    summary_path = model_dir / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except Exception:
            summary = None
    
    return {
        "predictions_mean": mean_pred,
        "predictions_std": std_pred,
        "coordinates": coordinates.tolist() if hasattr(coordinates, 'tolist') else list(coordinates),
        "summary": summary,
        "model_dir": str(model_dir)
    }

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

def _compare_models(
    results: Dict[str, Dict[str, object]],
    output_dir: Path,
    run_logger: "_RunLogger"
) -> None:
    """Compare predictions from gA and gAB models."""
    if plt is None:
        return
    
    gA_pred = results["gA"]["predictions_mean"]
    gAB_pred = results["gAB"]["predictions_mean"]
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot: gA vs gAB predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Direct comparison
    axes[0].scatter(gA_pred, gAB_pred, alpha=0.5, s=30)
    axes[0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0].set_xlabel('gA Predictions')
    axes[0].set_ylabel('gAB Predictions')
    axes[0].set_title('Model Prediction Comparison')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Difference map
    diff = gAB_pred - gA_pred
    axes[1].hist(diff, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', label='No difference')
    axes[1].set_xlabel('Difference (gAB - gA)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Differences Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute comparison metrics
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff**2).mean()))
    corr = float(np.corrcoef(gA_pred, gAB_pred)[0, 1])
    
    comparison_stats = {
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse,
        "correlation": corr,
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std()),
    }
    
    with open(comparison_dir / "comparison_stats.json", 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    run_logger.log(f"[inference] Model comparison: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")

#####################################################################################   

@torch.no_grad()
def infer_in_bc(encA, encB, agg, pA, pB, gAB, xA, xB_set, bmask=None):
    zA = encA(xA.cuda()); u = pA(zA)
    zB = encB(xB_set.view(-1, *xB_set.shape[2:]).cuda()).view(xB_set.size(0), xB_set.size(1), -1)
    bar_zB = agg(zA, zB, key_padding_mask=bmask.cuda() if bmask is not None else None)
    v = pB(bar_zB)
    p = gAB(u, v, torch.zeros(u.size(0),1, device=u.device))
    return p  # PN probability in BC using A+B

@torch.no_grad()
def infer_outside_bc(encA, pA, gA, xA):
    zA = encA(xA.cuda()); u = pA(zA)
    return gA(u)  # PN probability outside BC using A-only

#####################################################################################   

class PNHeadAOnly(nn.Module):
    def __init__(self, d, hidden=256, out_logits=True, prior_pi=None):
        super().__init__()
        self.out_logits = out_logits
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)
        # Optional: initialize bias to the prior logit to avoid all-negative start
        if prior_pi is not None and 0.0 < prior_pi < 1.0:
            with torch.no_grad():
                self.out.bias.fill_(math.log(prior_pi/(1.0-prior_pi)))

    def forward(self, u):
        z = self.out(self.mlp(u)).squeeze(-1)  # logits
        return z if self.out_logits else torch.sigmoid(z)  # toggle if needed

def _ensure_logits(y: torch.Tensor) -> torch.Tensor:
    """
    Accepts network output that might be logits (preferred) or probs in [0,1].
    Returns logits tensor of shape [B].
    """
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    # if looks like probs, convert to logits
    if torch.isfinite(y).all() and y.min() >= 0.0 and y.max() <= 1.0:
        y = torch.logit(y.clamp(1e-6, 1 - 1e-6))
    return y

def nnpu_basic_loss_from_logits(logits_p, logits_u, pi_p: float):
    """
    Classic non-negative PU risk (Kiryo et al., 2017) on logits.
    """
    pos_loss = F.softplus(-logits_p)   # ell^+(z)
    neg_on_p = F.softplus( logits_p)   # ell^-(z) on P
    neg_on_u = F.softplus( logits_u)   # ell^-(z) on U

    R_p = pi_p * pos_loss.mean()
    R_n = torch.clamp(neg_on_u.mean() - pi_p * neg_on_p.mean(), min=0.0)
    return R_p + R_n

def nnpu_weighted_loss_from_logits(
    logits_p, logits_u, pi_p: float,
    w_p: float = 10.0, w_n: float = 1.0,
    focal_gamma: float | None = 1.5,
    prior_penalty: float | None = 5.0
):
    """
    Weighted nnPU risk with optional focalization (positives) and prior matching on U.
    """
    pos_loss = F.softplus(-logits_p)   # ell^+(z)
    neg_on_p = F.softplus( logits_p)   # ell^-(z)
    neg_on_u = F.softplus( logits_u)   # ell^-(z)

    # focalize only the positive term (helps rare positives)
    if focal_gamma is not None:
        with torch.no_grad():
            p_pos = torch.sigmoid(logits_p).clamp_(1e-6, 1-1e-6)
        pos_loss = ((1.0 - p_pos) ** float(focal_gamma)) * pos_loss

    R_p = pi_p * pos_loss.mean()
    R_n = torch.clamp(neg_on_u.mean() - pi_p * neg_on_p.mean(), min=0.0)
    loss = w_p * R_p + w_n * R_n

    # prior-matching penalty to avoid all-negative collapse
    if prior_penalty is not None and prior_penalty > 0:
        p_u = torch.sigmoid(logits_u).mean()
        loss = loss + prior_penalty * (p_u - float(pi_p))**2
    return loss

# Note: TargetSetDataset and _collate_target_sets moved to Common.Unifying.DCCA.method1_phi_TransAgg

#####################################################################################   

# Note: CrossAttentionAggregator and AggregatorTargetHead moved to Common.Unifying.DCCA.method1_phi_TransAgg


def _train_transformer_aggregator(
    anchor_vecs: Sequence[torch.Tensor],
    target_stack_per_anchor: Sequence[torch.Tensor],
    pair_metadata: Sequence[Dict[str, object]],
    *,
    validation_anchor_vecs: Optional[Sequence[torch.Tensor]] = None,
    validation_target_stack: Optional[Sequence[torch.Tensor]] = None,
    validation_metadata: Optional[Sequence[Dict[str, object]]] = None,
    device: torch.device,
    batch_size: int,
    steps: int,
    lr: float,
    agg_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    dcca_eps: float,
    drop_ratio: float,
    use_positional_encoding: bool,
    run_logger: "_RunLogger",
) -> Tuple[bool, Optional[nn.Module], Optional[AggregatorTargetHead], List[Dict[str, object]], int, Optional[str]]:

    if not anchor_vecs or not target_stack_per_anchor:
        failure = "Transformer aggregator requires non-empty anchor/target stacks."
        run_logger.log(f"[agg] {failure}")
        return False, None, None, [], agg_dim, failure

    train_dataset = TargetSetDataset(anchor_vecs, target_stack_per_anchor, pair_metadata)
    if len(train_dataset) == 0:
        failure = "Transformer aggregator received empty training dataset."
        run_logger.log(f"[agg] {failure}")
        return False, None, None, [], agg_dim, failure

    collate_fn = lambda batch: _collate_target_sets(batch, use_positional_encoding)
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    has_validation = (
        validation_anchor_vecs is not None
        and validation_target_stack is not None
        and validation_metadata is not None
        and len(validation_anchor_vecs) > 0
    )
    if has_validation:
        val_dataset = TargetSetDataset(validation_anchor_vecs, validation_target_stack, validation_metadata)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        val_loader = None

    anchor_dim = anchor_vecs[0].numel()
    target_dim = target_stack_per_anchor[0].size(1)
    print(f"[DEBUG] anchor_dim={anchor_dim}, target_dim={target_dim}, agg_dim={agg_dim}, num_layers={num_layers}")
    aggregator = CrossAttentionAggregator(
        anchor_dim,
        target_dim,
        agg_dim,
        num_layers=max(1, num_layers),
        num_heads=max(1, num_heads),
        dropout=dropout,
        use_positional_encoding=use_positional_encoding,
    ).to(device)
    proj_anchor = ProjectionHead(anchor_dim, agg_dim, num_layers=num_layers).to(device)
    proj_target = ProjectionHead(agg_dim, agg_dim, num_layers=num_layers).to(device)
    
    # Debug: Check if models are initialized properly
    agg_has_nan = any(torch.isnan(p).any().item() for p in aggregator.parameters())
    proj_a_has_nan = any(torch.isnan(p).any().item() for p in proj_anchor.parameters())
    proj_t_has_nan = any(torch.isnan(p).any().item() for p in proj_target.parameters())
    print(f"[DEBUG init] aggregator has NaN weights: {agg_has_nan}")
    print(f"[DEBUG init] proj_anchor has NaN weights: {proj_a_has_nan}")
    print(f"[DEBUG init] proj_target has NaN weights: {proj_t_has_nan}")
    
    optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(proj_anchor.parameters()) + list(proj_target.parameters()),
        lr=lr,
    )
    if steps <= 0:
        aggregator.eval()
        proj_anchor.eval()
        proj_target.eval()
        target_head = AggregatorTargetHead(aggregator, proj_target, use_positional_encoding=use_positional_encoding)
        target_head.eval()
        run_logger.log("[agg] No training steps requested; returning aggregator in eval mode.")
        return True, proj_anchor, target_head, [], agg_dim, None

    def _evaluate(loader: Optional[DataLoader]) -> Optional[Dict[str, object]]:
        if loader is None:
            return None
        
        # Debug: Check weights BEFORE setting to eval mode
        if not hasattr(_evaluate, "_checked_before_eval"):
            agg_has_nan = any(torch.isnan(p).any().item() for p in aggregator.parameters())
            proj_a_has_nan = any(torch.isnan(p).any().item() for p in proj_anchor.parameters())
            proj_t_has_nan = any(torch.isnan(p).any().item() for p in proj_target.parameters())
            print(f"[DEBUG eval-weights] BEFORE .eval(): aggregator NaN={agg_has_nan}, proj_anchor NaN={proj_a_has_nan}, proj_target NaN={proj_t_has_nan}")
            _evaluate._checked_before_eval = True
        
        aggregator.eval()
        proj_anchor.eval()
        proj_target.eval()
        losses: List[float] = []
        batches = 0
        singular_store: List[torch.Tensor] = []
        
        # Debug: Check weights after setting to eval mode
        if not hasattr(_evaluate, "_checked_eval_weights"):
            agg_has_nan = any(torch.isnan(p).any().item() for p in aggregator.parameters())
            proj_a_has_nan = any(torch.isnan(p).any().item() for p in proj_anchor.parameters())
            proj_t_has_nan = any(torch.isnan(p).any().item() for p in proj_target.parameters())
            print(f"[DEBUG eval-weights] AFTER .eval(): aggregator NaN={agg_has_nan}, proj_anchor NaN={proj_a_has_nan}, proj_target NaN={proj_t_has_nan}")
            _evaluate._checked_eval_weights = True
        
        with torch.no_grad():
            for anchor_batch, target_batch, mask_batch, pos_batch in loader:
                anchor_batch = anchor_batch.to(device)
                target_batch = target_batch.to(device)
                mask_batch = mask_batch.to(device)
                if pos_batch is not None:
                    pos_batch = pos_batch.to(device)
                fused = aggregator(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
                u = proj_anchor(anchor_batch)
                v = proj_target(fused)
                
                # Normalize projections to prevent numerical instability
                u = torch.nn.functional.normalize(u, p=2, dim=1)
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                
                # Debug: log first batch details
                # if batches == 0:
                #     print(f"[DEBUG _evaluate] Batch 1: anchor_batch.shape={anchor_batch.shape}, target_batch.shape={target_batch.shape}")
                #     print(f"[DEBUG _evaluate] Batch 1: anchor_batch stats: mean={anchor_batch.mean().item():.6f}, std={anchor_batch.std().item():.6f}")
                #     print(f"[DEBUG _evaluate] Batch 1: anchor_batch has NaN: {torch.isnan(anchor_batch).any().item()}, Inf: {torch.isinf(anchor_batch).any().item()}")
                #     print(f"[DEBUG _evaluate] Batch 1: target_batch stats: mean={target_batch.mean().item():.6f}, std={target_batch.std().item():.6f}")
                #     print(f"[DEBUG _evaluate] Batch 1: target_batch has NaN: {torch.isnan(target_batch).any().item()}, Inf: {torch.isinf(target_batch).any().item()}")
                #     print(f"[DEBUG _evaluate] Batch 1: fused.shape={fused.shape}")
                #     print(f"[DEBUG _evaluate] Batch 1: fused stats: mean={fused.mean().item():.6f}, std={fused.std().item():.6f}")
                #     print(f"[DEBUG _evaluate] Batch 1: fused has NaN: {torch.isnan(fused).any().item()}, Inf: {torch.isinf(fused).any().item()}")
                #     print(f"[DEBUG _evaluate] Batch 1: u.shape={u.shape}, v.shape={v.shape}")
                #     print(f"[DEBUG _evaluate] Batch 1: u stats: mean={u.mean().item():.6f}, std={u.std().item():.6f}, min={u.min().item():.6f}, max={u.max().item():.6f}")
                #     print(f"[DEBUG _evaluate] Batch 1: v stats: mean={v.mean().item():.6f}, std={v.std().item():.6f}, min={v.min().item():.6f}, max={v.max().item():.6f}")
                #     print(f"[DEBUG _evaluate] Batch 1: u has NaN: {torch.isnan(u).any().item()}, v has NaN: {torch.isnan(v).any().item()}")
                #     print(f"[DEBUG _evaluate] Batch 1: u has Inf: {torch.isinf(u).any().item()}, v has Inf: {torch.isinf(v).any().item()}")
                loss, singulars, loss_info = dcca_loss(u, v, eps=dcca_eps, drop_ratio=drop_ratio)
                losses.append(loss.item())
                batches += 1
                # Debug: log first batch results
                # if batches == 1:
                #     print(f"[DEBUG _evaluate] Batch 1: loss={loss.item()}, singulars.numel()={singulars.numel()}")
                #     print(f"[DEBUG _evaluate] Batch 1: loss_info={loss_info}")
                if singulars.numel() > 0:
                    singular_store.append(singulars.detach().cpu())
        if batches == 0:
            return None
        mean_loss = float(sum(losses) / batches)
        if singular_store:
            compiled = torch.cat(singular_store)
            mean_corr = float(compiled.mean().item())
            tcc_sum = float(compiled.sum().item())
            tcc_mean = float(compiled.mean().item())
            k_val = int(compiled.numel())
        else:
            mean_corr = None
            tcc_sum = None
            tcc_mean = None
            k_val = 0
        aggregator.train()
        proj_anchor.train()
        return {
            "loss": mean_loss,
            "mean_corr": mean_corr,
            "tcc_sum": tcc_sum,
            "tcc_mean": tcc_mean,
            "k": k_val,
            "batches": batches,
        }

    def _format(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "None"
        return f"{value:.6f}"

    epoch_history: List[Dict[str, object]] = []
    failure_reason: Optional[str] = None

    for epoch_idx in range(steps):
        aggregator.train()
        proj_anchor.train()
        proj_target.train()
        epoch_loss = 0.0
        epoch_batches = 0
        had_nan_grad = False
        ihad_nan_grad = 0
        for anchor_batch, target_batch, mask_batch, pos_batch in train_loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            if pos_batch is not None:
                pos_batch = pos_batch.to(device)
            fused = aggregator(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
            u = proj_anchor(anchor_batch)
            v = proj_target(fused)
            
            # Normalize projections to prevent numerical instability
            u = torch.nn.functional.normalize(u, p=2, dim=1)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            
            # Debug: check first training batch
            if epoch_batches == 0 and epoch_idx == 0:
                print(f"[DEBUG train] TRAINING Epoch {epoch_idx+1}, Batch 1:")
                print(f"[DEBUG train]   fused has NaN: {torch.isnan(fused).any().item()}")
                print(f"[DEBUG train]   u has NaN: {torch.isnan(u).any().item()}, u norm: {u.norm(dim=1).mean().item():.6f}")
                print(f"[DEBUG train]   v has NaN: {torch.isnan(v).any().item()}, v norm: {v.norm(dim=1).mean().item():.6f}")
            loss, _, _ = dcca_loss(u, v, eps=dcca_eps, drop_ratio=drop_ratio)
            # Debug: check loss and gradients
            if epoch_batches == 0 and epoch_idx == 0:
                print(f"[DEBUG train]   loss value: {loss.item()}")
                print(f"[DEBUG train]   loss has NaN: {torch.isnan(loss).any().item()}")
                print(f"[DEBUG train]   loss requires_grad: {loss.requires_grad}")
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients and skip update if found
            has_nan_grad = False
            for param in itertools.chain(
                aggregator.parameters(),
                proj_anchor.parameters(),
                proj_target.parameters()
            ):
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    had_nan_grad = True
                    ihad_nan_grad += 1
                    break
            
            if has_nan_grad:
                optimizer.zero_grad()  # Clear the bad gradients
                epoch_batches += 1
                continue
            
            # Debug: check gradients after backward
            if epoch_batches == 0 and epoch_idx == 0:
                total_grad_norm = 0.0
                nan_grad_count = 0
                for name, param in itertools.chain(
                    aggregator.named_parameters(),
                    proj_anchor.named_parameters(),
                    proj_target.named_parameters()
                ):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_grad_count += 1
                        total_grad_norm += param.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"[DEBUG train]   gradients: total_norm={total_grad_norm:.6f}, nan_count={nan_grad_count}")
            torch.nn.utils.clip_grad_norm_(
                list(aggregator.parameters()) + list(proj_anchor.parameters()) + list(proj_target.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            # Debug: check weights after optimizer step
            if epoch_batches == 0 and epoch_idx == 0:
                agg_has_nan = any(torch.isnan(p).any().item() for p in aggregator.parameters())
                proj_a_has_nan = any(torch.isnan(p).any().item() for p in proj_anchor.parameters())
                proj_t_has_nan = any(torch.isnan(p).any().item() for p in proj_target.parameters())
                print(f"[DEBUG train]   After step: aggregator NaN={agg_has_nan}, proj_anchor NaN={proj_a_has_nan}, proj_target NaN={proj_t_has_nan}")
            epoch_loss += loss.item()
            epoch_batches += 1


        if had_nan_grad:
            run_logger.log(f"[Trans-AGG-DCCA] INFO: NaN gradients detected in epoch {epoch_idx+1}. Corresponding {ihad_nan_grad} batches skipped among total {epoch_batches} batches.")

        train_metrics = _evaluate(train_eval_loader)
        val_metrics = _evaluate(val_loader) if has_validation else None
        
        # # Debug: Print what _evaluate returned
        # run_logger.log(f"[DEBUG] train_metrics = {train_metrics}")
        # run_logger.log(f"[DEBUG] val_metrics = {val_metrics}")

        train_loss_log = train_metrics.get("loss") if train_metrics else epoch_loss / max(1, epoch_batches)
        train_corr_log = train_metrics.get("mean_corr") if train_metrics else None
        train_tcc_log = train_metrics.get("tcc_sum") if train_metrics else None
        train_tcc_mean_log = train_metrics.get("tcc_mean") if train_metrics else None
        train_k_log = train_metrics.get("k") if train_metrics else None
        val_loss_log = val_metrics.get("loss") if val_metrics else None
        val_corr_log = val_metrics.get("mean_corr") if val_metrics else None
        val_tcc_log = val_metrics.get("tcc_sum") if val_metrics else None
        val_tcc_mean_log = val_metrics.get("tcc_mean") if val_metrics else None
        val_k_log = val_metrics.get("k") if val_metrics else None
        train_batches_count = train_metrics.get("batches") if train_metrics else epoch_batches
        val_batches_count = val_metrics.get("batches") if val_metrics else None
        run_logger.log(
            "[Trans-AGG-DCCA] epoch {epoch}: train_loss={train_loss}, train_mean_corr={train_corr}, "
            "[Trans-AGG-DCCA] train_TCC = {train_tcc}, val_loss={val_loss}, val_mean_corr={val_corr}, "
            "[Trans-AGG-DCCA] val_TCC = {val_tcc}, batches={batches}".format(
                epoch=epoch_idx + 1,
                train_loss=_format(train_loss_log), train_corr=_format(train_corr_log), train_tcc=_format(train_tcc_log),
                val_loss=_format(val_loss_log),     val_corr=_format(val_corr_log),     val_tcc=_format(val_tcc_log),
                batches=train_batches_count,
            )
        )

        epoch_history.append(
            {
                "epoch": epoch_idx + 1,
                "loss": float(train_loss_log) if train_loss_log is not None else None,
                "mean_correlation": float(train_corr_log) if train_corr_log is not None else None,
                "train_eval_loss": float(train_loss_log) if train_loss_log is not None else None,
                "train_eval_mean_correlation": float(train_corr_log) if train_corr_log is not None else None,
                "train_eval_tcc": float(train_tcc_log) if train_tcc_log is not None else None,
                "train_eval_tcc_mean": float(train_tcc_mean_log) if train_tcc_mean_log is not None else None,
                "train_eval_k": float(train_k_log) if train_k_log is not None else None,
                "val_eval_loss": float(val_loss_log) if val_loss_log is not None else None,
                "val_eval_mean_correlation": float(val_corr_log) if val_corr_log is not None else None,
                "val_eval_tcc": float(val_tcc_log) if val_tcc_log is not None else None,
                "val_eval_tcc_mean": float(val_tcc_mean_log) if val_tcc_mean_log is not None else None,
                "val_eval_k": float(val_k_log) if val_k_log is not None else None,
                "batches": int(train_batches_count),
                "val_batches": int(val_batches_count) if val_batches_count is not None else None,
                "projection_dim": int(agg_dim),
            }
        )
    aggregator.eval()
    proj_anchor.eval()
    proj_target.eval()
    target_head = AggregatorTargetHead(aggregator, proj_target, use_positional_encoding=use_positional_encoding)
    target_head.eval()
    return True, proj_anchor, target_head, epoch_history, agg_dim, failure_reason


def _apply_transformer_target_head(
    target_head: AggregatorTargetHead,
    anchor_vecs: Sequence[torch.Tensor],
    target_stack_per_anchor: Sequence[torch.Tensor],
    pair_metadata: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    dataset = TargetSetDataset(anchor_vecs, target_stack_per_anchor, pair_metadata)
    collate_fn = lambda batch: _collate_target_sets(batch, target_head.aggregator.use_positional_encoding)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, drop_last=False, collate_fn=collate_fn)
    fused_outputs: List[torch.Tensor] = []
    target_head.eval()
    with torch.no_grad():
        for anchor_batch, target_batch, mask_batch, pos_batch in loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            if pos_batch is not None:
                pos_batch = pos_batch.to(device)
            output = target_head(anchor_batch, target_batch, key_padding_mask=mask_batch, pos_encoding=pos_batch)
            fused_outputs.extend(output.detach().cpu())
    return fused_outputs

#####################################################################################   


def _dataset_pair(cfg: AlignmentConfig) -> Tuple[str, str]:
    if len(cfg.datasets) < 2:
        raise ValueError("Overlap alignment requires at least two datasets.")
    return cfg.datasets[0].name, cfg.datasets[1].name


def _compute_overlap_debug(
    pairs: Sequence[OverlapAlignmentPair],
    anchor_name: str,
    target_name: str,
) -> Dict[str, int]:
    overlap_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
    positive_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
    overlapping_positive_pairs = 0
    for pair in pairs:
        for dataset_name, record in (
            (pair.anchor_dataset, pair.anchor_record),
            (pair.target_dataset, pair.target_record),
        ):
            marker = record.tile_id if record.tile_id is not None else record.index
            overlap_tiles.setdefault(dataset_name, set()).add(marker)
            if int(record.label) == 1:
                positive_tiles.setdefault(dataset_name, set()).add(marker)
        if int(pair.anchor_record.label) == 1 and int(pair.target_record.label) == 1:
            overlapping_positive_pairs += 1
    return {
        "anchor_positive": len(positive_tiles.get(anchor_name, set())),
        "anchor_overlap": len(overlap_tiles.get(anchor_name, set())),
        "target_positive": len(positive_tiles.get(target_name, set())),
        "target_overlap": len(overlap_tiles.get(target_name, set())),
        "positive_pairs": overlapping_positive_pairs,
    }

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
