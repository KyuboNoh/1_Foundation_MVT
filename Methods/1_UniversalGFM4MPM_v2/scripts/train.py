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

from .config import AlignmentConfig, load_config
from Common.Unifying.Labels_TwoDatasets.datasets import (
    auto_coord_error,
    load_embedding_records,
    summarise_records,
    _load_pn_lookup,
    _count_pn_lookup, 
    _MaskStack,
)
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from Common.Unifying.Labels_TwoDatasets import (
    _normalise_cross_matches,
    _prepare_classifier_labels,
    _build_aligned_pairs_OneToOne,
    _build_aligned_pairs_SetToSet,
    _normalise_coord,
    _normalise_row_col,
    _apply_projector_based_PNlabels,
    _apply_projector_based_PUNlabels,
    _subset_classifier_sample,
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
    _resolve_dcca_weights_path,
    _train_DCCA,
    _prepare_output_dir,
    _persist_state,
    _persist_metrics,
    _maybe_save_debug_figures,
    _create_debug_alignment_figures,
    reembedding_DCCA,
    dcca_loss,
    ProjectionHead
)
from Common.cls.infer.infer_maps import (group_coords, mc_predict_map_from_embeddings, write_prediction_outputs,)
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

import torch, torch.nn as nn, torch.nn.functional as F

INFERENCE_BATCH_SIZE = 2048


def _progress_iter(iterable, desc: str, *, leave: bool = False, total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave, total=total)

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
    parser.add_argument("--run-inference", action=argparse.BooleanOptionalAction, default=False, help="Run inference on aligned datasets after training (default: false).",)
    parser.add_argument("--read-dcca",  action=argparse.BooleanOptionalAction, default=False,help="Load existing DCCA projection head weights before training (default: false).",)
    parser.add_argument("--dcca-weights-path", type=str, default=None, help="Optional path to a saved DCCA checkpoint used when --read-dcca is enabled.",)

    parser.add_argument("--train-cls-1", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Non-Overlapping part.",)
    parser.add_argument("--train-cls-2", action=argparse.BooleanOptionalAction, default=False, help="Train the PN classifier for Overlapping part.",)
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes for classifier MLP heads (space-separated).",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied to classifier MLP layers.",
    )
    parser.add_argument(
        "--mlp-dropout-passes",
        type=int,
        default=5,
        help="Number of Monte Carlo dropout passes for uncertainty estimation in classifier inference.",
    )

    parser.add_argument( "--use-transformer-aggregator", action=argparse.BooleanOptionalAction, default=True, help="Enable transformer-based aggregation before DCCA (set-to-set pairing only).",)
    parser.add_argument("--agg-trans-num-layers", type=int, default=4, help="Number of cross-attention layers in the aggregator.")
    parser.add_argument("--agg-trans-num-heads", type=int, default=4, help="Number of attention heads in the aggregator.")
    parser.add_argument("--agg-trans-dropout", type=float, default=0.1, help="Dropout used inside the transformer aggregator.")
    parser.add_argument("--agg-trans-pos-enc", action=argparse.BooleanOptionalAction, default=False, help="Use positional encoding based on anchor/target coordinate differences.",)
    # parser.add_argument("--dcca-eps", type=float, default=1e-5, help="Epsilon value for DCCA covariance regularization (default: 1e-5).")
    # parser.add_argument("--singular-value-drop-ratio", type=float, default=0.01, help="Ratio threshold for dropping small singular values in DCCA (default: 0.01).")
    # parser.add_argument("--tcc-ratio", type=float, default=1.0, help="Fraction of canonical correlations to include when computing TCC (0 < ratio <= 1).", )
    # parser.add_argument("--dcca-mlp-layers", type=int, default=4, help="Number of linear layers used in each DCCA projection head MLP (default: 4).",)
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
    debug_overlap_stats = _compute_overlap_debug(pairs, anchor_name, target_name) if debug_mode else None
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

    gA = None
    gAB = None  #Temp
    inference_results = {}
    #################################### Preprocess - Augmenting positive labels ####################################
    augmentation_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    augmentation_loaded_counts: Dict[str, int] = {}
    if cfg.use_positive_augmentation:
        for dataset_cfg in cfg.datasets:
            if dataset_cfg.pos_aug_path is None:
                continue
            aug_map, aug_count = _load_augmented_embeddings(dataset_cfg.pos_aug_path, dataset_cfg.region_filter)
            if aug_map:
                augmentation_by_dataset[dataset_cfg.name] = aug_map
                augmentation_loaded_counts[dataset_cfg.name] = aug_count
                print(
                    f"[info] Loaded {aug_count} augmented embeddings for dataset {dataset_cfg.name} "
                    f"from {dataset_cfg.pos_aug_path}"
                )
                print(
                    f"[warn] No augmented embeddings retained for dataset {dataset_cfg.name}; "
                    "region filter or bundle contents may have excluded all entries."
                )

    #################################### Preprocess - Alignment two overlapping data - getting embeddings (pairs) and labels for DCCA & cls ####################################
    pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]] = {
        dataset_cfg.name: _load_pn_lookup(dataset_cfg.pn_split_path) for dataset_cfg in cfg.datasets
    }
    dataset_region_filters: Dict[str, Optional[List[str]]] = {
        dataset_cfg.name: dataset_cfg.region_filter for dataset_cfg in cfg.datasets
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

    pairs_by_region = defaultdict(lambda: Counter())
    for meta in pair_metadata:
        anchor_region = meta.get("anchor_region", "UNKNOWN")
        for label in meta["target_labels"]:
            pairs_by_region[anchor_region][label] += 1

    if not anchor_vecs:
        print("[warn] No qualifying overlap groups were found for training.")
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
        return
    if cfg.alignment_objective.lower() == "dcca" and len(anchor_vecs) < 2:
        print(
            "[warn] Not enough positive overlap pairs to optimise DCCA "
            f"(need at least 2, found {len(anchor_vecs)}). Aborting training."
        )
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
        return

    if cfg.use_positive_augmentation:
        _print_augmentation_usage(
            anchor_name,
            target_name,
            augmentation_loaded_counts,
            augmentation_stats,
            total_pairs=len(anchor_vecs),
        )

    if debug_overlap_stats is not None:
        print(
            "[debug] overlap positives: "
            f"{anchor_name}={debug_overlap_stats['anchor_positive']} / {debug_overlap_stats['anchor_overlap']}, "
            f"{target_name}={debug_overlap_stats['target_positive']} / {debug_overlap_stats['target_overlap']}, "
            f"positive-overlap-pairs={debug_overlap_stats['positive_pairs']}"
        )
    #################################### Preprocess - Alignment two overlapping data - getting embeddings (pairs) and labels for DCCA & cls ####################################

    #################################### Training Overlap Alignment (Transformer-based Aggregation DCCA) ####################################
    if args.train_dcca:
        objective = cfg.alignment_objective.lower()
        # if objective not in {"dcca", "barlow"}:
        #     raise ValueError(f"Unsupported alignment objective: {objective}")
        if objective != "dcca":
            raise NotImplementedError("Only DCCA objective supports adaptive projection control.")

        drop_ratio_source = getattr(cfg.dcca_training, "singular_value_drop_ratio", None)
        if drop_ratio_source is None:
            drop_ratio_source = getattr(args, "singular_value_drop_ratio", None)
        if drop_ratio_source is None:
            drop_ratio_source = 0.01
        try:
            drop_ratio = float(drop_ratio_source)
        except (TypeError, ValueError):
            drop_ratio = 0.01
        if not math.isfinite(drop_ratio):
            drop_ratio = 0.01
        drop_ratio = max(min(drop_ratio, 1.0), 0.0)

        tcc_ratio_source = getattr(cfg.dcca_training, "tcc_ratio", None)
        if tcc_ratio_source is None:
            tcc_ratio_source = getattr(args, "tcc_ratio", None)
        if tcc_ratio_source is None:
            tcc_ratio_source = 1.0
        try:
            tcc_ratio = float(tcc_ratio_source)
        except (TypeError, ValueError):
            tcc_ratio = 1.0
        if not math.isfinite(tcc_ratio) or tcc_ratio <= 0.0:
            tcc_ratio = 1.0
        tcc_ratio = min(tcc_ratio, 1.0)

        mlp_layers_source = None
        if isinstance(getattr(cfg.dcca_training, "extras", None), dict):
            mlp_layers_source = cfg.dcca_training.extras.get("dcca_mlp_layers")
        if mlp_layers_source is None:
            mlp_layers_source = getattr(args, "dcca_mlp_layers", None)
        if mlp_layers_source is None:
            mlp_layers_source = 4
        mlp_layers = int(mlp_layers_source)
        if mlp_layers < 1:
            mlp_layers = 1

        dcca_eps_source = getattr(cfg.dcca_training, "dcca_eps", None)
        if dcca_eps_source is None:
            dcca_eps_source = getattr(args, "dcca_eps", None)
        if dcca_eps_source is None:
            dcca_eps_source = 1e-5
        try:
            dcca_eps_value = float(dcca_eps_source)
        except (TypeError, ValueError):
            dcca_eps_value = 1e-5

        total_pairs = len(anchor_vecs)
        validation_split = cfg.dcca_training.validation_fraction
        if not math.isfinite(validation_split):
            validation_split = 0.0
        validation_split = max(0.0, min(validation_split, 0.9))
        DCCA_val_count = int(total_pairs * validation_split)

        minimum_train = 2
        if total_pairs - DCCA_val_count < minimum_train and total_pairs >= minimum_train:
            DCCA_val_count = max(0, total_pairs - minimum_train)
        if DCCA_val_count < 2:
            DCCA_val_count = 0

        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))
        indices = torch.randperm(total_pairs, generator=generator)
        val_indices = indices[:DCCA_val_count] if DCCA_val_count > 0 else torch.empty(0, dtype=torch.long)
        train_indices = indices[DCCA_val_count:]
        if train_indices.numel() < minimum_train and val_indices.numel() > 0:
            needed = minimum_train - train_indices.numel()
            recover = val_indices[:needed]
            train_indices = torch.cat([train_indices, recover]) if train_indices.numel() else recover
            val_indices = val_indices[needed:]
        if train_indices.numel() < minimum_train:
            raise RuntimeError("Unable to allocate at least two samples for DCCA training after validation split.")

        train_indices_list = train_indices.tolist()
        val_indices_list = val_indices.tolist()
        for idx in train_indices_list:
            pair_metadata[idx]["split"] = "train"
        for idx in val_indices_list:
            pair_metadata[idx]["split"] = "val"

        anchor_vecs_train = [anchor_vecs[i] for i in train_indices_list]
        target_stack_train = [target_stack_per_anchor[i] for i in train_indices_list]

        metadata_train = [pair_metadata[i] for i in train_indices_list]

        anchor_vecs_val = [anchor_vecs[i] for i in val_indices_list] if val_indices_list else []
        target_stack_val = [target_stack_per_anchor[i] for i in val_indices_list] if val_indices_list else []
        metadata_val = [pair_metadata[i] for i in val_indices_list] if val_indices_list else []

        actual_validation_fraction = val_indices.numel() / total_pairs if val_indices.numel() else 0.0

        pre_fusion_anchor_tensor = (torch.stack([vec.detach().clone() for vec in anchor_vecs]) if anchor_vecs else torch.empty((0,)))
        pre_fusion_target_tensor = (torch.stack([vec.detach().clone() for vec in target_vecs]) if target_vecs else torch.empty((0,)))
        fused_target_vecs: List[torch.Tensor] = list(target_vecs)
        agg_epoch_history: List[Dict[str, object]] = []
        agg_proj_dim = cfg.projection_dim or (target_vecs[0].numel() if target_vecs else 0)
        agg_failure_reason: Optional[str] = None
        projector_a_head: Optional[nn.Module] = nn.Identity()
        target_head: Optional[nn.Module] = nn.Identity()
        agg_success = True

        if use_transformer_agg and target_stack_per_anchor:
            agg_num_layers = int(args.agg_trans_num_layers) if args.agg_trans_num_layers is not None else 4
            agg_num_layers = max(1, agg_num_layers)
            agg_num_heads = int(args.agg_trans_num_heads) if args.agg_trans_num_heads is not None else 4
            agg_num_heads = max(1, agg_num_heads)
            agg_steps_raw = getattr(args, "agg_trans_steps", None)
            if agg_steps_raw is None:
                agg_steps_raw = getattr(cfg.dcca_training, "epochs", 0)
            try:
                agg_steps = int(agg_steps_raw)
            except (TypeError, ValueError):
                agg_steps = getattr(cfg.dcca_training, "epochs", 0)
            agg_steps = max(0, agg_steps)
            agg_dropout = max(0.0, float(getattr(args, "agg_trans_dropout", 0.1)))
            train_batch_size = min(max(1, cfg.dcca_training.batch_size), len(anchor_vecs_train)) if anchor_vecs_train else 1

            if cfg.projection_dim % agg_num_heads != 0:
                cfg.projection_dim = agg_num_heads * max(1, math.ceil(cfg.projection_dim / agg_num_heads))
                run_logger.log(f"[agg] Adjusted projection_dim from {old_dim} to {cfg.projection_dim} (must be divisible by {agg_num_heads} heads)")
            try:
                agg_success = False
                projector_a_head = None
                projector_b_head = None
                agg_epoch_history = []
                fused_target_vecs = []
                agg_proj_dim = cfg.projection_dim
                run_logger.log(
                    "[agg] Training transformer aggregator (dimension = {dim}, layers={layers}, heads={heads}, steps={steps}, "
                    "batch_size={batch}, dropout={drop:.3f})".format(
                        dim=cfg.projection_dim,
                        layers=agg_num_layers,
                        heads=agg_num_heads,
                        steps=agg_steps,
                        batch=train_batch_size,
                        drop=agg_dropout,
                    )
                )
                # TODO: cfg.projection_dim is the critical parameter here; it often leads to issues for 0-losses 
                agg_success, projector_a_head, projector_b_head, agg_epoch_history, agg_proj_dim, agg_failure_reason = _train_transformer_aggregator(
                    anchor_vecs_train,
                    target_stack_train,
                    metadata_train,
                    validation_anchor_vecs=anchor_vecs_val,
                    validation_target_stack=target_stack_val,
                    validation_metadata=metadata_val,
                    device=device,
                    batch_size=train_batch_size,
                    steps=agg_steps,
                    lr=cfg.dcca_training.lr,
                    agg_dim=cfg.projection_dim,
                    num_layers=agg_num_layers,
                    num_heads=agg_num_heads,
                    dropout=agg_dropout,
                    dcca_eps=dcca_eps_value,
                    drop_ratio=drop_ratio,
                    use_positional_encoding=bool(getattr(args, "agg_trans_pos_enc", False)),
                    run_logger=run_logger,
                )
                if not agg_success or projector_a_head is None or projector_b_head is None:
                    run_logger.log(f"[agg] Transformer aggregator training failed: {agg_failure_reason or 'unknown error'}")
                else:
                    agg_success = True
                    fused_outputs = _apply_transformer_target_head(
                        projector_b_head,
                        anchor_vecs,
                        target_stack_per_anchor,
                        pair_metadata,
                        batch_size=train_batch_size,
                        device=device,
                    )
                    if fused_outputs and len(fused_outputs) == len(target_vecs):
                        fused_target_vecs = fused_outputs
                        target_vecs = fused_outputs
                        run_logger.log(f"[agg] Transformer aggregator applied to {len(target_vecs)} pairs (dim={cfg.projection_dim}).")
                    else:
                        run_logger.log("[agg] Transformer aggregator output size mismatch; skipping fused targets.")
                if agg_success and projector_b_head is not None and agg_epoch_history:
                    run_logger.log(
                        "[agg] Completed transformer aggregator training: final_loss={loss}, final_mean_corr={corr}".format(
                            loss=_format_optional_scalar(agg_epoch_history[-1].get("train_eval_loss")),
                            corr=_format_optional_scalar(agg_epoch_history[-1].get("train_eval_mean_correlation")),
                        )
                    )
            except Exception as exc:
                run_logger.log(f"[agg] Transformer aggregator failed: {exc}")


        last_history_agg = agg_epoch_history[-1] if agg_epoch_history else {}
        trans_agg_dcca_summary = {
            "objective": objective,
            "aggregator": cfg.aggregator,
            "use_positive_only": cfg.use_positive_only,
            "use_positive_augmentation": cfg.use_positive_augmentation,
            "projection_dim": agg_proj_dim,
            "projection_layers": mlp_layers,
            "train_dcca": train_dcca,
            "read_dcca": read_dcca,
            "dcca_weights_path": str(weights_path) if weights_path is not None else None,
            "pretrained_projection_dim": int(pretrained_summary.get("projection_dim"))
            if isinstance(pretrained_summary, dict) and pretrained_summary.get("projection_dim") is not None
            else None,
            "pretrained_projection_layers": int(pretrained_summary.get("projection_layers"))
            if isinstance(pretrained_summary, dict) and pretrained_summary.get("projection_layers") is not None
            else None,
            # "label_histogram": dict(label_hist),
            "num_pairs": len(anchor_vecs),
            "epochs": cfg.dcca_training.epochs,
            "batch_size": train_batch_size,
            "lr": cfg.dcca_training.lr,
            "max_coord_error": max_coord_error,
            "final_loss": last_history_agg.get("train_eval_loss", last_history_agg.get("loss")),
            "final_mean_correlation": last_history_agg.get("train_eval_mean_correlation", last_history_agg.get("mean_correlation")),
            "final_train_tcc": last_history_agg.get("train_eval_tcc"),
            "final_train_tcc_mean": last_history_agg.get("train_eval_tcc_mean"),
            "final_train_tcc_k": int(last_history_agg["train_eval_k"]) if last_history_agg.get("train_eval_k") is not None else None,
            "train_pairs": len(train_indices_list),
            "validation_pairs": len(val_indices_list) if val_indices_list is not None else 0,
            "validation_fraction": actual_validation_fraction,
            "tcc_ratio": tcc_ratio,
            "dcca_eps": dcca_eps_value,
            "singular_value_drop_ratio": drop_ratio,
            "augmentation_stats": augmentation_stats,
            # "pn_index_summary": pn_index_summary,
            # "pn_label_counts": {
            #     "anchor": _count_pn_lookup(pn_label_maps.get(anchor_name), dataset_region_filters.get(anchor_name)),
            #     "target": _count_pn_lookup(pn_label_maps.get(target_name), dataset_region_filters.get(target_name)),
            # },
            # "cross_index_matches": label_cross_matches,
        }

        if not agg_success or projector_a_head is None or projector_b_head is None:
            raise RuntimeError(f"Transformer aggregator training failed: {agg_failure_reason or 'unknown error'}")

        _persist_state(cfg, projector_a_head, projector_b_head, filename="overlap_alignment_stage1_dcca.pt")
        _persist_metrics(
            cfg,
            trans_agg_dcca_summary,
            agg_epoch_history,
            filename="overlap_alignment_stage1_dcca_metrics.json",
        )
        projector_a = projector_a_head
        projector_b = projector_b_head

        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)

            anchor_pre_tensor = pre_fusion_anchor_tensor
            anchor_post_tensor = _project_in_batches(
                anchor_pre_tensor,
                projector_a_head,
                device,
                max(32, cfg.dcca_training.batch_size),
            )
            target_pre_tensor = pre_fusion_target_tensor
            if fused_target_vecs:
                target_post_tensor = torch.stack([vec.detach().clone() for vec in fused_target_vecs])
            else:
                target_post_tensor = target_pre_tensor

            _create_debug_alignment_figures(
                cfg=cfg,
                projector_a=nn.Identity(),
                projector_b=nn.Identity(),
                anchor_tensor=anchor_pre_tensor,
                target_tensor=target_pre_tensor,
                anchor_tensor_post=anchor_post_tensor,
                target_tensor_post=target_post_tensor,
                pair_metadata=pair_metadata,
                run_logger=run_logger,
                drop_ratio=drop_ratio,
                tcc_ratio=tcc_ratio,
                dcca_eps=dcca_eps_value,
                device=device,
                sample_seed=int(cfg.seed),
            )
    #################################### Training Overlap Alignment (Transformer-based Aggregation DCCA) END ####################################

    #################################### Training Classifier after Overlap Alignment ####################################
    if args.train_cls_1 or args.train_cls_2:
        
        #################################### Training Cls - 0) GET Training data ready (START) ####################################
        if not train_dcca:
            # Ensure projection heads are available when skipping fresh training.
            projector_missing = ('projector_a' not in locals() or projector_a is None or
                                 'projector_b' not in locals() or projector_b is None)
            weights_candidate = _resolve_dcca_weights_path(cfg, args.dcca_weights_path)
            if projector_missing:
                if weights_candidate is None or not weights_candidate.exists():
                    raise FileNotFoundError(
                        "DCCA projection heads are required for classifier training but no weights were found. "
                        "Provide --dcca-weights-path or ensure overlap_alignment_stage1.pt exists."
                    )
                run_logger.log(f"[cls] Loading DCCA projection heads from {weights_candidate}")
                (state_a_dict, state_b_dict), _ = _load_pretrained_dcca_state(weights_candidate)
                projector_a = _projection_head_from_state(state_a_dict).to(device)
                # projector_b = _projection_head_from_state(state_b_dict).to(device)

                # projector_b may be either a plain ProjectionHead or an AggregatorTargetHead when a transformer aggregator was used during stage-1. Detect that case by
                # checking for 'aggregator.' keys in the saved state dict and reconstruct the AggregatorTargetHead so it can be called with (anchor_batch, target_batch).
                if any(k.startswith("aggregator.") for k in state_b_dict.keys()):
                    run_logger.log("[cls] Detected aggregator-wrapped projection head for target dataset; reconstructing AggregatorTargetHead.")
                    # Infer agg_dim from proj_target first linear weight if available
                    if "proj_target.net.0.weight" in state_b_dict:
                        agg_dim = int(state_b_dict["proj_target.net.0.weight"].size(1))
                    else:
                        # fallback: use last linear weight's output dim if present
                        weight_keys = [k for k in state_b_dict.keys() if k.endswith(".weight")]
                        last_w = state_b_dict[sorted(weight_keys)[-1]] if weight_keys else None
                        agg_dim = int(last_w.size(0)) if last_w is not None and hasattr(last_w, "size") else cfg.projection_dim

                    # infer whether positional encoding was used by comparing key_proj input dim
                    key_proj = state_b_dict.get("aggregator.key_proj.weight")
                    target_dim = None
                    try:
                        target_dim = int(target_stack_per_anchor[0].size(1)) if target_stack_per_anchor else None
                    except Exception:
                        target_dim = None
                    anchor_dim = int(anchor_vecs[0].numel()) if anchor_vecs else agg_dim
                    pos_enc = args.agg_trans_pos_enc

                    # construct aggregator and proj_target with inferred sizes
                    try:
                        aggregator = CrossAttentionAggregator(
                            anchor_dim=anchor_dim,
                            target_dim=(kv_dim - 2) if pos_enc and key_proj is not None else (kv_dim if key_proj is not None else target_dim or anchor_dim),
                            agg_dim=agg_dim,
                            num_layers=max(1, args.agg_trans_num_layers),
                            num_heads=max(1, args.agg_trans_num_heads),
                            dropout=float(getattr(cfg.dcca_training, "agg_dropout", 0.1)),
                            use_positional_encoding=bool(pos_enc),
                        )
                    except Exception:
                        # Fallback conservative construction
                        aggregator = CrossAttentionAggregator(anchor_dim, target_dim or agg_dim, agg_dim, num_layers=max(1, args.agg_trans_num_layers), num_heads=max(1, args.agg_trans_num_heads), use_positional_encoding=bool(pos_enc))

                    proj_target = ProjectionHead(agg_dim, agg_dim, num_layers=args.agg_trans_num_layers)
                    target_head = AggregatorTargetHead(aggregator, proj_target, use_positional_encoding=bool(pos_enc))
                    # load state (should contain keys prefixed with 'aggregator.' and 'proj_target.')
                    try:
                        target_head.load_state_dict(state_b_dict, strict=True)
                    except Exception:
                        # try non-strict load if strict fails
                        target_head.load_state_dict(state_b_dict, strict=False)
                    projector_b = target_head.to(device)
                else:
                    projector_b = _projection_head_from_state(state_b_dict).to(device)
            else:
                projector_a = projector_a.to(device)
                projector_b = projector_b.to(device)

        projector_a.eval()
        projector_b.eval()
        projector_map = {anchor_name: projector_a, target_name: projector_b,}

        overlap_mask_info = _load_overlap_mask_data(cfg.overlap_mask_path)
        if overlap_mask_info is None:
            run_logger.log("[inference] No overlap mask available")
            raise RuntimeError("Overlap mask is required for classifier training but none was provided.")
        stack = _MaskStack(overlap_mask_info)


        sample_sets: Dict[str, Dict[str, object]] = {}
        label_summary_meta: Dict[str, List[str]] = {}
        anchor_overlap_samples: Optional[Dict[str, object]] = None
        anchor_non_overlap_samples: Optional[Dict[str, object]] = None
        target_overlap_samples: Optional[Dict[str, object]] = None
        target_non_overlap_samples: Optional[Dict[str, object]] = None

        def _summarise_sample_labels(sample_entry: Optional[Dict[str, object]]) -> List[str]:
            if not sample_entry:
                return []
            labels_tensor = sample_entry.get("labels")
            if labels_tensor is None:
                return []
            if hasattr(labels_tensor, "tolist"):
                raw = labels_tensor.tolist()
            else:
                raw = list(labels_tensor)
            summary: List[str] = []
            for value in raw:
                try:
                    summary.append("positive" if int(value) > 0 else "negative")
                except Exception:
                    summary.append("unknown")
            return summary

        # TODO: Make it more memory efficient by not storing all datasets at once
        # TODO: Currently, overlap_mask is only used for anchor dataset; need updates to handle target dataset overlap_mask filtering
        # Get all the data ready
        for dataset_name, projector in projector_map.items():
            run_logger.log(f"[cls] Preparing classifier samples for dataset {dataset_name}...")
            if projector is None:
                run_logger.log(f"[cls] projector unavailable for dataset {dataset_name}; skipping.")
                continue

            sample_set = _apply_projector_based_PUNlabels(
                workspace=workspace,
                dataset_name=dataset_name,
                pn_lookup=pn_label_maps.get(dataset_name),
                projector=projector,
                device=device,
                run_logger=run_logger,
                overlap_mask=overlap_mask_info,
                apply_overlap_filter=False,
            )

            run_logger.log(f"[cls] Preparing classifier samples for dataset {dataset_name}... done")

            if sample_set:
                sample_sets[dataset_name] = sample_set
                label_summary_meta[dataset_name] = _summarise_sample_labels(sample_set)

            # Get overlap-region anchor, anchor (global, overlap) data ready
            if dataset_name == anchor_name and sample_set:
                anchor_overlap_samples = _apply_projector_based_PUNlabels(
                    workspace=workspace, dataset_name=dataset_name, pn_lookup=pn_label_maps.get(dataset_name),
                    projector=projector, device=device, run_logger=run_logger,
                    overlap_mask=overlap_mask_info, apply_overlap_filter=True,
                )
                overlap_indices = set(anchor_overlap_samples.get("indices", [])) if anchor_overlap_samples else set()
                if overlap_indices:
                    keep_mask = [idx not in overlap_indices for idx in sample_set.get("indices", [])]
                    anchor_non_overlap_samples = _subset_classifier_sample(sample_set, keep_mask, subset_tag="non_overlap")

                # print("anchor_all", int(sample_set["labels"].numel()))
                # print("anchor_overlap", int(anchor_overlap_samples["labels"].numel()))
                # print("anchor_non_overlap", int(anchor_non_overlap_samples["labels"].numel()))

            # Get overlap-region anchor, target (global, overlap) data ready
            if dataset_name == target_name and sample_set:
                run_logger.log( "[DEVVVVV] ")
                run_logger.log(f"[DEVVVVV] The code currently assumes that the target dataset falls within the overlap region entirely.")   # YOU NEED THROUGH UPDATES FOR OVERLAP_MASK UPDATE IN integrate_stac.py
                run_logger.log( "[DEVVVVV] ")
                target_overlap_samples = _apply_projector_based_PUNlabels(
                    workspace=workspace, dataset_name=dataset_name, pn_lookup=pn_label_maps.get(dataset_name),
                    projector=projector, device=device, run_logger=run_logger,
                    overlap_mask=None, apply_overlap_filter=False,
                )
        #################################### Training Cls - 0) GET Training data ready (END) ####################################

    if args.train_cls_1:        
        #################################### Training Cls - 1) GLOBAL\OVERLAP using GLOBAL data (START) ####################################
        # Stat training of nnPU from (Global\Overlap) Anchor (Global) data
        if anchor_non_overlap_samples is None or anchor_non_overlap_samples is None:
            raise RuntimeError("Anchor non-overlap samples unavailable for classifier training; aborting.")
        uA = anchor_non_overlap_samples["features"].float()                                 # DDCA-projected features (u)
        yA_pu = torch.where(anchor_non_overlap_samples["labels"] >= 0.9, 1, -1).long()      # nnPU labels (+1/-1) from PN labels


        if debug_mode:
            # After line 881 in train.py
            temp_crd = anchor_non_overlap_samples.get("coords")  # Get coordinates
            temp_labels = anchor_non_overlap_samples["labels"]
            yA_pu = torch.where(temp_labels >= 0.9, 1, -1).long()

            # Create scatter plot of coordinates colored by labels
            if plt is not None and temp_crd is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Separate positive and negative samples
                pos_mask = (yA_pu == 1).cpu().numpy()
                neg_mask = (yA_pu == -1).cpu().numpy()
                
                # Extract coordinates
                coords_array = np.array(temp_crd)
                
                # Plot negative samples (unlabeled)
                if neg_mask.any():
                    ax.scatter(coords_array[neg_mask, 0], coords_array[neg_mask, 1], 
                            c='blue', s=5, alpha=0.5, label=f'Unlabeled (n={neg_mask.sum()})', 
                            marker='H')
                
                # Plot positive samples
                if pos_mask.any():
                    ax.scatter(coords_array[pos_mask, 0], coords_array[pos_mask, 1], 
                            c='red', s=5, alpha=0.7, label=f'Positive (n={pos_mask.sum()})', 
                            marker='H')
                
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title(f'Anchor Non-Overlap Samples Distribution\n'
                            f'Total: {len(yA_pu)}, Positive: {pos_mask.sum()}, Unlabeled: {neg_mask.sum()}')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='datalim')
                
                # Save plot
                debug_plot_path = Path(cfg.output_dir) / "debug_cls1_anchor_non_overlap_distribution.png"
                debug_plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                run_logger.log(f"[cls-1] Saved anchor non-overlap distribution plot to {debug_plot_path}")
                
                # Print statistics
                run_logger.log(f"[cls-1] Anchor non-overlap samples: total={len(yA_pu)}, "
                            f"positive={pos_mask.sum()}, unlabeled={neg_mask.sum()}, "
                            f"positive_ratio={pos_mask.sum()/len(yA_pu):.4f}")
        
        # Sanity-check shapes before creating DataLoader to avoid silent StopIteration later.
        try:
            n_u = int(uA.size(0))
        except Exception:
            n_u = 0
        try:
            n_y = int(yA_pu.size(0))
        except Exception:
            n_y = 0
        if n_u != n_y or n_u <= 0:
            run_logger.log(f"[cls][error] Feature/label size mismatch or empty: features={n_u}, labels={n_y}")
            raise RuntimeError(f"Classifier dataset empty or mismatched (features={n_u}, labels={n_y}).")

        # -- Create train/validation split for classifier training (mirror DCCA split logic)
        # prefer cfg.cls_training.validation_fraction if available, else default to 0.2
        val_frac = None
        try:
            val_frac = float(getattr(cfg, "cls_training", {}).validation_fraction)
        except Exception:
            val_frac = None
        if val_frac is None or not math.isfinite(val_frac):
            # fallback to arg parser level or default 0.2
            val_frac = float(getattr(cfg.dcca_training, "validation_fraction", 0.2) if getattr(cfg, "dcca_training", None) else 0.2)
        val_frac = max(0.0, min(val_frac, 0.9))

        total = n_u
        val_count = int(total * val_frac)
        # ensure at least one training sample if possible
        minimum_train = 1
        if total - val_count < minimum_train and total >= minimum_train:
            val_count = max(0, total - minimum_train)

        if val_count <= 0:
            train_indices = torch.arange(total, dtype=torch.long)
            val_indices = torch.empty(0, dtype=torch.long)
        else:
            gen = torch.Generator()
            try:
                gen.manual_seed(int(cfg.seed))
            except Exception:
                gen.manual_seed(0)
            indices = torch.randperm(total, generator=gen)
            val_indices = indices[:val_count]
            train_indices = indices[val_count:]

        train_idx_list = train_indices.tolist() if train_indices.numel() else []
        val_idx_list = val_indices.tolist() if val_indices.numel() else []

        uA_train = uA[train_idx_list].contiguous() if train_idx_list else uA.new_empty((0, uA.size(1)))
        yA_pu_train = yA_pu[train_idx_list].contiguous() if train_idx_list else yA_pu.new_empty((0,))
        uA_val = uA[val_idx_list].contiguous() if val_idx_list else uA.new_empty((0, uA.size(1)))
        yA_pu_val = yA_pu[val_idx_list].contiguous() if val_idx_list else yA_pu.new_empty((0,))

        train_ds = TensorDataset(uA_train, yA_pu_train)
        val_ds   = TensorDataset(uA_val,   yA_pu_val)

        # --- indices for P and U ---
        pos_idx = (yA_pu_train == 1).nonzero(as_tuple=True)[0].tolist()
        unl_idx = (yA_pu_train != 1).nonzero(as_tuple=True)[0].tolist()
        if len(pos_idx) == 0:
            raise RuntimeError("No positives in training set; cannot construct PU batches.")

        # choose at least 2 positive per batch:
        k_pos = min(int(max(2, math.ceil(0.05 * cfg.cls_training.batch_size))) , len(pos_idx))  # e.g., 5% of batch or at least 2

        k_unl = cfg.cls_training.batch_size - k_pos
        PUsampler = PUBatchSampler(pos_idx, unl_idx, k_pos=k_pos, k_unl=k_unl, seed=cfg.seed)

        # IMPORTANT: when using batch_sampler, do not pass batch_size/shuffle
        train_loader = DataLoader(train_ds, batch_sampler=PUsampler, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.cls_training.batch_size, shuffle=False, num_workers=0)

        # --- global prior (use your dataset-wide estimate; don't recompute per batch) ---
        pi_p = float((yA_pu == 1).sum().item()) / max(1, int(yA_pu.numel()))
        pi_p = float(min(max(pi_p, 1e-4), 0.2))  # clamp only to keep numerics sane

        encA = nn.Identity(); pA = nn.Identity()    # we are already in u-space
        gA = train_nnpu_a_only(encA, pA, train_loader, val_loader, pi_p, lr=cfg.dcca_training.lr, epochs=cfg.cls_training.epochs, device=device,)
        
        # After training gA inference
        uA_overlap = anchor_overlap_samples["features"].float()
        inference_results["gA_Overlap"] = run_inference_gA_only(
            uA=uA_overlap, 
            samples=anchor_overlap_samples,
            gA=gA,
            device=device,
            cfg=cfg,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "Overlap",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes
        )

        inference_results["gA_Non_Overlap"] = run_inference_gA_only(
            uA=uA, 
            samples=anchor_non_overlap_samples,
            gA=gA,
            device=device,
            cfg=cfg,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "Non_Overlap",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes
        )
        #################################### Training Cls - 1) GLOBAL\OVERLAP using GLOBAL data (END) ####################################

    if args.train_cls_1 or args.train_cls_2:
        # After training gA (around line 1037)
        if gA is not None:
            # Get input dimension from the first layer of gA's network
            d = gA.net[0].in_features  # Get dimension from first linear layer            

            # Save A-only PN head
            gA_state = {
                'state_dict': gA.state_dict(),
                'input_shape': d,  # feature dimension
                'architecture': 'PNHeadAOnly'
            }
            save_path = Path(cfg.output_dir) / "cls_1_nonoverlap.pt"
            torch.save(gA_state, save_path)
            run_logger.log(f"[cls] Saved A-only PN head (input dim={d}) to {save_path}")        

        # After gA training and before gAB training 
        if gA is None:
            # Try to load saved A-only PN head
            gA_path = Path(cfg.output_dir) / "cls_1_nonoverlap.pt"
            if gA_path.exists():
                try:
                    gA_state = torch.load(gA_path)
                    d = gA_state['input_shape']
                    gA = PNHeadAOnly(d=d).to(device)
                    gA.load_state_dict(gA_state['state_dict'])
                    gA.eval()
                    run_logger.log(f"[cls] Loaded A-only PN head (input dim={d}) from {gA_path}")
                except Exception as e:
                    run_logger.log(f"[cls] Failed to load A-only PN head from {gA_path}: {str(e)}")
                    raise RuntimeError("Could not train or load A-only PN head required for unified training")
            else:
                raise RuntimeError("A-only PN head not found and training failed; cannot proceed with unified training")

        #################################### Training Cls - 2) GLOBAL\OVERLAP using GLOBAL data (START) ####################################
        # ---------- simple dataset for precomputed u/v ----------
    class UVDataset(Dataset):
            """
            u: [N, d] (required)
            v: [N, d] (optional; if None, uses zeros like u)
            b_missing: [N, 1] float/bool mask: 1 if B missing for that sample; if None, infers 0 when v is given
            """
            def __init__(self, u, v=None, b_missing=None):
                assert u.ndim == 2, "u must be [N, d]"
                self.u = u
                if v is None:
                    self.v = torch.zeros_like(u)
                    self.b_missing = torch.ones(u.size(0), 1, device=u.device) if b_missing is None else b_missing.float()
                else:
                    assert v.shape == u.shape, f"v shape {v.shape} must match u {u.shape}"
                    self.v = v
                    if b_missing is None:
                        self.b_missing = torch.zeros(u.size(0), 1, device=u.device)
                    else:
                        self.b_missing = b_missing.float()
                assert self.b_missing.shape == (u.size(0), 1), "b_missing must be [N,1]"
            def __len__(self): return self.u.size(0)
            def __getitem__(self, i): return self.u[i], self.v[i], self.b_missing[i]

    def build_b_to_a_mapping(target_coords: List[Tuple[float, float]], 
                                anchor_coords: List[Tuple[float, float]], 
                                max_coord_error: float) -> torch.Tensor:
            """
            Build mapping from target (B) samples to anchor (A) samples based on coordinate proximity.
            
            Args:
                target_coords: List of (x,y) coordinates for target samples
                anchor_coords: List of (x,y) coordinates for anchor samples 
                max_coord_error: Maximum allowed coordinate distance for matching
                
            Returns:
                torch.Tensor of shape (Nb,) where each element is:
                - Index of matched anchor sample if match found
                - -1 if no match found within max_coord_error
            """
            Nb = len(target_coords)
            Na = len(anchor_coords)
            b_to_a = torch.full((Nb,), -1, dtype=torch.long)
            
            # Convert coords to numpy for efficient distance computation
            target_coords_np = np.array(target_coords)
            anchor_coords_np = np.array(anchor_coords)
            
            # For each target coord, find closest anchor coord within threshold
            for b_idx in range(Nb):
                b_coord = target_coords_np[b_idx]
                
                # Compute distances to all anchor coords
                distances = np.sqrt(np.sum((anchor_coords_np - b_coord)**2, axis=1))
                
                # Find closest anchor within threshold
                valid_matches = distances <= max_coord_error
                # if np.any(valid_matches):
                #     closest_a_idx = np.argmin(distances)
                #     b_to_a[b_idx] = closest_a_idx

                if np.any(valid_matches):
                    vv = np.flatnonzero(valid_matches)
                    closest_a_idx = vv[np.argmin(distances[vv])]
                    b_to_a[b_idx] = closest_a_idx

            return b_to_a

    def expand_pairs(u_bc, v_raw, b_to_a):
            """
            Return expanded (u_exp, v_exp, b_missing_exp) where each B row becomes one training pair.
            Unmatched A rows can be added with v=0 and b_missing=1 if you want to include them too.
            """
            valid = b_to_a >= 0
            idx = b_to_a[valid]
            u_exp = u_bc[idx]                     # [Nb_valid, d]
            v_exp = v_raw[valid]                  # [Nb_valid, d]
            b_missing_exp = torch.zeros(u_exp.size(0), 1, device=u_bc.device)
            return u_exp, v_exp, b_missing_exp

    def _infer_half_window_error_from_meta(workspace, anchor_name: str, target_name: str) -> Optional[float]:
            """Prefer dataset_window_spacing from integration metadata json."""
            try:
                # Direct path construction
                project_dir = Path("/home/wslqubuntu24/Research/Data/1_Foundation_MVT_Result")
                meta_path = project_dir / "2_Integrate_MVT_gcs_bcgs_occ/combined_metadata.json"

                print(f"[debug] Looking for metadata at: {meta_path}")
                
                # Load metadata file
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        print(f"[debug] Loaded metadata successfully")
                else:
                    print(f"[debug] Metadata file not found at {meta_path}")
                    return None
                    
                # Get window spacing
                spacing_map = meta.get("dataset_window_spacing", {})
                print(f"[debug] Found spacing map: {spacing_map}")
                
                wa = spacing_map.get(anchor_name)
                wb = spacing_map.get(target_name)
                print(f"[debug] Window sizes - {anchor_name}: {wa}, {target_name}: {wb}")
                
                vals = [float(v) for v in (wa, wb) if v is not None]
                if vals:
                    half_win = float(max(vals)) / 2.0
                    print(f"[debug] Using half window size: {half_win}")
                    return half_win
                    
            except Exception as e:
                print(f"[debug] Error reading window sizes: {str(e)}")
                import traceback
                traceback.print_exc()
                
            return None

    if args.train_cls_2:
        # u_overlap = anchor_overlap_samples["features"].float()
        # v_overlap = target_overlap_samples["features"].float()
        # # make b_to_a that is an integer vector of length Nb where each B sample is mapped to the row index of its matched A sample
        # half_win_err = _infer_half_window_error_from_meta(workspace, anchor_name, target_name)
        # # print("[DEVVVVVVVVVVVVVVVVVVVVVVVV] half_win_err is NOW TEMPORALY 20 times ", half_win_err*20)
        # # half_win_err = half_win_err * 20
        # if half_win_err is not None and math.isfinite(half_win_err) and half_win_err > 0:
        #     max_coord_error = half_win_err
        #     run_logger.log(f"[cls] Using half window spacing as matching radius: {max_coord_error}")
        # else:
        #     max_coord_error = auto_coord_error(workspace, anchor_name, target_name)
        #     run_logger.log(f"[cls] Falling back to auto_coord_error: {max_coord_error}")

        # if max_coord_error is None or not math.isfinite(max_coord_error) or max_coord_error <= 0:
        #     raise ValueError(f"Could not determine valid coordinate matching radius (half_win_err={half_win_err}, auto_coord_error={max_coord_error})")

        # anchor_coords = anchor_overlap_samples["coords"]
        # target_coords = target_overlap_samples["coords"]
        # b_to_a = build_b_to_a_mapping(target_coords, anchor_coords, max_coord_error)
        # u_exp, v_exp, bmiss_exp = expand_pairs(u_overlap, v_overlap, b_to_a)
        # run_logger.log(f"[cls] For cls-2; Pair expansion ended up from {u_overlap.shape[0], v_overlap.shape[0]} to {u_exp.shape[0], v_exp.shape[0]}")

        u_exp, v_exp, bmiss_exp = build_cls2_dataset_from_dcca_pairs(anchor_vecs, target_stack_per_anchor, projector_a, projector_b, device)
        run_logger.log(f"[cls] For cls-2; Pair expansion ended up from {len(anchor_vecs), len(target_stack_per_anchor)} to {u_exp.shape[0], v_exp.shape[0]}")

        dataset_uv = UVDataset(u_exp, v_exp, bmiss_exp)
        # gAB = fit_unified_head_OVERLAP_from_uv(gA, DataLoader(dataset_uv, batch_size=cfg.cls_training.batch_size, shuffle=True), 
        #                                         d_u=u_exp.size(1), device=device, lr=cfg.cls_training.lr, steps=cfg.cls_training.epochs, 
        #                                         view_dropout=0.3, noise_sigma=0.0)

        for iepoch in range(1, 10):
            gAB = fit_unified_head_OVERLAP_from_uv(gA, DataLoader(dataset_uv, batch_size=cfg.cls_training.batch_size, shuffle=True), 
                                                d_u=u_exp.size(1), device=device, lr=cfg.cls_training.lr, 
                                                # steps=cfg.cls_training.epochs
                                                steps=iepoch
                                                )

            # Run gAB (unified) inference
            inference_results["gAB_Overlap"] = run_overlap_inference_gAB_from_pairs(
                anchor_vecs=anchor_vecs,
                target_stack_per_anchor=target_stack_per_anchor,
                pair_metadata=pair_metadata,
                projector_a=projector_a,
                projector_b=projector_b,  # AggregatorTargetHead if you trained with the transformer aggregator
                gAB=gAB,
                device=device,
                cfg=cfg,
                output_dir=Path(cfg.output_dir) / "cls_2_inference_results" / f"epoch_{iepoch}",
                run_logger=run_logger,
                passes=cfg.cls_training.mc_dropout_passes,
                target_vecs=target_vecs,  # if projector_b is a plain ProjectionHead (no transformer)
                batch_size=INFERENCE_BATCH_SIZE,
            )

        # gAB = fit_unified_head_OVERLAP_from_uv(gA, DataLoader(dataset_uv, batch_size=cfg.cls_training.batch_size, shuffle=True), 
        #                                       d_u=u_exp.size(1), device=device, lr=cfg.cls_training.lr, steps=cfg.cls_training.epochs
        #                                       )

        # # Run gAB (unified) inference
        # inference_results["gAB_Overlap"] = run_overlap_inference_gAB_from_pairs(
        #     anchor_vecs=anchor_vecs,
        #     target_stack_per_anchor=target_stack_per_anchor,
        #     pair_metadata=pair_metadata,
        #     projector_a=projector_a,
        #     projector_b=projector_b,  # AggregatorTargetHead if you trained with the transformer aggregator
        #     gAB=gAB,
        #     device=device,
        #     cfg=cfg,
        #     output_dir=Path(cfg.output_dir) / "cls_2_inference_results" / "Overlap",
        #     run_logger=run_logger,
        #     passes=cfg.cls_training.mc_dropout_passes,
        #     target_vecs=target_vecs,  # if projector_b is a plain ProjectionHead (no transformer)
        #     batch_size=INFERENCE_BATCH_SIZE,
        # )

        # inference_results["gAB_Overlap"] = run_overlap_inference_gAB(
        #     u_overlap=u_overlap, v_overlap=v_overlap,b_to_a=b_to_a,
        #     anchor_overlap_samples=anchor_overlap_samples,
        #     target_overlap_samples=target_overlap_samples,
        #     gAB=gAB, device=device, cfg=cfg,
        #     output_dir=Path(cfg.output_dir) / "cls_2_inference_results" / "Overlap",
        #     run_logger=run_logger,
        #     passes=cfg.cls_training.mc_dropout_passes
        # )
        #################################### Training Cls - 2) GLOBAL\OVERLAP using GLOBAL data (END) ####################################

    if args.train_cls_1 and args.train_cls_2:
        # After training gAB (around line)
        if gAB is not None:
            # Get input dimension from the first layer of gA's network
            d_u = gAB.net[0].in_features  # Get dimension from first linear layer

            # Save unified PN head
            gAB_state = {
                'state_dict': gAB.state_dict(),
                'input_shape': d_u,  # feature dimension
                'architecture': 'PNHeadUnified'
            }
            save_path = Path(cfg.output_dir) / "cls_2_unified.pt"
            torch.save(gAB_state, save_path)
            run_logger.log(f"[cls] Saved unified PN head (input dim={d_u}) to {save_path}")

        # After gAB training
        if gAB is None:
            # Try to load saved A-only PN head
            gA_path = Path(cfg.output_dir) / "cls_2_unified.pt"
            if gA_path.exists():
                try:
                    gAB_state = torch.load(gA_path)
                    d = gAB_state['input_shape']
                    gAB = PNHeadAOnly(d=d).to(device)
                    gAB.load_state_dict(gAB_state['state_dict'])
                    gAB.eval()
                    run_logger.log(f"[cls] Loaded unified PN head (input dim={d}) from {gA_path}")
                except Exception as e:
                    run_logger.log(f"[cls] Failed to load unified PN head from {gA_path}: {str(e)}")
                    raise RuntimeError("Could not train or load unified PN head required for unified training")
            else:
                raise RuntimeError("A-only PN head not found and training failed; cannot proceed with unified training")

    #################################### Temporal Debugging Module ####################################
    _debug_root = Path(cfg.output_dir)
    debug_plot_dir = _debug_root / "debug_classifier_inspection"
    debug_plot_dir.mkdir(parents=True, exist_ok=True)

    print("debug_plot_dir=", debug_plot_dir)

    def _clean_point(point: Optional[Sequence[object]]) -> Optional[Tuple[float, float]]:
        if point is None:
            return None
        if not isinstance(point, (tuple, list)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (x, y)

    def _debug_plot_points(name: str, points_by_label: Dict[Optional[int], List[Tuple[float, float]]]) -> None:
        if debug_plot_dir is None or plt is None:
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        palette = {
            1: "tab:orange",
            0: "tab:blue",
            None: "tab:gray",
        }
        plotted = False
        for label, pts in points_by_label.items():
            valid_pts = [_clean_point(pt) for pt in pts]
            valid_pts = [pt for pt in valid_pts if pt is not None]
            if not valid_pts:
                continue
            xs, ys = zip(*valid_pts)
            label_name = "pos" if label == 1 else ("neg" if label == 0 else "unlabeled")
            ax.scatter(xs, ys, s=16, alpha=0.75, label=label_name, c=palette.get(label, "tab:gray"))
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best")
        fig.tight_layout()
        safe_name = name.replace(" ", "_").replace("/", "_")
        fig.savefig(debug_plot_dir / f"{safe_name}.png", dpi=200)
        plt.close(fig)

    def _debug_check_summarize(indices, tag):
        counts = Counter()
        regions = defaultdict(Counter)
        coords = []
        anchor_points: Dict[Optional[int], List[Tuple[float, float]]] = defaultdict(list)
        target_points: Dict[Optional[int], List[Tuple[float, float]]] = defaultdict(list)
        for idx in indices:
            meta = pair_metadata[idx]
            label = pair_labels[idx]["combined_label"]
            counts[label] += 1
            region = meta.get("anchor_region")
            regions[region][label] += 1
            anchor_coord = meta.get("anchor_coord")
            if anchor_coord is not None:
                anchor_points[label].append(anchor_coord)
            target_coords = meta.get("target_coords") or []
            filtered_targets = []
            for coord in target_coords:
                filtered_targets.append(coord)
                if coord is not None:
                    target_points[label].append(coord)
            coords.append((anchor_coord, filtered_targets))
        print(f"[unified {tag}] label counts {counts}")
        print(f"[unified {tag}] by anchor region {dict(regions)}")
        if indices:
            print("Call _debug_plot_points in _debug_check_summarize to plot points.")
            _debug_plot_points(f"unified_{tag}_anchor", anchor_points)
            _debug_plot_points(f"unified_{tag}_target", target_points)
        return coords  # keep if you want to inspect/export later

    def _debug_check_summarize_view(info, label_key, dataset_tag):
        train_indices_info = info.get("train_indices", [])
        val_indices_info = info.get("val_indices", [])
        counts = Counter()
        regions = defaultdict(Counter)
        val_counts = Counter()
        val_regions = defaultdict(Counter)
        train_points: Dict[Optional[int], List[Tuple[float, float]]] = defaultdict(list)
        val_points: Dict[Optional[int], List[Tuple[float, float]]] = defaultdict(list)
        train_coords = []
        for idx in train_indices_info:
            meta = pair_metadata[idx]
            label = pair_labels[idx][label_key]
            counts[label] += 1
            region = meta.get("anchor_region") if dataset_tag == "anchor" else meta.get("target_regions", [None])[0]
            regions[region][label] += 1
            coord = meta.get("anchor_coord") if dataset_tag == "anchor" else (meta.get("target_coords") or [None])[0]
            train_coords.append(coord)
            if coord is not None:
                train_points[label].append(coord)
        val_coords = []
        for idx in val_indices_info:
            meta = pair_metadata[idx]
            label = pair_labels[idx][label_key]
            val_counts[label] += 1
            region = meta.get("anchor_region") if dataset_tag == "anchor" else meta.get("target_regions", [None])[0]
            val_regions[region][label] += 1
            coord = meta.get("anchor_coord") if dataset_tag == "anchor" else (meta.get("target_coords") or [None])[0]
            val_coords.append(coord)
            if coord is not None:
                val_points[label].append(coord)
        print(f"[method2 {dataset_tag} train] label counts {counts}")
        print(f"[method2 {dataset_tag} train] by region {dict(regions)}")
        if val_indices_info:
            print(f"[method2 {dataset_tag} val] label counts {val_counts}")
            print(f"[method2 {dataset_tag} val] by region {dict(val_regions)}")
        if train_indices_info:
            _debug_plot_points(f"method2_{dataset_tag}_train", train_points)
        if val_indices_info:
            _debug_plot_points(f"method2_{dataset_tag}_val", val_points)
        return {"train": train_coords, "val": val_coords}
    #################################### Temporal Debugging Module ####################################

    return
#####################################################################################   #####################################################################################   #####################################################################################   

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
def run_inference_gA_only(
    uA: torch.Tensor,
    samples: Dict,
    gA: nn.Module,
    device: torch.device,
    cfg: AlignmentConfig,
    output_dir: Path,
    run_logger: "_RunLogger",
    passes: int = 10
) -> Dict[str, object]:
    """Run inference on overlap data using only the gA (anchor-only) model."""
    
   
    # Get corresponding metadata
    matched_coords = samples["coords"]
    
    run_logger.log(f"[inference-gA] Processing {len(matched_coords)} anchor samples")
    
    # Prepare output directory
    model_dir = output_dir / "overlap_inference" / "gA"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # MC Dropout inference
    gA.train()  # Enable dropout
    predictions_list = []
    
    with torch.no_grad():
        for _ in range(passes):
            pred = gA(uA)
            predictions_list.append(pred.cpu().numpy())
    
    gA.eval()  # Disable dropout
    
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
        model_name="gA",
        coords=matched_coords,
        mean_pred=mean_pred,
        std_pred=std_pred
    )

    # print("TODO: USE write_prediction_outputs function here to save outputs consistently.")
    # write_prediction_outputs(
    #         prediction,
    #         stack,
    #         out_dir = model_dir,
    #         pos_coords_by_region=pos_coords_by_region,
    #         neg_coords_by_region=None,
    #     )

    # Compute summary statistics
    summary = _compute_inference_summary(
        model_name="gA",
        mean_pred=mean_pred,
        std_pred=std_pred,
        labels=None,  # gA doesn't have labels available
    )
    
    # Save summary
    with open(model_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    run_logger.log(f"[inference-gA] Saved results to {model_dir}")
    
    return {
        "predictions_mean": mean_pred,
        "predictions_std": std_pred,
        "coordinates": matched_coords,
        "summary": summary
    }

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
    batch_size: int = INFERENCE_BATCH_SIZE
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
    model_dir = output_dir / "overlap_inference" / "gAB"
    model_dir.mkdir(parents=True, exist_ok=True)

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
                out = gAB(u_b, v_b, bm_b)  # [B]
                batch_out.append(out.detach().cpu())
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
        std_pred=std_pred
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


def run_overlap_inference_gAB(
    u_overlap: torch.Tensor,
    v_overlap: torch.Tensor,
    b_to_a: torch.Tensor,
    anchor_overlap_samples: Dict,
    target_overlap_samples: Dict,
    gAB: nn.Module,
    device: torch.device,
    cfg: AlignmentConfig,
    output_dir: Path,
    run_logger: "_RunLogger",
    passes: int = 10
) -> Dict[str, object]:
    """Run inference on overlap data using the gAB (unified) model."""
    
    # Align v features with u using b_to_a mapping
    valid_matches = b_to_a >= 0
    u_matched = u_overlap[b_to_a[valid_matches]].to(device)
    v_matched = v_overlap[valid_matches].to(device)
    
    # Get corresponding metadata
    matched_indices = b_to_a[valid_matches].cpu().numpy()
    matched_coords = [anchor_overlap_samples["coords"][int(idx)] for idx in matched_indices]
    
    run_logger.log(f"[inference-gAB] Processing {len(matched_coords)} matched pairs")
    
    # Prepare output directory
    model_dir = output_dir / "overlap_inference" / "gAB"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # MC Dropout inference
    gAB.train()  # Enable dropout
    predictions_list = []
    
    with torch.no_grad():
        for _ in range(passes):
            # For unified model, use proper forward signature
            b_missing = torch.zeros(u_matched.size(0), 1, device=device)
            pred = gAB(u_matched, v_matched, b_missing)
            predictions_list.append(pred.cpu().numpy())
    
    gAB.eval()  # Disable dropout
    
    print("TODO: USE write_prediction_outputs function here to save outputs consistently.")

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
        model_name="gAB",
        coords=matched_coords,
        mean_pred=mean_pred,
        std_pred=std_pred
    )
    
    # Compute summary statistics
    summary = _compute_inference_summary(
        model_name="gAB",
        mean_pred=mean_pred,
        std_pred=std_pred,
        labels=None,  # gAB (this version) doesn't have labels available
    )
    
    # Save summary
    with open(model_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    run_logger.log(f"[inference-gAB] Saved results to {model_dir}")
    
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
    std_pred: np.ndarray
) -> None:
    """Create and save scatter plots for inference results."""
    if plt is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mean predictions
    coords_array = np.array(coords)
    # sc1 = axes[0].scatter(coords_array[:, 0], coords_array[:, 1], c=mean_pred, cmap='RdYlGn', s=5, marker='H', alpha=1.0, vmin=0, vmax=1)
    sc1 = axes[0].scatter(coords_array[:, 0], coords_array[:, 1], c=mean_pred, cmap='RdYlGn', s=5, marker='H', alpha=1.0)
    axes[0].set_title(f'{model_name} - Mean Prediction')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_aspect('equal', adjustable='datalim')
    plt.colorbar(sc1, ax=axes[0], label='Probability')
    
    # Plot uncertainty (std)
    sc2 = axes[1].scatter(coords_array[:, 0], coords_array[:, 1], c=std_pred, cmap='viridis', s=5, marker='H', alpha=1.0)
    axes[1].set_title(f'{model_name} - Uncertainty (Std)')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].set_aspect('equal', adjustable='datalim')
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
    
    comparison_dir = output_dir / "overlap_inference" / "comparison"
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


# TODO: UPDATE FROM CONFIG
class PNHeadUnified(nn.Module):
    """Unified PN head on φ(u,v,mask)."""
    def __init__(self, d, hidden=256):
        super().__init__()
        in_dim = 4*d + 2  # [u, v, |u-v|, u*v, cos, mask]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, u, v=None, b_missing=None):
        if b_missing is None:
            b_missing = torch.zeros(u.size(0), 1, device=u.device)
        phi = build_phi(u, v, b_missing)
        return torch.sigmoid(self.net(phi)).squeeze(-1)

# --------------------------- utils ---------------------------
def cosine_sim(u, v, eps=1e-6):
    return F.cosine_similarity(u, v, dim=-1, eps=eps).unsqueeze(-1)

def build_phi(u, v, b_missing):
    # u,v: [B, d], b_missing: [B,1] in {0,1}
    if v is None:  # A-only fallback (outside BC)
        v = torch.zeros_like(u)
        b_missing = torch.ones(u.size(0), 1, device=u.device)
    feats = [u, v, torch.abs(u - v), u * v, cosine_sim(u, v), b_missing]
    return torch.cat(feats, dim=-1)  # [B, 4d + 2]

# ---------- unified head training from u/v ----------
import math
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

def fit_unified_head_OVERLAP_from_uv(
    gA,
    data_loader_uv,
    d_u=None,
    device=None,
    *,
    # optimization
    lr=5e-4,
    steps=30,
    eps=1e-6,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    # KD scheduling
    kd_max=0.20,
    kd_warmdown_epochs=5,
    # loss weights
    lambda_cons=1.0,
    lambda_ent=0.10,
    lambda_prior=5.0,
    lambda_delta=0.10,
    delta_margin=0.08,
    # stochastic B views
    jitter_sigma=0.05,
    jitter_drop=0.20,
    noise_sigma=None,
    view_dropout=None,
    # prior smoothing
    pi_ema_alpha=0.9,
    pi_clip_max=0.2
):
    """Train a unified PN head gAB(u, v, b_missing) on overlap pairs."""

    # Helpers
    def safe_logit(p, eps_=eps):
        p = p.clamp(eps_, 1.0 - eps_)
        return torch.log(p) - torch.log(1.0 - p)

    def to_prob(z, eps_=eps):
        return torch.sigmoid(z).clamp(eps_, 1.0 - eps_)

    def bin_entropy_from_logit(z):
        p = to_prob(z)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    # Accept legacy kwargs
    if noise_sigma is not None:
        jitter_sigma = float(noise_sigma)
    if view_dropout is not None:
        jitter_drop = float(view_dropout)

    def jitter(v, sigma=jitter_sigma, drop=jitter_drop):
        if sigma and sigma > 0.0:
            v = v + sigma * torch.randn_like(v)
        if drop and drop > 0.0:
            mask = (torch.rand_like(v) > drop).float()
            v = v * mask
        return v

    # Infer dims / device
    u0, v0, b0 = next(iter(data_loader_uv))
    if d_u is None:
        d_u = int(u0.size(-1))
    if device is None:
        device = u0.device

    # Models / optimizer
    gA = gA.to(device).eval()
    gAB = PNHeadUnified(d=d_u).to(device).train()
    opt = torch.optim.AdamW(gAB.parameters(), lr=lr, weight_decay=weight_decay)

    # Prior target (EMA)
    pi_ema = None

    # Training loop
    epochs = int(steps)
    for epoch in range(1, epochs + 1):
        # Cosine warm-down for KD
        if kd_warmdown_epochs <= 0:
            lambda_kd = 0.0
        else:
            t = min(epoch - 1, kd_warmdown_epochs) / max(1, kd_warmdown_epochs)
            lambda_kd = kd_max * 0.5 * (1.0 + math.cos(math.pi * t))

        epoch_loss = epoch_kd = epoch_ent = epoch_cons = epoch_prior = epoch_delta = 0.0
        batches = 0

        pbar = tqdm(data_loader_uv, desc=f"    [cls - 2] Epoch {epoch:3d}/{epochs}", leave=False)

        for u, v, bmiss in pbar:
            u, v, bmiss = u.to(device), v.to(device), bmiss.to(device)
            
            # ✅ FIX: Ensure bmiss is [B, 1] and create mB as [B] for scalar operations
            if bmiss.ndim == 1:
                bmiss = bmiss.unsqueeze(-1)
            
            mB = (1.0 - bmiss).squeeze(-1)  # ✅ [B] - scalar mask for element-wise ops
            hasB = bool(mB.sum().item() > 0.0)

            # Teacher (probabilities + logits)
            with torch.no_grad():
                t_p = gA(u).clamp(eps, 1 - eps)  # [B] (probabilities, already squeezed by gA)
                t_logit = safe_logit(t_p)        # [B]
                pi_batch = float(t_p.mean().item())
                pi_batch = float(min(max(pi_batch, 1e-4), pi_clip_max))
                pi_ema = (pi_batch if pi_ema is None else 
                         pi_ema_alpha * pi_ema + (1 - pi_ema_alpha) * pi_batch)

            # Student passes
            # No-B path
            z_noB = gAB(u, torch.zeros_like(v), torch.ones_like(bmiss))  # [B] logits
            p_noB = to_prob(z_noB)  # [B]

            # Two stochastic B views
            v1 = jitter(v)
            v2 = jitter(v)
            # Mask views where B is missing
            v1 = v1 * mB.unsqueeze(-1)  # [B, d]
            v2 = v2 * mB.unsqueeze(-1)  # [B, d]

            z_B1 = gAB(u, v1, bmiss)  # [B] logits
            z_B2 = gAB(u, v2, bmiss)  # [B] logits
            p_B1 = to_prob(z_B1)      # [B]
            p_B2 = to_prob(z_B2)      # [B]

            # ========== Losses ==========
            # 1) KD on logits for no-B path
            loss_kd = F.mse_loss(z_noB, t_logit)

            # 2) Consistency on logits (where B exists)
            if hasB:
                # ✅ Both z_B1 and mB are now [B], so element-wise multiplication works
                loss_cons = F.mse_loss(z_B1 * mB, (z_B2.detach()) * mB)
            else:
                loss_cons = z_B1.sum() * 0.0

            # 3) Entropy maximization on B path (where B exists)
            if hasB:
                H = bin_entropy_from_logit(z_B1)  # [B]
                loss_ent = -(H * mB).sum() / mB.sum().clamp_min(1.0)
            else:
                loss_ent = z_B1.sum() * 0.0

            # 4) Prior matching (where B exists)
            if hasB:
                p_mean = (p_B1 * mB).sum() / mB.sum().clamp_min(1.0)
                loss_prior = (p_mean - float(pi_ema)) ** 2
            else:
                loss_prior = z_B1.sum() * 0.0

            # 5) Delta loss: ensure B changes prediction (where B exists)
            if hasB:
                dB = (p_B1 - p_noB).abs()  # [B]
                loss_delta = (F.relu(delta_margin - dB) * mB).mean()
            else:
                loss_delta = z_B1.sum() * 0.0

            # Total loss
            loss = (
                lambda_kd   * loss_kd +
                lambda_cons * loss_cons +
                lambda_ent  * loss_ent +
                lambda_prior* loss_prior +
                lambda_delta* loss_delta
            )

            # Optimization step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gAB.parameters(), max_grad_norm)
            opt.step()

            # Bookkeeping
            batches += 1
            epoch_loss  += float(loss.item())
            epoch_kd    += float(loss_kd.item())
            epoch_cons  += float(loss_cons.item())
            epoch_ent   += float((-loss_ent).item())
            epoch_prior += float(loss_prior.item())
            epoch_delta += float(loss_delta.item())

            # Live stats
            with torch.no_grad():
                if hasB:
                    dB_mean = ((p_B1 - p_noB).abs() * mB).sum() / mB.sum().clamp_min(1.0)
                    pB_mean = (p_B1 * mB).sum() / mB.sum().clamp_min(1.0)
                else:
                    dB_mean = torch.tensor(0.0, device=device)
                    pB_mean = torch.tensor(0.0, device=device)
                p0_mean = p_noB.mean()

            pbar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'kd': f'{(lambda_kd*loss_kd).item():.4e}',
                'ent(H+)': f'{(-loss_ent).item():.4e}',
                'cons': f'{(lambda_cons*loss_cons).item():.4e}',
                'ΔB': f'{float(dB_mean):.3f}',
                'π̂': f'{float(pi_ema):.3f}',
                'p(B)': f'{float(pB_mean):.3f}',
                'p(noB)': f'{float(p0_mean):.3f}',
            })

        pbar.close()

        # Epoch summary
        denom = max(1, batches)
        print(
            f"    [cls - 2] Epoch {epoch:3d} | "
            f"loss={(epoch_loss/denom):.4e} | "
            f"kd={(epoch_kd/denom):.4e} | "
            f"ent(H+)={(epoch_ent/denom):.4e} | "
            f"cons={(epoch_cons/denom):.4e} | "
            f"prior={(epoch_prior/denom):.4e} | "
            f"delta={(epoch_delta/denom):.4e} | "
            f"batches={batches}"
        )

        # Sanity check after epoch
        with torch.no_grad():
            try:
                u_s, v_s, b_s = next(iter(data_loader_uv))
                u_s, v_s, b_s = u_s.to(device), v_s.to(device), b_s.to(device)
                if b_s.ndim == 1:
                    b_s = b_s.unsqueeze(-1)
                m_s = (1.0 - b_s).squeeze(-1)  # [B]
                
                z0 = gAB(u_s, torch.zeros_like(v_s), torch.ones_like(b_s))  # [B]
                zB = gAB(u_s, v_s * m_s.unsqueeze(-1), b_s)  # [B]
                p0 = to_prob(z0)  # [B]
                pB = to_prob(zB)  # [B]
                
                if m_s.sum() > 0:
                    pB_mean = (pB * m_s).sum() / m_s.sum()
                    dB_mean = ((pB - p0).abs() * m_s).sum() / m_s.sum()
                else:
                    pB_mean = torch.tensor(0.0, device=device)
                    dB_mean = torch.tensor(0.0, device=device)
                print(f"[sanity] mean p(noB)={p0.mean().item():.3f}, mean p(B)={pB_mean.item():.3f}, ΔB={dB_mean.item():.3f}")
            except StopIteration:
                pass

    print(f"\n    [cls - 2] Training completed after {epochs} epochs")
    return gAB.eval()




# def fit_unified_head_OVERLAP_from_uv(gA, data_loader_uv, d_u=None, device=None, 
#                                     lr=5e-4, steps=1500, view_dropout=0.3, noise_sigma=0.0, eps=1e-7, lambda_ent=0.1, lambda_cons=0.25):
#     """
#     gA: trained A-only PN head (expects u)
#     data_loader_uv: yields (u, v, b_missing)
#     d_u: feature dim; if None inferred from first batch
#     steps: number of epochs (renamed from steps for clarity)
#     """
#     # infer dim and device
#     u0, v0, _ = next(iter(data_loader_uv))
#     if d_u is None: d_u = u0.size(-1)
#     if device is None: device = u0.device

#     gAB = PNHeadUnified(d=d_u).to(device).train()
#     opt = torch.optim.AdamW(gAB.parameters(), lr=lr, weight_decay=1e-4)
#     gA = gA.to(device).eval()

#     epochs = steps  # treating steps as epochs for clarity
    
#     # Training loop with epoch-based progress
#     for epoch in range(1, epochs + 1):
#         epoch_loss = 0.0
#         epoch_loss_kd = 0.0
#         epoch_loss_ent = 0.0
#         epoch_loss_cons = 0.0
#         batch_count = 0
        
#         # Progress bar for batches within epoch
#         pbar = tqdm(data_loader_uv, desc=f"    [cls - 2] Epoch {epoch:3d}/{epochs}", leave=False)
        
#         for u, v, bmiss in pbar:
#             u, v, bmiss = u.to(device), v.to(device), bmiss.to(device)

#             with torch.no_grad():
#                 # optional tiny noise to stabilize (set noise_sigma>0 to enable)
#                 if noise_sigma > 0:
#                     u_noisy = u + noise_sigma*torch.randn_like(u)
#                     v_noisy = v + noise_sigma*torch.randn_like(v)
#                 else:
#                     u_noisy, v_noisy = u, v

#                 teacher = gA(u_noisy).detach()  # A-only teacher

#             # view-dropout to teach robustness when B is absent
#             if torch.rand(1).item() < view_dropout:
#                 v_in = torch.zeros_like(v_noisy)
#                 b_in = torch.ones(u_noisy.size(0), 1, device=device)
#             else:
#                 v_in = v_noisy
#                 b_in = bmiss  # use real mask (probably zeros if v present)

#             # student forward
#             p_student = gAB(u_noisy, v_in, b_in)

#             # consistency target (weak vs strong): here just no-drop vs drop mask
#             p_cons = gAB(u_noisy, v, torch.zeros_like(b_in))

#             # losses (same as your recipe)
#             loss_kd  = F.mse_loss(p_student, teacher)
#             loss_ent = -(p_student*torch.log(p_student+eps)+ (1-p_student)*torch.log(1-p_student+eps)).mean()
#             loss_cons = F.mse_loss(p_student, p_cons.detach())
#             loss = loss_kd + lambda_ent*loss_ent + lambda_cons*loss_cons

#             opt.zero_grad()
#             loss.backward()
#             opt.step()
            
#             epoch_loss += loss.item()
#             epoch_loss_kd += loss_kd.item()
#             epoch_loss_ent += loss_ent.item()
#             epoch_loss_cons += loss_cons.item()
#             batch_count += 1
            
#             # Update progress bar with current batch metrics
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4e}',
#                 'kd': f'{loss_kd.item():.4e}',
#                 'ent': f'{loss_ent.item():.4e}',
#                 'cons': f'{loss_cons.item():.4e}'
#             })
        
#         pbar.close()
        
#         # Compute epoch averages
#         avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
#         avg_kd = epoch_loss_kd / batch_count if batch_count > 0 else 0.0
#         avg_ent = epoch_loss_ent / batch_count if batch_count > 0 else 0.0
#         avg_cons = epoch_loss_cons / batch_count if batch_count > 0 else 0.0
        
#         # Print epoch summary (matching cls-1 style)
#         print(f"    [cls - 2] Epoch {epoch:3d} | "
#               f"loss={avg_epoch_loss:.4e} | "
#               f"kd={avg_kd:.4e} | "
#               f"ent={avg_ent:.4e} | "
#               f"cons={avg_cons:.4e} | "
#               f"batches={batch_count}")

#     print(f"\n    [cls - 2] Training completed after {epochs} epochs")
#     return gAB.eval()
    
#####################################################################################   

class PUBatchSampler(Sampler[list]):
    def __init__(self, pos_idx, unl_idx, k_pos, k_unl, epoch_len=None, seed=0):
        self.pos_idx = pos_idx
        self.unl_idx = unl_idx
        self.k_pos = k_pos
        self.k_unl = k_unl
        self.seed = seed
        # how many batches per epoch (by unlabeled supply)
        self.epoch_len = epoch_len or (len(unl_idx) // k_unl)

    def __iter__(self):
        rng = random.Random(self.seed)
        # reshuffle every epoch
        pos = self.pos_idx[:] ; unl = self.unl_idx[:]
        rng.shuffle(pos) ; rng.shuffle(unl)
        # cycle positives if we run out
        p_ptr = 0
        for b in range(self.epoch_len):
            if (b * self.k_unl + self.k_unl) > len(unl):
                break
            batch_unl = unl[b*self.k_unl : (b+1)*self.k_unl]
            # take k_pos positives, cycling if necessary
            if p_ptr + self.k_pos > len(pos):
                rng.shuffle(pos)
                p_ptr = 0
            batch_pos = pos[p_ptr : p_ptr + self.k_pos]
            p_ptr += self.k_pos
            yield batch_pos + batch_unl

    def __len__(self):
        return self.epoch_len


class PNHeadAOnly(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, u): return torch.sigmoid(self.net(u)).squeeze(-1)

class NNPULoss(nn.Module):
    def __init__(self, pi_p): super().__init__(); self.pi_p = float(pi_p); self.eps=1e-6
    def logloss(self,p,y): return -(y*torch.log(p+self.eps)+(1-y)*torch.log(1-p+self.eps))
    def forward(self,p,labels):
        Pm=(labels==1).float(); Um=(labels==-1).float()
        Lp=(self.logloss(p, torch.ones_like(p))*Pm).sum()  /(Pm.sum()+self.eps)
        LnU=(self.logloss(p, torch.zeros_like(p))*Um).sum()/(Um.sum()+self.eps)
        LnP=(self.logloss(p, torch.zeros_like(p))*Pm).sum()/(Pm.sum()+self.eps)
        return self.pi_p*Lp + torch.clamp(LnU - self.pi_p*LnP, min=0.0)

def train_nnpu_a_only(encA, pA, train_loader, val_loader, pi_p, lr=1e-3, epochs=3, device='cuda'):
    # Probe a batch to infer feature dimension; handle empty loader gracefully.
    try:
        it = iter(train_loader)
        x0, _ = next(it)
    except StopIteration:
        raise RuntimeError("Training DataLoader is empty; cannot train PN head.")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch a probe batch from train_loader: {exc}")

    with torch.no_grad():
        u0 = pA(encA(x0.to(device)))
    if u0.ndim < 2:
        raise RuntimeError("Projected features have unexpected shape; expected (B, D).")
    d = int(u0.shape[1])

    gA = PNHeadAOnly(d=d).to(device)
    opt = torch.optim.AdamW(gA.parameters(), lr=lr, weight_decay=1e-4)
    nnpu = NNPULoss(pi_p)

    encA = encA.to(device);  pA = pA.to(device)

    # simple history tracking similar to v1
    best = {"f1": -1.0, "state_dict": None}
    history = []
    for ep in range(1, epochs + 1):
        gA.train()
        running_loss = 0.0
        sample_count = 0
        for xA, pu in _progress_iter(train_loader, desc=f"CLS epoch {ep}"):
            xA, pu = xA.to(device), pu.to(device)
            with torch.no_grad():
                u = pA(encA(xA))
            p = gA(u)
            loss = nnpu(p, pu)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * int(pu.size(0))
            sample_count += int(pu.size(0))

        avg_loss = running_loss / sample_count if sample_count else float('nan')

        # Evaluate on train and val using helper from v1
        gA.eval()

        class _EncWrapper:
            def __init__(self, enc, p):
                self.enc = enc
                self.p = p
            def eval(self):
                return self
            def encode(self, x):
                with torch.no_grad():
                    return self.p(self.enc(x))

        encoder_wrapper = _EncWrapper(encA, pA)
        train_targets, train_probs = _collect_outputs(encoder_wrapper, gA, train_loader, device)
        val_targets, val_probs = _collect_outputs(encoder_wrapper, gA, val_loader, device)

        # PN labels in the pipeline are coded as +1 (positive) and -1 (unlabeled);
        # convert to 0/1 for sklearn metrics and BCE loss to avoid multiclass issues
        try:
            train_targets = (train_targets > 0).long()
        except Exception:
            train_targets = train_targets
        try:
            val_targets = (val_targets > 0).long()
        except Exception:
            val_targets = val_targets

        train_metrics = _compute_metrics(train_targets, train_probs)
        val_metrics = _compute_metrics(val_targets, val_probs)

        if val_targets.numel():
            try:
                val_loss = torch.nn.functional.binary_cross_entropy(val_probs, val_targets.float()).item()
            except Exception:
                val_loss = float('nan')
        else:
            val_loss = float('nan')

        comparable_f1 = val_metrics.get("f1", float('nan'))
        if not math.isnan(comparable_f1) and comparable_f1 > best["f1"]:
            best = {"f1": comparable_f1, "state_dict": gA.state_dict()}

        # Format loss separately in scientific notation
        train_loss_str = f"{avg_loss:.4e}" if math.isfinite(avg_loss) else "nan"
        val_loss_str = f"{val_loss:.4e}" if math.isfinite(val_loss) else "nan"

        # Create log dictionaries without loss first
        train_log = {**train_metrics}
        val_log = {**val_metrics}

        # Print custom formatted output with loss in scientific notation
        print(f"    [cls - 1] Epoch {ep:3d} | loss={train_loss_str}/{val_loss_str} | ")
        print(f"    [cls - 1] TRAIN | loss={train_loss_str} | " + 
            " | ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) and math.isfinite(v) else f"{k}={v}" for k, v in train_log.items()))
        print(f"    [cls - 1] VAL   | loss={val_loss_str} | " + 
            " | ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) and math.isfinite(v) else f"{k}={v}" for k, v in val_log.items()))

        # Still log to history with original numeric values for later analysis
        history.append({
            "epoch": int(ep), 
            "train": {"loss": float(avg_loss), **normalize_metrics(train_log)}, 
            "val": {"loss": float(val_loss), **normalize_metrics(val_log)}
        })

    if best["state_dict"] is not None:
        gA.load_state_dict(best["state_dict"])

    return gA.eval().to(device)



class TargetSetDataset(Dataset):
    def __init__(self, anchor_vecs: Sequence[torch.Tensor], target_stack_per_anchor: Sequence[torch.Tensor], metadata: Sequence[Dict[str, object]]):
        self.anchor_vecs = anchor_vecs
        self.target_stack_per_anchor = target_stack_per_anchor
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.anchor_vecs)

    def __getitem__(self, idx: int):
        return self.anchor_vecs[idx], self.target_stack_per_anchor[idx], self.metadata[idx]


def _collate_target_sets(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]], use_positional_encoding: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    anchors, target_stack_per_anchor, metadata = zip(*batch)
    anchor_batch = torch.stack(anchors, dim=0)
    max_len = max(t.size(0) for t in target_stack_per_anchor)
    feature_dim = target_stack_per_anchor[0].size(1)
    target_batch = anchor_batch.new_zeros((len(batch), max_len, feature_dim))
    mask_batch = torch.ones(len(batch), max_len, dtype=torch.bool)
    pos_batch: Optional[torch.Tensor] = None
    if use_positional_encoding:
        pos_batch = anchor_batch.new_zeros((len(batch), max_len, 2))
    for idx, (stack, meta) in enumerate(zip(target_stack_per_anchor, metadata)):
        length = stack.size(0)
        target_batch[idx, :length] = stack
        mask_batch[idx, :length] = False
        if use_positional_encoding and pos_batch is not None:
            anchor_coord = meta.get("anchor_coord")
            target_coords = meta.get("target_coords") or []
            diffs: List[List[float]] = []
            for pos_idx in range(length):
                coord = target_coords[pos_idx] if pos_idx < len(target_coords) else None
                if anchor_coord is not None and coord is not None:
                    diffs.append([float(coord[0] - anchor_coord[0]), float(coord[1] - anchor_coord[1])])
                else:
                    diffs.append([0.0, 0.0])
            pos_batch[idx, :length] = torch.tensor(diffs, dtype=anchor_batch.dtype)
    return anchor_batch, target_batch, mask_batch, pos_batch

#####################################################################################   

class CrossAttentionAggregator(nn.Module):
    def __init__(
        self,
        anchor_dim: int,
        target_dim: int,
        agg_dim: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        if agg_dim % num_heads != 0:
            raise ValueError("Aggregator hidden dimension must be divisible by the number of heads.")
        self.use_positional_encoding = bool(use_positional_encoding)
        pos_dim = 2 if self.use_positional_encoding else 0
        kv_dim = target_dim + pos_dim
        self.query_proj = nn.Linear(anchor_dim, agg_dim)
        self.key_proj = nn.Linear(kv_dim, agg_dim)
        self.value_proj = nn.Linear(kv_dim, agg_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(agg_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(max(1, num_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(agg_dim) for _ in range(len(self.layers))])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(agg_dim, agg_dim)

    def forward(
        self,
        anchor_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_positional_encoding and pos_encoding is not None:
            kv_input = torch.cat([target_batch, pos_encoding], dim=-1)
        else:
            kv_input = target_batch
        query = self.query_proj(anchor_batch).unsqueeze(1)
        keys = self.key_proj(kv_input)
        values = self.value_proj(kv_input)
        x = query
        for attn_layer, norm in zip(self.layers, self.norms):
            attn_out, _ = attn_layer(x, keys, values, key_padding_mask=key_padding_mask)
            x = norm(x + self.dropout(attn_out))
        fused = self.output_proj(x.squeeze(1))
        return fused


class AggregatorTargetHead(nn.Module):
    def __init__(self, aggregator: CrossAttentionAggregator, proj_target: nn.Module, *, use_positional_encoding: bool) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.proj_target = proj_target
        self.use_positional_encoding = bool(use_positional_encoding)

    def forward(
        self,
        anchor_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = self.aggregator(
            anchor_batch,
            target_batch,
            key_padding_mask=key_padding_mask,
            pos_encoding=pos_encoding if self.aggregator.use_positional_encoding else None,
        )
        return self.proj_target(fused)


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
            run_logger.log(f"[agg] WARNING: NaN gradients detected in epoch {epoch_idx+1}. Corresponding {ihad_nan_grad} batches skipped among total {epoch_batches} batches.")

        train_metrics = _evaluate(train_eval_loader)
        val_metrics = _evaluate(val_loader) if has_validation else None
        
        # Debug: Print what _evaluate returned
        run_logger.log(f"[DEBUG] train_metrics = {train_metrics}")
        run_logger.log(f"[DEBUG] val_metrics = {val_metrics}")

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
            "[agg] epoch {epoch}: train_loss={train_loss}, train_mean_corr={train_corr}, "
            "train_TCC = {train_tcc}, val_loss={val_loss}, val_mean_corr={val_corr}, "
            "val_TCC = {val_tcc}, batches={batches}".format(
                epoch=epoch_idx + 1,
                train_loss=_format(train_loss_log),
                train_corr=_format(train_corr_log),
                train_tcc=_format(train_tcc_log),
                val_loss=_format(val_loss_log),
                val_corr=_format(val_corr_log),
                val_tcc=_format(val_tcc_log),
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
