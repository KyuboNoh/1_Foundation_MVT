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
    _resolve_dcca_weights_path,
    _prepare_output_dir,
    _persist_state,
    _persist_metrics,
    _maybe_save_debug_figures,
    _create_debug_alignment_figures,
    dcca_loss,
    reembedding_DCCA,
    _train_DCCA,
    ProjectionHead
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

import torch, torch.nn as nn, torch.nn.functional as F


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
    if args.train_cls_1 or args.train_cls_2 or args.read_inference:
        
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
                            # target_dim=(kv_dim - 2) if pos_enc and key_proj is not None else (kv_dim if key_proj is not None else target_dim or anchor_dim),
                            target_dim=target_dim,
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


        dcca_sets: Dict[str, Dict[str, object]] = {}
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
                batch_size=cfg.dcca_training.batch_size,
                device=device,
                run_logger=run_logger,
                overlap_mask=overlap_mask_info,
                apply_overlap_filter=False,
            )

            run_logger.log(f"[cls] Preparing classifier samples for dataset {dataset_name}... done")

            if sample_set:
                dcca_sets[dataset_name] = sample_set
                label_summary_meta[dataset_name] = _summarise_sample_labels(sample_set)

            # Get overlap-region anchor, anchor (global, overlap) data ready
            if dataset_name == anchor_name and sample_set:
                anchor_overlap_samples = _apply_projector_based_PUNlabels(
                    workspace=workspace, dataset_name=dataset_name, pn_lookup=pn_label_maps.get(dataset_name),
                    projector=projector, batch_size=cfg.dcca_training.batch_size, device=device, run_logger=run_logger,
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
                    projector=projector, batch_size=cfg.dcca_training.batch_size, device=device, run_logger=run_logger,
                    overlap_mask=None, apply_overlap_filter=False,
                )

        # Coordinates of positive samples for plotting
        temp_crd = anchor_non_overlap_samples.get("coords")  # Get coordinates
        temp_labels = anchor_non_overlap_samples["labels"]
        yA_pu = torch.where(temp_labels >= 0.9, 1, -1).long()        
        coords_array = np.array(temp_crd)
        pos_mask = (yA_pu == 1).cpu().numpy()                   # Separate positive and negative samples
        pos_crd_anchor_non_overlap_plot = [coords_array[i] for i in range(len(coords_array)) if pos_mask[i]]        # Example: Extract coordinates of positive samples (labels == 1)

        temp_crd = anchor_overlap_samples.get("coords")  # Get coordinates
        temp_labels = anchor_overlap_samples["labels"]
        yA_pu = torch.where(temp_labels >= 0.9, 1, -1).long()        
        coords_array = np.array(temp_crd)
        pos_mask = (yA_pu == 1).cpu().numpy()                   # Separate positive and negative samples
        pos_crd_anchor_overlap_plot = [coords_array[i] for i in range(len(coords_array)) if pos_mask[i]]        # Example: Extract coordinates of positive samples (labels == 1)

        #################################### Training Cls - 0) GET Training data ready (END) ####################################

    if args.train_cls_1:
        #################################### Training Cls - 1) GLOBAL\OVERLAP using GLOBAL data (START) ####################################
        # data_use = anchor_non_overlap_samples
        data_use = dcca_sets[anchor_name]

        # Stat training of nnPU from (Global\Overlap) Anchor (Global) data
        if data_use is None:
            raise RuntimeError("Anchor non-overlap samples unavailable for classifier training; aborting.")
        features = data_use["features"].float().to(device)                                 # DDCA-projected features (u)
        yA_pu = torch.where(data_use["labels"] >= 0.9, 1, -1).long().to(device)      # nnPU labels (+1/-1) from PN labels

        temp_crd = data_use.get("coords")  # Get coordinates
        temp_labels = data_use["labels"]
        coords_array = np.array(temp_crd)
        pos_mask = (yA_pu == 1).cpu().numpy()                   # Separate positive and negative samples
        pos_crd_plot = [coords_array[i] for i in range(len(coords_array)) if pos_mask[i]]        # Example: Extract coordinates of positive samples (labels == 1)

        # -- Create train/validation split for classifier training
        val_frac = max(0.0, min(cfg.cls_training.validation_fraction, 0.9))

        if args.train_cls_1_Method == "PU":
            total = int(features.size(0))
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

            features_train = features[train_idx_list].contiguous() if train_idx_list else features.new_empty((0, features.size(1)))
            yA_pu_train = yA_pu[train_idx_list].contiguous() if train_idx_list else yA_pu.new_empty((0,))
            features_val = features[val_idx_list].contiguous() if val_idx_list else features.new_empty((0, features.size(1)))
            yA_pu_val = yA_pu[val_idx_list].contiguous() if val_idx_list else yA_pu.new_empty((0,))

            train_ds = TensorDataset(features_train, yA_pu_train)
            val_ds   = TensorDataset(features_val,   yA_pu_val)

            # --- indices for P and U ---
            pos_idx = (yA_pu_train == 1).nonzero(as_tuple=True)[0].tolist()
            unl_idx = (yA_pu_train != 1).nonzero(as_tuple=True)[0].tolist()
            if len(pos_idx) == 0:
                raise RuntimeError("No positives in training set; cannot construct PU batches.")

            # choose at least 2 positive per batch:
            # k_pos = min(int(max(2, math.ceil(0.05 * cfg.cls_training.batch_size))) , len(pos_idx))  # e.g., 5% of batch or at least 2
            k_pos = min(int(max(2, math.ceil(0.25 * cfg.cls_training.batch_size))) , len(pos_idx))  # e.g., 25% of batch or at least 2

            k_unl = cfg.cls_training.batch_size - k_pos
            PUsampler = PUBatchSampler(pos_idx, unl_idx, k_pos=k_pos, k_unl=k_unl, seed=cfg.seed)

            # IMPORTANT: when using batch_sampler, do not pass batch_size/shuffle
            train_loader = DataLoader(train_ds, batch_sampler=PUsampler, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=cfg.cls_training.batch_size, shuffle=False, num_workers=0)

            # --- global prior (use your dataset-wide estimate; don't recompute per batch) ---
            pi_p = float((yA_pu == 1).sum().item()) / max(1, int(yA_pu.numel()))
            pi_p = float(min(max(pi_p, 1e-4), 0.1))     # clamp only to keep numerics sane

            encA = nn.Identity(); pA = nn.Identity()    # we are already in u-space

            # TODO: Add loss_type in input configuration

            # basic loss_type:
            # gA = train_nnpu_a_only(encA, pA, train_loader, val_loader, pi_p, lr=cfg.dcca_training.lr, epochs=cfg.cls_training.epochs, device=device, loss_type='basic')
            
            # weighted loss_type:
            gA = train_nnpu_a_only(encA, pA, train_loader, val_loader, pi_p, lr=cfg.dcca_training.lr, epochs=cfg.cls_training.epochs, device=device, loss_type='weighted',
                                    alpha=0.9, w_p_max=20.0,
                                    focal_gamma=0.5, prior_penalty=10.0,
                                    logit_adjust_tau=0.0)   # optional for metrics calibration
            # gA = train_nnpu_a_only(encA, pA, train_loader, val_loader, pi_p, lr=cfg.dcca_training.lr, epochs=cfg.cls_training.epochs, device=device, loss_type='weighted',
            #                         alpha=0.75, w_p_max=25.0,
            #                         focal_gamma=1.5, prior_penalty=5.0,
            #                         logit_adjust_tau=1.0)   # optional for metrics calibration

        elif args.train_cls_1_Method == "PN":
            # Phase 1: Negative Selection & Dataset Preparation
            pos_idx = (yA_pu == 1).nonzero(as_tuple=True)[0].tolist()
            unl_idx = (yA_pu != 1).nonzero(as_tuple=True)[0].tolist()
            pos_idx_arr = np.asarray(pos_idx, dtype=int)
            unk_idx_arr = np.asarray(unl_idx, dtype=int)

            if debug_mode:
                run_logger.log(f"[cls-1-PN] Starting negative selection: {len(pos_idx_arr)} positives, {len(unk_idx_arr)} unlabeled")

            run_logger.log(f"[cls-1-PN] Converting features to numpy...")
            run_logger.log(f"[cls-1-PN] Starting negative selection...")

            neg_idx_region = pu_select_negatives(
                Z_all = features.cpu().numpy(),  
                pos_idx = pos_idx_arr,
                unk_idx = unk_idx_arr,
                filter_top_pct = args.filter_top_pct,
                negatives_per_pos = args.negs_per_pos,
            )
            if debug_mode:
                run_logger.log(f"[cls-1-PN] Finished negative selection: selected {len(neg_idx_region)} negatives")

            neg_idx_arr = np.asarray(neg_idx_region, dtype=int)

            # Phase 2: Create Balanced P+N Dataset
            # Create balanced PN dataset
            pn_indices = np.concatenate([pos_idx_arr, neg_idx_arr], axis=0)  # Selected P+N
            inf_indices = np.setdiff1d(np.arange(len(features)), pn_indices)  # Remaining for inference

            # Extract features and labels for P+N
            features_pn = features[pn_indices]  # [N_pn, d]
            yA_pn = torch.where(
                torch.tensor([i in pos_idx for i in pn_indices], dtype=torch.bool),
                1,  # Positive
                0   # Negative (changed from -1 to 0 for BCE loss)
            ).long()

            # Extract coordinates for dataloaders
            coords_pn = [data_use["coords"][i] for i in pn_indices]

            # Phase 3: Train/Val Split on P+N
            # Split P+N dataset into train/val
            total_pn = len(pn_indices)
            val_count = int(total_pn * val_frac)

            gen = torch.Generator()
            gen.manual_seed(int(cfg.seed))
            indices_pn = torch.randperm(total_pn, generator=gen)

            val_indices_pn = indices_pn[:val_count].tolist()
            train_indices_pn = indices_pn[val_count:].tolist()

            # Create train/val data
            Xtr = features_pn[train_indices_pn]  # Features OR coordinates
            ytr = yA_pn[train_indices_pn]  # Labels (0/1)
            Xval = features_pn[val_indices_pn]
            yval = yA_pn[val_indices_pn]

            # Phase 4: Create DataLoaders
            # Option A: Use embeddings directly (recommended since you already have features)
            dl_tr, dl_va, metrics_summary_append = dataloader_metric_inputORembedding(
                Xtr=Xtr,  # [N_tr, d] tensor
                Xval=Xval,  # [N_val, d] tensor
                ytr=ytr,  # [N_tr] tensor
                yval=yval,  # [N_val] tensor
                batch_size=cfg.cls_training.batch_size,
                positive_augmentation=False,  
                augmented_patches_all=None,
                pos_coord_to_index=None,
                window_size=None,
                stack=None,  # No need for MaskStack since using embeddings
                embedding=True,  
                epochs=cfg.cls_training.epochs
            )

            # Phase 5 : Build Model & Train
            # Determine input dimension
            in_dim = features.size(1)  # Already in projected space (e.g., 128)

            # Build classifier (matching reference)
            hidden_dims = mlp_hidden_dims  # From args, e.g., [256, 128]
            gA = MLPDropout(in_dim=in_dim, hidden_dims=hidden_dims, p=float(mlp_dropout)).to(device)

            # Encoder is identity since we're already in projected space
            encA = nn.Identity().to(device)

            # Train classifier (matching reference)
            print("[info] Training PN classifier...")
            gA, epoch_history = train_classifier(
                encA,  # Identity encoder
                gA,    # MLPDropout classifier
                dl_tr,
                dl_va,
                epochs=cfg.cls_training.epochs,
                return_history=True,
                loss_weights={'bce': 1.0},  # Binary cross-entropy
            )

        else:
            raise RuntimeError(f"Unsupported cls_1 training method: {args.train_cls_1_Method}")

        # After training gA inference
        # inference_results["gA_Overlap"] = run_inference_gA_only(
        #     samples=anchor_overlap_samples,
        #     gA=gA,
        #     device=device,
        #     cfg=cfg,
        #     output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "Overlap",
        #     run_logger=run_logger,
        #     passes=cfg.cls_training.mc_dropout_passes,
        #     pos_crd = pos_crd_anchor_overlap_plot
        # )

        # inference_results["gA_Non_Overlap"] = run_inference_gA_only(
        #     samples=anchor_non_overlap_samples,
        #     gA=gA,
        #     device=device,
        #     cfg=cfg,
        #     output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "Non_Overlap",
        #     run_logger=run_logger,
        #     passes=cfg.cls_training.mc_dropout_passes,
        #     pos_crd = pos_crd_anchor_non_overlap_plot
        # )

        inference_results["gA_Overlap"] = run_inference_gA_only(
            samples=anchor_overlap_samples,
            gA=gA,
            device=device,
            cfg=cfg,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "All_Overlap",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes,
            pos_crd = pos_crd_anchor_overlap_plot
        )

        inference_results["gA_All"] = run_inference_gA_only(
            samples=data_use,
            gA=gA,
            device=device,
            cfg=cfg,
            output_dir=Path(cfg.output_dir) / "cls_1_inference_results" / "All",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes,
            pos_crd = pos_crd_plot
        )
        #################################### Training Cls - 1) GLOBAL\OVERLAP using GLOBAL data (END) ####################################

    if args.train_cls_1 or args.train_cls_2:
        # After training gA (around line 1037)
        if gA is not None:
            # Get input dimension based on the model type
            if isinstance(gA, PNHeadAOnly):
                # PNHeadAOnly has .mlp[0] structure
                d = gA.mlp[0].in_features  # Get dimension from first linear layer
                architecture = 'PNHeadAOnly'
            elif isinstance(gA, MLPDropout):
                # MLPDropout has .net[0] structure
                d = gA.net[0].in_features  # Get dimension from first linear layer
                architecture = 'MLPDropout'
            else:
                raise RuntimeError(f"Unknown gA model type: {type(gA)}")

            # Save A-only PN head
            gA_state = {
                'state_dict': gA.state_dict(),
                'input_shape': d,  # feature dimension
                'architecture': architecture  # Store actual architecture type
            }
            save_path = Path(cfg.output_dir) / "cls_1_nonoverlap.pt"
            torch.save(gA_state, save_path)
            run_logger.log(f"[cls] Saved A-only PN head (architecture={architecture}, input dim={d}) to {save_path}")        

        # After gA training and before gAB training 
        if gA is None:
            # Try to load saved A-only PN head
            gA_path = Path(cfg.output_dir) / "cls_1_nonoverlap.pt"
            if gA_path.exists():
                try:
                    gA_state = torch.load(gA_path)
                    d = gA_state['input_shape']
                    architecture = gA_state.get('architecture', 'PNHeadAOnly')  # Default to old format
                    
                    # Reconstruct the correct model type
                    if architecture == 'PNHeadAOnly':
                        gA = PNHeadAOnly(d=d).to(device)
                    elif architecture == 'MLPDropout':
                        # Need to infer hidden_dims from the saved state
                        # For now, use default hidden dims
                        gA = MLPDropout(in_dim=d, hidden_dims=mlp_hidden_dims, p=mlp_dropout).to(device)
                    else:
                        raise ValueError(f"Unknown architecture type: {architecture}")
                    
                    gA.load_state_dict(gA_state['state_dict'])
                    gA.eval()
                    run_logger.log(f"[cls] Loaded A-only PN head (architecture={architecture}, input dim={d}) from {gA_path}")
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

    if args.train_cls_2:
        u_exp, v_exp, bmiss_exp = build_cls2_dataset_from_dcca_pairs(anchor_vecs, target_stack_per_anchor, projector_a, projector_b, device)
        run_logger.log(f"[cls] For cls-2; Pair expansion ended up from {len(anchor_vecs), len(target_stack_per_anchor)} to {u_exp.shape[0], v_exp.shape[0]}")

        dataset_uv = UVDataset(u_exp, v_exp, bmiss_exp)
        gAB = fit_unified_head_OVERLAP_from_uv(gA, DataLoader(dataset_uv, batch_size=cfg.cls_training.batch_size, shuffle=True),
                                               d_u=u_exp.size(1), device=device, lr=cfg.cls_training.lr, 
                                               steps=cfg.cls_training.epochs
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
            output_dir=Path(cfg.output_dir) / "cls_2_inference_results" / "Overlap",
            run_logger=run_logger,
            passes=cfg.cls_training.mc_dropout_passes,
            target_vecs=target_vecs,  # if projector_b is a plain ProjectionHead (no transformer)
            batch_size=cfg.cls_training.batch_size,
            pos_crd=pos_crd_anchor_overlap_plot
        )

        #################################### Training Cls - 2) GLOBAL\OVERLAP using GLOBAL data (END) ####################################

    if args.train_cls_2:
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
            # Try to load saved unified PN head
            gAB_path = Path(cfg.output_dir) / "cls_2_unified.pt"
            if gAB_path.exists():
                try:
                    gAB_state = torch.load(gAB_path)
                    d = gAB_state['input_shape']
                    gAB = PNHeadUnified(d=d).to(device)  # Fixed: use PNHeadUnified instead of PNHeadAOnly
                    gAB.load_state_dict(gAB_state['state_dict'])
                    gAB.eval()
                    run_logger.log(f"[cls] Loaded unified PN head (input dim={d}) from {gAB_path}")
                except Exception as e:
                    run_logger.log(f"[cls] Failed to load unified PN head from {gAB_path}: {str(e)}")
                    raise RuntimeError("Could not train or load unified PN head required for unified training")
            else:
                raise RuntimeError("Unified PN head not found and training failed; cannot proceed with unified training")

        yA_pu = torch.where(anchor_non_overlap_samples["labels"] >= 0.9, 1, -1).long().to(device)      # nnPU labels (+1/-1) from PN labels

    if args.read_inference:
        # Read gA inference results
        output_dir = Path(cfg.output_dir) / "cls_1_inference_results" / "Non_Overlap"
        inference_results["gA_Non_Overlap"] = _read_inference(
            cfg=cfg,
            output_dir=output_dir,
        )

        _create_inference_plots(
            model_dir=Path(inference_results["gA_Non_Overlap"]["model_dir"]),
            model_name="gA",
            coords=inference_results["gA_Non_Overlap"]["coordinates"],
            mean_pred=inference_results["gA_Non_Overlap"]["predictions_mean"],
            std_pred=inference_results["gA_Non_Overlap"]["predictions_std"],
            pos_crd=pos_crd_anchor_non_overlap_plot
        )

        # Read gAB inference results
        output_dir_gAB = Path(cfg.output_dir) / "cls_2_inference_results" / "Overlap"
        inference_results["gAB_Overlap"] = _read_inference(
            cfg=cfg,
            output_dir=output_dir_gAB,
        )

        _create_inference_plots(
            model_dir=Path(inference_results["gAB_Overlap"]["model_dir"]),
            model_name="gAB",
            coords=inference_results["gAB_Overlap"]["coordinates"],
            mean_pred=inference_results["gAB_Overlap"]["predictions_mean"],
            std_pred=inference_results["gAB_Overlap"]["predictions_std"],
            pos_crd=pos_crd_anchor_overlap_plot
        )

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
    samples: Dict,
    gA: nn.Module,
    device: torch.device,
    output_dir: Path,
    run_logger: "_RunLogger",
    passes: int = 10,
    pos_crd: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, object]:
    """Run inference on overlap data using only the gA (anchor-only) model."""
    
   
    # Get corresponding metadata
    matched_coords = samples["coords"]
    features = samples["features"].float().to(device)
    run_logger.log(f"[inference-gA] Processing {len(matched_coords)} anchor samples")
    
    # Prepare output directory
    model_dir = output_dir / "gA"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # MC Dropout inference
    gA.train()  # Enable dropout
    predictions_list = []
    
    # ✅ Check if model outputs probabilities or logits (To handle both PN and PU cases)
    outputs_probs = isinstance(gA, MLPDropout)  # MLPDropout outputs probabilities directly

    with torch.no_grad():
        for _ in range(passes):
            output = gA(features)
            
            if outputs_probs:
                # ✅ Model already outputs probabilities, use directly
                pred = output
            else:
                # ✅ Model outputs logits, apply sigmoid
                pred = torch.sigmoid(output)
            
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
        std_pred=std_pred,
        pos_crd=pos_crd
    )

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
    gA: nn.Module,
    data_loader_uv: DataLoader,
    d_u: int,
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 10,
    lambda_cons: float = 0.5,
    lambda_focal: float = 2.0,    # ✅ NEW: focal weight for positives
    view_dropout: float = 0.1,
    noise_sigma: float = 0.0
) -> nn.Module:
    
    # ✅ Check what type of outputs gA produces
    gA_outputs_probs = isinstance(gA, MLPDropout)  # MLPDropout outputs probs
    
    # Build unified head
    gAB = PNHeadUnified(d=d_u).to(device)
    gAB_outputs_probs = True  # PNHeadUnified outputs probabilities
    
    opt = torch.optim.AdamW(gAB.parameters(), lr=lr, weight_decay=1e-4)
    eps = 1e-6
    
    for epoch in range(1, steps + 1):
        epoch_loss = 0.0
        epoch_loss_kd = 0.0
        epoch_loss_focal = 0.0
        epoch_loss_ent = 0.0
        epoch_loss_cons = 0.0
        batch_count = 0
        
        pbar = tqdm(data_loader_uv, desc=f"    [cls - 2] Epoch {epoch:3d}/{steps}", leave=False)
        
        for u, v, bmiss in pbar:
            u, v, bmiss = u.to(device), v.to(device), bmiss.to(device)
            
            with torch.no_grad():
                # Optional noise for robustness
                if noise_sigma > 0:
                    u_noisy = u + noise_sigma * torch.randn_like(u)
                    v_noisy = v + noise_sigma * torch.randn_like(v)
                else:
                    u_noisy, v_noisy = u, v
                
                # ✅ Get teacher predictions in PROBABILITY space
                teacher_output = gA(u_noisy)
                if gA_outputs_probs:
                    teacher = teacher_output.detach()
                else:
                    teacher = torch.sigmoid(teacher_output).detach()  # Convert logits to probs
            
            # View dropout: teach robustness when B is absent
            if torch.rand(1).item() < view_dropout:
                v_in = torch.zeros_like(v_noisy)
                b_in = torch.ones(u_noisy.size(0), 1, device=device)
            else:
                v_in = v_noisy
                b_in = bmiss
            
            # ✅ Student predictions (ensure probabilities)
            student_output = gAB(u_noisy, v_in, b_in)
            if gAB_outputs_probs:
                p_student = student_output
            else:
                p_student = torch.sigmoid(student_output)
            
            # ✅ Consistency target (ensure probabilities)
            cons_output = gAB(u_noisy, v, torch.zeros_like(b_in))
            if gAB_outputs_probs:
                p_cons = cons_output
            else:
                p_cons = torch.sigmoid(cons_output)

            # ✅ NEW: Focal-weighted KD loss
            # Upweight samples where teacher predicts high probability
            with torch.no_grad():
                # Focal weight: higher for teacher's positive predictions
                focal_weight = teacher.pow(2.0)  # Weight = teacher²
                # Normalize so mean weight ≈ 1 (prevents loss scale issues)
                focal_weight = focal_weight / (focal_weight.mean() + eps)

            # Inside training loop, add after getting teacher predictions:
            if batch_count == 0:  # First batch of epoch
                print(f"    [debug] Teacher predictions: mean={teacher.mean().item():.3f}, "
                    f"std={teacher.std().item():.3f}, min={teacher.min().item():.3f}, max={teacher.max().item():.3f}")
                print(f"    [debug] Student predictions: mean={p_student.mean().item():.3f}, "
                    f"std={p_student.std().item():.3f}, min={p_student.min().item():.3f}, max={p_student.max().item():.3f}")

            # ✅ All losses now operate on probabilities [0,1]
            kd_errors = (p_student - teacher).pow(2)
            loss_kd_base = kd_errors.mean()  # Base MSE
            loss_focal = (focal_weight * kd_errors).mean()  # Focal MSE
            loss_cons = F.mse_loss(p_student, p_cons.detach())
            # loss_kd_base = F.binary_cross_entropy(p_student, teacher)     # Alternative: BCE
            
            # ✅ Total loss (removed entropy, added focal)
            loss = loss_kd_base + lambda_focal * loss_focal + lambda_cons * loss_cons
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            epoch_loss_kd += loss_kd_base.item()
            epoch_loss_focal += loss_focal.item()
            # epoch_loss_ent += loss_ent.item()
            epoch_loss_cons += loss_cons.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'kd': f'{loss_kd_base.item():.4e}',
                'focal': f'{loss_focal.item():.4e}',
                # 'ent': f'{loss_ent.item():.4e}',
                'cons': f'{loss_cons.item():.4e}'
            })
        
        pbar.close()
        
        # Print epoch summary
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        avg_kd = epoch_loss_kd / batch_count if batch_count > 0 else 0.0
        avg_focal = epoch_loss_focal / batch_count if batch_count > 0 else 0.0
        # avg_ent = epoch_loss_ent / batch_count if batch_count > 0 else 0.0
        avg_cons = epoch_loss_cons / batch_count if batch_count > 0 else 0.0
        
        print(f"    [cls - 2] Epoch {epoch:3d} | "
              f"loss={avg_loss:.4e} | "
              f"kd={avg_kd:.4e} | "
              f"focal={avg_focal:.4e} | "
            #   f"ent={avg_ent:.4e} | "
              f"cons={avg_cons:.4e}")
    
    print(f"\n    [cls - 2] Training completed after {steps} epochs")
    return gAB.eval()


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

def train_nnpu_a_only(
    encA, pA, train_loader, val_loader, pi_p,
    lr: float = 1e-3, epochs: int = 3, device: str = "cuda",
    loss_type: str = "weighted",         # 'basic' or 'weighted'
    # weighted-loss knobs:
    alpha: float = 0.75,                 # w_p ~ (U/P)^alpha
    w_p_max: float = 25.0,               # cap on w_p
    focal_gamma: float | None = 1.5,     # None to disable
    prior_penalty: float | None = 5.0,   # None/0 to disable
    # extras:
    grad_clip: float | None = 2.0,
    logit_adjust_tau: float | None = None  # prior-aware bias for metrics; None to skip
):                    
    # Probe a batch to infer feature dimension; handle empty loader gracefully.

    # Trains PNHeadAOnly on projected features u = pA(encA(x)).
    # Labels are +1 (positive) and -1 (unlabeled) in the loaders.
    # loss_type='basic'   -> classic non-negative PU (no weights/focal/prior)
    # loss_type='weighted'-> positive-upweighted,

    try:
        it = iter(train_loader)
        x0, _ = next(it)
    except StopIteration:
        raise RuntimeError("Training DataLoader is empty; cannot train PN head.")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch a probe batch from train_loader: {exc}")

    encA = encA.to(device);  pA = pA.to(device)
    with torch.no_grad():
        u0 = pA(encA(x0.to(device)))
    if u0.ndim < 2:
        raise RuntimeError("Projected features have unexpected shape; expected (B, D).")
    d = int(u0.shape[1])

    gA = PNHeadAOnly(d=d).to(device)
    opt = torch.optim.AdamW(gA.parameters(), lr=lr, weight_decay=1e-4)
    
    if loss_type == 'basic':
        w_p = 1.0
        w_n = 1.0
        focal_gamma = None
        prior_penalty = None
    elif loss_type == 'weighted':
        # --- Estimate dataset-level P/U counts once to set w_p ---
        with torch.no_grad():
            P = 0; U = 0
            for _, pu in train_loader:
                pu = pu.to(device)
                P += int((pu == 1).sum().item())
                U += int((pu != 1).sum().item())
        ratio = (U / max(1, P)) if P > 0 else 100.0
        w_p = float(min(w_p_max, ratio**alpha))   # e.g., ~ (U/P)^0.75 but capped
        w_n = 1.0
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


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
            out = gA(u)
            logits = _ensure_logits(out)

            pos_mask = (pu == 1)
            unl_mask = ~pos_mask
            if not pos_mask.any() or not unl_mask.any():
                # ensure sampler includes both; if not, skip batch
                continue
            logits_p = logits[pos_mask]
            logits_u = logits[unl_mask]

            if loss_type == 'weighted':
                loss = nnpu_weighted_loss_from_logits(
                    logits_p, logits_u, pi_p,
                    w_p=w_p, w_n=w_n,
                    focal_gamma=focal_gamma,
                    prior_penalty=prior_penalty
                )
            elif loss_type == 'basic':
                loss = nnpu_basic_loss_from_logits(logits_p, logits_u, pi_p)

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(gA.parameters(), grad_clip)            
            opt.step()

            running_loss += float(loss.item()) * int(xA.size(0))
            sample_count += int(xA.size(0))

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

        # optional prior-aware logit adjustment for metrics only
        if logit_adjust_tau is not None:
            # If you pass a probability here (e.g., 0.5 for balanced), compute delta from priors;
            # otherwise treat logit_adjust_tau as the direct log-odds shift.
            if 0.0 < float(logit_adjust_tau) < 1.0:
                pi_s = float(pi_p)
                pi_t = float(logit_adjust_tau)
                delta = math.log(pi_t/(1.0-pi_t)) - math.log(pi_s/(1.0-pi_s))
            else:
                delta = float(logit_adjust_tau)

            train_probs = torch.sigmoid(_ensure_logits(train_probs) + delta)
            val_probs   = torch.sigmoid(_ensure_logits(val_probs)   + delta)
        else:
            # ensure probabilities for metrics/BCE
            train_probs = torch.sigmoid(_ensure_logits(train_probs))
            val_probs   = torch.sigmoid(_ensure_logits(val_probs))

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

        # monitor predicted positive rate on U (val set proxy)
        with torch.no_grad():
            mu_pred = float(val_probs.mean().item()) if val_probs.numel() else float('nan')
        prior_mse = (mu_pred - float(pi_p))**2 if math.isfinite(mu_pred) else float('nan')

        # BCE for logging only (targets are 0/1 here)
        try:
            import torch.nn.functional as F
            val_loss_bce = F.binary_cross_entropy(val_probs, val_targets.float()).item() if val_targets.numel() else float('nan')
        except Exception:
            val_loss_bce = float('nan')

        # model selection by F1 on val
        val_f1 = val_metrics.get("f1", float('nan'))
        if not math.isnan(val_f1) and val_f1 > best["f1"]:
            best = {"f1": val_f1, "state_dict": gA.state_dict()}

        # Format loss separately in scientific notation
        train_loss_str = f"{avg_loss:.4e}" if math.isfinite(avg_loss) else "nan"
        val_loss_str = f"{val_loss_bce:.4e}" if math.isfinite(val_loss_bce) else "nan"

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
            "val": {"loss": float(val_loss_bce), **normalize_metrics(val_log)}
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
