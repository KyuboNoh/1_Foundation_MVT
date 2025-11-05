# Note: For training cls, pn_index_summary is critical to track positive/negative sample counts. by K.N. 30Oct2025
# Note: The terminology of "anchor" and "target" datasets is used throughout this module to refer to the two datasets. 
#       (Dataset 1 is the anchor and 2 is the target. No semantic meaning beyond that.) by K.N. 30Oct2025

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np
from affine import Affine
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:  # optional plotting for debug mode
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:  # optional raster IO
    import rasterio
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore[assignment]

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency
    CRS = Transformer = None  # type: ignore[assignment]

from .config import AlignmentConfig, load_config
from Common.Unifying.Labels_TwoDatasets.fusion_utils.workspace import (
    DatasetBundle,
    OverlapAlignmentLabels,
    OverlapAlignmentPair,
    OverlapAlignmentWorkspace,
)
from Common.Unifying.Labels_TwoDatasets.datasets import (
    auto_coord_error,
    load_embedding_records,
    summarise_records,
)
from Common.Unifying.Labels_TwoDatasets.overlaps import load_overlap_pairs
from sklearn.model_selection import train_test_split

from Common.cls.infer.infer_maps import (group_coords, mc_predict_map_from_embeddings, write_prediction_outputs,)
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.training.train_cls import (dataloader_metric_inputORembedding, eval_classifier, train_classifier,)
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from Common.Unifying.Labels_TwoDatasets import (
    _normalise_cross_matches,
    _prepare_classifier_labels,
    _build_aligned_pairs_OneToOne,
    _build_aligned_pairs_SetToSet,
    _normalise_coord,
    _normalise_row_col,
)
from Common.Unifying.Labels_TwoDatasets.fusion_utils import (
    align_overlap_embeddings_for_pn_one_to_one as _align_overlap_embeddings_for_pn_OneToOne,
    prepare_fusion_overlap_dataset_one_to_one as _prepare_fusion_overlap_dataset_OneToOne,
    prepare_fusion_overlap_dataset_for_inference as _prepare_fusion_overlap_dataset_for_inference,
)
from Common.Augmentation import (
    _load_augmented_embeddings,
    _print_augmentation_usage,
)
from Common.Unifying.DCCA import (
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
)
from Common.Unifying.Labels_TwoDatasets.splits import _overlap_split_indices


INFERENCE_BATCH_SIZE = 2048


def _progress_iter(iterable, desc: str, *, leave: bool = False, total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave, total=total)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer (positive-only)")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of aligned pairs reserved for validation evaluation (set to 0 to disable).", )
    parser.add_argument("--train-dcca", action=argparse.BooleanOptionalAction, default=True, help="Train the DCCA projection heads (default: true). Use --no-train-dcca to disable.",)
    parser.add_argument("--run-inference", action=argparse.BooleanOptionalAction, default=False, help="Run inference on aligned datasets after training (default: false).",)
    parser.add_argument("--read-dcca",  action=argparse.BooleanOptionalAction, default=False,help="Load existing DCCA projection head weights before training (default: false).",)
    parser.add_argument("--dcca-weights-path", type=str, default=None, help="Optional path to a saved DCCA checkpoint used when --read-dcca is enabled.",)

    parser.add_argument(
        "--train-cls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the PN classifier after DCCA training (default: false).",
    )
    parser.add_argument(
        "--train-cls-method",
        type=int,
        choices=[1, 2],
        default=1,
        help="PN classifier training method to use (1 [Overlap region; Use labels from chosen 'overlapregion_label' in config] or 2 [No label in overlapping region]; default: 1).",
    )
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

    # parser.add_argument("--dcca-eps", type=float, default=1e-5, help="Epsilon value for DCCA covariance regularization (default: 1e-5).")
    # parser.add_argument("--singular-value-drop-ratio", type=float, default=0.01, help="Ratio threshold for dropping small singular values in DCCA (default: 0.01).")
    # parser.add_argument("--tcc-ratio", type=float, default=1.0, help="Fraction of canonical correlations to include when computing TCC (0 < ratio <= 1).", )
    # parser.add_argument("--dcca-mlp-layers", type=int, default=4, help="Number of linear layers used in each DCCA projection head MLP (default: 4).",)

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    print(cfg.dcca_training)
    print(cfg.cls_training)
    

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
    if args.aggregator is not None:
        cfg.aggregator = args.aggregator.lower()
    if args.use_positive_only:
        cfg.use_positive_only = True
    if args.use_positive_augmentation:
        cfg.use_positive_augmentation = True
    debug_mode = bool(args.debug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger = _RunLogger(cfg)

    train_dcca = bool(args.train_dcca)
    read_dcca = bool(args.read_dcca)
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

    workspace = OverlapAlignmentWorkspace(cfg)
    anchor_name, target_name = _dataset_pair(cfg)

    max_coord_error = auto_coord_error(workspace, anchor_name, target_name)
    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    if workspace.overlap is None:
        raise RuntimeError("Overlap pairs are unavailable; ensure overlap_pairs_path points to the integrate_stac output JSON.")
    debug_overlap_stats = _compute_overlap_debug(pairs, anchor_name, target_name) if debug_mode else None
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

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
    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
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
    pairing_mode = (cfg.pairing_mode or "one_to_one").lower()
    if pairing_mode == "set_to_set":
        pair_builder = _build_aligned_pairs_SetToSet
    elif pairing_mode == "one_to_one":
        pair_builder = _build_aligned_pairs_OneToOne
    else:
        raise ValueError(f"Unsupported pairing_mode '{cfg.pairing_mode}'. Expected 'one_to_one' or 'set_to_set'.")

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
    label_matcher = OverlapAlignmentLabels(workspace, pn_label_maps)
    label_cross_matches = label_matcher.build_label_matches(
        pairs=pairs,
        max_coord_error=max_coord_error,
    )
    label_cross_matches = _normalise_cross_matches(label_cross_matches, anchor_name, target_name)

    if debug_mode:
        def _count_labels(entries: List[Dict[str, object]], field: str) -> Counter:
            counter: Counter = Counter()
            for entry in entries:
                value = entry.get(field)
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    for item in value:
                        counter[int(item)] += 1
                else:
                    counter[int(value)] += 1
            return counter
        dataset1_match_counts = _count_labels(label_cross_matches, "label_in_dataset_1")
        dataset2_match_counts = _count_labels(label_cross_matches, "label_in_dataset_2")
        print(f"[debug] label matches in {anchor_name} using label in {target_name}: {dict(dataset1_match_counts)}")
        print(f"[debug] label matches in {target_name} using label in {target_name}: {dict(dataset2_match_counts)}")

    pairs_by_region = defaultdict(lambda: Counter())
    for meta in pair_metadata:
        anchor_region = meta.get("anchor_region", "UNKNOWN")
        for label in meta["target_labels"]:
            pairs_by_region[anchor_region][label] += 1

    pn_index_sets: Dict[str, Dict[str, set[int]]] = {}
    for dataset, regions in pn_index_summary.items():
        pos_union: set[int] = set()
        neg_union: set[int] = set()
        for region, payload in regions.items():
            pos_indices = [int(idx) for idx in payload.get("pos_original_indices", [])]
            neg_indices = [int(idx) for idx in payload.get("neg_original_indices", [])]
            pos_union.update(pos_indices)
            neg_union.update(neg_indices)
        pn_index_sets[dataset] = {"pos": pos_union, "neg": neg_union}

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




    #################################### Training Overlap Alignment (DCCA) ####################################
    if args.train_dcca:
        anchor_tensor = torch.stack(anchor_vecs)
        target_tensor = torch.stack(target_vecs)
        total_pairs = anchor_tensor.size(0)

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

        train_anchor = anchor_tensor[train_indices]
        train_target = target_tensor[train_indices]
        train_dataset = TensorDataset(train_anchor, train_target)

        train_indices_list = train_indices.tolist()
        val_indices_list = val_indices.tolist()
        for idx in train_indices_list:
            pair_metadata[idx]["split"] = "train"
        for idx in val_indices_list:
            pair_metadata[idx]["split"] = "val"

        validation_dataset: Optional[TensorDataset]
        if val_indices.numel() >= 2:
            val_anchor = anchor_tensor[val_indices]
            val_target = target_tensor[val_indices]
            validation_dataset = TensorDataset(val_anchor, val_target)
            actual_validation_fraction = val_indices.numel() / total_pairs
        else:
            validation_dataset = None
            actual_validation_fraction = 0.0

        train_batch_size = min(cfg.dcca_training.batch_size, len(train_dataset))

        objective = cfg.alignment_objective.lower()
        if objective not in {"dcca", "barlow"}:
            raise ValueError(f"Unsupported alignment objective: {objective}")

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

        if read_dcca and weights_path is not None:
            run_logger.log(f"Reading DCCA weights from {weights_path}")
            run_logger.log(
                "Starting stage-1 overlap alignment with initial projection_dim={proj}; "
                "train_pairs={train}, val_pairs={val}, validation_fraction={vf:.3f}, "
                "drop_ratio={drop:.4f}, tcc_ratio={tcc:.4f}, mlp_layers={layers}, "
                "train_dcca={train_flag}, read_dcca={read_flag}".format(
                    proj=cfg.projection_dim,
                    train=len(train_dataset),
                    val=len(validation_dataset) if validation_dataset is not None else 0,
                    vf=actual_validation_fraction,
                    drop=drop_ratio,
                    tcc=tcc_ratio,
                    layers=mlp_layers,
                    train_flag=train_dcca,
                    read_flag=read_dcca,
                )
            )
        
        if objective != "dcca":
            raise NotImplementedError("Only DCCA objective supports adaptive projection control.")

        train_success, projector_a, projector_b, epoch_history, final_proj_dim, failure_reason = _train_DCCA(
            cfg=cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            batch_size=train_batch_size,
            device=device,
            anchor_dim=train_anchor.size(1),
            target_dim=train_target.size(1),
            dcca_eps=dcca_eps_value,            
            run_logger=run_logger,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
            mlp_layers=mlp_layers,
            train_dcca=train_dcca,
            pretrained_state=pretrained_state,
            pretrained_summary=pretrained_summary,
            pretrained_path=weights_path,
        )

        last_history_dcca = epoch_history[-1] if epoch_history else {}
        dcca_summary = {
            "objective": objective,
            "aggregator": cfg.aggregator,
            "use_positive_only": cfg.use_positive_only,
            "use_positive_augmentation": cfg.use_positive_augmentation,
            "projection_dim": final_proj_dim,
            "projection_layers": mlp_layers,
            "train_dcca": train_dcca,
            "read_dcca": read_dcca,
            "dcca_weights_path": str(weights_path) if weights_path is not None else None,
            "pretrained_projection_dim": int(pretrained_summary.get("projection_dim")) if isinstance(pretrained_summary, dict) and pretrained_summary.get("projection_dim") is not None else None,
            "pretrained_projection_layers": int(pretrained_summary.get("projection_layers")) if isinstance(pretrained_summary, dict) and pretrained_summary.get("projection_layers") is not None else None,
            "label_histogram": dict(label_hist),
            "num_pairs": len(anchor_vecs),
            "epochs": cfg.dcca_training.epochs,
            "batch_size": train_batch_size,
            "lr": cfg.dcca_training.lr,
            "max_coord_error": max_coord_error,
            "final_loss": last_history_dcca.get("train_eval_loss", last_history_dcca.get("loss")),
            "final_mean_correlation": last_history_dcca.get("train_eval_mean_correlation", last_history_dcca.get("mean_correlation")),
            "final_train_tcc": last_history_dcca.get("train_eval_tcc"),
            "final_train_tcc_mean": last_history_dcca.get("train_eval_tcc_mean"),
            "final_train_tcc_k": int(last_history_dcca["train_eval_k"]) if last_history_dcca.get("train_eval_k") is not None else None,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset) if validation_dataset is not None else 0,
            "validation_fraction": actual_validation_fraction,
            "tcc_ratio": tcc_ratio,
            "dcca_eps": dcca_eps_value,
            "singular_value_drop_ratio": drop_ratio,
            "augmentation_stats": augmentation_stats,
            "pn_index_summary": pn_index_summary,
            "pn_label_counts": {
                "anchor": _count_pn_lookup(pn_label_maps.get(anchor_name), dataset_region_filters.get(anchor_name)),
                "target": _count_pn_lookup(pn_label_maps.get(target_name), dataset_region_filters.get(target_name)),
            },
            "cross_index_matches": label_cross_matches,
        }

        if not train_success or projector_a is None or projector_b is None:
            raise RuntimeError(f"DCCA training failed: {failure_reason or 'unknown error'}")    

        _persist_state(cfg, projector_a, projector_b, filename="overlap_alignment_stage1_dcca.pt")
        _persist_metrics(
            cfg,
            dcca_summary,
            epoch_history,
            filename="overlap_alignment_stage1_dcca_metrics.json",
        )

        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
            _create_debug_alignment_figures(
                cfg=cfg,
                projector_a=projector_a,
                projector_b=projector_b,
                anchor_tensor=anchor_tensor,
                target_tensor=target_tensor,
                pair_metadata=pair_metadata,
                run_logger=run_logger,
                drop_ratio=drop_ratio,
                tcc_ratio=tcc_ratio,
                dcca_eps=dcca_eps_value,
                device=device,
                sample_seed=int(cfg.seed),
            )
    #################################### Training Overlap Alignment (DCCA) END ####################################

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

    #################################### Training Classifier after Overlap Alignment ####################################
    if args.train_cls:
        
        if not train_dcca:
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
                projector_b = _projection_head_from_state(state_b_dict).to(device)
            else:
                projector_a = projector_a.to(device)
                projector_b = projector_b.to(device)

        classifier_results: Dict[str, Dict[str, object]] = {}
        projector_a.eval()
        projector_b.eval()
        methods_to_run = sorted({int(args.train_cls_method)})
        if debug_mode:
            methods_to_run = [1, 2]
        projector_map = {
            anchor_name: projector_a,
            target_name: projector_b,
        }

        overlap_mask_info = _load_overlap_mask_data(cfg.overlap_mask_path)
        if overlap_mask_info is None and cfg.overlap_mask_path is not None:
            run_logger.log("[cls] Overlap mask unavailable or invalid; proceeding without mask filtering.")

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

        sample_sets: Dict[str, Dict[str, object]] = {}
        label_summary_meta: Dict[str, List[str]] = {}
        for dataset_name, projector in projector_map.items():
            if projector is None:
                run_logger.log(f"[cls] projector unavailable for dataset {dataset_name}; skipping.")
                continue

            sample_set = _collect_classifier_samples(
                workspace=workspace,
                dataset_name=dataset_name,
                pn_lookup=pn_label_maps.get(dataset_name),
                projector=projector,
                device=device,
                run_logger=run_logger,
                overlap_mask=overlap_mask_info,
                apply_overlap_filter=False,
            )

            if sample_set:
                sample_sets[dataset_name] = sample_set
                label_summary_meta[dataset_name] = _summarise_sample_labels(sample_set)

        # 1. compute u from embeddings of dataset A (all region - but overlap-aware) and trained DCCA
        DCCAEmbedding_anchor_all = reembedding_DCCA(workspace, anchor_name, projector_a, device, overlap_mask=overlap_mask_info, mask_only=False, )
        # 1-2. read u (take only overlap region) (simplefusion_anchor)
        DCCAEmbedding_anchor_overlap = _extract_reembedding_overlap(DCCAEmbedding_anchor_all, overlap_mask_info)
        # 2. compute v from embeddings of dataset B (all region - but overlap-aware) and trained DCCA
        DCCAEmbedding_target_all = reembedding_DCCA(workspace, target_name, projector_b, device, overlap_mask=overlap_mask_info, mask_only=False, )
        # 2-2. read v (take only overlap region) (simplefusion_target)
        DCCAEmbedding_target_overlap = _extract_reembedding_overlap(DCCAEmbedding_target_all, overlap_mask_info)

        (
            index_label_anchor_all,
            index_label_anchor_overlap,
            index_label_target_all,
            index_label_target_overlap,
        ) = _prepare_classifier_labels(
            meta_anchor_all=DCCAEmbedding_anchor_all,
            meta_anchor_overlap=DCCAEmbedding_anchor_overlap,
            meta_target_all=DCCAEmbedding_target_all,
            meta_target_overlap=DCCAEmbedding_target_overlap,
            overlapregion_label=cfg.cls_training.overlapregion_label,
            label_cross_matches=label_cross_matches,
            sample_sets=sample_sets,
            pn_label_maps=pn_label_maps,
            anchor_name=anchor_name,
            target_name=target_name,
            debug=debug_mode,
            run_logger=run_logger,
            label_matcher=label_matcher,
        )

        if debug_mode:
            def _summarise_label_indices(tag: str, payload: Dict[str, object]) -> None:
                labels_raw = payload.get("labels") if payload else []
                if isinstance(labels_raw, np.ndarray):
                    labels_list = labels_raw.tolist()
                else:
                    labels_list = list(labels_raw or [])
                counts = _count_label_distribution(labels_list)
                coords_raw = payload.get("coords") if payload else []
                pos_coords: List[Tuple[float, float]] = []
                neg_coords: List[Tuple[float, float]] = []
                for lbl, coord in zip(labels_list, coords_raw or []):
                    if coord is None:
                        continue
                    try:
                        if int(lbl) > 0:
                            pos_coords.append(coord)
                        else:
                            neg_coords.append(coord)
                    except Exception:
                        continue
                run_logger.log(
                    "[debug] {tag} counts {counts}, pos_coords={pos}, neg_coords={neg}".format(
                        tag=tag,
                        counts=counts,
                        pos=len(pos_coords),
                        neg=len(neg_coords),
                    )
                )

            _summarise_label_indices("anchor_all", index_label_anchor_all)
            _summarise_label_indices("anchor_overlap", index_label_anchor_overlap)
            _summarise_label_indices("target_all", index_label_target_all)
            _summarise_label_indices("target_overlap", index_label_target_overlap)

        pn_counts_anchor = _count_pn_lookup(pn_label_maps.get(anchor_name), dataset_region_filters.get(anchor_name))
        pn_counts_target = _count_pn_lookup(pn_label_maps.get(target_name), dataset_region_filters.get(target_name))
        if debug_mode:
            print(f"[debug] PN label counts anchor: {pn_counts_anchor}, target: {pn_counts_target}")

        if not sample_sets:
            run_logger.log("PN classifier skipped: no PN-labelled samples available for classifier training.")
        else:
            method_iter = _progress_iter(methods_to_run, "CLS methods", leave=False, total=len(methods_to_run))

            for method in method_iter:
                run_logger.log(f"Training PN classifier method {method}")
                method_dir = _prepare_classifier_dir(cfg, method, anchor_name, target_name)
                if method_dir is None:
                    run_logger.log(f"Skipping classifier method {method}: unable to resolve output directory.")
                    continue
                metadata_payload: Dict[str, object] = {}
                metadata_payload["overlap_mask_path"] = str(cfg.overlap_mask_path) if cfg.overlap_mask_path is not None else None
                metadata_payload["overlap_mask"] = overlap_mask_info
                if label_summary_meta:
                    metadata_payload["label_summary"] = dict(label_summary_meta)
                metadata_payload["pn_label_counts"] = {
                    "anchor": pn_counts_anchor,
                    "target": pn_counts_target,
                }
                results: Dict[str, object] = {}
                
                if method == 1:
                    #################################### Training Classifier using strong fusion for overlap region ####################################
                    # Construct MLP classifier using "strong fusion" unified method - ϕ=[u;v;∣u−v∣;u⊙v;cos(u,v)] inputs - overlap region only
                    run_logger.log(f"Training PN classifier method {method} - strong fusion head for overlap region")

                    # 1. read u (take only overlap region) 
                    # 2. read v (take only overlap region)
                    (
                        aligned_embedding_anchor_overlap,
                        aligned_embedding_target_overlap,
                        aligned_labels_anchor_overlap,
                        aligned_labels_target_overlap,
                    ) = _align_overlap_embeddings_for_pn_OneToOne(
                        DCCAEmbedding_anchor_overlap,
                        DCCAEmbedding_target_overlap,
                        index_label_anchor_overlap=index_label_anchor_overlap,
                        index_label_target_overlap=index_label_target_overlap,
                        anchor_name=anchor_name,
                        target_name=target_name,
                    )

                    # 3. prepare ϕ (overlap region) and prepare label (overlap region) (synthesize positive (positivie_all=[positive from A, positive from B]) and negative samples)
                    fusion_dataset = _prepare_fusion_overlap_dataset_OneToOne(
                        aligned_embedding_anchor_overlap,
                        aligned_embedding_target_overlap,
                        aligned_labels_anchor_overlap, 
                        aligned_labels_target_overlap,
                        anchor_name=anchor_name,
                        target_name=target_name,
                        method_id="strong"
                    )

                    if debug_mode:
                        print(
                            "[debug] overlap embedding sizes:",
                            DCCAEmbedding_anchor_overlap["features"].shape if DCCAEmbedding_anchor_overlap else None,
                            DCCAEmbedding_target_overlap["features"].shape if DCCAEmbedding_target_overlap else None,
                            len(index_label_anchor_overlap["labels"]) if index_label_anchor_overlap else 0,
                            len(index_label_target_overlap["labels"]) if index_label_target_overlap else 0,
                        )
                        print(
                            "[debug] overlap alignment sizes:",
                            aligned_embedding_anchor_overlap["features"].shape if aligned_embedding_anchor_overlap else None,
                            aligned_embedding_target_overlap["features"].shape if aligned_embedding_target_overlap else None,
                            aligned_labels_anchor_overlap.size,
                            aligned_labels_target_overlap.size,
                        )
                        print(
                            "[debug] prepared data sizes:",
                            fusion_dataset["features"].shape if fusion_dataset else None,
                            fusion_dataset["labels"].size,
                        )

                    if fusion_dataset is None:
                        run_logger.log("[strongfusion] Dataset preparation failed; skipping strong fusion head.")
                    else:
                        train_idx, val_idx = _overlap_split_indices(fusion_dataset, validation_fraction=cfg.cls_training.validation_fraction, seed = cfg.seed, )
                        if train_idx is None and val_idx is None:
                            run_logger.log("[strongfusion] Insufficient class diversity for training; skipping strong fusion head.")
                        else:
                            # 4. Split the data like this:
                            dl_tr, dl_val, metrics_summary_strong = _build_dataloaders(fusion_dataset, train_idx, val_idx, cfg,)

                            if len(dl_tr.dataset) == 0:
                                run_logger.log("[strongfusion] Training loader is empty; skipping strong fusion head.")
                            else:
                                # 5. Train MLP (mlp) with dropout
                                mlp_strong, history_payload, evaluation_summary = _fusion_train_model(fusion_dataset, dl_tr, dl_val, cfg, device, mlp_hidden_dims, mlp_dropout, run_logger,)
                                fusion_dir = Path(method_dir) / "Fusion_Method_1_2_Unified_head_strongfusion"

                                # 6. run inference across overlap datasets after training
                                if args.run_inference:
                                    # Prepare fusion dataset for inference; mind that the number of embeddings are different within overlaping region by datasets due to different resolution. 
                                    # Mind that the all data should be matched to construct phi (phi = [u;v] or phi = [u; v; |u-v|; u*v; cosine(u,v)]) pairs for inference.
                                    # The matching should be based on coordinates within DCCAEmbedding_anchor_overlap, DCCAEmbedding_target_overlap.
                                    fusion_dataset_for_inference = _prepare_fusion_overlap_dataset_for_inference(
                                        DCCAEmbedding_anchor_overlap,
                                        DCCAEmbedding_target_overlap,
                                        method_id="v1_strong",
                                    )
                                    if fusion_dataset_for_inference is not None:
                                        inference_outputs = _fusion_run_inference(
                                            fusion_dataset_for_inference,
                                            mlp_strong,
                                            overlap_mask_info,
                                            device,
                                            cfg,
                                            run_logger,
                                            method_id="strong",
                                        )
                                else:
                                    inference_outputs = {}

                                # 7. Plot results using Common/cls/infer/infer_maps.write_prediction_outputs like write_prediction_outputs(prediction,stack,out_dir, pos_coords_by_region=pos_coord_map, neg_coords_by_region=neg_coord_map, )
                                metrics_summary_strong = dict(metrics_summary_strong)
                                metrics_summary_strong["train_size"] = int(train_idx.size)
                                metrics_summary_strong["val_size"] = int(val_idx.size)
                                fusion_summary = _fusion_export_results(fusion_dir, mlp_strong, history_payload, evaluation_summary, metrics_summary_strong, inference_outputs,)
                                metadata_payload.setdefault("fusion 1-2 strong", fusion_summary)

                    #################################### Training Classifier using simple fusion ####################################
                    # Construct MLP classifier using "simple fusion" unified method - [u, v] inputs
                    run_logger.log(f"Training PN classifier method {method} - simple fusion head")
                    # 1. read u (take only overlap region) 
                    # 2. read v (take only overlap region)
                    (
                        aligned_embedding_anchor_overlap,
                        aligned_embedding_target_overlap,
                        aligned_labels_anchor_overlap,
                        aligned_labels_target_overlap,
                    ) = _align_overlap_embeddings_for_pn_OneToOne(
                        DCCAEmbedding_anchor_overlap,
                        DCCAEmbedding_target_overlap,
                        index_label_anchor_overlap=index_label_anchor_overlap,
                        index_label_target_overlap=index_label_target_overlap,
                        anchor_name=anchor_name,
                        target_name=target_name,
                    )

                    # 3. concatenate [u, v] and prepare label (synthesize positive (positivie_all=[positive from A, positive from B]) and negative samples)
                    fusion_dataset = _prepare_fusion_overlap_dataset_OneToOne(
                        aligned_embedding_anchor_overlap,
                        aligned_embedding_target_overlap,
                        aligned_labels_anchor_overlap, 
                        aligned_labels_target_overlap,
                        anchor_name=anchor_name,
                        target_name=target_name,
                        method_id="simple"
                    )

                    if debug_mode:
                        print(
                            "[debug] overlap embedding sizes:",
                            DCCAEmbedding_anchor_overlap["features"].shape if DCCAEmbedding_anchor_overlap else None,
                            DCCAEmbedding_target_overlap["features"].shape if DCCAEmbedding_target_overlap else None,
                            len(index_label_anchor_overlap["labels"]) if index_label_anchor_overlap else 0,
                            len(index_label_target_overlap["labels"]) if index_label_target_overlap else 0,
                        )
                        print(
                            "[debug] overlap alignment sizes:",
                            aligned_embedding_anchor_overlap["features"].shape if aligned_embedding_anchor_overlap else None,
                            aligned_embedding_target_overlap["features"].shape if aligned_embedding_target_overlap else None,
                            aligned_labels_anchor_overlap.size,
                            aligned_labels_target_overlap.size,
                        )
                        print(
                            "[debug] prepared data sizes:",
                            fusion_dataset["features"].shape if fusion_dataset else None,
                            fusion_dataset["labels"].size,
                        )

                    if fusion_dataset is None:
                        run_logger.log("[fusion] Dataset preparation failed; skipping simple fusion head.")
                    else:
                        train_idx, val_idx = _overlap_split_indices(fusion_dataset, validation_fraction=cfg.cls_training.validation_fraction, seed = cfg.seed, )
                        if train_idx is None and val_idx is None:
                            run_logger.log("[fusion] Insufficient class diversity for training; skipping simple fusion head.")
                        else:
                            # 4. Split the data like this:
                            dl_tr, dl_val, metrics_summary_simple = _build_dataloaders(fusion_dataset, train_idx, val_idx, cfg,)

                            if len(dl_tr.dataset) == 0:
                                run_logger.log("[fusion] Training loader is empty; skipping simple fusion head.")
                            else:
                                # 5. Train MLP (mlp) with dropout
                                mlp_simple, history_payload, evaluation_summary = _fusion_train_model(fusion_dataset, dl_tr, dl_val, cfg, device, mlp_hidden_dims, mlp_dropout, run_logger,)

                                fusion_dir = Path(method_dir) / "Fusion_Method_1_1_Unified_head_simplefusion"

                                # 6. run inference across overlap datasets after training
                                if args.run_inference:
                                    # Prepare fusion dataset for inference; mind that the number of embeddings are different within overlaping region by datasets due to different resolution. 
                                    # Mind that the all data should be matched to construct phi (phi = [u;v] or phi = [u; v; |u-v|; u*v; cosine(u,v)]) pairs for inference.
                                    # The matching should be based on coordinates within DCCAEmbedding_anchor_overlap, DCCAEmbedding_target_overlap.
                                    fusion_dataset_for_inference = _prepare_fusion_overlap_dataset_for_inference(
                                        DCCAEmbedding_anchor_overlap,
                                        DCCAEmbedding_target_overlap,
                                        method_id="v1_simple"
                                    )

                                    if fusion_dataset_for_inference is not None:
                                        inference_outputs = _fusion_run_inference(
                                            fusion_dataset_for_inference,
                                            mlp_simple,
                                            overlap_mask_info,
                                            device,
                                            cfg,
                                            run_logger,
                                            method_id="simple",
                                        )
                                else:
                                    inference_outputs = {}

                                # 7. Plot results using Common/cls/infer/infer_maps.write_prediction_outputs like write_prediction_outputs(prediction,stack,out_dir, pos_coords_by_region=pos_coord_map, neg_coords_by_region=neg_coord_map, )
                                metrics_summary_simple = dict(metrics_summary_simple)
                                metrics_summary_simple["train_size"] = int(train_idx.size)
                                metrics_summary_simple["val_size"] = int(val_idx.size)

                                fusion_summary = _fusion_export_results(fusion_dir, mlp_simple, history_payload, evaluation_summary, metrics_summary_simple, inference_outputs,)
                                metadata_payload.setdefault("fusion 1-1 simple", fusion_summary)


                    # #################################### Training Classifier using strong fusion for all region ####################################
                    # NOT IMPLEMENTED YET
                    #
                    
                elif method == 2:
                    raise NotImplementedError("Method 2 classifier training is not implemented yet.")

                

    return

















def _load_pn_lookup(path: Optional[Path]) -> Optional[Dict[str, set[Tuple[str, int, int]]]]:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    pos_entries = payload.get("pos") or []
    neg_entries = payload.get("neg") or []
    def _to_key(entry: Dict[str, object]) -> Optional[Tuple[str, int, int]]:
        region = entry.get("region")
        row = entry.get("row")
        col = entry.get("col")
        if region is None or row is None or col is None:
            return None
        try:
            return (str(region).upper(), int(row), int(col))
        except Exception:
            return None
    pos_set: set[Tuple[str, int, int]] = set()
    neg_set: set[Tuple[str, int, int]] = set()
    for collection, target in ((pos_entries, pos_set), (neg_entries, neg_set)):
        if not isinstance(collection, list):
            continue
        for entry in collection:
            if isinstance(entry, dict):
                key = _to_key(entry)
                if key is not None:
                    target.add(key)
    return {"pos": pos_set, "neg": neg_set}


def _lookup_pn_label(region: Optional[str], row_col: Optional[Tuple[int, int]], lookup: Optional[Dict[str, set[Tuple[str, int, int]]]]) -> Optional[int]:
    if lookup is None or region is None or row_col is None:
        return None
    region_key = str(region).upper()
    row, col = row_col
    key = (region_key, int(row), int(col))
    if key in lookup["pos"]:
        return 1
    if key in lookup["neg"]:
        return 0
    return None


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_mult: float = 2.0):
        super().__init__()
        hidden = max(32, int(in_dim * hidden_mult))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _prepare_classifier_dir(
    cfg: AlignmentConfig,
    method_id: int,
    *name_tokens: str,
) -> Optional[Path]:
    
    primary = cfg.output_dir if cfg.output_dir is not None else cfg.log_dir
    suffix = f"CLS_Method_{method_id}"
    sanitized: List[str] = []
    for token in name_tokens:
        text = str(token).strip()
        if not text:
            continue
        text = text.replace("\\", "_").replace("/", "_").replace(" ", "_")
        sanitized.append(text)
    if sanitized:
        suffix += "_Data_" + "_".join(sanitized)
    return _prepare_output_dir(primary, suffix)



def _compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    interior = mask.copy()
    interior[:-1, :] &= mask[1:, :]
    interior[1:, :] &= mask[:-1, :]
    interior[:, :-1] &= mask[:, 1:]
    interior[:, 1:] &= mask[:, :-1]
    boundary &= ~interior
    return boundary


def _load_overlap_mask_data(mask_path: Optional[Path]) -> Optional[Dict[str, object]]:
    if mask_path is None:
        return None
    if rasterio is None:
        print(f"[warn] rasterio unavailable; cannot apply overlap mask filtering ({mask_path}).")
        return None
    try:
        with rasterio.open(mask_path) as src:
            mask_array = src.read(1)
            return {
                "array": np.asarray(mask_array),
                "transform": src.transform,
                "shape": mask_array.shape,
                "nodata": src.nodata,
            }
    except Exception as exc:
        print(f"[warn] Unable to load overlap mask {mask_path}: {exc}")
        return None

def _mask_contains_coord(coord: Optional[Tuple[float, float]], mask_info: Optional[Dict[str, object]]) -> bool:
    if mask_info is None:
        return True
    if coord is None or rasterio is None:
        return False
    transform = mask_info.get("transform")
    array = mask_info.get("array")
    shape = mask_info.get("shape")
    if transform is None or array is None or shape is None:
        return True
    try:
        row, col = rasterio.transform.rowcol(transform, coord[0], coord[1])
    except Exception:
        return False
    height, width = shape
    if not (0 <= row < height and 0 <= col < width):
        return False
    value = array[row, col]
    nodata = mask_info.get("nodata")
    try:
        if nodata is not None and np.isfinite(nodata):
            if np.isfinite(value) and np.isclose(value, nodata):
                return False
            if not np.isfinite(value):
                return False
    except Exception:
        pass
    return bool(value)






def _extract_reembedding_overlap(
    entry: Optional[Dict[str, object]],
    overlap_mask: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if entry is None:
        return None
    mask_flags = entry.get("mask_flags")
    if overlap_mask is None:
        return entry
    if mask_flags is None:
        return entry
    keep_idx = [i for i, flag in enumerate(mask_flags) if flag]
    if not keep_idx:
        return None
    index_tensor = torch.as_tensor(keep_idx, dtype=torch.long)
    if isinstance(entry["features"], torch.Tensor):
        features = entry["features"].index_select(0, index_tensor)
    else:
        features = entry["features"][keep_idx]
    labels = entry.get("labels")
    def _slice_list(values):
        if values is None:
            return None
        return [values[i] for i in keep_idx]
    indices_slice = _slice_list(entry.get("indices")) or []
    coords_slice = _slice_list(entry.get("coords")) or []
    metadata_slice = _slice_list(entry.get("metadata")) or []
    row_cols_mask_slice = _slice_list(entry.get("row_cols_mask")) or []
    if isinstance(labels, np.ndarray):
        labels_slice = labels[keep_idx]
    else:
        label_list = _slice_list(labels) or []
        labels_slice = np.asarray(label_list, dtype=np.int16)
    filtered = {
        "dataset": entry.get("dataset"),
        "features": features,
        "indices": list(indices_slice),
        "coords": list(coords_slice),
        "metadata": list(metadata_slice),
        "labels": labels_slice,
        "row_cols": list(row_cols_mask_slice),
        "mask_flags": [True] * len(keep_idx),
        "row_cols_mask": list(row_cols_mask_slice),
    }
    return filtered


class _MaskStack:
    def __init__(self, mask_info: Dict[str, object]):
        shape = mask_info.get("shape")
        self.height = int(shape[0]) if shape is not None else 0
        self.width = int(shape[1]) if shape is not None else 0
        self.transform = mask_info.get("transform")
        self.crs = None

    def grid_centers(self, stride: int) -> List[Tuple[int, int]]:
        centers: List[Tuple[int, int]] = []
        for r in range(0, self.height, stride):
            for c in range(0, self.width, stride):
                centers.append((r, c))
        return centers


def _make_mask_reference(mask_info: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if mask_info is None:
        return None
    mask_array = mask_info.get("array")
    mask_transform = mask_info.get("transform")
    mask_shape = mask_info.get("shape")
    mask_nodata = mask_info.get("nodata")
    if mask_array is None or mask_transform is None or mask_shape is None:
        return None
    mask_arr = np.asarray(mask_array)
    valid_mask = np.isfinite(mask_arr)
    if mask_nodata is not None and np.isfinite(mask_nodata):
        valid_mask &= ~np.isclose(mask_arr, mask_nodata)
    valid_mask &= mask_arr != 0
    height = int(mask_shape[0])
    width = int(mask_shape[1])
    if valid_mask.shape != (height, width):
        valid_mask = valid_mask.reshape((height, width))
    boundary_mask = _compute_boundary_mask(valid_mask) if valid_mask.any() else None
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "transform": mask_transform,
    }
    return {
        "profile": profile,
        "transform": mask_transform,
        "crs": None,
        "valid_mask": valid_mask if valid_mask.any() else None,
        "boundary_mask": boundary_mask,
    }


def _fusion_run_inference(
    dataset: Optional[Dict[str, object]],
    mlp: nn.Module,
    overlap_mask: Optional[Dict[str, object]],
    device: torch.device,
    cfg: AlignmentConfig,
    run_logger: "_RunLogger",
    *,
    method_id: str = "simple",
) -> Dict[str, Dict[str, object]]:
    outputs: Dict[str, Dict[str, object]] = {}

    print("[info] Inference for fusion")
    if dataset is None or dataset.get("features") is None:
        run_logger.log("[fusion] Fusion dataset unavailable; skipping inference.")
        return outputs
    if overlap_mask is None:
        run_logger.log("[fusion] Overlap mask unavailable; skipping inference exports.")
        return outputs
    default_reference = _make_mask_reference(overlap_mask)
    if default_reference is None:
        run_logger.log("[fusion] Failed to build mask reference; skipping inference exports.")
        return outputs
    stack = _MaskStack(overlap_mask)

    features = dataset.get("features")
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(features, np.ndarray):
        features_np = features.astype(np.float32, copy=False)
    else:
        features_np = np.asarray(features, dtype=np.float32)
        if features_np.ndim == 1:
            features_np = features_np.reshape(-1, features_np.shape[0])
    if features_np.size == 0:
        run_logger.log("[fusion] Fusion dataset empty; skipping inference.")
        return outputs

    row_cols_raw = dataset.get("row_cols") or []
    coords_raw = dataset.get("coords") or []
    metadata_raw = dataset.get("metadata") or []
    labels_raw = dataset.get("labels")
    if labels_raw is None:
        labels_np = np.zeros(features_np.shape[0], dtype=np.int16)
    elif isinstance(labels_raw, np.ndarray):
        labels_np = labels_raw.astype(np.int16, copy=False)
    else:
        labels_np = np.asarray(labels_raw, dtype=np.int16)

    embedding_vectors: List[np.ndarray] = []
    lookup: Dict[Tuple[int, int], int] = {}
    coord_list: List[Tuple[int, int]] = []
    metadata_list: List[Dict[str, object]] = []
    labels_list: List[int] = []
    coords_list: List[Optional[Tuple[float, float]]] = []

    for idx, row_col in enumerate(row_cols_raw):
        rc_val = row_col
        if rc_val is None and idx < len(metadata_raw) and isinstance(metadata_raw[idx], dict):
            rc_meta = metadata_raw[idx].get("row_col") or metadata_raw[idx].get("row_col_mask")
            if isinstance(rc_meta, (list, tuple)) and len(rc_meta) >= 2:
                rc_val = rc_meta
        if not isinstance(rc_val, (list, tuple)) or len(rc_val) < 2:
            continue
        try:
            rc_tuple = (int(rc_val[0]), int(rc_val[1]))
        except Exception:
            continue
        if rc_tuple in lookup or idx >= features_np.shape[0]:
            continue
        lookup[rc_tuple] = len(embedding_vectors)
        embedding_vectors.append(features_np[idx])
        coord_list.append(rc_tuple)
        label_val = int(labels_np[idx]) if idx < labels_np.shape[0] else 0
        labels_list.append(label_val)
        meta_entry = metadata_raw[idx] if idx < len(metadata_raw) and isinstance(metadata_raw[idx], dict) else {}
        metadata_list.append(meta_entry)
        coord_val = coords_raw[idx] if idx < len(coords_raw) else None
        if isinstance(coord_val, (list, tuple)) and len(coord_val) >= 2:
            coords_list.append((float(coord_val[0]), float(coord_val[1])))
        else:
            coords_list.append(None)

    if not embedding_vectors:
        run_logger.log("[fusion] No valid overlap samples for inference; skipping.")
        return outputs

    embedding_array = np.vstack(embedding_vectors).astype(np.float32, copy=False)
    passes = max(1, int(getattr(cfg.cls_training, "mc_dropout_passes", 30)))
    prediction = mc_predict_map_from_embeddings(
        {"GLOBAL": (embedding_array, lookup, coord_list)},
        mlp,
        stack,
        passes=passes,
        device=str(device),
        show_progress=True,
    )
    if isinstance(prediction, tuple):
        mean_map, std_map = prediction
        prediction_payload = {"GLOBAL": {"mean": mean_map, "std": std_map}}
    else:
        mean_map = prediction.get("GLOBAL", {}).get("mean") if isinstance(prediction, dict) else None
        std_map = prediction.get("GLOBAL", {}).get("std") if isinstance(prediction, dict) else None
        prediction_payload = prediction

    pos_coords = []
    neg_coords = []
    for label_val, rc, meta in zip(labels_list, coord_list, metadata_list):
        region = meta.get("region") if isinstance(meta, dict) else None
        region = region or "GLOBAL"
        if label_val > 0:
            pos_coords.append((region, rc[0], rc[1]))
        elif label_val <= 0:
            neg_coords.append((region, rc[0], rc[1]))

    pos_map = group_coords(pos_coords, stack) if pos_coords else {}
    neg_map = group_coords(neg_coords, stack) if neg_coords else {}

    mean_values: List[float] = []
    std_values: List[float] = []
    for r, c in coord_list:
        mean_values.append(float(mean_map[r, c]) if mean_map is not None else float("nan"))
        std_values.append(float(std_map[r, c]) if std_map is not None else float("nan"))

    dataset_label = dataset.get("name") or f"fusion_{method_id.lower()}"
    outputs[dataset_label] = {
        "prediction": prediction_payload,
        "default_reference": default_reference,
        "pos_map": pos_map,
        "neg_map": neg_map,
        "counts": {
            "pos": int(sum(len(group) for group in pos_map.values())),
            "neg": int(sum(len(group) for group in neg_map.values())),
        },
        "row_cols": coord_list,
        "coords": coords_list,
        "labels": labels_list,
        "metadata": metadata_list,
        "mean_values": mean_values,
        "std_values": std_values,
    }
    return outputs


def _build_dataloaders(
    dataset: Dict[str, object],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: AlignmentConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    features = dataset["features"]
    labels = dataset["labels"]
    ytr = labels[train_idx]
    yval = labels[val_idx] if val_idx.size else np.empty(0, dtype=int)
    dl_tr, dl_val, metrics_summary = dataloader_metric_inputORembedding(
        train_idx,
        val_idx,
        ytr,
        yval,
        cfg.cls_training.batch_size,
        positive_augmentation=False,
        embedding=features,
        epochs=cfg.cls_training.epochs,
    )
    return dl_tr, dl_val, metrics_summary


def _fusion_train_model(
    dataset: Dict[str, object],
    dl_tr: DataLoader,
    dl_val: DataLoader,
    cfg: AlignmentConfig,
    device: torch.device,
    mlp_hidden_dims: Sequence[int],
    mlp_dropout: float,
    run_logger: "_RunLogger",
) -> Tuple[nn.Module, List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    class _IdentityEncoder(nn.Module):
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return x

    encoder = _IdentityEncoder()
    in_dim = int(dataset["features"].shape[1])
    hidden_dims = tuple(int(h) for h in mlp_hidden_dims if int(h) > 0)
    mlp = MLPDropout(in_dim=in_dim, hidden_dims=hidden_dims, p=float(mlp_dropout))
    mlp, epoch_history = train_classifier(
        encoder,
        mlp,
        dl_tr,
        dl_val,
        epochs=cfg.cls_training.epochs,
        lr=cfg.cls_training.lr,
        device=str(device),
        return_history=True,
    )
    train_eval = eval_classifier(encoder, mlp, dl_tr, device=str(device))
    val_eval = eval_classifier(encoder, mlp, dl_val, device=str(device)) if len(dl_val.dataset) else {
        "loss": float("nan"),
        "weighted_loss": float("nan"),
    }
    run_logger.log("[fusion] Training metrics:")
    log_metrics("fusion train", train_eval, order=DEFAULT_METRIC_ORDER)
    if len(dl_val.dataset):
        log_metrics("fusion val", val_eval, order=DEFAULT_METRIC_ORDER)
    history_payload: List[Dict[str, object]] = []
    for record in epoch_history:
        history_payload.append(
            {
                "epoch": int(record.get("epoch", 0)),
                "train": normalize_metrics(record.get("train", {})),
                "val": normalize_metrics(record.get("val", {})),
            }
        )
    evaluation_summary = {
        "train": normalize_metrics(train_eval),
        "val": normalize_metrics(val_eval),
    }
    return mlp, history_payload, evaluation_summary


def _fusion_export_results(
    fusion_dir: Path,
    mlp: nn.Module,
    history_payload: List[Dict[str, object]],
    evaluation_summary: Dict[str, Dict[str, float]],
    metrics_summary: Dict[str, object],
    inference_outputs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    fusion_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = dict(metrics_summary)
    metrics_payload["evaluation"] = evaluation_summary
    metrics_payload["history"] = history_payload
    metrics_path = fusion_dir / "metrics.json"
    try:
        save_metrics_json(metrics_payload, metrics_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save fusion metrics: {exc}")
    state_path = fusion_dir / "classifier.pt"
    torch.save({"state_dict": mlp.state_dict()}, state_path)
    inference_summary: Dict[str, object] = {}
    for dataset_name, payload in inference_outputs.items():
        out_dir = fusion_dir / dataset_name
        write_prediction_outputs(
            payload["prediction"],
            payload["default_reference"],
            out_dir,
            pos_coords_by_region=payload["pos_map"],
            neg_coords_by_region=payload["neg_map"],
        )
        # predictions_path = out_dir / "predictions.npy"
        predictions_path = out_dir / "predictions.npz"
        try:
            prediction_payload = payload.get("prediction")
            if isinstance(prediction_payload, dict):
                global_payload = prediction_payload.get("GLOBAL") or {}
                mean_map = global_payload.get("mean")
                std_map = global_payload.get("std")
            else:
                mean_map = prediction_payload
                std_map = None
            pos_map = payload.get("pos_map") or {}
            neg_map = payload.get("neg_map") or {}
            pos_coords = [
                (region, int(r), int(c))
                for region, coords in pos_map.items()
                for r, c in coords
            ]
            neg_coords = [
                (region, int(r), int(c))
                for region, coords in neg_map.items()
                for r, c in coords
            ]
            row_cols = [tuple(rc) if isinstance(rc, (list, tuple)) else rc for rc in (payload.get("row_cols") or [])]
            coords_list = payload.get("coords") or [None] * len(row_cols)
            labels = payload.get("labels") or [0] * len(row_cols)
            metadata = payload.get("metadata") or [None] * len(row_cols)
            mean_values = payload.get("mean_values")
            std_values = payload.get("std_values")
            if mean_values is None and mean_map is not None and row_cols:
                mean_values = [float(mean_map[r, c]) for r, c in row_cols]
            if std_values is None and std_map is not None and row_cols:
                std_values = [float(std_map[r, c]) for r, c in row_cols]
            data_payload = {
                "mean": np.asarray(mean_map, dtype=np.float32) if mean_map is not None else None,
                "std": np.asarray(std_map, dtype=np.float32) if std_map is not None else None,
                "row_cols": row_cols,
                "coords": coords_list,
                "labels": labels,
                "metadata": metadata,
                "mean_values": np.asarray(mean_values, dtype=np.float32) if mean_values is not None else None,
                "std_values": np.asarray(std_values, dtype=np.float32) if std_values is not None else None,
                "pos_coords": pos_coords,
                "neg_coords": neg_coords,
            }
            # np.save(predictions_path, data_payload, allow_pickle=True)
            np.savez_compressed(predictions_path, predictions=data_payload)

        except Exception as exc:
            print(f"[warn] Failed to save prediction array for {dataset_name}: {exc}")
        inference_summary[dataset_name] = {
            "output_dir": str(out_dir),
            "npy_path": str(predictions_path),
            "positive_count": payload["counts"]["pos"],
            "negative_count": payload["counts"]["neg"],
        }
    return {
        "metrics_path": str(metrics_path),
        "state_dict_path": str(state_path),
        "evaluation": evaluation_summary,
        "history": history_payload,
        "outputs": inference_summary,
    }

def _collect_classifier_samples(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    pn_lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
    projector: nn.Module,
    device: torch.device,
    run_logger: "_RunLogger",
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
    apply_overlap_filter: bool = False,
) -> Optional[Dict[str, object]]:
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None:
        run_logger.log(f"[cls] dataset {dataset_name} not found in workspace; skipping classifier samples.")
        return None
    if pn_lookup is None or (not pn_lookup.get("pos") and not pn_lookup.get("neg")):
        run_logger.log(f"[cls] PN lookup unavailable for dataset {dataset_name}; skipping classifier samples.")
        return None
    if apply_overlap_filter and overlap_mask is None:
        run_logger.log(f"[cls] overlap mask unavailable; cannot filter samples for dataset {dataset_name}.")
        apply_overlap_filter = False
    matched_records: List[EmbeddingRecord] = []
    labels: List[int] = []
    coords: List[Optional[Tuple[float, float]]] = []
    regions: List[Optional[str]] = []
    row_cols: List[Optional[Tuple[int, int]]] = []
    indices: List[int] = []
    metadata: List[Dict[str, object]] = []
    for record in bundle.records:
        label = _lookup_pn_label(record.region, record.row_col, pn_lookup)
        if label is None:
            continue
        label_int = 1 if int(label) > 0 else 0
        coord_norm = _normalise_coord(record.coord)
        if apply_overlap_filter and not _mask_contains_coord(coord_norm, overlap_mask):
            continue
        matched_records.append(record)
        labels.append(label_int)
        coords.append(coord_norm)
        regions.append(getattr(record, "region", None))
        row_cols.append(getattr(record, "row_col", None))
        indices.append(int(getattr(record, "index", len(indices))))
        metadata.append(
            {
                "dataset": dataset_name,
                "label": label_int,
                "coord": coord_norm,
                "region": getattr(record, "region", None),
                "row_col": getattr(record, "row_col", None),
                "embedding_index": int(getattr(record, "index", len(indices))),
                "tile_id": getattr(record, "tile_id", None),
                "overlap_filtered": bool(apply_overlap_filter),
            }
        )
    if not matched_records:
        run_logger.log(f"[cls] No PN-labelled samples found for dataset {dataset_name}; skipping classifier samples.")
        return None
    embeddings = np.stack([np.asarray(rec.embedding, dtype=np.float32) for rec in matched_records])
    projector = projector.to(device)
    projector.eval()
    with torch.no_grad():
        embed_tensor = torch.from_numpy(embeddings).to(device)
        projected = projector(embed_tensor).detach().cpu()
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return {
        "dataset": dataset_name,
        "features": projected,
        "labels": label_tensor,
        "metadata": metadata,
        "coords": coords,
        "regions": regions,
        "row_cols": row_cols,
        "indices": indices,
    }


def _count_pn_lookup(
    pn_lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
    region_filter: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    if not pn_lookup:
        return {"positive": 0, "negative": 0}

    def _normalise_region(value: Optional[object]) -> str:
        if value is None:
            return "NONE"
        return str(value).upper()

    allowed_regions: Optional[set[str]] = None
    if region_filter:
        allowed_regions = {str(region).upper() for region in region_filter if region is not None}

    def _count(entries: Iterable[Tuple[str, int, int]]) -> int:
        if entries is None:
            return 0
        if not allowed_regions:
            return sum(1 for _ in entries)

        total = 0
        for region_value, _, _ in entries:
            region_key = _normalise_region(region_value)
            if region_key in allowed_regions:
                total += 1
            elif region_key == "NONE" and ("GLOBAL" in allowed_regions or "ALL" in allowed_regions):
                total += 1
        return total

    pos = _count(pn_lookup.get("pos", ()))
    neg = _count(pn_lookup.get("neg", ()))
    return {"positive": int(pos), "negative": int(neg)}


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
