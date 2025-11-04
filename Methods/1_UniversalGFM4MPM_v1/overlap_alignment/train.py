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
from .workspace import (DatasetBundle, OverlapAlignmentPair, OverlapAlignmentWorkspace, OverlapAlignmentLabels,)
from .datasets import auto_coord_error
from Common.overlap_debug_plot import save_overlap_debug_plot
from sklearn.model_selection import train_test_split

from Common.cls.infer.infer_maps import (group_coords, mc_predict_map_from_embeddings, write_prediction_outputs,)
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.training.train_cls import (dataloader_metric_inputORembedding, eval_classifier, train_classifier,)
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from Common.Unifying.Labels_TwoDatasets import (
    _normalise_cross_matches, _prepare_classifier_labels, _build_aligned_pairs, _serialise_sample, _normalise_coord, _normalise_row_col)
from Common.Unifying.Labels_TwoDatasets.fusion_utils import (
    align_overlap_embeddings_for_pn_one_to_one as _align_overlap_embeddings_for_pn_OneToOne,
    prepare_fusion_overlap_dataset_one_to_one as _prepare_fusion_overlap_dataset_OneToOne,
    prepare_fusion_overlap_dataset_for_inference as _prepare_fusion_overlap_dataset_for_inference,
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
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
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

    (anchor_vecs, target_vecs, label_hist, debug_data, 
     augmentation_stats, pair_metadata, pn_index_summary) = _build_aligned_pairs(
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





















if torch is not None:
    class ProjectionHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, *, num_layers: int = 2):
            super().__init__()
            if num_layers < 1:
                raise ValueError("ProjectionHead requires at least one layer.")
            hidden_dim = max(out_dim, min(in_dim, 512))
            layers: List[nn.Module] = []
            prev_dim = in_dim
            if num_layers == 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU(inplace=True))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class ProjectionHead:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ProjectionHead requires PyTorch. Install torch before training.")


def _filter_singular_values(singular_values: torch.Tensor, drop_ratio: float) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {
        "total": int(singular_values.numel()),
        "dropped_count": 0,
        "dropped_indices": [],
        "dropped_values": [],
        "threshold": None,
        "all_dropped": False,
    }
    if singular_values.numel() == 0:
        info["all_dropped"] = True
        return singular_values, info

    max_val = float(singular_values.max().item())
    if not math.isfinite(max_val):
        max_val = 0.0
    ratio_threshold = max_val * drop_ratio if max_val > 0 else 0.0
    info["threshold"] = float(ratio_threshold)

    keep_mask = singular_values >= ratio_threshold
    if keep_mask.all():
        drop_count = max(1, int(math.floor(info["total"] * drop_ratio)))
        if drop_count >= info["total"]:
            drop_count = info["total"] - 1
        if drop_count > 0:
            sorted_vals, sorted_idx = torch.sort(singular_values)
            drop_indices = sorted_idx[:drop_count]
            keep_mask[drop_indices] = False

    drop_indices = (~keep_mask).nonzero(as_tuple=False).flatten()
    if drop_indices.numel() > 0:
        info["dropped_count"] = int(drop_indices.numel())
        info["dropped_indices"] = drop_indices.cpu().tolist()
        info["dropped_values"] = singular_values[drop_indices].detach().cpu().tolist()

    if not keep_mask.any():
        info["all_dropped"] = True
        return singular_values.new_empty(0), info

    filtered = singular_values[keep_mask]
    info["kept_count"] = int(filtered.numel())
    if filtered.numel() > 0:
        info["kept_min"] = float(filtered.min().item())
        info["kept_max"] = float(filtered.max().item())
        info["kept_min"] = None
        info["kept_max"] = None
    return filtered, info


def dcca_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    eps: float = 1e-5,
    drop_ratio: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        dummy = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        info = {"total": 0, "dropped_count": 0, "all_dropped": True}
        return dummy, torch.zeros(0, device=z_a.device), info

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    n = z_a.size(0)
    cov_aa = (z_a.T @ z_a) / (n - 1) + eps * torch.eye(z_a.size(1), device=z_a.device)
    cov_bb = (z_b.T @ z_b) / (n - 1) + eps * torch.eye(z_b.size(1), device=z_b.device)
    cov_ab = (z_a.T @ z_b) / (n - 1)

    inv_sqrt_aa = _matrix_inverse_sqrt(cov_aa, eps)
    inv_sqrt_bb = _matrix_inverse_sqrt(cov_bb, eps)
    t_mat = inv_sqrt_aa @ cov_ab @ inv_sqrt_bb
    jitter = _adaptive_tmat_jitter(t_mat)
    if jitter > 0:
        eye = torch.eye(t_mat.size(-1), device=t_mat.device, dtype=t_mat.dtype)
        t_mat = t_mat + jitter * eye
    try:
        singular_values = torch.linalg.svdvals(t_mat)
    except RuntimeError:
        zero = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        info = {
            "total": 0,
            "dropped_count": 0,
            "all_dropped": True,
            "error": "svd_failed",
        }
        return zero, torch.zeros(0, device=z_a.device), info

    filtered, filter_info = _filter_singular_values(singular_values, drop_ratio)
    filter_info["kept_count"] = int(filtered.numel())
    filter_info["total"] = int(singular_values.numel())
    filter_info["original_max"] = float(singular_values.max().item()) if singular_values.numel() else None
    filter_info["original_min"] = float(singular_values.min().item()) if singular_values.numel() else None

    if filter_info.get("all_dropped"):
        zero = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        return zero, filtered, filter_info

    loss = -filtered.sum()
    return loss, filtered, filter_info


def _format_optional_scalar(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6f}"


def _train_DCCA(
    *,
    cfg: AlignmentConfig,
    train_dataset: TensorDataset,
    validation_dataset: Optional[TensorDataset],
    batch_size: int,
    device: torch.device,
    anchor_dim: int,
    target_dim: int,
    dcca_eps: float,
    run_logger: "_RunLogger",
    drop_ratio: float,
    tcc_ratio: float,
    mlp_layers: int,
    train_dcca: bool,
    pretrained_state: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    pretrained_summary: Optional[Dict[str, object]],
    pretrained_path: Optional[Path],
) -> Tuple[bool, Optional[nn.Module], Optional[nn.Module], List[Dict[str, object]], int, Optional[str]]:
    projection_dim = int(cfg.projection_dim)
    attempt = 1
    failure_reason: Optional[str] = None
    max_attempts = 10

    pretrained_proj_dim: Optional[int] = None
    pretrained_layers: Optional[int] = None
    if isinstance(pretrained_summary, dict):
        proj_value = pretrained_summary.get("projection_dim")
        layer_value = pretrained_summary.get("projection_layers")
        try:
            pretrained_proj_dim = int(proj_value) if proj_value is not None else None
        except (TypeError, ValueError):
            pretrained_proj_dim = None
        try:
            pretrained_layers = int(layer_value) if layer_value is not None else None
        except (TypeError, ValueError):
            pretrained_layers = None

    warned_dim_mismatch = False
    warned_layer_mismatch = False

    if not train_dcca:
        if pretrained_state is None:
            failure_reason = "Requested --read-dcca without providing pretrained weights."
            run_logger.log(failure_reason)
            return False, None, None, [], projection_dim, failure_reason
        if pretrained_proj_dim is not None:
            projection_dim = int(pretrained_proj_dim)
        current_dim = max(int(projection_dim), 1)
        if pretrained_layers is not None and pretrained_layers != mlp_layers:
            failure_reason = (
                f"Pretrained DCCA weights expect {pretrained_layers} layers but {mlp_layers} were requested."
            )
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
        projector_a = ProjectionHead(anchor_dim, current_dim, num_layers=mlp_layers).to(device)
        projector_b = ProjectionHead(target_dim, current_dim, num_layers=mlp_layers).to(device)
        try:
            projector_a.load_state_dict(pretrained_state[0], strict=True)
            projector_b.load_state_dict(pretrained_state[1], strict=True)
        except RuntimeError as exc:
            failure_reason = f"Failed to load pretrained DCCA weights: {exc}"
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
        train_eval = _evaluate_dcca(
            projector_a,
            projector_b,
            train_dataset,
            device=device,
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
            batch_size=batch_size,
        )
        if train_eval is None:
            failure_reason = "Unable to evaluate pretrained DCCA weights on training data."
            run_logger.log(failure_reason)
            return False, None, None, [], current_dim, failure_reason
        val_eval = (
            _evaluate_dcca(
                projector_a,
                projector_b,
                validation_dataset,
                device=device,
                eps=dcca_eps,
                drop_ratio=drop_ratio,
                tcc_ratio=tcc_ratio,
                batch_size=batch_size,
            )
            if validation_dataset is not None and len(validation_dataset) >= 2
            else None
        )
        train_loss_val = train_eval.get("loss")
        train_corr_val = train_eval.get("mean_corr")
        train_tcc_val = train_eval.get("tcc_sum")
        train_tcc_mean_val = train_eval.get("tcc_mean")
        train_k_val = train_eval.get("k")
        val_loss_val = val_eval.get("loss") if val_eval else None
        val_corr_val = val_eval.get("mean_corr") if val_eval else None
        val_tcc_val = val_eval.get("tcc_sum") if val_eval else None
        val_tcc_mean_val = val_eval.get("tcc_mean") if val_eval else None
        val_k_val = val_eval.get("k") if val_eval else None
        epoch_history: List[Dict[str, object]] = [
            {
                "epoch": 0,
                "loss": float(train_loss_val) if train_loss_val is not None else None,
                "mean_correlation": float(train_corr_val) if train_corr_val is not None else None,
                "train_eval_loss": float(train_loss_val) if train_loss_val is not None else None,
                "train_eval_mean_correlation": float(train_corr_val) if train_corr_val is not None else None,
                "train_eval_tcc": float(train_tcc_val) if train_tcc_val is not None else None,
                "train_eval_tcc_mean": float(train_tcc_mean_val) if train_tcc_mean_val is not None else None,
                "train_eval_k": float(train_k_val) if train_k_val is not None else None,
                "val_eval_loss": float(val_loss_val) if val_loss_val is not None else None,
                "val_eval_mean_correlation": float(val_corr_val) if val_corr_val is not None else None,
                "val_eval_tcc": float(val_tcc_val) if val_tcc_val is not None else None,
                "val_eval_tcc_mean": float(val_tcc_mean_val) if val_tcc_mean_val is not None else None,
                "val_eval_k": float(val_k_val) if val_k_val is not None else None,
                "batches": 0,
                "projection_dim": current_dim,
            }
        ]
        source_desc = f" from {pretrained_path}" if pretrained_path is not None else ""
        run_logger.log(f"Loaded pretrained DCCA weights{source_desc} without additional training.")
        return True, projector_a, projector_b, epoch_history, current_dim, None

    while projection_dim > 0 and attempt <= max_attempts:
        current_dim = max(projection_dim, 1)
        run_logger.log(f"Attempt {attempt}: projection_dim={current_dim}, mlp_layers={mlp_layers}")
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > 1,
        )
        projector_a = ProjectionHead(anchor_dim, current_dim, num_layers=mlp_layers).to(device)
        projector_b = ProjectionHead(target_dim, current_dim, num_layers=mlp_layers).to(device)

        state_for_attempt = None
        if pretrained_state is not None:
            if pretrained_proj_dim is not None and int(pretrained_proj_dim) != current_dim:
                if not warned_dim_mismatch:
                    run_logger.log(
                        f"Pretrained DCCA projection_dim {pretrained_proj_dim} does not match attempt dimension {current_dim}; skipping weight initialisation."
                    )
                    warned_dim_mismatch = True
            elif pretrained_layers is not None and pretrained_layers != mlp_layers:
                if not warned_layer_mismatch:
                    run_logger.log(
                        f"Pretrained DCCA projection layers {pretrained_layers} do not match configured {mlp_layers}; skipping weight initialisation."
                    )
                    warned_layer_mismatch = True
                state_for_attempt = pretrained_state

        if state_for_attempt is not None:
            try:
                projector_a.load_state_dict(state_for_attempt[0], strict=True)
                projector_b.load_state_dict(state_for_attempt[1], strict=True)
                source_desc = f" from {pretrained_path}" if pretrained_path is not None else ""
                run_logger.log(f"Loaded pretrained DCCA weights{source_desc} for attempt {attempt}.")
            except RuntimeError as exc:
                run_logger.log(f"Failed to load pretrained DCCA weights at attempt {attempt}: {exc}")

        optimizer = torch.optim.Adam(
            list(projector_a.parameters()) + list(projector_b.parameters()),
            lr=cfg.dcca_training.lr,
        )

        epoch_history: List[Dict[str, object]] = []
        attempt_success = True

        for epoch in range(cfg.dcca_training.epochs):
            running_loss = 0.0
            running_corr = 0.0
            batches = 0
            iterator = loader
            if tqdm is not None:
                iterator = tqdm(
                    loader,
                    desc=f"epoch {epoch+1}/{cfg.dcca_training.epochs} (proj={current_dim})",
                    leave=False,
                )

            for batch_index, (anchor_batch, target_batch) in enumerate(iterator):
                anchor_batch = anchor_batch.to(device)
                target_batch = target_batch.to(device)
                if anchor_batch.size(0) < 2:
                    if tqdm is not None:
                        iterator.set_postfix({"skip": "batch<2"})
                    continue

                optimizer.zero_grad()
                z_a = projector_a(anchor_batch)
                z_b = projector_b(target_batch)

                loss, singular_vals, filter_info = dcca_loss(z_a, z_b, eps=dcca_eps, drop_ratio=drop_ratio)

                if filter_info.get("all_dropped"):
                    failure_reason = (
                        f"All singular values filtered out at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                if not torch.isfinite(loss):
                    failure_reason = (
                        f"Non-finite loss detected at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                loss.backward()

                if _has_nonfinite_gradients((projector_a, projector_b)):
                    failure_reason = (
                        f"Non-finite gradients detected at attempt {attempt}, epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                    run_logger.log(failure_reason)
                    attempt_success = False
                    optimizer.zero_grad(set_to_none=True)
                    break

                optimizer.step()

                running_loss += float(loss.item())
                running_corr += float(singular_vals.mean().item()) if singular_vals.numel() else 0.0
                batches += 1

                if tqdm is not None:
                    iterator.set_postfix({"loss": loss.item(), "corr": running_corr / max(batches, 1)})

            if not attempt_success:
                break

            if batches == 0:
                failure_reason = f"No batches processed at attempt {attempt}, epoch {epoch + 1}"
                run_logger.log(failure_reason)
                attempt_success = False
                break

            avg_loss = running_loss / batches
            avg_corr = running_corr / batches if batches else float("nan")
            train_eval = _evaluate_dcca(
                projector_a,
                projector_b,
                train_dataset,
                device=device,
                eps=dcca_eps,
                drop_ratio=drop_ratio,
                tcc_ratio=tcc_ratio,
                batch_size=batch_size,
            )
            val_eval = (
                _evaluate_dcca(
                    projector_a,
                    projector_b,
                    validation_dataset,
                    device=device,
                    eps=dcca_eps,
                    drop_ratio=drop_ratio,
                    tcc_ratio=tcc_ratio,
                    batch_size=batch_size,
                )
                if validation_dataset is not None and len(validation_dataset) >= 2
                else None
            )
            train_loss_log = float(train_eval["loss"]) if train_eval and "loss" in train_eval else avg_loss
            train_corr_log = float(train_eval["mean_corr"]) if train_eval and "mean_corr" in train_eval else avg_corr
            train_tcc_log = train_eval.get("tcc_sum") if train_eval else None
            train_tcc_mean_log = train_eval.get("tcc_mean") if train_eval else None
            train_k_log = train_eval.get("k") if train_eval else None
            val_loss_log = val_eval.get("loss") if val_eval else None
            val_corr_log = val_eval.get("mean_corr") if val_eval else None
            val_tcc_log = val_eval.get("tcc_sum") if val_eval else None
            val_tcc_mean_log = val_eval.get("tcc_mean") if val_eval else None
            val_k_log = val_eval.get("k") if val_eval else None

            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(avg_loss),
                    "mean_correlation": float(avg_corr),
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
                    "batches": batches,
                    "projection_dim": current_dim,
                }
            )
            run_logger.log(
                "Attempt {attempt} epoch {epoch}: "
                "train_loss={train_loss:.6f}, train_mean_corr={train_corr:.6f}, train_TCC = {train_tcc}, "
                "val_loss={val_loss}, val_mean_corr={val_corr}, val_TCC = {val_tcc}, batches={batches}".format(
                    attempt=attempt,
                    epoch=epoch + 1,
                    train_loss=train_loss_log,
                    train_corr=train_corr_log,
                    train_tcc=_format_optional_scalar(train_tcc_log),
                    val_loss=_format_optional_scalar(val_loss_log),
                    val_corr=_format_optional_scalar(val_corr_log),
                    val_tcc=_format_optional_scalar(val_tcc_log),
                    batches=batches,
                )
            )

        if attempt_success:
            run_logger.log(f"Attempt {attempt} succeeded with projection_dim={current_dim}")
            return True, projector_a, projector_b, epoch_history, current_dim, None

        run_logger.log(
            f"Attempt {attempt} failed with projection_dim={current_dim}; reason={failure_reason or 'unspecified'}"
        )
        prev_dim = current_dim
        projection_dim = max(int(math.floor(prev_dim * 0.8)), 1)
        if projection_dim == prev_dim:
            projection_dim = max(prev_dim - 1, 1)
            if projection_dim == prev_dim:
                failure_reason = failure_reason or "Reached minimum projection dimension without success."
                break
        run_logger.log(f"Reducing projection_dim from {prev_dim} to {projection_dim}")
        attempt += 1

    if failure_reason is None:
        failure_reason = "Exceeded maximum adaptive attempts."
    return False, None, None, [], max(projection_dim, 1), failure_reason


def _has_nonfinite_gradients(modules: Sequence[nn.Module]) -> bool:
    for module in modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
    return False


def _evaluate_dcca(
    projector_a: nn.Module,
    projector_b: nn.Module,
    dataset: Optional[TensorDataset],
    *,
    device: torch.device,
    eps: float,
    drop_ratio: float,
    tcc_ratio: float,
    batch_size: int,
) -> Optional[Dict[str, object]]:
    if dataset is None or len(dataset) < 2:
        return None

    eval_batch = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=eval_batch, shuffle=False, drop_last=False)

    was_training_a = projector_a.training
    was_training_b = projector_b.training
    projector_a.eval()
    projector_b.eval()

    projections_a: List[torch.Tensor] = []
    projections_b: List[torch.Tensor] = []
    with torch.no_grad():
        for anchor_batch, target_batch in loader:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            proj_a = projector_a(anchor_batch).detach()
            proj_b = projector_b(target_batch).detach()
            projections_a.append(proj_a)
            projections_b.append(proj_b)

    projector_a.train(was_training_a)
    projector_b.train(was_training_b)

    if not projections_a or not projections_b:
        return None

    z_a = torch.cat(projections_a, dim=0)
    z_b = torch.cat(projections_b, dim=0)
    metrics = _canonical_metrics(z_a, z_b, eps=eps, drop_ratio=drop_ratio, tcc_ratio=tcc_ratio)
    if metrics is None:
        return None
    normalised: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalised[key] = float(value)
            normalised[key] = value
    return normalised


def _canonical_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    eps: float,
    drop_ratio: float,
    tcc_ratio: float,
) -> Optional[Dict[str, object]]:
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return None

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    n = z_a.size(0)
    cov_aa = (z_a.T @ z_a) / (n - 1) + eps * torch.eye(z_a.size(1), device=z_a.device, dtype=z_a.dtype)
    cov_bb = (z_b.T @ z_b) / (n - 1) + eps * torch.eye(z_b.size(1), device=z_b.device, dtype=z_b.dtype)
    cov_ab = (z_a.T @ z_b) / (n - 1)

    inv_sqrt_aa = _matrix_inverse_sqrt(cov_aa, eps)
    inv_sqrt_bb = _matrix_inverse_sqrt(cov_bb, eps)
    t_mat = inv_sqrt_aa @ cov_ab @ inv_sqrt_bb
    jitter = _adaptive_tmat_jitter(t_mat)
    if jitter > 0:
        t_mat = t_mat + jitter * torch.eye(t_mat.size(-1), device=t_mat.device, dtype=t_mat.dtype)

    try:
        singular_values = torch.linalg.svdvals(t_mat)
    except RuntimeError:
        return None

    filtered, filter_info = _filter_singular_values(singular_values, drop_ratio)
    if filter_info.get("all_dropped") or filtered.numel() == 0:
        return None

    sorted_vals, _ = torch.sort(filtered, descending=True)
    total = int(sorted_vals.numel())
    if total == 0:
        return None

    ratio = float(max(min(tcc_ratio, 1.0), 1e-6))
    k = max(1, min(total, int(math.ceil(total * ratio))))
    topk = sorted_vals[:k]

    metrics: Dict[str, object] = {
        "loss": -float(filtered.sum().item()),
        "mean_corr": float(filtered.mean().item()),
        "tcc_sum": float(topk.sum().item()),
        "tcc_mean": float(topk.mean().item()),
        "k": int(k),
        "canonical_correlations": int(total),
    }
    metrics["drop_info"] = filter_info
    metrics["canonical_correlations_sorted"] = [float(val) for val in sorted_vals.detach().cpu().tolist()]
    return metrics


def _project_in_batches(
    tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    progress_desc: Optional[str] = None,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty_like(tensor)
    outputs: List[torch.Tensor] = []
    batch_size = max(1, batch_size)
    with torch.no_grad():
        iterator = range(0, tensor.size(0), batch_size)
        if progress_desc and tqdm is not None:
            total = math.ceil(tensor.size(0) / batch_size)
            iterator = tqdm(iterator, desc=progress_desc, leave=False, total=total)
        for start in iterator:
            end = min(start + batch_size, tensor.size(0))
            chunk = tensor[start:end].to(device)
            projected = model(chunk).detach().cpu()
            outputs.append(projected)
    return torch.cat(outputs, dim=0)


def _select_subset_indices(total: int, max_points: int, seed: int) -> torch.Tensor:
    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    if total <= max_points:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator)
    return perm[:max_points]


def _reduce_dimensionality(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE  # type: ignore

        perplexity = max(5, min(30, features.shape[0] // 3))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca")
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    try:
        from sklearn.decomposition import PCA  # type: ignore

        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features)
        if embedding.shape[1] == 2:
            return embedding
    except Exception:
        pass
    if features.shape[1] >= 2:
        return features[:, :2]
    pad = np.zeros((features.shape[0], 2), dtype=np.float32)
    pad[:, 0] = features[:, 0]
    return pad


def _create_debug_alignment_figures(
    *,
    cfg: AlignmentConfig,
    projector_a: nn.Module,
    projector_b: nn.Module,
    anchor_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    pair_metadata: Sequence[Dict[str, object]],
    run_logger: "_RunLogger",
    drop_ratio: float,
    tcc_ratio: float,
    dcca_eps: float,
    device: torch.device,
    sample_seed: int,
) -> None:
    if plt is None:
        run_logger.log("matplotlib is unavailable; skipping advanced debug figures.")
        return

    analysis_primary = None
    if cfg.output_dir is not None:
        analysis_primary = cfg.output_dir / "overlap_visualizations" / "alignment_analysis"
    elif cfg.log_dir is not None:
        analysis_primary = cfg.log_dir / "overlap_visualizations" / "alignment_analysis"
    analysis_dir = _prepare_output_dir(analysis_primary, "overlap_alignment_outputs/alignment_analysis")
    if analysis_dir is None:
        run_logger.log("Unable to prepare alignment analysis directory; skipping figures.")
        return

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        run_logger.log(f"Failed to create alignment analysis directory {analysis_dir}: {exc}")
        return

    if anchor_tensor.size(0) == 0 or target_tensor.size(0) == 0:
        run_logger.log("No samples available for debug plotting.")
        return

    analysis_batch = max(32, min(2048, anchor_tensor.size(0)))
    projected_anchor = _project_in_batches(anchor_tensor, projector_a, device, analysis_batch)
    projected_target = _project_in_batches(target_tensor, projector_b, device, analysis_batch)

    # t-SNE / UMAP figure
    try:
        subset_indices = _select_subset_indices(anchor_tensor.size(0), max_points=1000, seed=sample_seed)
        subset_list = subset_indices.tolist()
        if subset_indices.numel() >= 2:
            anchor_raw = anchor_tensor.detach().cpu()
            target_raw = target_tensor.detach().cpu()
            proj_anchor_raw = projected_anchor.detach().cpu()
            proj_target_raw = projected_target.detach().cpu()

            before_features = torch.cat(
                [anchor_raw[subset_indices], target_raw[subset_indices]], dim=0
            ).numpy()
            after_features = torch.cat(
                [proj_anchor_raw[subset_indices], proj_target_raw[subset_indices]], dim=0
            ).numpy()

            embedding_before = _reduce_dimensionality(before_features, random_state=sample_seed)
            embedding_after = _reduce_dimensionality(after_features, random_state=sample_seed + 1)

            num_subset = subset_indices.numel()
            coords_before_anchor = embedding_before[:num_subset]
            coords_before_target = embedding_before[num_subset:]
            coords_after_anchor = embedding_after[:num_subset]
            coords_after_target = embedding_after[num_subset:]

            fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            colors = {"anchor": "#1f77b4", "target": "#ff7f0e"}
            for ax, coords_anchor, coords_target, title in (
                (axes[0], coords_before_anchor, coords_before_target, "Embeddings (pre-DCCA)"),
                (axes[1], coords_after_anchor, coords_after_target, "Embeddings (post-DCCA)"),
            ):
                ax.scatter(
                    coords_anchor[:, 0],
                    coords_anchor[:, 1],
                    c=colors["anchor"],
                    label="Anchor",
                    alpha=0.6,
                    s=20,
                )
                ax.scatter(
                    coords_target[:, 0],
                    coords_target[:, 1],
                    c=colors["target"],
                    label="Target",
                    alpha=0.6,
                    s=20,
                    marker="s",
                )
                val_anchor_x = []
                val_anchor_y = []
                val_target_x = []
                val_target_y = []
                for i, idx in enumerate(subset_list):
                    if i >= coords_anchor.shape[0]:
                        break
                    split = pair_metadata[idx].get("split") if idx < len(pair_metadata) else None
                    if split == "val":
                        val_anchor_x.append(coords_anchor[i, 0])
                        val_anchor_y.append(coords_anchor[i, 1])
                        val_target_x.append(coords_target[i, 0])
                        val_target_y.append(coords_target[i, 1])
                    ax.plot(
                        [coords_anchor[i, 0], coords_target[i, 0]],
                        [coords_anchor[i, 1], coords_target[i, 1]],
                        color="gray",
                        alpha=0.2,
                        linewidth=0.5,
                    )
                if val_anchor_x:
                    ax.scatter(
                        val_anchor_x,
                        val_anchor_y,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        s=40,
                        label="Anchor (val)" if ax is axes[0] else None,
                    )
                if val_target_x:
                    ax.scatter(
                        val_target_x,
                        val_target_y,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        s=40,
                        marker="s",
                        label="Target (val)" if ax is axes[0] else None,
                    )
                ax.set_title(title)
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
            axes[1].legend(loc="upper right")
            tsne_path = analysis_dir / "alignment_tsne_umap.png"
            fig.suptitle("Paired Embeddings Before vs After DCCA")
            fig.savefig(tsne_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            run_logger.log(f"Saved t-SNE/UMAP alignment figure to {tsne_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create t-SNE/UMAP debug figure: {exc}")

    # Canonical correlation spectrum
    try:
        metrics_before = _canonical_metrics(
            anchor_tensor.detach().to(projected_anchor.dtype),
            target_tensor.detach().to(projected_target.dtype),
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
        )
        metrics_after = _canonical_metrics(
            projected_anchor,
            projected_target,
            eps=dcca_eps,
            drop_ratio=drop_ratio,
            tcc_ratio=tcc_ratio,
        )
        if metrics_before and metrics_after:
            vals_before = metrics_before.get("canonical_correlations_sorted") or []
            vals_after = metrics_after.get("canonical_correlations_sorted") or []
            if vals_before and vals_after:
                vals_before = np.asarray(vals_before, dtype=float)
                vals_after = np.asarray(vals_after, dtype=float)
                k_plot = int(min(len(vals_before), len(vals_after), 128))
                if k_plot > 0:
                    idx = np.arange(1, k_plot + 1)
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                    for ax, values, title in (
                        (axes[0], vals_before[:k_plot], "Pre-DCCA"),
                        (axes[1], vals_after[:k_plot], "Post-DCCA"),
                    ):
                        ax.bar(idx, values, color="#4c72b0", alpha=0.7)
                        cumulative = np.cumsum(values)
                        ax.plot(idx, cumulative, color="#dd8452", linewidth=1.5, label="Cumulative sum")
                        ax.set_title(f"{title} Canonical Spectrum (Top {k_plot})")
                        ax.set_xlabel("Canonical component")
                        ax.set_ylabel("Correlation")
                        ax.set_ylim(0, 1.05)
                        ax.legend(loc="upper right")
                    spectrum_path = analysis_dir / "canonical_correlation_spectrum.png"
                    fig.savefig(spectrum_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    run_logger.log(f"Saved canonical correlation spectrum to {spectrum_path}")
    except Exception as exc:
        run_logger.log(f"Failed to generate canonical correlation spectrum: {exc}")

    # Geospatial distance heatmap
    try:
        geo_indices: List[int] = []
        geo_coords: List[Tuple[float, float]] = []
        geo_splits: List[str] = []
        for idx, meta in enumerate(pair_metadata):
            coord = meta.get("anchor_coord")
            if coord is None:
                coord = meta.get("target_weighted_coord")
            if coord is None:
                continue
            try:
                x, y = float(coord[0]), float(coord[1])
            except Exception:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            geo_indices.append(idx)
            geo_coords.append((x, y))
            geo_splits.append(str(meta.get("split") or "train"))
        if geo_indices:
            pre_dist = torch.linalg.norm(anchor_tensor - target_tensor, dim=1).detach().cpu().numpy()
            post_dist = torch.linalg.norm(projected_anchor - projected_target, dim=1).detach().cpu().numpy()
            pre_geo = pre_dist[geo_indices]
            post_geo = post_dist[geo_indices]
            geo_x = np.asarray([coord[0] for coord in geo_coords], dtype=float)
            geo_y = np.asarray([coord[1] for coord in geo_coords], dtype=float)
            vmin = float(min(pre_geo.min(), post_geo.min()))
            vmax = float(max(pre_geo.max(), post_geo.max()))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharex=True, sharey=True)
            for ax, values, title in (
                (axes[0], pre_geo, "Pre-DCCA"),
                (axes[1], post_geo, "Post-DCCA"),
            ):
                val_mask = np.array([split == "val" for split in geo_splits], dtype=bool)
                sc = ax.scatter(
                    geo_x,
                    geo_y,
                    c=values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    s=40,
                    alpha=0.9,
                )
                if val_mask.any():
                    ax.scatter(
                        geo_x[val_mask],
                        geo_y[val_mask],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.8,
                        s=60,
                        label="Validation",
                    )
                    ax.legend(loc="upper right")
                ax.set_title(f"{title} Pairwise Distance")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                color_label = (
                    "||Latent_base(A) - Latent_base(B)||" if title == "Pre-DCCA" else "||Latent_DCCA(A) - Latent_DCCA(B)||"
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)
            geo_path = analysis_dir / "geospatial_distance_map.png"
            fig.suptitle("Pairwise Distance Heatmap (Anchor vs Target)")
            fig.savefig(geo_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            run_logger.log(f"Saved geospatial distance heatmap to {geo_path}")
    except Exception as exc:
        run_logger.log(f"Failed to create geospatial distance map: {exc}")


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


def _resolve_pair_labels(
    pair_metadata: Sequence[Dict[str, object]],
    anchor_dataset: str,
    target_dataset: str,
    pn_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]],
    pn_index_sets: Optional[Dict[str, Dict[str, set[int]]]] = None,
) -> List[Dict[str, Optional[int]]]:
    resolved: List[Dict[str, Optional[int]]] = []
    anchor_lookup = pn_maps.get(anchor_dataset)
    target_lookup = pn_maps.get(target_dataset)
    pn_index_sets = pn_index_sets or {}
    anchor_index_lookup = pn_index_sets.get(anchor_dataset, {"pos": set(), "neg": set()})
    target_index_lookup = pn_index_sets.get(target_dataset, {"pos": set(), "neg": set()})

    def _label_from_index(idx: Optional[int], lookup_sets: Dict[str, set[int]]) -> Optional[int]:
        if idx is None:
            return None
        if idx in lookup_sets.get("pos", set()):
            return 1
        if idx in lookup_sets.get("neg", set()):
            return 0
        return None

    def _label_from_indices(indices: Sequence[Optional[int]], lookup_sets: Dict[str, set[int]]) -> Optional[int]:
        if not indices:
            return None
        pos = False
        neg = False
        pos_set = lookup_sets.get("pos", set())
        neg_set = lookup_sets.get("neg", set())
        for idx in indices:
            if idx is None:
                continue
            if idx in pos_set:
                pos = True
            elif idx in neg_set:
                neg = True
        if pos:
            return 1
        if neg:
            return 0
        return None

    for meta in pair_metadata:
        anchor_index = meta.get("anchor_index")
        target_indices = meta.get("target_indices") or []
        anchor_region = meta.get("anchor_region")
        anchor_rc = meta.get("anchor_row_col")
        anchor_final = _label_from_index(anchor_index, anchor_index_lookup)
        if anchor_final is None:
            anchor_label_pn = _lookup_pn_label(anchor_region, anchor_rc, anchor_lookup)
            if anchor_label_pn is not None:
                anchor_final = 1 if int(anchor_label_pn) > 0 else 0

        target_regions = meta.get("target_regions") or []
        target_row_cols = meta.get("target_row_cols") or []
        target_final = _label_from_indices(target_indices, target_index_lookup)
        if target_final is None and isinstance(target_row_cols, list):
            target_label_pn = None
            for idx_rc, rc in enumerate(target_row_cols):
                if rc is None:
                    continue
                region_candidate = target_regions[idx_rc] if idx_rc < len(target_regions) else None
                if region_candidate is None:
                    region_candidate = anchor_region
                label_candidate = _lookup_pn_label(
                    region_candidate,
                    rc,
                    target_lookup,
                )
                if label_candidate is not None:
                    target_label_pn = label_candidate
                    break
            if target_label_pn is not None:
                target_final = 1 if int(target_label_pn) > 0 else 0

        if anchor_final is not None and target_final is not None:
            combined = int(max(anchor_final, target_final))
        elif anchor_final is not None:
            combined = anchor_final
            combined = target_final

        resolved.append(
            {
                "anchor_label": anchor_final,
                "target_label": target_final,
                "combined_label": combined,
            }
        )
    return resolved


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


class _LogitMLPWrapper(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        base = MLPDropout(in_dim=in_dim, hidden_dims=tuple(hidden_dims), p=dropout)
        layers = list(base.net.children())
        if layers and isinstance(layers[-1], nn.Sigmoid):
            layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def _build_mlp_classifier(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Module:
    return _LogitMLPWrapper(in_dim, hidden_dims, dropout)


def _mc_dropout_statistics(
    model: nn.Module,
    features: torch.Tensor,
    *,
    num_passes: int = 5,
    progress_desc: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Monte-Carlo dropout over ``num_passes`` stochastic forward passes.

    Returns mean and standard deviation tensors (CPU, 1-D) aligned with the input rows.
    """
    if features.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty

    device = features.device
    dropout_modules: List[nn.Module] = [
        module
        for module in model.modules()
        if isinstance(module, nn.Dropout)
    ]

    def _set_dropout_modules(mode: bool) -> None:
        for module in dropout_modules:
            module.train(mode)

    model.eval()
    _set_dropout_modules(False)
    with torch.no_grad():
        logits = model(features)
        base_probs = torch.sigmoid(logits).squeeze(1)

    if num_passes <= 1 or not dropout_modules:
        return base_probs.detach().cpu(), torch.zeros_like(base_probs, device="cpu")

    samples: List[torch.Tensor] = []
    iterator = range(num_passes)
    if num_passes > 1:
        desc = progress_desc or "MC dropout"
        iterator = _progress_iter(iterator, desc, leave=False, total=num_passes)
    with torch.no_grad():
        for _ in iterator:
            _set_dropout_modules(True)
            logits = model(features)
            probs = torch.sigmoid(logits).squeeze(1)
            samples.append(probs.detach().clone())
        _set_dropout_modules(False)

    stacked = torch.stack(samples, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return mean.detach().cpu(), std.detach().cpu()


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



def _dataset_boundary_path(dataset_meta_map: Dict[str, Dict[str, object]], dataset_name: str) -> Optional[Path]:
    meta = dataset_meta_map.get(dataset_name, {}) or {}
    boundaries = meta.get("boundaries") or {}
    project_entries = boundaries.get("project") or []
    for entry in project_entries:
        candidate = entry.get("path_resolved") or entry.get("path")
        if candidate is None:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            root = meta.get("root")
            if root:
                candidate_path = Path(root) / candidate
                candidate_path = Path(candidate).resolve()
        if candidate_path.exists():
            return candidate_path
    return None


def _plot_training_curves(history: Sequence[Dict[str, object]], out_path: Path, title: str) -> None:
    if plt is None or not history:
        return
    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    train_loss = [entry.get("train", {}).get("loss") for entry in history]
    val_loss = [entry.get("val", {}).get("loss") if entry.get("val") else None for entry in history]
    train_acc = [entry.get("train", {}).get("accuracy") for entry in history]
    val_acc = [entry.get("val", {}).get("accuracy") if entry.get("val") else None for entry in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    if any(value is not None for value in val_loss):
        ax.plot(epochs, [v if v is not None else float("nan") for v in val_loss], label="val_loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    if any(value is not None for value in train_acc):
        ax2 = ax.twinx()
        ax2.plot(epochs, [v if v is not None else float("nan") for v in train_acc], label="train_acc", color="#2ca02c", linestyle="--")
        if any(value is not None for value in val_acc):
            ax2.plot(epochs, [v if v is not None else float("nan") for v in val_acc], label="val_acc", color="#d62728", linestyle="--")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _resolve_binary_label(value: Optional[object], default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return 1 if int(value) > 0 else 0
    except Exception:
        return int(default)


def _estimate_pixel_size(
    dataset_meta_map: Dict[str, Dict[str, object]],
    dataset: str,
    span_x: float,
    span_y: float,
) -> float:
    meta = dataset_meta_map.get(dataset, {}) or {}
    for key in ("pixel_resolution", "min_resolution", "window_spacing"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    approx_span = max(span_x, span_y)
    if approx_span <= 0 or not math.isfinite(approx_span):
        return 1.0
    return max(approx_span / 256.0, 1.0)


def _compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    interior = mask.copy()
    interior[:-1, :] &= mask[1:, :]
    interior[1:, :] &= mask[:-1, :]
    interior[:, :-1] &= mask[:, 1:]
    interior[:, 1:] &= mask[:, :-1]
    boundary &= ~interior
    return boundary


def _build_prediction_payload_from_samples(
    dataset: str,
    samples: Sequence[object],
    dataset_meta_map: Dict[str, Dict[str, object]],
    run_logger: "_RunLogger",
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
) -> Optional[Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, object],
    Dict[str, List[Tuple[int, int]]],
    Dict[str, List[Tuple[int, int]]],
]]:
    if not samples:
        return None

    xs_list: List[float] = []
    ys_list: List[float] = []
    prob_list: List[float] = []
    label_list: List[int] = []

    for entry in samples:
        if isinstance(entry, dict):
            x = entry.get("x")
            y = entry.get("y")
            prob = entry.get("prob")
            label = entry.get("label", 1)
        elif isinstance(entry, (list, tuple, np.ndarray)):
            if len(entry) < 3:
                continue
            x, y, prob = entry[:3]
            label = entry[3] if len(entry) >= 4 else 1
        else:
            continue
        try:
            x_f = float(x)
            y_f = float(y)
            prob_f = float(prob)
        except Exception:
            continue
        if not (math.isfinite(x_f) and math.isfinite(y_f) and math.isfinite(prob_f)):
            continue
        xs_list.append(x_f)
        ys_list.append(y_f)
        prob_list.append(prob_f)
        label_list.append(_resolve_binary_label(label))
    if not xs_list:
        run_logger.log(f"Probability export skipped for dataset {dataset}: insufficient coordinate data.")
        return None

    xs = np.asarray(xs_list, dtype=np.float64)
    ys = np.asarray(ys_list, dtype=np.float64)
    probs = np.asarray(prob_list, dtype=np.float64)
    labels = np.asarray(label_list, dtype=np.int16)

    if overlap_mask is not None and rasterio is not None:
        mask_array = overlap_mask.get("array")
        mask_transform = overlap_mask.get("transform")
        mask_shape = overlap_mask.get("shape")
        mask_nodata = overlap_mask.get("nodata")
        if mask_array is not None and mask_transform is not None and mask_shape is not None:
            try:
                mask_rows, mask_cols = rasterio.transform.rowcol(mask_transform, xs, ys)
                mask_rows = np.asarray(mask_rows, dtype=np.int64)
                mask_cols = np.asarray(mask_cols, dtype=np.int64)
                inside = (
                    (mask_rows >= 0)
                    & (mask_rows < int(mask_shape[0]))
                    & (mask_cols >= 0)
                    & (mask_cols < int(mask_shape[1]))
                )
                if inside.any():
                    vals = np.asarray(mask_array)[mask_rows[inside], mask_cols[inside]]
                    finite = np.isfinite(vals)
                    if mask_nodata is not None and np.isfinite(mask_nodata):
                        finite &= ~np.isclose(vals, mask_nodata)
                    valid = finite & (vals != 0)
                    mask_filter = np.zeros_like(inside, dtype=bool)
                    mask_filter[np.where(inside)[0][valid]] = True
                else:
                    mask_filter = inside
            except Exception:
                mask_filter = None
            else:
                if mask_filter is not None:
                    xs = xs[mask_filter]
                    ys = ys[mask_filter]
                    probs = probs[mask_filter]
                    labels = labels[mask_filter]
                    if xs.size == 0:
                        run_logger.log(f"Probability export skipped for dataset {dataset}: no samples within overlap mask.")
                        return None

    boundary_path = _dataset_boundary_path(dataset_meta_map, dataset)
    if boundary_path is not None and rasterio is not None:
        try:
            with rasterio.open(boundary_path) as src:
                transform = src.transform
                rows, cols = rasterio.transform.rowcol(transform, xs, ys)
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)
                inside = (
                    (rows >= 0)
                    & (rows < src.height)
                    & (cols >= 0)
                    & (cols < src.width)
                )
                if inside.any():
                    rows = rows[inside]
                    cols = cols[inside]
                    probs_inside = probs[inside]
                    labels_inside = labels[inside]

                    sum_map = np.zeros((src.height, src.width), dtype=np.float64)
                    sumsq_map = np.zeros_like(sum_map)
                    counts = np.zeros_like(sum_map, dtype=np.int32)

                    np.add.at(sum_map, (rows, cols), probs_inside)
                    np.add.at(sumsq_map, (rows, cols), probs_inside * probs_inside)
                    np.add.at(counts, (rows, cols), 1)

                    mean_map = np.full((src.height, src.width), np.nan, dtype=np.float32)
                    std_map = np.full((src.height, src.width), np.nan, dtype=np.float32)
                    mask = counts > 0
                    if mask.any():
                        mean_vals = sum_map[mask] / counts[mask]
                        var_vals = np.clip(sumsq_map[mask] / counts[mask] - mean_vals ** 2, 0.0, None)
                        mean_map[mask] = mean_vals.astype(np.float32)
                        std_map[mask] = np.sqrt(var_vals).astype(np.float32)

                    boundary_data = src.read(1, masked=False)
                    valid_mask = np.asarray(boundary_data != 0, dtype=bool)
                    boundary_mask = _compute_boundary_mask(valid_mask) if valid_mask.any() else None

                    profile = src.profile.copy()
                    reference = {
                        "profile": profile,
                        "transform": transform,
                        "crs": src.crs,
                        "valid_mask": valid_mask if valid_mask.any() else None,
                        "boundary_mask": boundary_mask,
                    }

                    mean_map, std_map, valid_mask, boundary_mask = _apply_overlap_mask_to_grid(
                    mean_map,
                    std_map,
                    valid_mask,
                    boundary_mask,
                    overlap_mask,
                    transform,
                )
                    pos_pixels: set[Tuple[int, int]] = set()
                    neg_pixels: set[Tuple[int, int]] = set()
                    for r, c, lbl in zip(rows.tolist(), cols.tolist(), labels_inside.tolist()):
                        if lbl > 0:
                            pos_pixels.add((int(r), int(c)))
                            neg_pixels.add((int(r), int(c)))

                    return (
                        {
                            dataset: {
                                "mean": mean_map,
                                "std": std_map,
                                "reference": reference,
                            }
                        },
                        reference,
                        {dataset: list(pos_pixels)},
                        {dataset: list(neg_pixels)},
                    )
        except Exception as exc:
            run_logger.log(f"Failed to use boundary raster for dataset {dataset}: {exc}")

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    span_x = max(x_max - x_min, 0.0)
    span_y = max(y_max - y_min, 0.0)
    pixel_size = _estimate_pixel_size(dataset_meta_map, dataset, span_x, span_y)
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        pixel_size = 1.0

    width = int(math.ceil(span_x / pixel_size)) + 1 if span_x > 0 else 1
    height = int(math.ceil(span_y / pixel_size)) + 1 if span_y > 0 else 1
    width = max(width, 1)
    height = max(height, 1)

    cols = np.floor((xs - x_min) / pixel_size).astype(int)
    rows = np.floor((y_max - ys) / pixel_size).astype(int)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    sum_map = np.zeros((height, width), dtype=np.float64)
    sumsq_map = np.zeros_like(sum_map)
    counts = np.zeros_like(sum_map, dtype=np.int32)

    np.add.at(sum_map, (rows, cols), probs)
    np.add.at(sumsq_map, (rows, cols), probs * probs)
    np.add.at(counts, (rows, cols), 1)

    mean_map = np.full((height, width), np.nan, dtype=np.float32)
    std_map = np.full((height, width), np.nan, dtype=np.float32)
    mask = counts > 0
    if mask.any():
        mean_vals = sum_map[mask] / counts[mask]
        var_vals = np.clip(sumsq_map[mask] / counts[mask] - mean_vals ** 2, 0.0, None)
        mean_map[mask] = mean_vals.astype(np.float32)
        std_map[mask] = np.sqrt(var_vals).astype(np.float32)

    valid_mask = mask.astype(bool)
    boundary_mask = _compute_boundary_mask(valid_mask) if valid_mask.any() else None

    transform = Affine(pixel_size, 0.0, x_min, 0.0, -pixel_size, y_max)
    meta_entry = dataset_meta_map.get(dataset, {}) or {}
    crs = meta_entry.get("crs") or meta_entry.get("spatial_ref")
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs

    reference = {
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "valid_mask": valid_mask if valid_mask.any() else None,
        "boundary_mask": boundary_mask,
    }

    pos_pixels: set[Tuple[int, int]] = set()
    neg_pixels: set[Tuple[int, int]] = set()
    for r, c, lbl in zip(rows.tolist(), cols.tolist(), labels.tolist()):
        if lbl > 0:
            pos_pixels.add((int(r), int(c)))
        else:
            neg_pixels.add((int(r), int(c)))

    return (
        {
            dataset: {
                "mean": mean_map,
                "std": std_map,
                "reference": reference,
            }
        },
        reference,
        {dataset: list(pos_pixels)},
        {dataset: list(neg_pixels)},
    )



def _apply_overlap_mask_to_grid(
    mean_map: np.ndarray,
    std_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    boundary_mask: Optional[np.ndarray],
    mask_info: Optional[Dict[str, object]],
    grid_transform: Affine,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if mask_info is None or rasterio is None:
        return mean_map, std_map, valid_mask, boundary_mask
    mask_array = mask_info.get("array")
    mask_transform = mask_info.get("transform")
    mask_shape = mask_info.get("shape")
    nodata = mask_info.get("nodata")
    if mask_array is None or mask_transform is None or mask_shape is None:
        return mean_map, std_map, valid_mask, boundary_mask
    height, width = mean_map.shape
    if height == 0 or width == 0:
        return mean_map, std_map, valid_mask, boundary_mask
    rr, cc = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    try:
        xs, ys = rasterio.transform.xy(grid_transform, rr, cc, offset="center")
        xs = np.asarray(xs).ravel()
        ys = np.asarray(ys).ravel()
        mask_rows, mask_cols = rasterio.transform.rowcol(mask_transform, xs, ys)
    except Exception:
        return mean_map, std_map, valid_mask, boundary_mask
    mask_bool = np.zeros(height * width, dtype=bool)
    mask_rows = np.asarray(mask_rows)
    mask_cols = np.asarray(mask_cols)
    valid = (mask_rows >= 0) & (mask_rows < mask_shape[0]) & (mask_cols >= 0) & (mask_cols < mask_shape[1])
    if valid.any():
        values = mask_array[mask_rows[valid], mask_cols[valid]]
        finite = np.isfinite(values)
        if nodata is not None and np.isfinite(nodata):
            finite &= ~np.isclose(values, nodata)
        mask_bool[valid] = finite & (values != 0)
    mask_bool = mask_bool.reshape(height, width)
    if valid_mask is not None:
        valid_mask = valid_mask & mask_bool
    else:
        valid_mask = mask_bool
    if not valid_mask.any():
        mean_map = np.full_like(mean_map, np.nan)
        std_map = np.full_like(std_map, np.nan)
        boundary_mask = None
        return mean_map, std_map, valid_mask, boundary_mask
    mean_map = np.where(valid_mask, mean_map, np.nan)
    std_map = np.where(valid_mask, std_map, np.nan)
    boundary_mask = _compute_boundary_mask(valid_mask) if valid_mask.any() else None
    return mean_map, std_map, valid_mask, boundary_mask

def _plot_probability_map(
    dataset: str,
    samples: Sequence[object],
    out_dir: Path,
    run_logger: "_RunLogger",
    dataset_meta_map: Dict[str, Dict[str, object]],
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
) -> None:
    payload = _build_prediction_payload_from_samples(
        dataset,
        samples,
        dataset_meta_map,
        run_logger,
        overlap_mask=overlap_mask,
    )
    if payload is None:
        run_logger.log(f"No probability samples available for dataset {dataset}; skipping exports.")
        return
    result_payload, default_reference, pos_map, neg_map = payload
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_outputs(
        result_payload,
        default_reference,
        out_dir,
        pos_coords_by_region=pos_map,
        neg_coords_by_region=neg_map,
    )


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

def _save_classifier_outputs(
    cfg: AlignmentConfig,
    run_logger: "_RunLogger",
    method_id: int,
    anchor_name: str,
    target_name: str,
    method_dir: Path,
    results: Dict[str, object],
    classifier_metadata: Dict[str, object],
    dataset_meta_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    summary_payload: Dict[str, object] = {}
    method_dir = method_dir.resolve()
    method_dir.mkdir(parents=True, exist_ok=True)
    overlap_mask_info = classifier_metadata.get("overlap_mask") if classifier_metadata else None

    def _write_json(path: Path, payload: object) -> None:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            run_logger.log(f"Failed to write {path}: {exc}")

    def _export_state_dict(state_dict: Optional[Dict[str, torch.Tensor]], out_path: Path) -> None:
        if state_dict is None:
            return
        try:
            state_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
            torch.save({"state_dict": state_cpu}, out_path)
        except Exception as exc:
            run_logger.log(f"Failed to save classifier weights to {out_path}: {exc}")

    def _plot_for_dataset(
        dataset: str,
        samples: List[Dict[str, object]],
        out_dir: Path,
        run_logger: "_RunLogger",
        dataset_meta_map: Dict[str, Dict[str, object]],
        *,
        overlap_mask: Optional[Dict[str, object]] = None,
    ) -> None:
        if not samples:
            run_logger.log(f"No samples available for dataset {dataset}; skipping probability export.")
            return
        _plot_probability_map(
            dataset,
            samples,
            out_dir,
            run_logger,
            dataset_meta_map,
            overlap_mask=overlap_mask,
        )

    if method_id == 1:
        method_1_dir = Path(method_dir) / 'method_1_classifiers'
        method_1_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = method_1_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_data: Dict[str, object] = {}
        history_data: Dict[str, object] = {}
        method1_data: Optional[Dict[str, object]] = classifier_metadata.get("method1") if classifier_metadata else None
        for tag, info in results.items():
            info = dict(info)
            state_dict = info.pop('state_dict', None)
            metrics = info.get('metrics', {})
            history = info.get('history', [])
            metrics_data[tag] = metrics
            history_data[tag] = history
            _write_json(method_1_dir / f'predictions_{tag}.json', info)
            _plot_training_curves(history, plots_dir / f'training_curve_{tag}.png', f'Method 1 ({tag}) Training')
            _export_state_dict(state_dict, method_1_dir / f'classifier_{tag}.pt')
        strong_info = results.get('strong') or next(iter(results.values()), None)
        if strong_info and method1_data:
            valid_indices = strong_info.get('valid_indices', [])
            all_probs = strong_info.get('all_probs', [])
            mc_info = strong_info.get('mc_dropout') or {}
            mc_all_mean = mc_info.get('all_mean') or []
            mc_all_std = mc_info.get('all_std') or []
            mc_passes = mc_info.get('num_passes')
            train_set = set(method1_data.get('train_indices', []))
            val_set = set(method1_data.get('val_indices', []))
            metadata_list: Sequence[Dict[str, object]] = method1_data.get('metadata', [])
            samples_by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for pos, idx in enumerate(valid_indices):
                if pos >= len(all_probs):
                    continue
                prob = all_probs[pos]
                if idx >= len(metadata_list):
                    continue
                meta = metadata_list[idx]
                coord = meta.get('coord')
                if coord is None or not isinstance(coord, (tuple, list)):
                    continue
                dataset_key = str(meta.get('dataset') or anchor_name)
                split = 'val' if idx in val_set else ('train' if idx in train_set else 'unlabeled')
                mc_mean_val = mc_all_mean[pos] if pos < len(mc_all_mean) else None
                mc_std_val  = mc_all_std[pos]  if pos < len(mc_all_std) else None
                sample_entry = {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "prob": float(prob),
                    "label": _resolve_binary_label(meta.get('label'), default=0),
                    "split": split,
                }
                if mc_mean_val is not None:
                    sample_entry["prob_mc_mean"] = float(mc_mean_val)
                if mc_std_val is not None:
                    sample_entry["prob_mc_std"] = float(mc_std_val)
                if mc_passes:
                    sample_entry["mc_passes"] = int(mc_passes)
                samples_by_dataset[dataset_key].append(sample_entry)
            for dataset_key, dataset_samples in samples_by_dataset.items():
                dataset_plot_dir = plots_dir / f"probability_{dataset_key}"
                _plot_for_dataset(
                    dataset_key,
                    dataset_samples,
                    dataset_plot_dir,
                    run_logger,
                    dataset_meta_map,
                    overlap_mask=overlap_mask_info,
                )
                _write_json(plots_dir / f"probability_{dataset_key}.json", dataset_samples)
            
            inference_map = strong_info.get("inference") or {}
            if inference_map:
                full_plots_dir = method_1_dir / "plots_full"
                full_plots_dir.mkdir(parents=True, exist_ok=True)
                for dataset_key, payload in inference_map.items():
                    samples_full = list(payload.get("samples") or [])
                    if not samples_full:
                        continue
                    dataset_plot_dir = full_plots_dir / f"probability_{dataset_key}"
                    _plot_for_dataset(
                        dataset_key,
                        samples_full,
                        dataset_plot_dir,
                        run_logger,
                        dataset_meta_map,
                        overlap_mask=overlap_mask_info,
                    )
                    _write_json(full_plots_dir / f"probability_{dataset_key}.json", samples_full)
                    metadata_full = payload.get("metadata")
                    if metadata_full:
                        _write_json(full_plots_dir / f"metadata_{dataset_key}.json", metadata_full)
        _write_json(method_1_dir / 'training_metrics.json', metrics_data)
        _write_json(method_1_dir / 'training_history.json', history_data)
        log_lines = [f'Method {method_id} classifier summary']
        for key, metric in metrics_data.items():
            log_lines.append(f'[{key}] {json.dumps(metric)}')
        (method_1_dir / 'training.log').write_text('\n'.join(log_lines), encoding='utf-8')
        summary_payload['output_dir'] = str(method_1_dir)
        summary_payload['metrics'] = metrics_data
        if method1_data:
            summary_payload['train_size'] = len(method1_data.get('train_indices', []))
            summary_payload['val_size'] = len(method1_data.get('val_indices', []))
        return summary_payload

    print("[error] Unsupported classifier method ID:", method_id)
    if method_id == 2:
        anchor_info = dict(results.get('anchor') or {})
        target_info = dict(results.get('target') or {})
        summary_payload['anchor'] = {}
        summary_payload['target'] = {}
        overview: Dict[str, object] = {}

        if anchor_info:
            anchor_dir = Path(method_dir) / ('method_2_' + anchor_name)
            anchor_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = anchor_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            anchor_state = anchor_info.pop('state_dict', None)
            anchor_predictions = dict(anchor_info)
            anchor_predictions.pop("metadata", None)
            _write_json(anchor_dir / 'predictions_anchor.json', anchor_predictions)
            _export_state_dict(anchor_state, anchor_dir / 'classifier.pt')
            metrics = anchor_info.get('metrics', {})
            history = anchor_info.get('history', [])
            _write_json(anchor_dir / 'training_metrics.json', metrics)
            _write_json(anchor_dir / 'training_history.json', history)
            _plot_training_curves(history, plots_dir / 'training_curve_anchor.png', f'Method 2 ({anchor_name}) Training')
            anchor_samples: List[Dict[str, object]] = []
            anchor_metadata: Sequence[Dict[str, object]] = (classifier_metadata.get('anchor') or {}).get('metadata', [])
            train_set = set(anchor_info.get('train_indices', []))
            val_set = set(anchor_info.get('val_indices', []))
            mc_info = anchor_info.get('mc_dropout') or {}
            mc_all_mean = mc_info.get('all_mean') or []
            mc_all_std = mc_info.get('all_std') or []
            mc_passes = mc_info.get('num_passes')
            valid_indices_anchor = anchor_info.get('valid_indices', [])
            all_probs_anchor = anchor_info.get('all_probs', [])
            for pos, idx in enumerate(valid_indices_anchor):
                if pos >= len(all_probs_anchor):
                    continue
                prob = all_probs_anchor[pos]
                if idx >= len(anchor_metadata):
                    continue
                meta = anchor_metadata[idx]
                coord = meta.get('coord')
                if coord is None or not isinstance(coord, (tuple, list)):
                    continue
                split = 'val' if idx in val_set else ('train' if idx in train_set else 'unlabeled')
                mc_mean_val = mc_all_mean[pos] if pos < len(mc_all_mean) else None
                mc_std_val = mc_all_std[pos] if pos < len(mc_all_std) else None
                sample_entry = {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "prob": float(prob),
                    "label": _resolve_binary_label(meta.get('label'), default=0),
                    "split": split,
                }
                if mc_mean_val is not None:
                    sample_entry["prob_mc_mean"] = float(mc_mean_val)
                if mc_std_val is not None:
                    sample_entry["prob_mc_std"] = float(mc_std_val)
                if mc_passes:
                    sample_entry["mc_passes"] = int(mc_passes)
                anchor_samples.append(sample_entry)
            dataset_plot_dir = plots_dir / f"probability_{anchor_name}"
            _plot_for_dataset(
                anchor_name,
                anchor_samples,
                dataset_plot_dir,
                run_logger,
                dataset_meta_map,
                overlap_mask=overlap_mask_info,
            )
            _write_json(anchor_dir / f"probability_{anchor_name}.json", anchor_samples)
            inference_payload = anchor_info.get("inference") or {}
            if inference_payload:
                full_plots_dir = plots_dir / "full_dataset"
                full_plots_dir.mkdir(parents=True, exist_ok=True)
                full_samples = list(inference_payload.get("samples") or [])
                if full_samples:
                    dataset_plot_dir_full = full_plots_dir / f"probability_{anchor_name}"
                    _plot_for_dataset(
                        anchor_name,
                        full_samples,
                        dataset_plot_dir_full,
                        run_logger,
                        dataset_meta_map,
                        overlap_mask=overlap_mask_info,
                    )
                    _write_json(full_plots_dir / f"probability_{anchor_name}.json", full_samples)
                metadata_full = inference_payload.get("metadata")
                if metadata_full:
                    _write_json(full_plots_dir / f"metadata_{anchor_name}.json", metadata_full)
            (anchor_dir / 'training.log').write_text('\n'.join([f'Method {method_id} anchor summary', json.dumps(metrics)]), encoding='utf-8')
            summary_payload['anchor']['output_dir'] = str(anchor_dir)
            summary_payload['anchor']['metrics'] = metrics
            summary_payload['anchor']['train_size'] = len(train_set)
            summary_payload['anchor']['val_size'] = len(val_set)
            overview['anchor_train_size'] = len(train_set)
            overview['anchor_val_size'] = len(val_set)

        if target_info:
            target_dir = Path(method_dir) / ('method_2_' + target_name)
            target_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = target_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            target_state = target_info.pop('state_dict', None)
            target_predictions = dict(target_info)
            target_predictions.pop("metadata", None)
            _write_json(target_dir / 'predictions_target.json', target_predictions)
            _export_state_dict(target_state, target_dir / 'classifier.pt')
            metrics = target_info.get('metrics', {})
            history = target_info.get('history', [])
            _write_json(target_dir / 'training_metrics.json', metrics)
            _write_json(target_dir / 'training_history.json', history)
            _plot_training_curves(history, plots_dir / 'training_curve_target.png', f'Method 2 ({target_name}) Training')
            target_samples: List[Dict[str, object]] = []
            target_metadata: Sequence[Dict[str, object]] = (classifier_metadata.get('target') or {}).get('metadata', [])
            train_set = set(target_info.get('train_indices', []))
            val_set = set(target_info.get('val_indices', []))
            mc_info = target_info.get('mc_dropout') or {}
            mc_all_mean = mc_info.get('all_mean') or []
            mc_all_std = mc_info.get('all_std') or []
            mc_passes = mc_info.get('num_passes')
            valid_indices_target = target_info.get('valid_indices', [])
            all_probs_target = target_info.get('all_probs', [])
            for pos, idx in enumerate(valid_indices_target):
                if pos >= len(all_probs_target):
                    continue
                prob = all_probs_target[pos]
                if idx >= len(target_metadata):
                    continue
                meta = target_metadata[idx]
                coord = meta.get('coord')
                if coord is None or not isinstance(coord, (tuple, list)):
                    continue
                split = 'val' if idx in val_set else ('train' if idx in train_set else 'unlabeled')
                mc_mean_val = mc_all_mean[pos] if pos < len(mc_all_mean) else None
                mc_std_val = mc_all_std[pos] if pos < len(mc_all_std) else None
                sample_entry = {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "prob": float(prob),
                    "label": _resolve_binary_label(meta.get('label'), default=0),
                    "split": split,
                }
                if mc_mean_val is not None:
                    sample_entry["prob_mc_mean"] = float(mc_mean_val)
                if mc_std_val is not None:
                    sample_entry["prob_mc_std"] = float(mc_std_val)
                if mc_passes:
                    sample_entry["mc_passes"] = int(mc_passes)
                target_samples.append(sample_entry)
            dataset_plot_dir = plots_dir / f"probability_{target_name}"
            _plot_for_dataset(
                target_name,
                target_samples,
                dataset_plot_dir,
                run_logger,
                dataset_meta_map,
                overlap_mask=overlap_mask_info,
            )
            _write_json(target_dir / f"probability_{target_name}.json", target_samples)
            inference_payload = target_info.get("inference") or {}
            if inference_payload:
                full_plots_dir = plots_dir / "full_dataset"
                full_plots_dir.mkdir(parents=True, exist_ok=True)
                full_samples = list(inference_payload.get("samples") or [])
                if full_samples:
                    dataset_plot_dir_full = full_plots_dir / f"probability_{target_name}"
                    _plot_for_dataset(
                        target_name,
                        full_samples,
                        dataset_plot_dir_full,
                        run_logger,
                        dataset_meta_map,
                        overlap_mask=overlap_mask_info,
                    )
                    _write_json(full_plots_dir / f"probability_{target_name}.json", full_samples)
                metadata_full = inference_payload.get("metadata")
                if metadata_full:
                    _write_json(full_plots_dir / f"metadata_{target_name}.json", metadata_full)
            (target_dir / 'training.log').write_text('\n'.join([f'Method {method_id} target summary', json.dumps(metrics)]), encoding='utf-8')
            summary_payload['target']['output_dir'] = str(target_dir)
            summary_payload['target']['metrics'] = metrics
            summary_payload['target']['train_size'] = len(train_set)
            summary_payload['target']['val_size'] = len(val_set)
            overview['target_train_size'] = len(train_set)
            overview['target_val_size'] = len(val_set)

        if overview:
            _write_json(method_dir / 'summary.json', overview)

        if not summary_payload['anchor']:
            summary_payload.pop('anchor', None)
        if not summary_payload['target']:
            summary_payload.pop('target', None)
        return summary_payload

    run_logger.log(f'Classifier outputs skipped for unsupported method {method_id}.')
    return summary_payload

def _build_unified_features(
    projected_anchor: torch.Tensor,
    projected_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    simple = torch.cat([projected_anchor, projected_target], dim=1)
    diff = torch.abs(projected_anchor - projected_target)
    prod = projected_anchor * projected_target
    cosine = nn.functional.cosine_similarity(projected_anchor, projected_target, dim=1, eps=1e-8).unsqueeze(1)
    strong = torch.cat([projected_anchor, projected_target, diff, prod, cosine], dim=1)
    return simple, strong


def _get_bundle_embeddings(bundle: DatasetBundle) -> torch.Tensor:
    if bundle.embeddings is not None and bundle.embeddings.numel() > 0:
        return bundle.embeddings.detach().cpu().float()
    tensors: List[torch.Tensor] = []
    for record in bundle.records:
        emb_arr = np.asarray(record.embedding, dtype=np.float32)
        tensors.append(torch.from_numpy(emb_arr))
    if not tensors:
        return torch.empty(0, 0, dtype=torch.float32)
    return torch.stack(tensors, dim=0)


def _lookup_dataset_entry(dataset_meta_map: Dict[str, Dict[str, object]], dataset_name: str) -> Dict[str, object]:
    if not dataset_meta_map:
        return {}
    entry = dataset_meta_map.get(dataset_name)
    if isinstance(entry, dict) and "dataset_id" not in entry:
        return entry
    datasets_list = dataset_meta_map.get("datasets") if isinstance(dataset_meta_map, dict) else None
    if isinstance(datasets_list, list):
        for candidate in datasets_list:
            if isinstance(candidate, dict) and candidate.get("dataset_id") == dataset_name:
                return candidate
    return {}


def _dataset_native_crs(dataset_meta_map: Dict[str, Dict[str, object]], dataset_name: str) -> Optional[str]:
    native_map = dataset_meta_map.get("native_crs") if isinstance(dataset_meta_map, dict) else None
    if isinstance(native_map, dict):
        native_entry = native_map.get(dataset_name)
        if native_entry:
            return str(native_entry)
    entry = _lookup_dataset_entry(dataset_meta_map, dataset_name)
    native_entry = entry.get("native_crs") if isinstance(entry, dict) else None
    return str(native_entry) if native_entry else None


def _load_unified_mask_data(dataset_meta_map: Dict[str, Dict[str, object]], dataset_name: str) -> Optional[Dict[str, object]]:
    if rasterio is None:
        return None
    entry = _lookup_dataset_entry(dataset_meta_map, dataset_name)
    if not isinstance(entry, dict):
        return None
    boundaries = entry.get("boundaries")
    if not isinstance(boundaries, dict):
        return None
    unified_records = boundaries.get("Unified CRS")
    if not isinstance(unified_records, list):
        return None
    root_path = entry.get("root")
    root_resolved = Path(str(root_path)).resolve() if root_path else None
    for record in unified_records:
        if not isinstance(record, dict):
            continue
        path_value = record.get("path_resolved") or record.get("path")
        if not path_value:
            continue
        mask_path = Path(path_value)
        if not mask_path.is_absolute() and root_resolved is not None:
            mask_path = (root_resolved / mask_path).resolve()
        mask_path = mask_path.resolve()
        if not mask_path.exists():
            continue
        try:
            with rasterio.open(mask_path) as ds:
                array = ds.read(1)
                if array is None or array.size == 0:
                    continue
                mask = np.asarray(array != 0, dtype=bool)
                info = {
                    "array": mask,
                    "transform": ds.transform,
                    "shape": (ds.height, ds.width),
                    "nodata": ds.nodata,
                    "crs": ds.crs,
                    "path": str(mask_path),
                }
                return info
        except Exception:
            continue
    return None


def _build_transformer(native_crs: Optional[str], unified_crs: Optional[str]) -> Optional[Transformer]:
    if Transformer is None or CRS is None:
        return None
    if not native_crs or not unified_crs:
        return None
    try:
        native = CRS.from_user_input(native_crs)
        unified = CRS.from_user_input(unified_crs)
        if native == unified:
            return None
        return Transformer.from_crs(native, unified, always_xy=True)
    except Exception:
        return None


def _collect_full_dataset_projection(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    projector: Optional[nn.Module],
    device: torch.device,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    overlap_mask: Optional[Dict[str, object]] = None,
    mask_only: bool = True,
) -> Optional[Tuple[DatasetBundle, torch.Tensor, List[int], List[Optional[Tuple[int, int]]], List[bool]]]:
    if projector is None:
        return None
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None or not bundle.records:
        return None
    embeddings = _get_bundle_embeddings(bundle)
    if embeddings.numel() == 0:
        return None
    selected_indices: List[int] = []
    selected_rowcols: List[Optional[Tuple[int, int]]] = []
    mask_flags: List[bool] = []
    mask_array = overlap_mask.get("array") if overlap_mask else None
    mask_transform = overlap_mask.get("transform") if overlap_mask else None
    mask_shape = overlap_mask.get("shape") if overlap_mask else None
    mask_nodata = overlap_mask.get("nodata") if overlap_mask else None
    use_mask = mask_array is not None and mask_transform is not None and mask_shape is not None and rasterio is not None
    for idx, record in enumerate(bundle.records):
        coord = _normalise_coord(getattr(record, "coord", None))
        if coord is None:
            continue
        mask_inside = False
        mask_rowcol: Optional[Tuple[int, int]] = None
        include = True
        if use_mask:
            try:
                row, col = rasterio.transform.rowcol(mask_transform, coord[0], coord[1])
            except Exception:
                include = not mask_only
            else:
                include = 0 <= row < int(mask_shape[0]) and 0 <= col < int(mask_shape[1])
                if include:
                    value = mask_array[row, col]
                    if mask_nodata is not None and np.isfinite(mask_nodata):
                        if np.isclose(value, mask_nodata) or not np.isfinite(value):
                            value = 0
                    mask_inside = bool(value != 0)
                    mask_rowcol = (int(row), int(col))
                    if mask_only and not mask_inside:
                        include = False
                else:
                    include = not mask_only
        else:
            include = True

        if not include:
            continue

        selected_indices.append(idx)
        if use_mask and mask_rowcol is not None:
            selected_rowcols.append(mask_rowcol if mask_inside else None)
        else:
            rc = getattr(record, "row_col", None)
            selected_rowcols.append(_normalise_row_col(rc))
        mask_flags.append(mask_inside)
    if not selected_indices:
        return None
    index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
    if index_tensor.numel() == 0:
        return None
    embeddings = embeddings.index_select(0, index_tensor)
    if embeddings.numel() == 0:
        return None
    projected = _project_in_batches(
        embeddings,
        projector,
        device,
        batch_size,
        progress_desc=f"Project {dataset_name}",
    )
    return bundle, projected, selected_indices, selected_rowcols, mask_flags


def reembedding_DCCA(
    workspace: OverlapAlignmentWorkspace,
    dataset_name: str,
    projector: Optional[nn.Module],
    device: torch.device,
    *,
    overlap_mask: Optional[Dict[str, object]] = None,
    batch_size: int = INFERENCE_BATCH_SIZE,
    mask_only: bool = True,
) -> Optional[Dict[str, object]]:
    collected = _collect_full_dataset_projection(
        workspace,
        dataset_name,
        projector,
        device,
        batch_size=batch_size,
        overlap_mask=overlap_mask,
        mask_only=mask_only,
    )
    if collected is None:
        return None
    bundle, projected, record_indices, record_rowcols, mask_flags = collected
    projected_cpu = projected.detach().cpu()
    metadata_entries: List[Dict[str, object]] = []
    labels: List[int] = []
    coords: List[Optional[Tuple[float, float]]] = []
    for pos, (record_idx, row_col, mask_flag) in enumerate(zip(record_indices, record_rowcols, mask_flags)):
        record = bundle.records[record_idx]
        coord = _normalise_coord(getattr(record, "coord", None))
        label_val = int(getattr(record, "label", 0))
        coords.append(coord)
        metadata_entries.append(
            {
                "dataset": dataset_name,
                "index": int(getattr(record, "index", record_idx)),
                "record_index": int(record_idx),
                "coord": coord,
                "region": getattr(record, "region", None),
                "row_col": getattr(record, "row_col", None),
                "row_col_mask": [int(row_col[0]), int(row_col[1])] if row_col is not None else None,
                "tile_id": getattr(record, "tile_id", None),
                "position": int(pos),
                "label": label_val,
                "overlap_flag": bool(mask_flag),
            }
        )
        labels.append(label_val)
    return {
        "dataset": dataset_name,
        "features": projected_cpu,
        "indices": [int(idx) for idx in record_indices],
        "coords": coords,
        "metadata": metadata_entries,
        "labels": np.asarray(labels, dtype=np.int16),
        "row_cols": [tuple(rc) if rc is not None else None for rc in record_rowcols],
        "mask_flags": [bool(flag) for flag in mask_flags],
        "row_cols_mask": [tuple(rc) if rc is not None else None for rc in record_rowcols],
    }


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


def _prepare_strongfusion_dataset_all(
    cfg: AlignmentConfig,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_name: str,
    target_name: str,
    anchor_all: Optional[Dict[str, object]],
    target_all: Optional[Dict[str, object]],
    run_logger: "_RunLogger",
) -> Optional[Tuple[Dict[str, object], Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]]:
    if anchor_all is None and target_all is None:
        run_logger.log("[strongfusion] No embeddings available for either dataset; skipping strong fusion head.")
        return None
    unified_crs = dataset_meta_map.get("unified_crs") if isinstance(dataset_meta_map, dict) else None
    if unified_crs is None:
        run_logger.log("[strongfusion] Unified CRS unavailable; skipping strong fusion head.")
        return None
    mask_info_anchor = _load_unified_mask_data(dataset_meta_map, anchor_name)
    mask_info_target = _load_unified_mask_data(dataset_meta_map, target_name)
    if mask_info_anchor is None or mask_info_target is None:
        run_logger.log("[strongfusion] Unified boundary masks missing; skipping strong fusion head.")
        return None
    native_anchor = _dataset_native_crs(dataset_meta_map, anchor_name)
    native_target = _dataset_native_crs(dataset_meta_map, target_name)
    transformer_anchor = _build_transformer(native_anchor, unified_crs)
    transformer_target = _build_transformer(native_target, unified_crs)

    def _build_lookup(entry: Optional[Dict[str, object]], transformer, mask_info: Dict[str, object]) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], Dict[str, object]], Dict[Tuple[int, int], int]]:
        feature_map: Dict[Tuple[int, int], np.ndarray] = {}
        metadata_map: Dict[Tuple[int, int], Dict[str, object]] = {}
        label_map: Dict[Tuple[int, int], int] = {}
        if entry is None:
            return feature_map, metadata_map, label_map
        features_tensor: torch.Tensor = entry.get("features")  # type: ignore[assignment]
        if features_tensor is None or not isinstance(features_tensor, torch.Tensor):
            return feature_map, metadata_map, label_map
        features_np = features_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        coords = entry.get("coords") or []
        metadata_list = entry.get("metadata") or []
        labels_arr = entry.get("labels")
        row_cols_mask = entry.get("row_cols_mask") or entry.get("row_cols") or []
        transform = mask_info.get("transform")
        shape = mask_info.get("shape")
        if transform is None or shape is None:
            return feature_map, metadata_map, label_map
        height, width = int(shape[0]), int(shape[1])
        for idx, feature in enumerate(features_np):
            key: Optional[Tuple[int, int]] = None
            if idx < len(row_cols_mask) and row_cols_mask[idx] is not None:
                rc = row_cols_mask[idx]
                key = (int(rc[0]), int(rc[1]))
            else:
                if idx >= len(coords):
                    continue
                coord = coords[idx]
                if coord is None:
                    continue
                if transformer is not None:
                    try:
                        x_u, y_u = transformer.transform(coord[0], coord[1])
                    except Exception:
                        continue
                else:
                    x_u, y_u = float(coord[0]), float(coord[1])
                try:
                    row, col = rasterio.transform.rowcol(transform, x_u, y_u)
                except Exception:
                    continue
                if not (0 <= row < height and 0 <= col < width):
                    continue
                key = (int(row), int(col))
            if key is None or key in feature_map:
                continue
            feature_map[key] = feature
            meta_entry = metadata_list[idx] if idx < len(metadata_list) else {}
            metadata_map[key] = meta_entry if isinstance(meta_entry, dict) else {}
            if isinstance(labels_arr, np.ndarray) and idx < len(labels_arr):
                label_map[key] = int(labels_arr[idx])
            else:
                label_map[key] = 0
        return feature_map, metadata_map, label_map

    anchor_lookup, anchor_meta_map, anchor_label_map = _build_lookup(anchor_all, transformer_anchor, mask_info_anchor)
    target_lookup, target_meta_map, target_label_map = _build_lookup(target_all, transformer_target, mask_info_target)

    if not anchor_lookup and not target_lookup:
        run_logger.log("[strongfusion] Unable to align embeddings to unified grid; skipping strong fusion head.")
        return None

    union_keys = sorted(set(anchor_lookup.keys()) | set(target_lookup.keys()))
    if not union_keys:
        run_logger.log("[strongfusion] No unified grid positions available; skipping strong fusion head.")
        return None

    dim_u = int(anchor_all["features"].shape[1]) if anchor_all and isinstance(anchor_all.get("features"), torch.Tensor) else (int(target_all["features"].shape[1]) if target_all and isinstance(target_all.get("features"), torch.Tensor) else 0)
    dim_v = int(target_all["features"].shape[1]) if target_all and isinstance(target_all.get("features"), torch.Tensor) else dim_u
    if dim_u <= 0 or dim_v <= 0:
        run_logger.log("[strongfusion] Invalid feature dimensions; skipping strong fusion head.")
        return None

    phi_map: Dict[Tuple[int, int], np.ndarray] = {}
    phi_rows: List[np.ndarray] = []
    label_rows: List[int] = []
    metadata_rows: List[Dict[str, object]] = []
    anchor_presence: List[bool] = []
    target_presence: List[bool] = []

    for key in union_keys:
        u_vec = anchor_lookup.get(key)
        v_vec = target_lookup.get(key)
        anchor_present = u_vec is not None
        target_present = v_vec is not None
        anchor_presence.append(anchor_present)
        target_presence.append(target_present)
        u_arr = u_vec.astype(np.float32, copy=False) if anchor_present else np.zeros(dim_u, dtype=np.float32)
        v_arr = v_vec.astype(np.float32, copy=False) if target_present else np.zeros(dim_v, dtype=np.float32)
        dim_min = min(len(u_arr), len(v_arr))
        diff_vec = np.abs(u_arr[:dim_min] - v_arr[:dim_min])
        prod_vec = u_arr[:dim_min] * v_arr[:dim_min]
        norm_u = np.linalg.norm(u_arr)
        norm_v = np.linalg.norm(v_arr)
        cosine_val = float(np.dot(u_arr, v_arr) / (norm_u * norm_v + 1e-8)) if norm_u > 0 and norm_v > 0 else 0.0
        missing_flag = 0.0 if anchor_present and target_present else 1.0
        phi = np.concatenate([
            u_arr,
            v_arr,
            diff_vec,
            prod_vec,
            np.array([cosine_val], dtype=np.float32),
            np.array([missing_flag], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        phi_map[key] = phi
        phi_rows.append(phi)
        anchor_label = anchor_label_map.get(key, 0)
        target_label = target_label_map.get(key, 0)
        label_rows.append(1 if (anchor_label > 0 or target_label > 0) else 0)
        meta_combined: Dict[str, object] = {
            "row": int(key[0]),
            "col": int(key[1]),
            "anchor_present": bool(anchor_present),
            "target_present": bool(target_present),
        }
        if anchor_present:
            meta_combined["anchor"] = anchor_meta_map.get(key, {})
        if target_present:
            meta_combined["target"] = target_meta_map.get(key, {})
        metadata_rows.append(meta_combined)

    features_array = np.asarray(phi_rows, dtype=np.float32)
    labels_array = np.asarray(label_rows, dtype=np.int16)
    anchor_keys = [key for key in union_keys if key in anchor_lookup]
    target_keys = [key for key in union_keys if key in target_lookup]

    def _build_inference_entry(keys: List[Tuple[int, int]], lookup: Dict[Tuple[int, int], np.ndarray], label_map: Dict[Tuple[int, int], int], meta_map: Dict[Tuple[int, int], Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not keys:
            return None
        embeddings = np.asarray([phi_map[key] for key in keys], dtype=np.float32)
        index_lookup = {key: idx for idx, key in enumerate(keys)}
        coords = [(int(r), int(c)) for r, c in keys]
        metadata = [meta_map.get(key, {}) for key in keys]
        labels = np.asarray([label_map.get(key, 0) for key in keys], dtype=np.int16)
        return {
            "tuple": (embeddings, index_lookup, coords),
            "metadata": metadata,
            "labels": labels,
            "coords": coords,
        }

    inference_map = {
        anchor_name: _build_inference_entry(anchor_keys, anchor_lookup, anchor_label_map, anchor_meta_map),
        target_name: _build_inference_entry(target_keys, target_lookup, target_label_map, target_meta_map),
    }

    dataset_payload = {
        "features": features_array,
        "labels": labels_array,
        "metadata": metadata_rows,
        "row_cols": union_keys,
        "dim_u": dim_u,
        "dim_v": dim_v,
        "anchor_presence": anchor_presence,
        "target_presence": target_presence,
    }

    mask_info_map = {
        anchor_name: mask_info_anchor,
        target_name: mask_info_target,
    }

    return dataset_payload, inference_map, mask_info_map


def _strongfusion_run_inference(
    strongfusion_dir: Path,
    inference_map: Dict[str, Optional[Dict[str, object]]],
    mask_info_map: Dict[str, Dict[str, object]],
    mlp: nn.Module,
    cfg: AlignmentConfig,
    device: torch.device,
    run_logger: "_RunLogger",
) -> Dict[str, Dict[str, object]]:
    outputs: Dict[str, Dict[str, object]] = {}
    passes = max(1, int(getattr(cfg.cls_training, "mc_dropout_passes", 30)))
    for dataset_name, entry in inference_map.items():
        if entry is None:
            run_logger.log(f"[strongfusion] No inference embeddings prepared for dataset {dataset_name}; skipping export.")
            continue
        mask_info = mask_info_map.get(dataset_name)
        if mask_info is None:
            run_logger.log(f"[strongfusion] Missing unified mask for dataset {dataset_name}; skipping export.")
            continue
        stack = _MaskStack(mask_info)
        default_reference = _make_mask_reference(mask_info)
        prediction = mc_predict_map_from_embeddings(
            entry["tuple"],
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
            prediction_payload = prediction
        labels_arr = entry.get("labels") if isinstance(entry.get("labels"), np.ndarray) else np.empty(0, dtype=np.int16)
        coords = entry.get("coords") or []
        metadata = entry.get("metadata") or []
        pos_coords = [
            (str(meta.get("region") or "GLOBAL"), int(coord[0]), int(coord[1]))
            for coord, meta, lbl in zip(coords, metadata, labels_arr)
            if lbl > 0
        ]
        neg_coords = [
            (str(meta.get("region") or "GLOBAL"), int(coord[0]), int(coord[1]))
            for coord, meta, lbl in zip(coords, metadata, labels_arr)
            if lbl <= 0
        ]
        pos_map = group_coords(pos_coords, stack) if pos_coords else {}
        neg_map = group_coords(neg_coords, stack) if neg_coords else {}
        outputs[dataset_name] = {
            "prediction": prediction_payload,
            "default_reference": default_reference,
            "pos_map": pos_map,
            "neg_map": neg_map,
            "counts": {
                "pos": int(sum(len(vals) for vals in pos_map.values())),
                "neg": int(sum(len(vals) for vals in neg_map.values())),
            },
        }
    return outputs


def _strongfusion_build_record_lookup(
    strong_dataset: Dict[str, object],
    anchor_name: str,
    target_name: str,
) -> Dict[str, Dict[int, int]]:
    lookup: Dict[str, Dict[int, int]] = {
        anchor_name: {},
        target_name: {},
    }
    metadata_rows: Sequence[Dict[str, object]] = strong_dataset.get("metadata", []) or []
    for row_idx, meta in enumerate(metadata_rows):
        anchor_entry = meta.get("anchor") if isinstance(meta, dict) else None
        target_entry = meta.get("target") if isinstance(meta, dict) else None
        if isinstance(anchor_entry, dict):
            for key_name in ("embedding_index", "record_index", "index"):
                value = anchor_entry.get(key_name)
                if value is not None:
                    lookup[anchor_name][int(value)] = row_idx
        if isinstance(target_entry, dict):
            for key_name in ("embedding_index", "record_index", "index"):
                value = target_entry.get(key_name)
                if value is not None:
                    lookup[target_name][int(value)] = row_idx
    return lookup


def _strongfusion_map_method1_indices(
    strong_dataset: Dict[str, object],
    method1_data: Dict[str, object],
    anchor_name: str,
    target_name: str,
    run_logger: "_RunLogger",
) -> Tuple[np.ndarray, np.ndarray]:
    metadata_list: Sequence[Dict[str, object]] = method1_data.get("metadata", []) or []
    train_source: Sequence[int] = method1_data.get("train_indices", []) or []
    val_source: Sequence[int] = method1_data.get("val_indices", []) or []
    record_lookup = _strongfusion_build_record_lookup(strong_dataset, anchor_name, target_name)

    def _map(indices: Sequence[int], tag: str) -> np.ndarray:
        mapped: List[int] = []
        missing = 0
        seen: set[int] = set()
        for idx in indices:
            if idx >= len(metadata_list):
                missing += 1
                continue
            meta = metadata_list[idx] or {}
            dataset = str(meta.get("dataset") or "")
            dataset_lookup = record_lookup.get(dataset)
            if dataset_lookup is None:
                if dataset.lower() == "anchor":
                    dataset_lookup = record_lookup.get(anchor_name)
                    dataset = anchor_name
                elif dataset.lower() == "target":
                    dataset_lookup = record_lookup.get(target_name)
                    dataset = target_name
            if dataset_lookup is None:
                missing += 1
                continue
            candidate_values: List[int] = []
            for key_name in ("embedding_index", "record_index", "index"):
                value = meta.get(key_name)
                if value is not None:
                    candidate_values.append(int(value))
            row_idx: Optional[int] = None
            for value in candidate_values:
                row_idx = dataset_lookup.get(value)
                if row_idx is not None:
                    break
            if row_idx is None:
                missing += 1
                continue
            if row_idx in seen:
                continue
            mapped.append(row_idx)
            seen.add(row_idx)
        if missing and mapped:
            run_logger.log(f"[strongfusion] {missing} {tag} samples could not be aligned to the unified grid.")
        return np.asarray(mapped, dtype=int) if mapped else np.empty(0, dtype=int)

    train_rows = _map(train_source, "training")
    val_rows = _map(val_source, "validation")
    return train_rows, val_rows


def _build_samples_from_projection(
    bundle: DatasetBundle,
    record_indices: Sequence[int],
    probs: Sequence[float],
    stds: Optional[Sequence[float]],
    dataset_name: str,
    *,
    mc_passes: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    samples: List[Dict[str, object]] = []
    metadata_entries: List[Dict[str, object]] = []
    probs_list = list(probs)
    stds_list = list(stds) if stds is not None else None
    paired_indices = list(enumerate(record_indices))
    if progress_desc:
        iterator: Iterable[Tuple[int, int]] = _progress_iter(paired_indices, progress_desc, leave=False, total=len(paired_indices))
    else:
        iterator = paired_indices
    for pos, record_idx in iterator:
        if pos >= len(probs_list):
            break
        record = bundle.records[record_idx]
        coord = _normalise_coord(getattr(record, "coord", None))
        label_val = _resolve_binary_label(getattr(record, "label", 0), default=0)
        record_index = int(getattr(record, "index", record_idx))
        metadata_entry: Dict[str, object] = {
            "dataset": dataset_name,
            "label": label_val,
            "split": "inference",
            "index": record_index,
        }
        if coord is not None:
            metadata_entry["coord"] = [float(coord[0]), float(coord[1])]
        row_col = getattr(record, "row_col", None)
        if row_col is not None:
            row_col_norm = _normalise_row_col(row_col)
            if row_col_norm is not None:
                metadata_entry["row_col"] = [int(row_col_norm[0]), int(row_col_norm[1])]
        tile_id = getattr(record, "tile_id", None)
        if tile_id is not None:
            metadata_entry["tile_id"] = tile_id
        pixel_res = getattr(record, "pixel_resolution", None)
        if pixel_res is not None and math.isfinite(pixel_res):
            metadata_entry["pixel_resolution"] = float(pixel_res)
        window_size = getattr(record, "window_size", None)
        if isinstance(window_size, (list, tuple)) and len(window_size) >= 2:
            try:
                metadata_entry["window_size"] = [int(window_size[0]), int(window_size[1])]
            except Exception:
                pass
        metadata_entries.append(metadata_entry)
        if coord is None:
            continue
        prob_val = probs_list[pos]
        sample_entry: Dict[str, object] = {
            "x": float(coord[0]),
            "y": float(coord[1]),
            "prob": float(prob_val),
            "label": label_val,
            "split": "inference",
            "dataset": dataset_name,
            "record_index": record_index,
        }
        region_val = getattr(record, "region", None)
        if region_val is not None:
            sample_entry["region"] = region_val
        if stds_list is not None and pos < len(stds_list):
            sample_entry["prob_mc_std"] = float(stds_list[pos])
        if mc_passes:
            sample_entry["mc_passes"] = int(mc_passes)
        samples.append(sample_entry)
    return samples, metadata_entries

def _binary_classification_metrics(targets: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    metrics = {
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "pos_rate": float("nan"),
    }
    if targets.numel() == 0 or probs.numel() == 0:
        return metrics
    targets_np = targets.detach().cpu().numpy().astype(int)
    probs_np = probs.detach().cpu().numpy()
    preds_np = (probs_np >= 0.5).astype(int)
    total = targets_np.size
    correct = (preds_np == targets_np).sum()
    tp = ((preds_np == 1) & (targets_np == 1)).sum()
    fp = ((preds_np == 1) & (targets_np == 0)).sum()
    fn = ((preds_np == 0) & (targets_np == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    metrics.update(
        {
            "accuracy": correct / total if total else float("nan"),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pos_rate": float(probs_np.mean()),
        }
    )
    return metrics


def _train_classifier(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    pos_weight: Optional[float] = None,
    *,
    model_factory: Optional[Callable[[int], nn.Module]] = None,
    progress_desc: str = "CLS",
) -> Tuple[nn.Module, List[Dict[str, object]], Dict[str, Dict[str, float]], torch.Tensor, torch.Tensor]:
    device = features_train.device
    input_dim = features_train.size(1)
    if model_factory is None:
        model = ClassifierHead(input_dim).to(device)
    else:
        model = model_factory(input_dim)
        if model is None:
            raise ValueError("model_factory returned None")
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pos_weight is not None and pos_weight > 0:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    history: List[Dict[str, object]] = []
    if tqdm is not None:
        epoch_iter = tqdm(range(1, epochs + 1), desc=progress_desc, leave=False)
    else:
        epoch_iter = range(1, epochs + 1)

    labels_val = labels_val if labels_val is not None else torch.empty(0, dtype=torch.float32, device=device)
    final_train_probs = torch.empty(0)
    final_val_probs = torch.empty(0)

    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        logits_train = model(features_train)
        loss = loss_fn(logits_train, labels_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_train_eval = model(features_train)
            probs_train = torch.sigmoid(logits_train_eval).squeeze(1)
            train_metrics = _binary_classification_metrics(labels_train, probs_train)
            train_metrics["loss"] = float(loss_fn(logits_train_eval, labels_train.unsqueeze(1)).item())
            probs_val = torch.empty(0, device=device)
            val_metrics: Dict[str, float] = {}
            if labels_val.numel() > 0 and features_val.numel() > 0:
                logits_val = model(features_val)
                probs_val = torch.sigmoid(logits_val).squeeze(1)
                val_metrics = _binary_classification_metrics(labels_val, probs_val)
                val_metrics["loss"] = float(loss_fn(logits_val, labels_val.unsqueeze(1)).item())

        history.append(
            {
                "epoch": int(epoch),
                "train": {key: float(value) for key, value in train_metrics.items()},
                "val": {key: float(value) for key, value in val_metrics.items()} if val_metrics else {},
            }
        )

        final_train_probs = probs_train.detach().cpu()
        final_val_probs = probs_val.detach().cpu() if probs_val.numel() else torch.empty(0)

        if tqdm is not None:
            postfix = {"loss": train_metrics["loss"]}
            if val_metrics:
                postfix["val_loss"] = val_metrics.get("loss")
            epoch_iter.set_postfix(postfix)

    final_metrics = {
        "train": history[-1]["train"] if history else {},
        "val": history[-1]["val"] if history and history[-1]["val"] else {},
    }
    return model, history, final_metrics, final_train_probs, final_val_probs


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


def _split_classifier_indices(
    total: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if total <= 0:
        return [], []
    validation_fraction = max(0.0, min(float(validation_fraction), 0.9))
    val_count = int(total * validation_fraction)
    if val_count > 0 and total - val_count < 1:
        val_count = max(0, total - 1)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator)
    val_indices = permutation[:val_count].tolist() if val_count > 0 else []
    train_indices = permutation[val_count:].tolist()
    if not train_indices and val_indices:
        train_indices.append(val_indices.pop())
    return train_indices, val_indices

def _count_label_distribution(labels: Sequence[int]) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0}
    for value in labels:
        try:
            if int(value) > 0:
                counts["positive"] += 1
            else:
                counts["negative"] += 1
        except Exception:
            counts.setdefault("unknown", 0)
            counts["unknown"] += 1
    return counts


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


def _resolve_dcca_weights_path(cfg: AlignmentConfig, override: Optional[str]) -> Optional[Path]:
    if override:
        try:
            candidate = Path(override).expanduser().resolve()
        except Exception:
            candidate = Path(override)
        if candidate.exists():
            return candidate
        return candidate if override else None
    candidates: List[Path] = []
    if cfg.output_dir is not None:
        candidates.append((cfg.output_dir / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
    if cfg.log_dir is not None:
        candidates.append((cfg.log_dir / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
    candidates.append((Path.cwd() / "overlap_alignment_outputs" / "overlap_alignment_stage1.pt").resolve())
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None

def _load_pretrained_dcca_state(path: Path) -> Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Optional[Dict[str, object]]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} does not contain a valid dictionary payload.")
    if "projection_head_a" not in checkpoint or "projection_head_b" not in checkpoint:
        raise KeyError(f"Checkpoint {path} is missing DCCA projection head weights.")
    state_a = checkpoint["projection_head_a"]
    state_b = checkpoint["projection_head_b"]
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        raise ValueError(f"Checkpoint {path} does not contain valid state dicts for DCCA projections.")
    summary = checkpoint.get("summary")
    if summary is not None and not isinstance(summary, dict):
        summary = None
    return (state_a, state_b), summary


def _projection_head_from_state(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    weight_keys = [key for key in state_dict.keys() if key.endswith(".weight")]
    if not weight_keys:
        raise ValueError("DCCA state dict does not contain linear layer weights.")
    def _layer_index(key: str) -> int:
        parts = key.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    sorted_keys = sorted(weight_keys, key=_layer_index)
    first_weight = state_dict[sorted_keys[0]]
    last_weight = state_dict[sorted_keys[-1]]
    if first_weight.ndim != 2 or last_weight.ndim != 2:
        raise ValueError("Unexpected weight tensor shape in DCCA state dict.")
    in_dim = int(first_weight.size(1))
    out_dim = int(last_weight.size(0))
    num_layers = len(sorted_keys)
    head = ProjectionHead(in_dim, out_dim, num_layers=num_layers)
    head.load_state_dict(state_dict, strict=True)
    return head

def _load_augmented_embeddings(path: Path, region_filter: Optional[Sequence[str]]) -> Tuple[Dict[str, List[np.ndarray]], int]:
    mapping: Dict[str, List[np.ndarray]] = {}
    count = 0
    region_filter_set: Optional[set[str]] = None
    if region_filter:
        region_filter_set = {str(entry).upper() for entry in region_filter}
    try:
        with np.load(path, allow_pickle=True) as bundle:
            embeddings = bundle.get("embeddings")
            metadata = bundle.get("metadata")
            flags = bundle.get("is_augmented")
            tile_ids = bundle.get("tile_ids")
            if embeddings is None or metadata is None or flags is None or tile_ids is None:
                print(f"[warn] positive augmentation bundle {path} is missing required arrays; ignoring.")
                return {}, 0
            for emb, meta, flag, tile_id in zip(embeddings, metadata, flags, tile_ids):
                if not bool(flag):
                    continue
                meta_dict: Dict[str, object] = {}
                if isinstance(meta, dict):
                    meta_dict = meta
                    try:
                        meta_dict = meta.item()
                    except Exception:
                        meta_dict = {}
                region_value = str(meta_dict.get("region") or meta_dict.get("Region") or "").upper()
                if region_filter_set and region_value and region_value not in region_filter_set:
                    continue
                source_tile_id = meta_dict.get("source_tile_id")
                if source_tile_id is None and isinstance(tile_id, str):
                    source_tile_id = tile_id.split("__")[0]
                if source_tile_id is None:
                    continue
                emb_arr = np.asarray(emb, dtype=np.float32)
                mapping.setdefault(str(source_tile_id), []).append(emb_arr)
                count += 1
    except FileNotFoundError:
        print(f"[warn] positive augmentation file {path} not found.")
        return {}, 0
    except Exception as exc:
        print(f"[warn] Failed to load positive augmentation file {path}: {exc}")
        return {}, 0
    return mapping, count

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


def _print_augmentation_usage(
    anchor_name: str,
    target_name: str,
    loaded_counts: Dict[str, int],
    usage_stats: Dict[str, int],
    total_pairs: int,
) -> None:
    anchor_loaded = loaded_counts.get(anchor_name, 0)
    target_loaded = loaded_counts.get(target_name, 0)
    anchor_used = int(usage_stats.get("anchor_augmented_pairs", 0))
    target_used = int(usage_stats.get("target_augmented_pairs", 0))
    print(
        "[info] augmentation usage: "
        f"{anchor_name} loaded={anchor_loaded} used={anchor_used}; "
        f"{target_name} loaded={target_loaded} used={target_used}; "
        f"total_pairs_with_aug={total_pairs}"
    )











def _matrix_inverse_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        eigvals, eigvecs = torch.linalg.eigh(mat)
    except RuntimeError:
        dim = mat.size(0)
        return torch.eye(dim, device=mat.device, dtype=mat.dtype)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return inv_sqrt


def _adaptive_tmat_jitter(mat: torch.Tensor, base: float = 1e-3, scale_ratio: float = 1e-2) -> float:
    if mat.numel() == 0:
        return 0.0
    try:
        fro_norm = torch.linalg.norm(mat, ord="fro")
    except RuntimeError:
        fro_norm = torch.sqrt(torch.sum(mat * mat))
    dim = float(mat.size(-1))
    if dim <= 0:
        return float(base)
    scaled = float(fro_norm.detach().cpu()) / math.sqrt(mat.numel())
    jitter = max(base, scaled * scale_ratio)
    return float(jitter)


def _prepare_output_dir(primary: Optional[Path], fallback_relative: str) -> Optional[Path]:
    if primary is not None:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except Exception as exc:
            print(f"[warn] Unable to create directory {primary}: {exc}. Falling back to local path.")
    fallback = Path.cwd() / fallback_relative
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception as exc:
        print(f"[warn] Unable to create fallback directory {fallback}: {exc}")
        return None


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


def _persist_state(
    cfg: AlignmentConfig,
    proj_a: nn.Module,
    proj_b: nn.Module,
    #summary: Dict[str, object],
    *,
    filename: str = "overlap_alignment_stage1.pt",
) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return

    state_path = target_dir / filename
    payload = {
        "projection_head_a": proj_a.state_dict(),
        "projection_head_b": proj_b.state_dict(),
    #    "summary": summary,
    }
    try:
        torch.save(payload, state_path)
        print(f"[info] saved stage-1 weights to {state_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist training state: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            alt_path = fallback_dir / state_path.name
            try:
                torch.save(payload, alt_path)
                print(f"[info] saved stage-1 weights to fallback location {alt_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist training state to fallback location: {exc_fallback}")


def _persist_metrics(
    cfg: AlignmentConfig,
    summary: Dict[str, object],
    epoch_history: Sequence[Dict[str, float]],
    *,
    filename: str = "overlap_alignment_stage1_metrics.json",
) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return
    metrics_path = target_dir / filename
    payload = {
        "summary": summary,
        "epoch_history": list(epoch_history),
    }

    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] saved metrics to {metrics_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist metrics: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            fallback_path = fallback_dir / metrics_path.name
            try:
                with fallback_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                print(f"[info] saved metrics to fallback location {fallback_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist metrics to fallback location: {exc_fallback}")


def _maybe_save_debug_figures(cfg: AlignmentConfig, debug_data: Optional[Dict[str, List[Dict[str, object]]]]) -> None:
    if debug_data is None:
        print("[warn] Debug data unavailable; skipping figure generation.")
        return
    viz_primary = None
    if cfg.output_dir is not None:
        viz_primary = cfg.output_dir / "overlap_visualizations" / "overlap"
    elif cfg.log_dir is not None:
        viz_primary = cfg.log_dir / "overlap_visualizations" / "overlap"
    viz_dir = _prepare_output_dir(viz_primary, "overlap_alignment_debug/overlap_visualizations/overlap")
    if viz_dir is None:
        return

    geometry = None
    overlap_json = cfg.overlap_pairs_path or cfg.overlap_pairs_augmented_path
    if overlap_json is not None:
        try:
            with Path(overlap_json).open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
            geometry = doc.get("overlap", {}).get("geometry")
        except Exception:
            geometry = None

    anchor_samples = debug_data.get("anchor_positive", []) or []
    target_samples = debug_data.get("target_positive", []) or []
    anchor_label = (
        anchor_samples[0].get("dataset")
        if anchor_samples and anchor_samples[0].get("dataset")
        else debug_data.get("anchor_name", "anchor")
    )
    target_label = (
        target_samples[0].get("dataset")
        if target_samples and target_samples[0].get("dataset")
        else debug_data.get("target_name", "target")
    )

    anchor_serialised = [_serialise_sample(sample) for sample in anchor_samples]
    target_serialised = [_serialise_sample(sample) for sample in target_samples]
    plot_path = viz_dir / "debug_positive_overlap.png"
    save_overlap_debug_plot(
        plot_path,
        geometry,
        anchor_serialised,
        target_serialised,
        title="Positive overlap",
        anchor_label=str(anchor_label),
        target_label=str(target_label),
        centroid_points=None,
    )
    print(f"[info] saved debug figure to {plot_path}")

    if cfg.use_positive_augmentation:
        aug_anchor = [sample for sample in anchor_serialised if sample.get("is_augmented")]
        aug_target = [sample for sample in target_serialised if sample.get("is_augmented")]
        if aug_anchor or aug_target:
            aug_path = viz_dir / "debug_positive_aug_overlap.png"
            save_overlap_debug_plot(
                aug_path,
                geometry,
                aug_anchor or anchor_serialised,
                aug_target or target_serialised,
                title="Positive overlap (augmented)",
                anchor_label=str(anchor_label),
                target_label=str(target_label),
                centroid_points=None,
            )
            print(f"[info] saved debug figure to {aug_path}")

# def _prepare_classifier_inputs(
#     anchor_name: str,
#     target_name: str,
#     sample_sets: Dict[str, Dict[str, object]],
#     validation_fraction: float,
#     seed: int,
# ) -> Tuple[Optional[Dict[str, object]], Dict[str, Dict[str, object]]]:
#     anchor_set = sample_sets.get(anchor_name)
#     target_set = sample_sets.get(target_name)
#     method1_data: Optional[Dict[str, object]] = None
#     if anchor_set or target_set:
#         anchor_features_list: List[torch.Tensor] = []
#         target_features_list: List[torch.Tensor] = []
#         labels_list: List[int] = []
#         metadata_list: List[Dict[str, object]] = []
#         if anchor_set:
#             features = anchor_set["features"]
#             zeros_target = torch.zeros_like(features)
#             anchor_features_list.append(features)
#             target_features_list.append(zeros_target)
#             for idx, meta in enumerate(anchor_set["metadata"]):
#                 label_int = int(anchor_set["labels"][idx].item())
#                 labels_list.append(label_int)
#                 meta_entry = dict(meta)
#                 meta_entry.setdefault("anchor_coord", meta_entry.get("coord"))
#                 meta_entry.setdefault("anchor_label", label_int)
#                 meta_entry.setdefault("anchor_region", meta_entry.get("region"))
#                 meta_entry.setdefault("target_coords", [])
#                 meta_entry.setdefault("target_labels", [])
#                 meta_entry.setdefault("target_regions", [])
#                 metadata_list.append(meta_entry)
#         if target_set:
#             features = target_set["features"]
#             zeros_anchor = torch.zeros_like(features)
#             anchor_features_list.append(zeros_anchor)
#             target_features_list.append(features)
#             for idx, meta in enumerate(target_set["metadata"]):
#                 label_int = int(target_set["labels"][idx].item())
#                 labels_list.append(label_int)
#                 coord_val = meta.get("coord")
#                 meta_entry = {
#                     **meta,
#                     "anchor_coord": None,
#                     "anchor_label": None,
#                     "anchor_region": None,
#                     "target_coords": [coord_val] if coord_val is not None else [],
#                     "target_labels": [label_int],
#                     "target_regions": [meta.get("region")],
#                 }
#                 metadata_list.append(meta_entry)
#         if labels_list:
#             anchor_tensor = torch.cat(anchor_features_list, dim=0) if anchor_features_list else torch.empty(0)
#             target_tensor = torch.cat(target_features_list, dim=0) if target_features_list else torch.empty(0)
#             labels_dicts = [{"combined_label": int(label)} for label in labels_list]
#             train_indices, val_indices = _split_classifier_indices(len(labels_list), validation_fraction, seed)
#             method1_data = {
#                 "anchor_features": anchor_tensor,
#                 "target_features": target_tensor,
#                 "labels": labels_dicts,
#                 "metadata": metadata_list,
#                 "train_indices": train_indices,
#                 "val_indices": val_indices,
#             }
#     method2_data: Dict[str, Dict[str, object]] = {}
#     if anchor_set:
#         train_idx, val_idx = _split_classifier_indices(len(anchor_set["labels"]), validation_fraction, seed + 101)
#         method2_data["anchor"] = {
#             **anchor_set,
#             "train_indices": train_idx,
#             "val_indices": val_idx,
#         }
#     if target_set:
#         train_idx, val_idx = _split_classifier_indices(len(target_set["labels"]), validation_fraction, seed + 202)
#         method2_data["target"] = {
#             **target_set,
#             "train_indices": train_idx,
#             "val_indices": val_idx,
#         }
#     return method1_data, method2_data


# def _train_unified_method(
#     cfg: AlignmentConfig,
#     run_logger: "_RunLogger",
#     projected_anchor: torch.Tensor,
#     projected_target: torch.Tensor,
#     labels: List[Dict[str, Optional[int]]],
#     train_indices: Sequence[int],
#     val_indices: Sequence[int],
#     device: torch.device,
#     total_epochs: int = 100,
#     *,
#     mlp_hidden_dims: Sequence[int],
#     mlp_dropout: float,
#     mc_passes: int = 30,
# ) -> Dict[str, Dict[str, object]]:
#     simple_features, strong_features = _build_unified_features(projected_anchor, projected_target)
#     valid_indices = [idx for idx, entry in enumerate(labels) if entry.get("combined_label") is not None]
#     if not valid_indices:
#         run_logger.log("Unified classifier skipped: no labeled pairs available.")
#         return {}
#     index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
#     label_tensor = torch.tensor(
#         [1 if labels[idx].get("combined_label") else 0 for idx in valid_indices],
#         dtype=torch.float32,
#     )
#     sel_train = [index_map[idx] for idx in train_indices if idx in index_map]
#     sel_val = [index_map[idx] for idx in val_indices if idx in index_map]
#     if not sel_train:
#         run_logger.log("Unified classifier skipped: no labeled training pairs available.")
#         return {}
#     def _select(source: torch.Tensor, idx_list: List[int]) -> torch.Tensor:
#         if not idx_list:
#             return torch.empty(0, source.size(1), device=device)
#         subset = source[valid_indices]
#         return subset[idx_list].to(device)
#     def _select_labels(idx_list: List[int]) -> torch.Tensor:
#         if not idx_list:
#             return torch.empty(0, device=device)
#         return label_tensor[idx_list].to(device)
#     results: Dict[str, Dict[str, float]] = {}
#     classifier_factory = lambda in_dim: _build_mlp_classifier(in_dim, mlp_hidden_dims, mlp_dropout)
#     feature_variants = [("simple", simple_features), ("strong", strong_features)]
#     for tag, feature_tensor in _progress_iter(feature_variants, "Method 1 heads", leave=False, total=len(feature_variants)):
#         x_train = _select(feature_tensor, sel_train)
#         x_val = _select(feature_tensor, sel_val)
#         y_train = _select_labels(sel_train)
#         y_val = _select_labels(sel_val)
#         if x_train.size(0) < 2:
#             run_logger.log(f"Unified classifier ({tag}) skipped: insufficient training samples.")
#             continue
#         pos_weight = None
#         pos_count = float(y_train.sum().item())
#         neg_count = float(y_train.numel() - pos_count)

#         if pos_count > 0 and neg_count > 0:
#             pos_weight = neg_count / max(pos_count, 1e-6)
#         model, history, metrics, train_probs, val_probs = _train_classifier(
#             x_train,
#             y_train,
#             x_val,
#             y_val if y_val.numel() > 0 else None,
#             epochs=total_epochs,
#             pos_weight=pos_weight,
#             model_factory=classifier_factory,
#             progress_desc=f"CLS method 1 ({tag})",
#         )
#         model.eval()
#         with torch.no_grad():
#             all_subset = feature_tensor[valid_indices].to(device)
#             all_probs = torch.sigmoid(model(all_subset)).cpu().squeeze(1)

#         mc_mean_all, mc_std_all = _mc_dropout_statistics(model, all_subset, num_passes=mc_passes)
#         mc_mean_np = mc_mean_all.numpy()
#         mc_std_np = mc_std_all.numpy()
#         def _slice_or_empty(arr: np.ndarray, idx: List[int]) -> List[float]:
#             if not idx:
#                 return []
#             return arr[idx].tolist()
#         result_entry: Dict[str, object] = {
#             "metrics": metrics,
#             "history": history,
#             "train_indices": [valid_indices[i] for i in sel_train],
#             "val_indices": [valid_indices[i] for i in sel_val],
#             "valid_indices": valid_indices,
#             "train_probs": train_probs.numpy().tolist(),
#             "val_probs": val_probs.numpy().tolist() if val_probs.numel() else [],
#             "all_probs": all_probs.numpy().tolist(),
#             "labels": [int(labels[idx].get("combined_label") or 0) for idx in valid_indices],
#             "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
#             "mc_dropout": {
#                 "num_passes": mc_passes,
#                 "all_mean": mc_mean_np.tolist(),
#                 "all_std": mc_std_np.tolist(),
#                 "train_mean": _slice_or_empty(mc_mean_np, sel_train),
#                 "train_std": _slice_or_empty(mc_std_np, sel_train),
#                 "val_mean": _slice_or_empty(mc_mean_np, sel_val),
#                 "val_std": _slice_or_empty(mc_std_np, sel_val),
#             },
#         }
#         results[tag] = result_entry
#     return results


# def _train_single_view_classifier(
#     features: torch.Tensor,
#     labels: List[Optional[int]],
#     train_indices: Sequence[int],
#     val_indices: Sequence[int],
#     device: torch.device,
#     epochs: int = 100,
#     *,
#     desc: str,
#     model_factory: Optional[Callable[[int], nn.Module]] = None,
#     mc_passes: int = 30,
# ) -> Tuple[Optional[nn.Module], Dict[str, object], List[int]]:
#     valid_idx = [idx for idx in range(features.size(0)) if labels[idx] is not None]
#     if not valid_idx:
#         return None, {}, []
#     label_tensor = torch.tensor([1 if labels[idx] else 0 for idx in valid_idx], dtype=torch.float32, device=device)
#     idx_map = {orig: new for new, orig in enumerate(valid_idx)}
#     train_sel = [idx_map[idx] for idx in train_indices if idx in idx_map]
#     val_sel = [idx_map[idx] for idx in val_indices if idx in idx_map]
#     if not train_sel:
#         return None, {}, valid_idx
#     x_valid = features[valid_idx].to(device)
#     x_train = x_valid[train_sel]
#     x_val = x_valid[val_sel] if val_sel else torch.empty(0, x_valid.size(1), device=device)
#     y_train = label_tensor[train_sel]
#     y_val = label_tensor[val_sel] if val_sel else torch.empty(0, device=device)
#     pos_weight = None
#     pos_count = float(y_train.sum().item())
#     neg_count = float(y_train.numel() - pos_count)
#     if pos_count > 0 and neg_count > 0:
#         pos_weight = neg_count / max(pos_count, 1e-6)
#     model, history, metrics, train_probs, val_probs = _train_classifier(
#         x_train,
#         y_train,
#         x_val,
#         y_val if y_val.numel() > 0 else None,
#         epochs=epochs,
#         pos_weight=pos_weight,
#         model_factory=model_factory,
#         progress_desc=desc,
#     )
#     model.eval()
#     with torch.no_grad():
#         all_probs = torch.sigmoid(model(x_valid)).cpu().squeeze(1)
#     mc_mean_all, mc_std_all = _mc_dropout_statistics(model, x_valid, num_passes=max(1, mc_passes))
#     mc_mean_np = mc_mean_all.numpy()
#     mc_std_np = mc_std_all.numpy()
#     def _slice(arr: np.ndarray, idx: List[int]) -> List[float]:
#         if not idx:
#             return []
#         return arr[idx].tolist()
#     info: Dict[str, object] = {
#         "metrics": metrics,
#         "history": history,
#         "train_indices": [valid_idx[i] for i in train_sel],
#         "val_indices": [valid_idx[i] for i in val_sel],
#         "valid_indices": valid_idx,
#         "train_probs": train_probs.numpy().tolist(),
#         "val_probs": val_probs.numpy().tolist() if val_probs.numel() else [],
#         "all_probs": all_probs.numpy().tolist(),
#         "labels": [int(labels[idx]) if labels[idx] is not None else 0 for idx in valid_idx],
#         "mc_dropout": {
#             "num_passes": int(max(1, mc_passes)),
#             "all_mean": mc_mean_np.tolist(),
#             "all_std": mc_std_np.tolist(),
#             "train_mean": _slice(mc_mean_np, train_sel),
#             "train_std": _slice(mc_std_np, train_sel),
#             "val_mean": _slice(mc_mean_np, val_sel),
#             "val_std": _slice(mc_std_np, val_sel),
#         },
#     }
#     return model, info, valid_idx


# def _train_dual_head_method(
#     cfg: AlignmentConfig,
#     run_logger: "_RunLogger",
#     anchor_data: Optional[Dict[str, object]],
#     target_data: Optional[Dict[str, object]],
#     device: torch.device,
#     epochs: int = 100,
#     *,
#     mlp_hidden_dims: Sequence[int],
#     mlp_dropout: float,
#     mc_passes: int = 5,
# ) -> Dict[str, object]:
#     results: Dict[str, object] = {}

#     classifier_factory = lambda in_dim: _build_mlp_classifier(in_dim, mlp_hidden_dims, mlp_dropout)

#     dataset_entries: List[Tuple[str, Dict[str, object]]] = []
#     if anchor_data:
#         dataset_entries.append(("anchor", anchor_data))
#     if target_data:
#         dataset_entries.append(("target", target_data))

#     for dataset_key, dataset_payload in _progress_iter(dataset_entries, "Method 2 datasets", leave=False, total=len(dataset_entries)):
#         labels_tensor = dataset_payload["labels"].tolist()
#         labels_int = [int(val) for val in labels_tensor]
#         model, info, _ = _train_single_view_classifier(
#             dataset_payload["features"],
#             labels_int,
#             dataset_payload["train_indices"],
#             dataset_payload["val_indices"],
#             device=device,
#             epochs=epochs,
#             desc=f"CLS method 2 ({dataset_key})",
#             model_factory=classifier_factory,
#             mc_passes=mc_passes,
#         )
#         if info:
#             info = dict(info)
#             info["metadata"] = dataset_payload["metadata"]
#             if model is not None:
#                 info["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
#             else:
#                 info["state_dict"] = None
#             results[dataset_key] = info
#         else:
#             run_logger.log(f"Dual-head classifier: {dataset_key} dataset unavailable or lacks PN-labelled samples.")

#     if not results:
#         run_logger.log("Dual-head classifier skipped: no classifiers were trained.")
#     return results


# def _run_unified_full_inference(
#     workspace: OverlapAlignmentWorkspace,
#     anchor_name: str,
#     target_name: str,
#     projector_a: Optional[nn.Module],
#     projector_b: Optional[nn.Module],
#     results: Dict[str, Dict[str, object]],
#     *,
#     device: torch.device,
#     mlp_hidden_dims: Sequence[int],
#     mlp_dropout: float,
#     batch_size: int = INFERENCE_BATCH_SIZE,
#     overlap_mask: Optional[Dict[str, object]] = None,
# ) -> None:
#     dataset_projectors = {
#         anchor_name: projector_a,
#         target_name: projector_b,
#     }
#     tag_entries = list(results.items())
#     for tag, info in _progress_iter(tag_entries, "Method 1 inference", leave=False, total=len(tag_entries)):
#         state_dict = info.get("state_dict")
#         if not isinstance(state_dict, dict):
#             continue
#         inference_payload: Dict[str, Dict[str, object]] = {}
#         mc_info = info.get("mc_dropout") or {}
#         mc_passes = int(mc_info.get("num_passes") or 0)
#         if mc_passes <= 0:
#             mc_passes = 1
#         dataset_items = list(dataset_projectors.items())
#         for dataset_name, projector in _progress_iter(dataset_items, f"{tag} datasets", leave=False, total=len(dataset_items)):
#             collected = _collect_full_dataset_projection(
#                 workspace,
#                 dataset_name,
#                 projector,
#                 device,
#                 batch_size=batch_size,
#                 overlap_mask=overlap_mask,
#             )
#             if collected is None:
#                 continue
#             bundle, projected, record_indices, record_rowcols = collected
#             if projected.numel() == 0:
#                 continue
#             zeros = torch.zeros_like(projected)
#             if dataset_name == anchor_name:
#                 proj_anchor = projected
#                 proj_target = zeros
#             else:
#                 proj_anchor = zeros
#                 proj_target = projected
#             simple_feats, strong_feats = _build_unified_features(proj_anchor, proj_target)
#             feature_tensor = strong_feats if tag == "strong" else simple_feats
#             if feature_tensor.numel() == 0:
#                 continue
#             model = _build_mlp_classifier(feature_tensor.size(1), mlp_hidden_dims, mlp_dropout)
#             model.load_state_dict(state_dict)
#             model = model.to(device)
#             mc_mean, mc_std = _mc_dropout_statistics(model, feature_tensor.to(device), num_passes=max(1, mc_passes))
#             model.cpu()
#             probs_np = mc_mean.numpy()
#             std_np = mc_std.numpy() if mc_std is not None else None
#             samples, metadata_entries = _build_samples_from_projection(
#                 bundle,
#                 record_indices,
#                 probs_np,
#                 std_np,
#                 dataset_name,
#                 mc_passes=mc_passes,
#                 progress_desc=f"Samples ({dataset_name})",
#             )
#             combined_metadata = []
#             for entry, rowcol in zip(metadata_entries, record_rowcols):
#                 if rowcol is not None:
#                     entry = dict(entry)
#                     entry.setdefault("row_col_mask", [int(rowcol[0]), int(rowcol[1])])
#                 combined_metadata.append(entry)
#             inference_payload[dataset_name] = {
#                 "samples": samples,
#                 "metadata": combined_metadata,
#                 "probs": probs_np.tolist(),
#                 "mc_mean": probs_np.tolist(),
#                 "mc_std": std_np.tolist() if std_np is not None else [],
#                 "mc_passes": mc_passes,
#                 "record_indices": record_indices,
#                 "row_cols": [tuple(rc) if rc is not None else None for rc in record_rowcols],
#             }
#         if inference_payload:
#             info["inference"] = inference_payload


# def _run_dual_head_full_inference(
#     workspace: OverlapAlignmentWorkspace,
#     dataset_name: str,
#     projector: Optional[nn.Module],
#     classifier_info: Dict[str, object],
#     *,
#     device: torch.device,
#     mlp_hidden_dims: Sequence[int],
#     mlp_dropout: float,
#     batch_size: int = INFERENCE_BATCH_SIZE,
#     overlap_mask: Optional[Dict[str, object]] = None,
# ) -> None:
#     state_dict = classifier_info.get("state_dict")
#     if not isinstance(state_dict, dict):
#         return
#     collected = _collect_full_dataset_projection(
#         workspace,
#         dataset_name,
#         projector,
#         device,
#         batch_size=batch_size,
#         overlap_mask=overlap_mask,
#     )
#     if collected is None:
#         return
#     bundle, projected, record_indices = collected
#     if projected.numel() == 0:
#         return
#     feature_tensor = projected
#     model = _build_mlp_classifier(feature_tensor.size(1), mlp_hidden_dims, mlp_dropout)
#     model.load_state_dict(state_dict)
#     model = model.to(device)
#     mc_info = classifier_info.get("mc_dropout") or {}
#     mc_passes = int(mc_info.get("num_passes") or 0)
#     if mc_passes <= 0:
#         mc_passes = 1
#     mc_mean, mc_std = _mc_dropout_statistics(model, feature_tensor.to(device), num_passes=max(1, mc_passes))
#     model.cpu()
#     probs_np = mc_mean.numpy()
#     std_np = mc_std.numpy() if mc_std is not None else None
#     samples, metadata_entries = _build_samples_from_projection(
#         bundle,
#         record_indices,
#         probs_np,
#         std_np,
#         dataset_name,
#         mc_passes=mc_passes,
#         progress_desc=f"Samples ({dataset_name})",
#     )
#     combined_metadata = []
#     for entry, rowcol in zip(metadata_entries, record_rowcols):
#         if rowcol is not None:
#             entry = dict(entry)
#             entry.setdefault("row_col_mask", [int(rowcol[0]), int(rowcol[1])])
#         combined_metadata.append(entry)
#     inference_payload = {
#         "samples": samples,
#         "metadata": combined_metadata,
#         "probs": probs_np.tolist(),
#         "mc_mean": probs_np.tolist(),
#         "mc_std": std_np.tolist() if std_np is not None else [],
#         "mc_passes": mc_passes,
#         "record_indices": record_indices,
#         "row_cols": [tuple(rc) if rc is not None else None for rc in record_rowcols],
#     }
#     classifier_info["inference"] = inference_payload



# def _prepare_inference_entry(
#     entry: Dict[str, object],
#     *,
#     dim_u: int,
#     dim_v: int,
#     role: str,
#     method: str = "simple",
# ) -> Optional[Dict[str, object]]:
#     features_t = entry.get("features")
#     if features_t is None:
#         return None
#     if isinstance(features_t, torch.Tensor):
#         base_tensor = features_t.detach().to(dtype=torch.float32)
#     elif isinstance(features_t, np.ndarray):
#         base_tensor = torch.from_numpy(features_t.astype(np.float32, copy=False))
#     else:
#         base_tensor = torch.as_tensor(features_t, dtype=torch.float32)
#     if base_tensor.ndim != 2 or base_tensor.numel() == 0:
#         return None

#     method_key = method.lower()
#     num_rows = base_tensor.shape[0]

#     def _pad_tensor(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
#         if target_dim <= 0:
#             return tensor.new_zeros(tensor.shape[0], 0)
#         current = tensor.shape[1]
#         if current == target_dim:
#             return tensor
#         if current > target_dim:
#             return tensor[:, :target_dim]
#         pad_width = target_dim - current
#         zeros = tensor.new_zeros(tensor.shape[0], pad_width)
#         return torch.cat([tensor, zeros], dim=1)

#     if role == "anchor":
#         anchor_vec = _pad_tensor(base_tensor, dim_u if dim_u > 0 else base_tensor.shape[1])
#         target_vec = base_tensor.new_zeros(num_rows, max(dim_v, 0))
#         anchor_present = True
#         target_present = False
#     else:
#         anchor_vec = base_tensor.new_zeros(num_rows, max(dim_u, 0))
#         target_vec = _pad_tensor(base_tensor, dim_v if dim_v > 0 else base_tensor.shape[1])
#         anchor_present = False
#         target_present = True

#     if method_key == "strong":
#         max_dim = max(dim_u, dim_v, base_tensor.shape[1], 1)
#         anchor_common = _pad_tensor(anchor_vec, max_dim)
#         target_common = _pad_tensor(target_vec, max_dim)
#         diff_vec = torch.abs(anchor_common - target_common)
#         prod_vec = anchor_common * target_common
#         norm_u = torch.linalg.norm(anchor_common, dim=1, keepdim=True)
#         norm_v = torch.linalg.norm(target_common, dim=1, keepdim=True)
#         dot = (anchor_common * target_common).sum(dim=1, keepdim=True)
#         cosine = dot / (norm_u * norm_v + 1e-8)
#         missing_flag_val = 0.0 if (anchor_present and target_present) else 1.0
#         missing_flag = torch.full_like(cosine, missing_flag_val)
#         phi_parts = [anchor_vec, target_vec, diff_vec, prod_vec, cosine.to(torch.float32), missing_flag.to(torch.float32)]
#         padded = torch.cat(phi_parts, dim=1)
#     else:
#         padded = torch.cat([anchor_vec, target_vec], dim=1)

#     row_cols = entry.get("row_cols_mask") or entry.get("row_cols") or []
#     metadata_list = entry.get("metadata") or []
#     labels_arr = entry.get("labels")
#     lookup: Dict[Tuple[int, int], int] = {}
#     coords: List[Tuple[int, int]] = []
#     kept_indices: List[int] = []
#     kept_labels: List[int] = []
#     kept_metadata: List[Dict[str, object]] = []
#     for idx, rowcol in enumerate(row_cols):
#         if rowcol is None:
#             continue
#         try:
#             r, c = int(rowcol[0]), int(rowcol[1])
#         except Exception:
#             continue
#         lookup[(r, c)] = len(kept_indices)
#         coords.append((r, c))
#         kept_indices.append(idx)
#         if labels_arr is not None and idx < len(labels_arr):
#             kept_labels.append(int(labels_arr[idx]))
#         else:
#             kept_labels.append(0)
#         if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
#             kept_metadata.append(metadata_list[idx])
#         else:
#             kept_metadata.append({})
#     if not kept_indices:
#         return None

#     index_tensor = torch.tensor(kept_indices, dtype=torch.long, device=padded.device)
#     embedding_tensor = padded.index_select(0, index_tensor).contiguous().detach()
#     labels_np = np.asarray(kept_labels, dtype=np.int16)
#     return {
#         "tuple": (embedding_tensor, lookup, coords),
#         "metadata": kept_metadata,
#         "labels": labels_np,
#         "coords": coords,
#     }

# def _no_fusion_run_inference(
#     dataset: Dict[str, object],
#     anchor_overlap: Optional[Dict[str, object]],
#     target_overlap: Optional[Dict[str, object]],
#     mlp: nn.Module,
#     overlap_mask: Optional[Dict[str, object]],
#     device: torch.device,
#     cfg: AlignmentConfig,
#     run_logger: "_RunLogger",
# ) -> Dict[str, Dict[str, object]]:
#     outputs: Dict[str, Dict[str, object]] = {}
#     if overlap_mask is None:
#         run_logger.log("[no_fusion] Overlap mask unavailable; skipping inference exports.")
#         return outputs
#     default_reference = _make_mask_reference(overlap_mask)
#     if default_reference is None:
#         run_logger.log("[no_fusion] Failed to build mask reference; skipping inference exports.")
#         return outputs
#     stack = _MaskStack(overlap_mask)

#     def _feature_dim(entry: Optional[Dict[str, object]]) -> Optional[int]:
#         if not entry:
#             return None
#         feats = entry.get("features")
#         if isinstance(feats, torch.Tensor):
#             return int(feats.shape[1])
#         if isinstance(feats, np.ndarray):
#             return int(feats.shape[1])
#         return None

#     dim_u = dataset.get("dim_u")
#     dim_v = dataset.get("dim_v")

#     anchor_dim = _feature_dim(anchor_overlap)
#     target_dim = _feature_dim(target_overlap)
#     dataset_features = dataset.get("features")
#     total_dim = int(dataset_features.shape[1]) if isinstance(dataset_features, np.ndarray) and dataset_features.ndim == 2 else None

#     if dim_u is None:
#         dim_u = anchor_dim
#     if dim_v is None:
#         dim_v = target_dim
#     if dim_u is None and dim_v is not None and total_dim is not None:
#         dim_u = max(total_dim - dim_v, 0)
#     if dim_v is None and dim_u is not None and total_dim is not None:
#         dim_v = max(total_dim - dim_u, 0)

#     if dim_u is None or dim_v is None:
#         run_logger.log("[no_fusion] Unable to infer feature dimensions; skipping inference exports.")
#         return outputs

#     # method_key = str(dataset.get("fusion_method") or "simple").lower()
#     method_key = str(dataset.get("fusion_method") or "simple").lower()
#     passes = max(1, int(getattr(cfg.cls_training, "mc_dropout_passes", 30)))
#     for role_name, entry, role_flag in (
#         ("anchor", anchor_overlap, "anchor"),
#         ("target", target_overlap, "target"),
#     ):
#         print("[info] Inference for :", role_name)
#         inference_entry = _prepare_inference_entry(
#             entry,
#             dim_u=dim_u,
#             dim_v=dim_v,
#             role=role_flag,
#             method=method_key,
#         ) if entry is not None else None
#         if inference_entry is None:
#             run_logger.log(f"[no_fusion] No inference embeddings available for dataset {role_name}; skipping.")
#             continue
#         embedding_array, lookup, coord_list = inference_entry["tuple"]
#         prediction = mc_predict_map_from_embeddings(
#             {"GLOBAL": (embedding_array, lookup, coord_list)},
#             mlp,
#             stack,
#             passes=passes,
#             device=str(device),
#             show_progress=True,
#         )
#         if isinstance(prediction, tuple):
#             mean_map, std_map = prediction
#             prediction_payload = {"GLOBAL": {"mean": mean_map, "std": std_map}}
#         else:
#             global_payload = prediction.get("GLOBAL") if isinstance(prediction, dict) else {}
#             mean_map = global_payload.get("mean") if isinstance(global_payload, dict) else None
#             std_map = global_payload.get("std") if isinstance(global_payload, dict) else None
#             prediction_payload = prediction
#         labels_arr = inference_entry["labels"]
#         coords = inference_entry["coords"]
#         metadata = inference_entry["metadata"]
#         pos_coords = [
#             (meta.get("region") or "GLOBAL", int(coord[0]), int(coord[1]))
#             for lbl, coord, meta in zip(labels_arr, coord_list, metadata)
#             if lbl > 0
#         ]
#         neg_coords = [
#             (meta.get("region") or "GLOBAL", int(coord[0]), int(coord[1]))
#             for lbl, coord, meta in zip(labels_arr, coord_list, metadata)
#             if lbl <= 0
#         ]
#         pos_map = group_coords(pos_coords, stack) if pos_coords else {}
#         neg_map = group_coords(neg_coords, stack) if neg_coords else {}
#         mean_values = [float(mean_map[r, c]) if mean_map is not None else float("nan") for r, c in coord_list]
#         std_values = [float(std_map[r, c]) if std_map is not None else float("nan") for r, c in coord_list]
#         outputs[role_name] = {
#             "prediction": prediction_payload,
#             "default_reference": default_reference,
#             "pos_map": pos_map,
#             "neg_map": neg_map,
#             "counts": {
#                 "pos": int(sum(len(group) for group in pos_map.values())),
#                 "neg": int(sum(len(group) for group in neg_map.values())),
#             },
#             "row_cols": coord_list,
#             "coords": coords,
#             "labels": labels_arr.tolist(),
#             "metadata": metadata,
#             "mean_values": mean_values,
#             "std_values": std_values,
#         }
#     return outputs

if __name__ == "__main__":
    main()
