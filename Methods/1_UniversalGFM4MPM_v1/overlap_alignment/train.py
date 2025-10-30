# Note: For training cls, pn_index_summary is critical to track positive/negative sample counts. by K.N. 30Oct2025

from __future__ import annotations

import argparse
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

from .config import AlignmentConfig, load_config
from .workspace import OverlapAlignmentPair, OverlapAlignmentWorkspace
from .datasets import auto_coord_error
from Common.overlap_debug_plot import save_overlap_debug_plot
from Common.cls.infer.infer_maps import write_prediction_outputs
from Common.cls.models.mlp_dropout import MLPDropout

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer (positive-only)")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    parser.add_argument("--dcca-eps", type=float, default=1e-5, help="Epsilon value for DCCA covariance regularization (default: 1e-5).")
    parser.add_argument("--singular-value-drop-ratio", type=float, default=0.01, help="Ratio threshold for dropping small singular values in DCCA (default: 0.01).")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of aligned pairs reserved for validation evaluation (set to 0 to disable).",
    )
    parser.add_argument(
        "--tcc-ratio",
        type=float,
        default=1.0,
        help="Fraction of canonical correlations to include when computing TCC (0 < ratio <= 1).",
    )
    parser.add_argument(
        "--dcca-mlp-layers",
        type=int,
        default=4,
        help="Number of linear layers used in each DCCA projection head MLP (default: 4).",
    )
    parser.add_argument(
        "--train-dcca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the DCCA projection heads (default: true). Use --no-train-dcca to disable.",
    )
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
        help="PN classifier training method to use (1 [Unified CLS] or 2 [Independent CLS]; default: 1).",
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
        "--read-dcca",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load existing DCCA projection head weights before training (default: false).",
    )
    parser.add_argument(
        "--dcca-weights-path",
        type=str,
        default=None,
        help="Optional path to a saved DCCA checkpoint used when --read-dcca is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if torch is None or nn is None or DataLoader is None:
        raise ImportError("Overlap alignment training requires PyTorch; install torch before running the trainer.")

    mlp_hidden_dims = tuple(int(h) for h in args.mlp_hidden_dims if int(h) > 0)
    if not mlp_hidden_dims:
        raise ValueError("At least one positive hidden dimension must be provided for classifier MLP heads.")
    mlp_dropout = float(args.mlp_dropout)
    if not (0.0 <= mlp_dropout < 1.0):
        raise ValueError("Classifier MLP dropout must be in the range [0.0, 1.0).")

    if args.objective is not None:
        cfg.alignment_objective = args.objective.lower()
    if args.aggregator is not None:
        cfg.aggregator = args.aggregator.lower()
    if args.use_positive_only:
        cfg.use_positive_only = True
    if args.use_positive_augmentation:
        cfg.use_positive_augmentation = True
    debug_mode = bool(args.debug)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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
            else:
                print(
                    f"[warn] No augmented embeddings retained for dataset {dataset_cfg.name}; "
                    "region filter or bundle contents may have excluded all entries."
                )

    #################################### Preprocess - Alignment two overlapping data - getting labels for cls ####################################
    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
    pn_label_maps: Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]] = {
        dataset_cfg.name: _load_pn_lookup(dataset_cfg.pn_split_path) for dataset_cfg in cfg.datasets
    }
    for dataset_cfg in cfg.datasets:
        lookup = pn_label_maps.get(dataset_cfg.name)
        if not lookup:
            print(f"[info] PN labels unavailable for dataset {dataset_cfg.name}")
            continue
        pos_count = len(lookup.get("pos", ()))
        neg_count = len(lookup.get("neg", ()))
        print(f"[info] PN label counts for {dataset_cfg.name}: pos={pos_count}, neg={neg_count}")

    (anchor_vecs, target_vecs, label_hist, debug_data, 
     augmentation_stats, pair_metadata, pn_index_summary,) = _build_aligned_pairs(
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
    print(f"[dev] label_histogram={dict(label_hist)}")



    pairs_by_region = defaultdict(lambda: Counter())
    for meta in pair_metadata:
        anchor_region = meta.get("anchor_region", "UNKNOWN")
        for label in meta["target_labels"]:
            pairs_by_region[anchor_region][label] += 1
    print("pairs grouped by anchor region:", dict(pairs_by_region))

    pn_index_sets: Dict[str, Dict[str, set[int]]] = {}
    for dataset, regions in pn_index_summary.items():
        pos_union: set[int] = set()
        neg_union: set[int] = set()
        print(f"  [{dataset}]")
        for region, payload in regions.items():
            pos_indices = [int(idx) for idx in payload.get("pos_original_indices", [])]
            neg_indices = [int(idx) for idx in payload.get("neg_original_indices", [])]
            pos_union.update(pos_indices)
            neg_union.update(neg_indices)
            print(f"  {region}: pos={payload['pos_count']} neg={payload['neg_count']} ")
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
            _persist_metrics(cfg, {"objective": cfg.alignment_objective.lower(), "max_coord_error": max_coord_error}, [], debug_data)
        return
    #################################### Preprocess - Alignment two overlapping data - getting labels for cls ####################################

    #################################### Training Overlap Alignment (DCCA) ####################################
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

    anchor_tensor = torch.stack(anchor_vecs)
    target_tensor = torch.stack(target_vecs)
    total_pairs = anchor_tensor.size(0)

    validation_split = float(args.validation_split)
    if not math.isfinite(validation_split):
        validation_split = 0.0
    validation_split = max(0.0, min(validation_split, 0.9))
    val_count = int(total_pairs * validation_split)
    minimum_train = 2
    if total_pairs - val_count < minimum_train and total_pairs >= minimum_train:
        val_count = max(0, total_pairs - minimum_train)
    if val_count < 2:
        val_count = 0

    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    indices = torch.randperm(total_pairs, generator=generator)
    val_indices = indices[:val_count] if val_count > 0 else torch.empty(0, dtype=torch.long)
    train_indices = indices[val_count:]
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

    train_batch_size = min(cfg.training.batch_size, len(train_dataset))

    objective = cfg.alignment_objective.lower()
    if objective not in {"dcca", "barlow"}:
        raise ValueError(f"Unsupported alignment objective: {objective}")

    drop_ratio = float(args.singular_value_drop_ratio)
    drop_ratio = min(drop_ratio, 1.0)

    tcc_ratio = float(args.tcc_ratio)
    if not math.isfinite(tcc_ratio) or tcc_ratio <= 0.0:
        tcc_ratio = 1.0
    tcc_ratio = min(tcc_ratio, 1.0)

    mlp_layers = int(args.dcca_mlp_layers)
    if mlp_layers < 1:
        mlp_layers = 1

    run_logger = _RunLogger(cfg)
    if read_dcca and weights_path is not None:
        run_logger.log(f"Reading DCCA weights from {weights_path}")
    else:
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

    train_success, projector_a, projector_b, epoch_history, final_proj_dim, failure_reason = _train_with_adaptive_projection(
        cfg=cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=train_batch_size,
        device=device,
        anchor_dim=train_anchor.size(1),
        target_dim=train_target.size(1),
        dcca_eps=args.dcca_eps,
        run_logger=run_logger,
        drop_ratio=drop_ratio,
        tcc_ratio=tcc_ratio,
        mlp_layers=mlp_layers,
        train_dcca=train_dcca,
        pretrained_state=pretrained_state,
        pretrained_summary=pretrained_summary,
        pretrained_path=weights_path,
    )
    #################################### Training Overlap Alignment (DCCA) END ####################################

    if plt is not None:
        if cfg.output_dir is not None:
            _debug_root = Path(cfg.output_dir)
        elif cfg.log_dir is not None:
            _debug_root = Path(cfg.log_dir)
        else:
            _debug_root = Path.cwd()
        debug_plot_dir = _debug_root / "debug_classifier_inspection"
        try:
            debug_plot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            debug_plot_dir = None
    else:
        debug_plot_dir = None

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

    #################################### Training Classifier after Overlap Alignment ####################################
    classifier_results: Dict[str, Dict[str, object]] = {}
    if args.train_cls:
        projector_a.eval()
        projector_b.eval()
        methods_to_run = sorted({int(args.train_cls_method)})
        if debug_mode:
            methods_to_run = [1, 2]
        projector_map = {
            anchor_name: projector_a,
            target_name: projector_b,
        }
        sample_sets: Dict[str, Dict[str, object]] = {}
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
            )
            if sample_set:
                sample_sets[dataset_name] = sample_set
        if not sample_sets:
            run_logger.log("PN classifier skipped: no PN-labelled samples available for classifier training.")
        else:
            method1_data, method2_data = _prepare_classifier_inputs(
                anchor_name=anchor_name,
                target_name=target_name,
                sample_sets=sample_sets,
                validation_fraction=validation_split,
                seed=int(cfg.seed),
            )

            def _debug_check_summarize(indices: Sequence[int], tag: str) -> None:
                if method1_data is None or not indices:
                    print(f"[unified {tag}] no samples")
                    return
                counts = Counter()
                by_dataset: Dict[str, Counter] = {}
                points_by_dataset: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
                metadata_list = method1_data["metadata"]
                for idx in indices:
                    if idx >= len(metadata_list):
                        continue
                    meta = metadata_list[idx]
                    dataset = str(meta.get("dataset"))
                    label = int(meta.get("label", 0))
                    region = meta.get("region")
                    counts[(dataset, label)] += 1
                    by_dataset.setdefault(dataset, Counter())[label] += 1
                    coord = meta.get("coord")
                    if coord is not None:
                        points_by_dataset.setdefault(dataset, {}).setdefault(label, []).append(coord)
                print(f"[unified {tag}] counts {dict(counts)}")
                print(f"[unified {tag}] per dataset { {ds: dict(cnt) for ds, cnt in by_dataset.items()} }")
                for ds, label_map in points_by_dataset.items():
                    _debug_plot_points(f"unified_{tag}_{ds}", label_map)

            def _debug_check_summarize_view(dataset_key: str) -> None:
                dataset_data = method2_data.get(dataset_key)
                if not dataset_data:
                    print(f"[method2 {dataset_key}] no samples")
                    return
                labels_tensor: torch.Tensor = dataset_data["labels"]
                metadata_list: List[Dict[str, object]] = dataset_data["metadata"]
                train_idx = dataset_data["train_indices"]
                val_idx = dataset_data["val_indices"]
                counts = Counter(int(labels_tensor[i].item()) for i in train_idx)
                print(f"[method2 {dataset_key} train] counts {dict(counts)}")
                val_counts = Counter(int(labels_tensor[i].item()) for i in val_idx)
                if val_counts:
                    print(f"[method2 {dataset_key} val] counts {dict(val_counts)}")
                train_points: Dict[int, List[Tuple[float, float]]] = {}
                val_points: Dict[int, List[Tuple[float, float]]] = {}
                for idx in train_idx:
                    if idx >= len(metadata_list):
                        continue
                    coord = metadata_list[idx].get("coord")
                    if coord is not None:
                        label = int(labels_tensor[idx].item())
                        train_points.setdefault(label, []).append(coord)
                for idx in val_idx:
                    if idx >= len(metadata_list):
                        continue
                    coord = metadata_list[idx].get("coord")
                    if coord is not None:
                        label = int(labels_tensor[idx].item())
                        val_points.setdefault(label, []).append(coord)
                if train_points:
                    _debug_plot_points(f"method2_{dataset_key}_train", train_points)
                if val_points:
                    _debug_plot_points(f"method2_{dataset_key}_val", val_points)

            for method in methods_to_run:
                run_logger.log(f"Training PN classifier method {method}")
                method_dir = _prepare_classifier_dir(cfg, method, anchor_name, target_name)
                if method_dir is None:
                    run_logger.log(f"Skipping classifier method {method}: unable to resolve output directory.")
                    continue
                metadata_payload: Dict[str, object] = {}
                if method == 1:
                    if method1_data is None or not method1_data["train_indices"]:
                        run_logger.log("Unified classifier skipped: no labelled samples available.")
                        continue
                    if debug_mode:
                        _debug_check_summarize(method1_data["train_indices"], "train")
                        _debug_check_summarize(method1_data["val_indices"], "val")
                    results = _train_unified_method(
                        cfg=cfg,
                        run_logger=run_logger,
                        projected_anchor=method1_data["anchor_features"],
                        projected_target=method1_data["target_features"],
                        labels=method1_data["labels"],
                        train_indices=method1_data["train_indices"],
                        val_indices=method1_data["val_indices"],
                        device=device,
                        mlp_hidden_dims=mlp_hidden_dims,
                        mlp_dropout=mlp_dropout,
                    )
                    metadata_payload["method1"] = method1_data
                elif method == 2:
                    anchor_data = method2_data.get("anchor")
                    target_data = method2_data.get("target")
                    if not anchor_data and not target_data:
                        run_logger.log("Dual-head classifier skipped: no labelled samples available for either dataset.")
                        continue
                    if debug_mode:
                        if anchor_data:
                            _debug_check_summarize_view("anchor")
                        if target_data:
                            _debug_check_summarize_view("target")
                    results = _train_dual_head_method(
                        cfg=cfg,
                        run_logger=run_logger,
                        anchor_data=anchor_data,
                        target_data=target_data,
                        device=device,
                        epochs=cfg.training.epochs,
                        mlp_hidden_dims=mlp_hidden_dims,
                        mlp_dropout=mlp_dropout,
                    )
                    metadata_payload["anchor"] = anchor_data
                    metadata_payload["target"] = target_data
                else:
                    run_logger.log(f"Unknown classifier method {method}; skipping.")
                    continue

                if results:
                    summary_entry = _save_classifier_outputs(
                        cfg=cfg,
                        run_logger=run_logger,
                        method_id=method,
                        anchor_name=anchor_name,
                        target_name=target_name,
                        method_dir=method_dir,
                        results=results,
                        classifier_metadata=metadata_payload,
                        dataset_meta_map=dataset_meta_map,
                    )
                    if summary_entry:
                        classifier_results[f"method_{method}"] = summary_entry

    if not train_success or projector_a is None or projector_b is None:
        raise RuntimeError(f"DCCA training failed: {failure_reason or 'unknown error'}")

    last_history = epoch_history[-1] if epoch_history else {}

    summary = {
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
        "epochs": cfg.training.epochs,
        "batch_size": train_batch_size,
        "lr": cfg.training.lr,
        "max_coord_error": max_coord_error,
        "final_loss": last_history.get("train_eval_loss", last_history.get("loss")),
        "final_mean_correlation": last_history.get("train_eval_mean_correlation", last_history.get("mean_correlation")),
        "final_train_tcc": last_history.get("train_eval_tcc"),
        "final_train_tcc_mean": last_history.get("train_eval_tcc_mean"),
        "final_train_tcc_k": int(last_history["train_eval_k"]) if last_history.get("train_eval_k") is not None else None,
        "final_val_loss": last_history.get("val_eval_loss"),
        "final_val_mean_correlation": last_history.get("val_eval_mean_correlation"),
        "final_val_tcc": last_history.get("val_eval_tcc"),
        "final_val_tcc_mean": last_history.get("val_eval_tcc_mean"),
        "final_val_tcc_k": int(last_history["val_eval_k"]) if last_history.get("val_eval_k") is not None else None,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset) if validation_dataset is not None else 0,
        "validation_fraction": actual_validation_fraction,
        "tcc_ratio": tcc_ratio,
        "singular_value_drop_ratio": drop_ratio,
    }
    if classifier_results:
        summary["classifier_results"] = classifier_results
    summary["selected_pairs_augmented"] = int(augmentation_stats.get("selected_pairs_augmented", 0))
    summary["anchor_augmented_pairs"] = int(augmentation_stats.get("anchor_augmented_pairs", 0))
    summary["target_augmented_pairs"] = int(augmentation_stats.get("target_augmented_pairs", 0))
    summary["pn_index_summary"] = pn_index_summary
    summary["pn_index_sets"] = {
        dataset: {
            "pos": sorted(posneg.get("pos", set())),
            "neg": sorted(posneg.get("neg", set())),
        }
        for dataset, posneg in pn_index_sets.items()
    }
    # print("[info] training summary:", summary)

    _persist_state(cfg, projector_a, projector_b, summary)
    _persist_metrics(cfg, summary, epoch_history, debug_data if debug_mode else None)
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
            dcca_eps=args.dcca_eps,
            device=device,
            sample_seed=int(cfg.seed),
        )



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
    else:
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


def _train_with_adaptive_projection(
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
            else:
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
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        epoch_history: List[Dict[str, object]] = []
        attempt_success = True

        for epoch in range(cfg.training.epochs):
            running_loss = 0.0
            running_corr = 0.0
            batches = 0
            iterator = loader
            if tqdm is not None:
                iterator = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.training.epochs} (proj={current_dim})", leave=False)

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

#                if filter_info.get("dropped_count", 0):
#                    run_logger.log(
#                        "Attempt {attempt} epoch {epoch} batch {batch}: dropped {count} singular values "
#                        "indices={indices};".format(
#                            attempt=attempt,
#                            epoch=epoch + 1,
#                            batch=batch_index + 1,
#                            count=filter_info["dropped_count"],
#                            indices=filter_info.get("dropped_indices"),
#                        )
#                    )

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
        else:
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


def _project_in_batches(tensor: torch.Tensor, model: nn.Module, device: torch.device, batch_size: int) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty_like(tensor)
    outputs: List[torch.Tensor] = []
    batch_size = max(1, batch_size)
    with torch.no_grad():
        for start in range(0, tensor.size(0), batch_size):
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
        else:
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
            else:
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
        else:
            try:
                x, y, prob = entry[:3]
                label = entry[3] if len(entry) >= 4 else 1
            except Exception:
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

                    pos_pixels: set[Tuple[int, int]] = set()
                    neg_pixels: set[Tuple[int, int]] = set()
                    for r, c, lbl in zip(rows.tolist(), cols.tolist(), labels_inside.tolist()):
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


def _plot_probability_map(
    dataset: str,
    samples: Sequence[object],
    out_dir: Path,
    run_logger: "_RunLogger",
    dataset_meta_map: Dict[str, Dict[str, object]],
) -> None:
    payload = _build_prediction_payload_from_samples(dataset, samples, dataset_meta_map, run_logger)
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


def _save_classifier_outputs(
    cfg: AlignmentConfig,
    run_logger: "_RunLogger",
    method_id: int,
    anchor_name: str,
    target_name: str,
    method_dir: Path,
    results: Dict[str, object],
    pair_metadata: Sequence[Dict[str, object]],
    dataset_meta_map: Dict[str, Dict[str, object]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> Dict[str, object]:
    summary_payload: Dict[str, object] = {}
    method_dir = method_dir.resolve()
    method_dir.mkdir(parents=True, exist_ok=True)

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

    def _plot_for_dataset(dataset: str, samples: List[Dict[str, object]], out_dir: Path) -> None:
        if not samples:
            run_logger.log(f"No samples available for dataset {dataset}; skipping probability export.")
            return
        _plot_probability_map(dataset, samples, out_dir, run_logger, dataset_meta_map)

    if method_id == 1:
        method_1_dir = Path(method_dir) / 'method_1_classifiers'
       
        plots_dir = method_1_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_data: Dict[str, object] = {}
        history_data: Dict[str, object] = {}
        for tag, info in results.items():
            info = dict(info)
            state_dict = info.pop('state_dict', None)
            metrics = info.get('metrics', {})
            history = info.get('history', [])
            metrics_data[tag] = metrics
            history_data[tag] = history
            _plot_training_curves(history, plots_dir / f'training_curve_{tag}.png', f'Method 1 ({tag}) Training')
            _export_state_dict(state_dict, method_1_dir / f'classifier_{tag}.pt')
        strong_info = results.get('strong') or next(iter(results.values()), None)
        if strong_info:
            valid_indices = strong_info.get('valid_indices', [])
            all_probs = strong_info.get('all_probs', [])
            train_set = set(strong_info.get('train_indices', []))
            val_set = set(strong_info.get('val_indices', []))
            anchor_samples: List[Dict[str, object]] = []
            target_samples: List[Dict[str, object]] = []
            for idx, prob in zip(valid_indices, all_probs):
                if idx >= len(pair_metadata):
                    continue
                meta = pair_metadata[idx]
                split = 'val' if idx in val_set else ('train' if idx in train_set else 'unlabeled')
                anchor_coord = meta.get('anchor_coord')
                if anchor_coord is not None and isinstance(anchor_coord, (tuple, list)):
                    anchor_label = _resolve_binary_label(meta.get('anchor_label'), default=0)
                    anchor_samples.append(
                        {
                            "x": float(anchor_coord[0]),
                            "y": float(anchor_coord[1]),
                            "prob": float(prob),
                            "label": anchor_label,
                            "split": split,
                        }
                    )
                coords_list = meta.get('target_coords') or []
                labels_list = meta.get('target_labels') or []
                added = False
                if isinstance(coords_list, (list, tuple)) and isinstance(labels_list, (list, tuple)):
                    for coord_val, label_val in zip(coords_list, labels_list):
                        if coord_val is None or not isinstance(coord_val, (tuple, list)):
                            continue
                        target_samples.append(
                            {
                                "x": float(coord_val[0]),
                                "y": float(coord_val[1]),
                                "prob": float(prob),
                                "label": _resolve_binary_label(label_val, default=0),
                                "split": split,
                            }
                        )
                        added = True
                if not added:
                    weighted_coord = meta.get('target_weighted_coord')
                    if weighted_coord is None and coords_list:
                        weighted_coord = coords_list[0]
                    if weighted_coord is not None and isinstance(weighted_coord, (tuple, list)):
                        fallback_label = labels_list[0] if labels_list else meta.get('combined_label')
                        target_samples.append(
                            {
                                "x": float(weighted_coord[0]),
                                "y": float(weighted_coord[1]),
                                "prob": float(prob),
                                "label": _resolve_binary_label(fallback_label, default=0),
                                "split": split,
                            }
                        )
            _plot_for_dataset(anchor_name, anchor_samples, plots_dir / f"probability_{anchor_name}")
            _plot_for_dataset(target_name, target_samples, plots_dir / f"probability_{target_name}")
        _write_json(method_1_dir / 'training_metrics.json', metrics_data)
        _write_json(method_1_dir / 'training_history.json', history_data)
        log_lines = [f'Method {method_id} classifier summary']
        for key, metric in metrics_data.items():
            log_lines.append(f'[{key}] {json.dumps(metric)}')
        (method_1_dir / 'training.log').write_text('\n'.join(log_lines), encoding='utf-8')
        summary_payload['output_dir'] = str(method_1_dir)
        summary_payload['metrics'] = metrics_data
        return summary_payload

    if method_id == 2:
        anchor_info = dict(results.get('anchor') or {})
        target_info = dict(results.get('target') or {})
        fusion_info = dict(results.get('fusion') or {})
        summary_payload['anchor'] = {}
        summary_payload['target'] = {}
        overview: Dict[str, object] = {'train_size': len(train_indices), 'val_size': len(val_indices)}

        if anchor_info:
            # anchor_dir = _prepare_classifier_dir(cfg, method_id, anchor_name) or _fallback_subdir(method_dir, anchor_name, 'anchor')
            anchor_dir = Path(method_dir) / ('method_2_' + anchor_name)

            anchor_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = anchor_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            anchor_state = anchor_info.pop('state_dict', None)
            _export_state_dict(anchor_state, anchor_dir / 'classifier.pt')
            metrics = anchor_info.get('metrics', {})
            history = anchor_info.get('history', [])
            _write_json(anchor_dir / 'training_metrics.json', metrics)
            _write_json(anchor_dir / 'training_history.json', history)
            _plot_training_curves(history, plots_dir / 'training_curve_anchor.png', f'Method 2 ({anchor_name}) Training')
            anchor_samples: List[Dict[str, object]] = []
            for idx, prob in zip(anchor_info.get('valid_indices', []), anchor_info.get('all_probs', [])):
                if idx >= len(pair_metadata):
                    continue
                meta = pair_metadata[idx]
                coord = meta.get('anchor_coord')
                if coord is None:
                    continue
                split = 'val' if idx in anchor_info.get('val_indices', []) else ('train' if idx in anchor_info.get('train_indices', []) else 'unlabeled')
                anchor_label = _resolve_binary_label(meta.get('anchor_label'), default=0)
                anchor_samples.append(
                    {
                        "x": float(coord[0]),
                        "y": float(coord[1]),
                        "prob": float(prob),
                        "label": anchor_label,
                        "split": split,
                    }
                )
            _plot_for_dataset(anchor_name, anchor_samples, plots_dir / f"probability_{anchor_name}")
            (anchor_dir / 'training.log').write_text('\n'.join([f'Method {method_id} anchor summary', json.dumps(metrics)]), encoding='utf-8')
            summary_payload['anchor']['output_dir'] = str(anchor_dir)
            summary_payload['anchor']['metrics'] = metrics
            overview['anchor'] = metrics

        if target_info:
            # target_dir = _prepare_classifier_dir(cfg, method_id, target_name) or _fallback_subdir(method_dir, target_name, 'target')
            target_dir = Path(method_dir) / ('method_2_' + target_name)

            target_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = target_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            target_state = target_info.pop('state_dict', None)
            _export_state_dict(target_state, target_dir / 'classifier.pt')
            metrics = target_info.get('metrics', {})
            history = target_info.get('history', [])
            _write_json(target_dir / 'training_metrics.json', metrics)
            _write_json(target_dir / 'training_history.json', history)
            _plot_training_curves(history, plots_dir / 'training_curve_target.png', f'Method 2 ({target_name}) Training')
            target_samples: List[Dict[str, object]] = []
            for idx, prob in zip(target_info.get('valid_indices', []), target_info.get('all_probs', [])):
                if idx >= len(pair_metadata):
                    continue
                meta = pair_metadata[idx]
                coords_list = meta.get('target_coords') or []
                labels_list = meta.get('target_labels') or []
                split = 'val' if idx in target_info.get('val_indices', []) else ('train' if idx in target_info.get('train_indices', []) else 'unlabeled')
                added = False
                if isinstance(coords_list, (list, tuple)) and isinstance(labels_list, (list, tuple)):
                    for coord_val, label_val in zip(coords_list, labels_list):
                        if coord_val is None or not isinstance(coord_val, (tuple, list)):
                            continue
                        target_samples.append(
                            {
                                "x": float(coord_val[0]),
                                "y": float(coord_val[1]),
                                "prob": float(prob),
                                "label": _resolve_binary_label(label_val, default=0),
                                "split": split,
                            }
                        )
                        added = True
                if not added:
                    weighted_coord = meta.get('target_weighted_coord')
                    if weighted_coord is None and coords_list:
                        weighted_coord = coords_list[0]
                    if weighted_coord is None or not isinstance(weighted_coord, (tuple, list)):
                        continue
                    fallback_label = labels_list[0] if labels_list else meta.get('combined_label')
                    target_samples.append(
                        {
                            "x": float(weighted_coord[0]),
                            "y": float(weighted_coord[1]),
                            "prob": float(prob),
                            "label": _resolve_binary_label(fallback_label, default=0),
                            "split": split,
                        }
                    )
            _plot_for_dataset(target_name, target_samples, plots_dir / f"probability_{target_name}")
            (target_dir / 'training.log').write_text('\n'.join([f'Method {method_id} target summary', json.dumps(metrics)]), encoding='utf-8')
            summary_payload['target']['output_dir'] = str(target_dir)
            summary_payload['target']['metrics'] = metrics
            overview['target'] = metrics

        # fusion_dir = _prepare_classifier_dir(cfg, method_id, anchor_name, target_name, 'fusion') or _fallback_subdir(method_dir, anchor_name, target_name, 'fusion')
        fusion_dir = Path(method_dir) / ('method_2_' + 'fusion')

        fusion_dir.mkdir(parents=True, exist_ok=True)
        fusion_metrics = fusion_info.get('metrics', {})
        fusion_params = fusion_info.get('fusion_params')
        _write_json(fusion_dir / 'training_metrics.json', fusion_metrics)
        if fusion_params:
            try:
                torch.save({'weights': list(fusion_params)}, fusion_dir / 'fusion_params.pt')
            except Exception as exc:
                run_logger.log(f'Failed to save fusion parameters: {exc}')
        (fusion_dir / 'training.log').write_text('\n'.join([f'Method {method_id} fusion summary', json.dumps(fusion_metrics)]), encoding='utf-8')
        summary_payload.setdefault('fusion', {})['output_dir'] = str(fusion_dir)
        summary_payload['fusion']['metrics'] = fusion_metrics
        summary_payload['fusion']['train_pairs'] = fusion_info.get('train_pairs', [])
        summary_payload['fusion']['val_pairs'] = fusion_info.get('val_pairs', [])
        overview['fusion'] = fusion_metrics
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
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
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
        model = model_factory(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pos_weight is not None and pos_weight > 0:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    history: List[Dict[str, object]] = []
    if tqdm is not None:
        epoch_iter: Iterable[int] = tqdm(range(1, epochs + 1), desc=progress_desc, leave=False)
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
) -> Optional[Dict[str, object]]:
    bundle = workspace.datasets.get(dataset_name)
    if bundle is None:
        run_logger.log(f"[cls] dataset {dataset_name} not found in workspace; skipping classifier samples.")
        return None
    if pn_lookup is None or (not pn_lookup.get("pos") and not pn_lookup.get("neg")):
        run_logger.log(f"[cls] PN lookup unavailable for dataset {dataset_name}; skipping classifier samples.")
        return None
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
        matched_records.append(record)
        labels.append(label_int)
        coord_norm = _normalise_coord(record.coord)
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


def _prepare_classifier_inputs(
    anchor_name: str,
    target_name: str,
    sample_sets: Dict[str, Dict[str, object]],
    validation_fraction: float,
    seed: int,
) -> Tuple[Optional[Dict[str, object]], Dict[str, Dict[str, object]]]:
    anchor_set = sample_sets.get(anchor_name)
    target_set = sample_sets.get(target_name)
    method1_data: Optional[Dict[str, object]] = None
    if anchor_set or target_set:
        anchor_features_list: List[torch.Tensor] = []
        target_features_list: List[torch.Tensor] = []
        labels_list: List[int] = []
        metadata_list: List[Dict[str, object]] = []
        if anchor_set:
            features = anchor_set["features"]
            zeros_target = torch.zeros_like(features)
            anchor_features_list.append(features)
            target_features_list.append(zeros_target)
            for idx, meta in enumerate(anchor_set["metadata"]):
                label_int = int(anchor_set["labels"][idx].item())
                labels_list.append(label_int)
                meta_entry = dict(meta)
                meta_entry.setdefault("anchor_coord", meta_entry.get("coord"))
                meta_entry.setdefault("anchor_label", label_int)
                meta_entry.setdefault("anchor_region", meta_entry.get("region"))
                meta_entry.setdefault("target_coords", [])
                meta_entry.setdefault("target_labels", [])
                meta_entry.setdefault("target_regions", [])
                metadata_list.append(meta_entry)
        if target_set:
            features = target_set["features"]
            zeros_anchor = torch.zeros_like(features)
            anchor_features_list.append(zeros_anchor)
            target_features_list.append(features)
            for idx, meta in enumerate(target_set["metadata"]):
                label_int = int(target_set["labels"][idx].item())
                labels_list.append(label_int)
                coord_val = meta.get("coord")
                meta_entry = {
                    **meta,
                    "anchor_coord": None,
                    "anchor_label": None,
                    "anchor_region": None,
                    "target_coords": [coord_val] if coord_val is not None else [],
                    "target_labels": [label_int],
                    "target_regions": [meta.get("region")],
                }
                metadata_list.append(meta_entry)
        if labels_list:
            anchor_tensor = torch.cat(anchor_features_list, dim=0) if anchor_features_list else torch.empty(0)
            target_tensor = torch.cat(target_features_list, dim=0) if target_features_list else torch.empty(0)
            labels_dicts = [{"combined_label": int(label)} for label in labels_list]
            train_indices, val_indices = _split_classifier_indices(len(labels_list), validation_fraction, seed)
            method1_data = {
                "anchor_features": anchor_tensor,
                "target_features": target_tensor,
                "labels": labels_dicts,
                "metadata": metadata_list,
                "train_indices": train_indices,
                "val_indices": val_indices,
            }
    method2_data: Dict[str, Dict[str, object]] = {}
    if anchor_set:
        train_idx, val_idx = _split_classifier_indices(len(anchor_set["labels"]), validation_fraction, seed + 101)
        method2_data["anchor"] = {
            **anchor_set,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }
    if target_set:
        train_idx, val_idx = _split_classifier_indices(len(target_set["labels"]), validation_fraction, seed + 202)
        method2_data["target"] = {
            **target_set,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }
    return method1_data, method2_data


def _train_unified_method(
    cfg: AlignmentConfig,
    run_logger: "_RunLogger",
    projected_anchor: torch.Tensor,
    projected_target: torch.Tensor,
    labels: List[Dict[str, Optional[int]]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    device: torch.device,
    total_epochs: int = 100,
    *,
    mlp_hidden_dims: Sequence[int],
    mlp_dropout: float,
) -> Dict[str, Dict[str, object]]:
    simple_features, strong_features = _build_unified_features(projected_anchor, projected_target)
    valid_indices = [idx for idx, entry in enumerate(labels) if entry.get("combined_label") is not None]
    if not valid_indices:
        run_logger.log("Unified classifier skipped: no labeled pairs available.")
        return {}
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
    label_tensor = torch.tensor(
        [1 if labels[idx].get("combined_label") else 0 for idx in valid_indices],
        dtype=torch.float32,
    )
    sel_train = [index_map[idx] for idx in train_indices if idx in index_map]
    sel_val = [index_map[idx] for idx in val_indices if idx in index_map]
    if not sel_train:
        run_logger.log("Unified classifier skipped: no labeled training pairs available.")
        return {}
    def _select(source: torch.Tensor, idx_list: List[int]) -> torch.Tensor:
        if not idx_list:
            return torch.empty(0, source.size(1), device=device)
        subset = source[valid_indices]
        return subset[idx_list].to(device)
    def _select_labels(idx_list: List[int]) -> torch.Tensor:
        if not idx_list:
            return torch.empty(0, device=device)
        return label_tensor[idx_list].to(device)
    results: Dict[str, Dict[str, float]] = {}
    classifier_factory = lambda in_dim: _build_mlp_classifier(in_dim, mlp_hidden_dims, mlp_dropout)
    for tag, feature_tensor in (("simple", simple_features), ("strong", strong_features)):
        x_train = _select(feature_tensor, sel_train)
        x_val = _select(feature_tensor, sel_val)
        y_train = _select_labels(sel_train)
        y_val = _select_labels(sel_val)
        if x_train.size(0) < 2:
            run_logger.log(f"Unified classifier ({tag}) skipped: insufficient training samples.")
            continue
        pos_weight = None
        pos_count = float(y_train.sum().item())
        neg_count = float(y_train.numel() - pos_count)

        if pos_count > 0 and neg_count > 0:
            pos_weight = neg_count / max(pos_count, 1e-6)
        model, history, metrics, train_probs, val_probs = _train_classifier(
            x_train,
            y_train,
            x_val,
            y_val if y_val.numel() > 0 else None,
            epochs=total_epochs,
            pos_weight=pos_weight,
            model_factory=classifier_factory,
            progress_desc=f"CLS method 1 ({tag})",
        )
        model.eval()
        with torch.no_grad():
            all_subset = feature_tensor[valid_indices].to(device)
            all_probs = torch.sigmoid(model(all_subset)).cpu().squeeze(1)
        result_entry: Dict[str, object] = {
            "metrics": metrics,
            "history": history,
            "train_indices": [valid_indices[i] for i in sel_train],
            "val_indices": [valid_indices[i] for i in sel_val],
            "valid_indices": valid_indices,
            "train_probs": train_probs.numpy().tolist(),
            "val_probs": val_probs.numpy().tolist() if val_probs.numel() else [],
            "all_probs": all_probs.numpy().tolist(),
            "labels": [int(labels[idx].get("combined_label") or 0) for idx in valid_indices],
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
        results[tag] = result_entry
    return results


def _train_single_view_classifier(
    features: torch.Tensor,
    labels: List[Optional[int]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    device: torch.device,
    epochs: int = 100,
    *,
    desc: str,
    model_factory: Optional[Callable[[int], nn.Module]] = None,
) -> Tuple[Optional[nn.Module], Dict[str, object], List[int]]:
    valid_idx = [idx for idx in range(features.size(0)) if labels[idx] is not None]
    if not valid_idx:
        return None, {}, []
    label_tensor = torch.tensor([1 if labels[idx] else 0 for idx in valid_idx], dtype=torch.float32, device=device)
    idx_map = {orig: new for new, orig in enumerate(valid_idx)}
    train_sel = [idx_map[idx] for idx in train_indices if idx in idx_map]
    val_sel = [idx_map[idx] for idx in val_indices if idx in idx_map]
    if not train_sel:
        return None, {}, valid_idx
    x_valid = features[valid_idx].to(device)
    x_train = x_valid[train_sel]
    x_val = x_valid[val_sel] if val_sel else torch.empty(0, x_valid.size(1), device=device)
    y_train = label_tensor[train_sel]
    y_val = label_tensor[val_sel] if val_sel else torch.empty(0, device=device)
    pos_weight = None
    pos_count = float(y_train.sum().item())
    neg_count = float(y_train.numel() - pos_count)
    if pos_count > 0 and neg_count > 0:
        pos_weight = neg_count / max(pos_count, 1e-6)
    model, history, metrics, train_probs, val_probs = _train_classifier(
        x_train,
        y_train,
        x_val,
        y_val if y_val.numel() > 0 else None,
        epochs=epochs,
        pos_weight=pos_weight,
        model_factory=model_factory,
        progress_desc=desc,
    )
    model.eval()
    with torch.no_grad():
        all_probs = torch.sigmoid(model(x_valid)).cpu().squeeze(1)
    info: Dict[str, object] = {
        "metrics": metrics,
        "history": history,
        "train_indices": [valid_idx[i] for i in train_sel],
        "val_indices": [valid_idx[i] for i in val_sel],
        "valid_indices": valid_idx,
        "train_probs": train_probs.numpy().tolist(),
        "val_probs": val_probs.numpy().tolist() if val_probs.numel() else [],
        "all_probs": all_probs.numpy().tolist(),
        "labels": [int(labels[idx]) if labels[idx] is not None else 0 for idx in valid_idx],
    }
    return model, info, valid_idx


def _train_dual_head_method(
    cfg: AlignmentConfig,
    run_logger: "_RunLogger",
    anchor_data: Optional[Dict[str, object]],
    target_data: Optional[Dict[str, object]],
    device: torch.device,
    epochs: int = 100,
    *,
    mlp_hidden_dims: Sequence[int],
    mlp_dropout: float,
) -> Dict[str, object]:
    results: Dict[str, object] = {}

    classifier_factory = lambda in_dim: _build_mlp_classifier(in_dim, mlp_hidden_dims, mlp_dropout)

    if anchor_data:
        anchor_labels = [int(val) for val in anchor_data["labels"].tolist()]
        anchor_model, anchor_info, _ = _train_single_view_classifier(
            anchor_data["features"],
            anchor_labels,
            anchor_data["train_indices"],
            anchor_data["val_indices"],
            device=device,
            epochs=epochs,
            desc="CLS method 2 (anchor)",
            model_factory=classifier_factory,
        )
        if anchor_info:
            anchor_info = dict(anchor_info)
            anchor_info["metadata"] = anchor_data["metadata"]
            anchor_info["state_dict"] = {k: v.detach().cpu() for k, v in anchor_model.state_dict().items()} if anchor_model is not None else None
            results["anchor"] = anchor_info
    else:
        run_logger.log("Dual-head classifier: anchor dataset unavailable or lacks PN-labelled samples.")

    if target_data:
        target_labels = [int(val) for val in target_data["labels"].tolist()]
        target_model, target_info, _ = _train_single_view_classifier(
            target_data["features"],
            target_labels,
            target_data["train_indices"],
            target_data["val_indices"],
            device=device,
            epochs=epochs,
            desc="CLS method 2 (target)",
            model_factory=classifier_factory,
        )
        if target_info:
            target_info = dict(target_info)
            target_info["metadata"] = target_data["metadata"]
            target_info["state_dict"] = {k: v.detach().cpu() for k, v in target_model.state_dict().items()} if target_model is not None else None
            results["target"] = target_info
    else:
        run_logger.log("Dual-head classifier: target dataset unavailable or lacks PN-labelled samples.")

    if not results:
        run_logger.log("Dual-head classifier skipped: no classifiers were trained.")
    return results

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
                else:
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

def _build_aligned_pairs(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    pn_label_maps: Optional[Dict[str, Optional[Dict[str, set[Tuple[str, int, int]]]]]] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
    Dict[str, int],
    List[Dict[str, object]],
    Dict[str, Dict[str, Dict[str, object]]],
]:
    anchor_augment_map = anchor_augment_map or {}
    target_augment_map = target_augment_map or {}
    pn_index_tracker: Dict[str, Dict[str, Dict[str, set[int]]]] = {}

    pn_label_maps = pn_label_maps or {}

    def _register_index(record, dataset_key: str) -> None:
        if not dataset_key:
            return
        lookup = pn_label_maps.get(dataset_key)
        if not lookup:
            return
        region_val = getattr(record, "region", None)
        row_col_val = getattr(record, "row_col", None)
        if region_val is None or row_col_val is None:
            return
        try:
            row_val = int(row_col_val[0])
            col_val = int(row_col_val[1])
        except Exception:
            return
        region_key = str(region_val).upper()
        key = (region_key, row_val, col_val)
        label_bucket: Optional[str] = None
        pos_lookup = lookup.get("pos")
        if pos_lookup and key in pos_lookup:
            label_bucket = "pos"
        else:
            neg_lookup = lookup.get("neg")
            if neg_lookup and key in neg_lookup:
                label_bucket = "neg"
        if label_bucket is None:
            return
        idx_val = getattr(record, "index", None)
        if idx_val is None:
            return
        try:
            idx_int = int(idx_val)
        except Exception:
            return
        dataset_bucket = pn_index_tracker.setdefault(dataset_key, {})
        region_bucket = dataset_bucket.setdefault(region_key, {"pos": set(), "neg": set()})
        region_bucket[label_bucket].add(idx_int)

    def _finalise_pn_index_summary() -> Dict[str, Dict[str, Dict[str, object]]]:
        summary: Dict[str, Dict[str, Dict[str, object]]] = {}
        for dataset_name, regions in pn_index_tracker.items():
            region_summary: Dict[str, Dict[str, object]] = {}
            for region_name, label_sets in regions.items():
                pos_sorted = sorted(label_sets.get("pos", ()))
                neg_sorted = sorted(label_sets.get("neg", ()))
                region_summary[region_name] = {
                    "pos_count": len(pos_sorted),
                    "neg_count": len(neg_sorted),
                    "pos_original_indices": pos_sorted,
                    "neg_original_indices": neg_sorted,
                    "pos_reindexed_pairs": [
                        {"reindexed": re_idx, "original": original_idx}
                        for re_idx, original_idx in enumerate(pos_sorted)
                    ],
                    "neg_reindexed_pairs": [
                        {"reindexed": re_idx, "original": original_idx}
                        for re_idx, original_idx in enumerate(neg_sorted)
                    ],
                }
            summary[dataset_name] = region_summary
        return summary

    grouped: Dict[str, Dict[str, object]] = {}
    anchor_aug_added = 0
    target_aug_added = 0
    label_hist: Counter = Counter()
    debug_data: Optional[Dict[str, object]] = None
    if debug:
        debug_data = {
            "anchor_positive": {},
            "target_positive": {},
            "selected_pairs": [],
            "anchor_name": anchor_name,
            "target_name": target_name,
        }

    for pair in pairs:
        anchor_record, target_record = pair.anchor_record, pair.target_record
        anchor_ds, target_ds = pair.anchor_dataset, pair.target_dataset

        if anchor_ds != anchor_name or target_ds != target_name:
            if pair.target_dataset == anchor_name and pair.anchor_dataset == target_name:
                anchor_record, target_record = pair.target_record, pair.anchor_record
                anchor_ds, target_ds = pair.target_dataset, pair.anchor_dataset
            else:
                continue

        anchor_label = int(anchor_record.label)
        target_label = int(target_record.label)

        if debug_data is not None:
            if anchor_label == 1:
                _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
            if target_label == 1:
                _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))

        if use_positive_only and anchor_label != 1:
            continue
        if use_positive_only and target_label != 1:
            continue

        _register_index(anchor_record, anchor_ds)
        _register_index(target_record, target_ds)

        label_key = _label_key(anchor_label, target_label, anchor_name, target_name)
        label_hist[label_key] += 1

        if debug_data is not None:
            if anchor_label == 1 and target_label == 1:
                anchor_sample = _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
                target_sample = _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))
                if anchor_sample is not None and target_sample is not None:
                    debug_data["selected_pairs"].append({"anchor": anchor_sample, "target": target_sample})

        tile_id = anchor_record.tile_id
        grouped.setdefault(tile_id, {
            "anchor": anchor_record,
            "targets": [],
            "weights": [],
        })
        grouped[tile_id]["targets"].append(target_record)
        grouped[tile_id]["weights"].append(_pair_weight(anchor_record, target_record, gaussian_sigma))

    anchor_vecs: List[torch.Tensor] = []
    target_vecs: List[torch.Tensor] = []
    pair_metadata: List[Dict[str, object]] = []

    for entry in grouped.values():
        targets: List = entry["targets"]
        if not targets:
            continue
        aggregated, normalized_weights, target_stack = _aggregate_targets(entry['anchor'], targets, entry['weights'], aggregator)
        if aggregated is None:
            continue
        anchor_tensor = torch.from_numpy(entry['anchor'].embedding).float()
        anchor_vecs.append(anchor_tensor)
        target_vecs.append(aggregated.clone())
        pair_metadata.append(
            _make_pair_meta(
                anchor_record=entry["anchor"],
                targets=targets,
                weights=normalized_weights,
                is_augmented=False,
            )
        )
        anchor_aug_list = anchor_augment_map.get(entry['anchor'].tile_id, [])
        for aug_emb in anchor_aug_list:
            aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
            if aug_tensor.shape != anchor_tensor.shape:
                continue
            anchor_vecs.append(aug_tensor)
            target_vecs.append(aggregated.clone())
            label_hist[label_key] += 1
            anchor_aug_added += 1
            pair_metadata.append(
                _make_pair_meta(
                    anchor_record=entry["anchor"],
                    targets=targets,
                    weights=normalized_weights,
                    is_augmented=True,
                    augmentation_role="anchor",
                )
            )

        for idx_target, target_record in enumerate(targets):
            aug_list = target_augment_map.get(target_record.tile_id, [])
            if not aug_list:
                continue
            for aug_emb in aug_list:
                aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
                if aug_tensor.shape != target_stack[idx_target].shape:
                    continue
                modified_stack = target_stack.clone()
                modified_stack[idx_target] = aug_tensor
                aggregated_aug = torch.matmul(normalized_weights.unsqueeze(0), modified_stack).squeeze(0)
                anchor_vecs.append(anchor_tensor.clone())
                target_vecs.append(aggregated_aug)
                label_hist[label_key] += 1
                target_aug_added += 1
                pair_metadata.append(
                    _make_pair_meta(
                        anchor_record=entry["anchor"],
                        targets=targets,
                        weights=normalized_weights,
                        is_augmented=True,
                        augmentation_role="target",
                    )
                )


    aug_stats = {
        "anchor_augmented_pairs": int(anchor_aug_added),
        "target_augmented_pairs": int(target_aug_added),
        "selected_pairs_augmented": int(anchor_aug_added + target_aug_added),
    }

    pn_index_summary = _finalise_pn_index_summary()

    if debug_data is not None:
        anchor_serialised = [_serialise_sample(sample) for sample in debug_data["anchor_positive"].values()]
        target_serialised = [_serialise_sample(sample) for sample in debug_data["target_positive"].values()]
        pair_serialised = [
            {
                "anchor": _serialise_sample(pair["anchor"]),
                "target": _serialise_sample(pair["target"]),
            }
            for pair in debug_data["selected_pairs"]
        ]
        debug_payload = {
            "anchor_positive": anchor_serialised,
            "target_positive": target_serialised,
            "selected_pairs": pair_serialised,
            "anchor_name": anchor_name,
            "target_name": target_name,
            'anchor_augmented_pairs': int(anchor_aug_added),
            'target_augmented_pairs': int(target_aug_added),
        }
        debug_payload["pn_index_summary"] = pn_index_summary
    else:
        debug_payload = None

    return anchor_vecs, target_vecs, label_hist, debug_payload, aug_stats, pair_metadata, pn_index_summary

def _add_debug_sample(
    store: Dict[str, Dict[str, object]],
    record,
    dataset_name: str,
    dataset_meta: Optional[Dict[str, object]],
    *,
    is_augmented: bool = False,
) -> Optional[Dict[str, object]]:
    if record.coord is None or any(math.isnan(c) for c in record.coord[:2]):
        return None
    coord = (float(record.coord[0]), float(record.coord[1]))
    sample = store.get(record.tile_id)
    if sample is None:
        sample = {
            "tile_id": record.tile_id,
            "coord": coord,
            "dataset": dataset_name,
        }
        window = record.window_size
        pixel_res = record.pixel_resolution
        if window is None and dataset_meta:
            spacing = dataset_meta.get("window_spacing") or dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
            if spacing is not None:
                pixel_res = float(spacing)
                window = (1, 1)
        if window is not None:
            sample["window_size"] = [int(window[0]), int(window[1])]
        if pixel_res is None and dataset_meta:
            pixel_res = dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
        if pixel_res is not None:
            sample["pixel_resolution"] = float(pixel_res)
        if window is not None and pixel_res is not None:
            width = window[1] * float(pixel_res)
            height = window[0] * float(pixel_res)
            sample["footprint"] = [width, height]
        sample["is_augmented"] = bool(is_augmented)
        store[record.tile_id] = sample
    return sample

def _serialise_sample(sample: Dict[str, object]) -> Dict[str, object]:
    coord = sample.get("coord")
    coord_list = [float(coord[0]), float(coord[1])] if coord is not None else None
    data: Dict[str, object] = {
        "tile_id": sample.get("tile_id"),
    }
    if coord_list is not None:
        data["coord"] = coord_list
    if sample.get("dataset") is not None:
        data["dataset"] = sample["dataset"]
    if "window_size" in sample and sample["window_size"] is not None:
        data["window_size"] = [int(sample["window_size"][0]), int(sample["window_size"][1])]
    if "pixel_resolution" in sample and sample["pixel_resolution"] is not None:
        data["pixel_resolution"] = float(sample["pixel_resolution"])
    if sample.get("footprint") is not None:
        data["footprint"] = [float(sample["footprint"][0]), float(sample["footprint"][1])]
    if sample.get("is_augmented"):
        data["is_augmented"] = True
    return data

def _label_key(anchor_label: int, target_label: int, anchor_name: str, target_name: str) -> str:
    if anchor_label == 1 and target_label == 1:
        return "positive_common"
    if anchor_label == 1 and target_label != 1:
        return f"positive_{anchor_name}"
    if anchor_label != 1 and target_label == 1:
        return f"positive_{target_name}"
    return "unlabelled"


def _pair_weight(anchor_record, target_record, gaussian_sigma: Optional[float]) -> float:
    if gaussian_sigma is None:
        return 1.0
    if anchor_record.coord is None or target_record.coord is None:
        return 1.0
    anchor_xy = np.asarray(anchor_record.coord, dtype=float)
    target_xy = np.asarray(target_record.coord, dtype=float)
    if anchor_xy.size < 2 or target_xy.size < 2:
        return 1.0
    dist = float(np.linalg.norm(anchor_xy[:2] - target_xy[:2]))
    if dist <= 0.0:
        return 1.0
    weight = math.exp(- (dist ** 2) / (2.0 * (gaussian_sigma ** 2)))
    return float(weight + 1e-8)


def _aggregate_targets(anchor_record, targets: Sequence, weights: Sequence[float], aggregator: str) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if aggregator != "weighted_pool":
        raise NotImplementedError(f"Aggregator '{aggregator}' is not implemented yet.")

    embeddings = [torch.from_numpy(target.embedding).float() for target in targets]
    if not embeddings:
        return None, torch.empty(0), torch.empty(0)
    stacked = torch.stack(embeddings)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.isnan(weight_tensor).any() or float(weight_tensor.sum()) <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    aggregated = torch.matmul(weight_tensor.unsqueeze(0), stacked).squeeze(0)
    return aggregated, weight_tensor, stacked


def _normalise_coord(coord: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if coord is None:
        return None
    try:
        x = float(coord[0])
        y = float(coord[1])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _normalise_row_col(value: Optional[Sequence[object]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        row = int(value[0])
        col = int(value[1])
    except Exception:
        return None
    return (row, col)


def _weighted_average_coord(
    coords: Sequence[Optional[Tuple[float, float]]],
    weights: Sequence[float],
) -> Optional[Tuple[float, float]]:
    total_weight = 0.0
    accum_x = 0.0
    accum_y = 0.0
    for coord, weight in zip(coords, weights):
        if coord is None:
            continue
        try:
            w = float(weight)
        except Exception:
            continue
        if not math.isfinite(w) or w <= 0:
            continue
        accum_x += coord[0] * w
        accum_y += coord[1] * w
        total_weight += w
    if total_weight <= 0:
        return None
    return (accum_x / total_weight, accum_y / total_weight)


def _make_pair_meta(
    anchor_record,
    targets: Sequence,
    weights: Optional[torch.Tensor],
    is_augmented: bool,
    augmentation_role: Optional[str] = None,
) -> Dict[str, object]:
    anchor_row_col = _normalise_row_col(getattr(anchor_record, "row_col", None))
    target_row_cols = [_normalise_row_col(getattr(target, "row_col", None)) for target in targets]
    target_coords = [_normalise_coord(getattr(target, "coord", None)) for target in targets]
    target_tile_ids = [getattr(target, "tile_id", None) for target in targets]
    target_labels = [int(getattr(target, "label", 0)) for target in targets]
    target_regions = [getattr(target, "region", None) for target in targets]
    try:
        anchor_index = int(getattr(anchor_record, "index"))
    except Exception:
        anchor_index = None
    target_indices: List[Optional[int]] = []
    for target in targets:
        try:
            target_indices.append(int(getattr(target, "index")))
        except Exception:
            target_indices.append(None)
    weights_list: Optional[List[float]] = None
    weighted_coord: Optional[Tuple[float, float]] = None
    if isinstance(weights, torch.Tensor) and weights.numel() == len(targets):
        weights_cpu = weights.detach().cpu()
        weights_list = [float(w) for w in weights_cpu.tolist()]
        weighted_coord = _weighted_average_coord(target_coords, weights_list)

    meta: Dict[str, object] = {
        "anchor_tile_id": getattr(anchor_record, "tile_id", None),
        "anchor_coord": _normalise_coord(getattr(anchor_record, "coord", None)),
        "anchor_region": getattr(anchor_record, "region", None),
        "anchor_label": int(getattr(anchor_record, "label", 0)),
        "anchor_row_col": anchor_row_col,
        "target_tile_ids": target_tile_ids,
        "target_coords": target_coords,
        "target_labels": target_labels,
        "target_regions": target_regions,
        "target_row_cols": target_row_cols,
        "target_weights": weights_list,
        "target_weighted_coord": weighted_coord,
        "is_augmented": bool(is_augmented),
        "anchor_index": anchor_index,
        "target_indices": target_indices,
    }
    if augmentation_role is not None:
        meta["augmentation_role"] = augmentation_role
    return meta


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
                for layer_idx in range(num_layers - 1):
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


def _persist_state(cfg: AlignmentConfig, proj_a: nn.Module, proj_b: nn.Module, summary: Dict[str, object]) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return

    state_path = target_dir / "overlap_alignment_stage1.pt"
    payload = {
        "projection_head_a": proj_a.state_dict(),
        "projection_head_b": proj_b.state_dict(),
        "summary": summary,
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
    debug_data: Optional[Dict[str, List[Dict[str, object]]]] = None,
) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return
    metrics_path = target_dir / "overlap_alignment_stage1_metrics.json"
    payload = {
        "summary": summary,
        "epoch_history": list(epoch_history),
    }
    if debug_data:
        filtered_debug = {key: value for key, value in debug_data.items() if value}
        if filtered_debug:
            payload["debug"] = filtered_debug
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


if __name__ == "__main__":
    main()

