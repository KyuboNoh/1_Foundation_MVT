# scripts/train_classifier.py
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.data_utils import (
    clamp_coords_to_window,
    filter_valid_raster_coords,
    load_split_stack,
    normalize_region_coord,
    prefilter_valid_window_coords,
)

from Common.data_utils import resolve_search_root, load_training_args, load_training_metadata, mae_kwargs_from_training_args, resolve_pretraining_patch, infer_region_from_name, resolve_label_rasters, collect_feature_rasters, read_stack_patch
from Common.debug_visualization import visualize_debug_features
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.models.mlp_dropout import MLPDropout
from src.gfm4mpm.training.train_cls import train_classifier, eval_classifier
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from src.gfm4mpm.infer.infer_maps import group_positive_coords, mc_predict_map, write_prediction_outputs

class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, window=32, extra_samples: Optional[Sequence[Tuple[np.ndarray, int]]] = None):
        self.stack = stack
        self.coords = list(coords)
        self.labels = list(labels)
        self.window = window
        self.extra: List[Tuple[np.ndarray, int]] = []
        if extra_samples:
            for patch, label in extra_samples:
                arr = np.asarray(patch, dtype=np.float32)
                self.extra.append((arr, int(label)))

    def __len__(self):
        return len(self.coords) + len(self.extra)

    def __getitem__(self, idx):
        if idx < len(self.coords):
            coord = self.coords[idx]
            x = read_stack_patch(self.stack, coord, self.window)
            y = self.labels[idx]
        else:
            patch, label = self.extra[idx - len(self.coords)]
            x = patch
            y = label
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--step1', required=True)
    ap.add_argument('--step2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)

    ap.add_argument('--stride', type=int, default=None, help='Stride for sliding window inference (in pixels)')
    ap.add_argument('--passes', type=int, default=10, help='Number of stochastic forward passes for MC-Dropout')
    ap.add_argument('--debug', action='store_true', help='Generate debug plots for labels and feature rasters')
    ap.add_argument('--positive-augmentation', action='store_true', help='Include augmented positive patches from the encoder directory.')

    ap.add_argument('--test-ratio', type=float, default=0.3, help='Fraction of data to use for validation')
    ap.add_argument('--random-seed', type=int, default=42, help='Random seed for data splits and shuffling')

    ap.add_argument('--save-prediction', dest='save_prediction', action='store_true', help='Save prediction results as ascii grid files')
    ap.add_argument('--plot-prediction', dest='plot_prediction', action='store_false', help='Plot prediction results')
    ap.set_defaults(save_prediction=True)
    ap.set_defaults(plot_prediction=False)

    args = ap.parse_args()
    metrics_summary: Dict[str, Any] = {}
    metrics_summary["test_ratio"] = float(args.test_ratio)
    metrics_summary["random_seed"] = int(args.random_seed)
    metrics_summary["save_prediction"] = bool(args.save_prediction)
    metrics_summary["plot_prediction"] = bool(args.plot_prediction)
    metrics_summary["passes"] = int(args.passes)
    metrics_summary["positive_augmentation"] = bool(args.positive_augmentation)

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands or one of --stac-root/--stac-table (exactly one input source)')

    encoder_input_path = Path(args.step1).resolve()
    if encoder_input_path.is_dir():
        encoder_dir = encoder_input_path
        weight_candidates = sorted(encoder_dir.glob('mae_encoder*.pth'))
        if not weight_candidates:
            raise FileNotFoundError(f"No mae_encoder*.pth found inside {encoder_dir}")
        encoder_weights_path = weight_candidates[0]
    else:
        encoder_weights_path = encoder_input_path
        encoder_dir = encoder_weights_path.parent
    if not encoder_weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found at {encoder_weights_path}")
    metrics_summary["encoder_weights"] = str(encoder_weights_path)
    label_column = 'Training_MVT_Deposit'

    load_result = load_split_stack(
        args.stac_root,
        args.stac_table,
        args.bands,
        encoder_weights_path,
        label_column,
        load_training_args,
        load_training_metadata,
        resolve_label_rasters,
        collect_feature_rasters,
        infer_region_from_name,
    )

    stack = load_result.stack
    augmented_patches_all: Optional[np.ndarray] = None
    augmented_sources_all: Optional[np.ndarray] = None
    augmented_embeddings_all: Optional[np.ndarray] = None
    augmented_plot_coords: List[Tuple[str, int, int]] = []

    if args.positive_augmentation:
        aug_path = Path(args.step2) / 'positive_augmentation.npz'
        if not aug_path.exists():
            print(f'[warn] positive_augmentation.npz not found at {aug_path}; disabling --positive-augmentation.')
            args.positive_augmentation = False
        else:
            try:
                with np.load(aug_path, allow_pickle=True) as aug_npz:
                    tile_data_all = np.asarray(aug_npz['tile_data'], dtype=object)
                    is_augmented = np.asarray(aug_npz['is_augmented'], dtype=bool)
                    if not is_augmented.any():
                        print('[warn] positive_augmentation.npz contains no augmented samples; disabling --positive-augmentation.')
                        args.positive_augmentation = False
                    else:
                        source_index_all = np.asarray(aug_npz['source_index'], dtype=int)
                        embeddings_all = np.asarray(aug_npz['embeddings'], dtype=np.float32)
                        tile_data_masked = tile_data_all[is_augmented]
                        augmented_patches_all = np.stack([np.asarray(arr, dtype=np.float32) for arr in tile_data_masked], axis=0)
                        augmented_sources_all = source_index_all[is_augmented]
                        augmented_embeddings_all = embeddings_all[is_augmented]
                        metadata_all = aug_npz.get('metadata')
                        augmentation_params_all = aug_npz.get('augmentation_params')
                        if metadata_all is not None and augmentation_params_all is not None:
                            meta_masked = np.asarray(metadata_all, dtype=object)[is_augmented]
                            params_masked = np.asarray(augmentation_params_all, dtype=object)[is_augmented]
                            for meta, params in zip(meta_masked, params_masked):
                                if not isinstance(meta, dict):
                                    continue
                                region = str(meta.get('region') or meta.get('Region') or 'GLOBAL')
                                row = meta.get('row')
                                col = meta.get('col')
                                try:
                                    row = int(row)
                                    col = int(col)
                                except Exception:
                                    continue
                                if isinstance(params, dict):
                                    try:
                                        row += int(params.get('shift_rows', 0))
                                        col += int(params.get('shift_cols', 0))
                                    except Exception:
                                        pass
                                augmented_plot_coords.append((region, row, col))
                        metrics_summary['positive_augmented_total'] = int(augmented_patches_all.shape[0])
            except Exception as exc:
                print(f'[warn] Failed to load positive_augmentation.npz: {exc}')
                args.positive_augmentation = False
                augmented_patches_all = None
                augmented_sources_all = None
                augmented_embeddings_all = None
    else:
        args.positive_augmentation = False
    training_args_data = load_result.training_args
    metrics_summary["input_mode"] = load_result.mode
    if load_result.training_args_path is not None:
        metrics_summary["pretrain_args_path"] = str(load_result.training_args_path)

    # if load_result.mode == 'table':
    #     default_patch = 1
    # else:
    #     default_patch = args.patch
    # patch_size = _resolve_pretraining_patch(training_args_data, default_patch)
    # if training_args_data and training_args_data.get('window'):
    #     window_size = int(training_args_data['window'])
    # else:
    #     window_size = default_patch if load_result.mode == 'table' else args.patch
    patch_size = int(training_args_data['patch'])
    window_size = int(training_args_data['window'])

    if load_result.mode == 'table' and window_size != 1:
        window_size = 1
    metrics_summary["patch_size"] = int(patch_size)
    metrics_summary["window_size"] = int(window_size)

    stride = args.stride if args.stride is not None else window_size
    metrics_summary["stride"] = int(stride)

    state_dict = torch.load(encoder_weights_path, map_location='cpu')
    patch_proj = state_dict.get('patch.proj.weight')
    if isinstance(patch_proj, torch.Tensor):
        expected_channels = patch_proj.shape[1]
        if stack.count != expected_channels:
            raise RuntimeError(
                f"MAE encoder expects {expected_channels} input channel(s) but raster stack provides {stack.count}; "
                "verify feature selection and metadata alignment."
                )

    mae_kwargs: Dict[str, Any] = {}
    if training_args_data:
        mae_kwargs = mae_kwargs_from_training_args(training_args_data)
    mae_kwargs.setdefault('patch_size', patch_size)
    mae_kwargs.setdefault('image_size', window_size)

    if hasattr(stack, "resolve_region_stack"):
        default_region = getattr(stack, "default_region", None)
        if default_region is None:
            regions = getattr(stack, "regions", [])
            default_region = regions[0] if regions else "GLOBAL"
    else:
        default_region = "GLOBAL"

    with open(Path(args.step2)/"splits.json") as f:
        sp = json.load(f)
    pos_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['pos']]
    neg_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['neg']]
    coords = pos_coords + neg_coords
    labels = [1]*len(pos_coords) + [0]*len(neg_coords)
    metrics_summary["initial_pos"] = len(pos_coords)
    metrics_summary["initial_neg"] = len(neg_coords)

    batch_size = args.batch
    print(f"[info] Using batch size {batch_size} from training_args_1_pretrain.json")

    coords_clamped, clamp_count = clamp_coords_to_window(coords, stack, window_size)
    if clamp_count:
        print(f"[info] Adjusted {clamp_count} split coordinate(s) to stay within window bounds")
    metrics_summary["clamp_count"] = int(clamp_count)
    coords = [normalize_region_coord(coord, default_region=default_region) for coord in coords_clamped]
    coords_with_labels = list(zip(coords, labels))

    if load_result.mode == 'raster':
        prefiltered = prefilter_valid_window_coords(stack, coords, window_size)
        removed = len(coords) - len(prefiltered)
        if removed:
            print(f"[info] Prefiltered {removed} coordinate(s) falling outside the project boundary")
        metrics_summary["prefilter_removed"] = int(removed)
        valid_prefilter = set(prefiltered)
        coords_with_labels = [(coord, lab) for coord, lab in coords_with_labels if coord in valid_prefilter]
        coords = [coord for coord, _ in coords_with_labels]
        filtered_coords, dropped_coords = filter_valid_raster_coords(stack, coords, window_size, min_valid_fraction=0.05)
        dropped_count = len(dropped_coords)
        if dropped_count:
            print(f"[info] Dropped {dropped_count} coordinate(s) due to insufficient valid pixels")
        metrics_summary["dropped_invalid"] = int(dropped_count)
        if not filtered_coords:
            raise RuntimeError("All training samples were filtered out; check raster coverage or label coordinates.")
        valid_set = set(filtered_coords)
        coords_with_labels = [(coord, lab) for coord, lab in coords_with_labels if coord in valid_set]
    else:
        metrics_summary["prefilter_removed"] = 0
        metrics_summary["dropped_invalid"] = 0

    coords = [coord for coord, _ in coords_with_labels]
    labels = [lab for _, lab in coords_with_labels]
    pos_coords_final = [coord for coord, lab in coords_with_labels if lab == 1]
    pos_coord_to_index = {tuple(coord): idx for idx, coord in enumerate(pos_coords_final)}
    neg_coords_final = [coord for coord, lab in coords_with_labels if lab == 0]

    if not coords_with_labels:
        raise RuntimeError("No training samples available after preprocessing.")

    print(f"[info] Training classifier with {len(coords)} samples ({len(pos_coords_final)} positive, {len(neg_coords_final)} negative)")
    metrics_summary["final_training_samples"] = len(coords)
    metrics_summary["final_pos"] = len(pos_coords_final)
    metrics_summary["final_neg"] = len(neg_coords_final)

    if args.debug:
        if load_result.mode != 'raster':
            print("[warn] Debug visualisations currently only implemented for raster stacks; skipping.")
        else:
            feature_list = load_result.feature_columns or getattr(stack, "feature_columns", None) or []
            try:
                visualize_debug_features(
                    stack,
                    feature_list,
                    pos_coords_final,
                    neg_coords_final,
                    Path(args.out),
                    augmented_coords=augmented_plot_coords if (args.positive_augmentation and augmented_plot_coords) else None,
                )
            except Exception as exc:
                print(f"[warn] Failed to generate debug visualisations: {exc}")


    # Split data
    Xtr, Xval, ytr, yval = train_test_split(coords, labels, test_size=args.test_ratio, stratify=labels, random_state=args.random_seed)
    train_pos = int(sum(ytr))
    val_pos = int(sum(yval))
    metrics_summary["train_samples"] = len(Xtr)
    metrics_summary["val_samples"] = len(Xval)
    metrics_summary["train_pos"] = train_pos
    metrics_summary["train_neg"] = len(Xtr) - train_pos
    metrics_summary["val_pos"] = val_pos
    metrics_summary["val_neg"] = len(Xval) - val_pos

    extra_train_samples: List[Tuple[np.ndarray, int]] = []
    train_aug_sources = np.array([], dtype=int)
    if args.positive_augmentation and augmented_patches_all is not None and augmented_sources_all is not None:
        train_pos_indices: set[int] = set()
        for coord, label in zip(Xtr, ytr):
            if label == 1:
                idx = pos_coord_to_index.get(tuple(coord))
                if idx is not None:
                    train_pos_indices.add(idx)
        if train_pos_indices:
            mask_train_aug = np.isin(augmented_sources_all, list(train_pos_indices))
            if mask_train_aug.any():
                train_aug_patches = augmented_patches_all[mask_train_aug]
                train_aug_sources = augmented_sources_all[mask_train_aug]
                extra_train_samples = [(patch, 1) for patch in train_aug_patches]
                metrics_summary['train_augmented'] = int(len(train_aug_sources))
        else:
            train_aug_sources = np.array([], dtype=int)
    else:
        train_aug_sources = np.array([], dtype=int)
    if 'train_augmented' not in metrics_summary:
        metrics_summary['train_augmented'] = int(len(extra_train_samples))
    metrics_summary['train_samples_with_aug'] = len(Xtr) + len(extra_train_samples)
    ds_tr = LabeledPatches(stack, Xtr, ytr, window=window_size, extra_samples=extra_train_samples)
    ds_va = LabeledPatches(stack, Xval, yval, window=window_size)
    if extra_train_samples:
        print(f'[info] Added {len(extra_train_samples)} augmented positive samples to training loader.')
    worker_count = 0 if getattr(stack, 'kind', None) == 'raster' else 8
    if worker_count == 0:
        print('[info] Using single-process data loading for raster stack')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=worker_count)
    metrics_summary["batch_size"] = int(batch_size)
    metrics_summary["epochs"] = int(args.epochs)

    encoder = MAEViT(in_chans=stack.count, **mae_kwargs)
    encoder.load_state_dict(state_dict)
    in_dim = encoder.blocks[0].attn.embed_dim

    # TODO: Add hyperparameters for probability, hidden dims...
    mlp = MLPDropout(in_dim=in_dim)

    # Train classifier
    mlp, epoch_history = train_classifier(
        encoder,
        mlp,
        dl_tr,
        dl_va,
        epochs=args.epochs,
        return_history=True,
        loss_weights={'bce': 1.0},
    )
    train_eval = eval_classifier(encoder, mlp, dl_tr)
    val_eval = eval_classifier(encoder, mlp, dl_va)
    log_metrics("final train", train_eval, order=DEFAULT_METRIC_ORDER)
    log_metrics("final val", val_eval, order=DEFAULT_METRIC_ORDER)

    metrics_summary["evaluation"] = {
        "train": normalize_metrics(train_eval),
        "val": normalize_metrics(val_eval),
    }
    history = []
    for rec in epoch_history:
        entry = {"epoch": rec["epoch"],
                 "train": normalize_metrics(rec["train"]),
                 "val": normalize_metrics(rec["val"])}
        if "train_weighted_loss" in rec:
            entry["train_weighted_loss"] = float(rec["train_weighted_loss"])
        if "val_weighted_loss" in rec:
            entry["val_weighted_loss"] = float(rec["val_weighted_loss"])
        history.append(entry)
    metrics_summary["epoch_metrics"] = history
    out_dir = Path(args.out) if args.out else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'mlp_classifier.pth'
    torch.save(mlp.state_dict(), out_path)
    print(f"[info] Saved classifier to {out_path}")
    outputs_summary = {
        "classifier_path": str(out_path),
        "saved_prediction_arrays": bool(args.save_prediction),
    }
    if args.positive_augmentation and augmented_patches_all is not None and augmented_patches_all.size:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            encoder.eval().to(device)
            mlp.eval().to(device)
            preds_list: List[np.ndarray] = []
            for start_idx in range(0, augmented_patches_all.shape[0], batch_size):
                patch_chunk = torch.from_numpy(augmented_patches_all[start_idx:start_idx + batch_size]).float().to(device)
                with torch.no_grad():
                    z_chunk = encoder.encode(patch_chunk)
                    p_chunk = mlp(z_chunk).cpu().numpy().reshape(-1)
                preds_list.append(p_chunk)
            if preds_list:
                aug_probs = np.concatenate(preds_list, axis=0)
            else:
                aug_probs = np.empty(0, dtype=np.float32)
            metrics_summary['augmented_predictions'] = {
                'count': int(aug_probs.size),
                'mean': float(np.mean(aug_probs)) if aug_probs.size else float('nan'),
                'min': float(np.min(aug_probs)) if aug_probs.size else float('nan'),
                'max': float(np.max(aug_probs)) if aug_probs.size else float('nan'),
            }
            try:
                fig, ax = plt.subplots(figsize=(9, 4))
                jitter = np.random.uniform(-0.1, 0.1, size=aug_probs.size) if aug_probs.size else np.array([])
                ax.scatter(augmented_sources_all + jitter, aug_probs, s=30, facecolors='none', edgecolors='orange', linewidths=0.8)
                ax.set_xlabel('Original positive index')
                ax.set_ylabel('Augmented prediction probability')
                ax.set_ylim(0, 1)
                ax.set_title('Augmented Positive Predictions')
                aug_plot_path = out_dir / 'augmented_positive_predictions.png'
                fig.tight_layout()
                fig.savefig(aug_plot_path, dpi=200)
                plt.close(fig)
                outputs_summary['augmented_prediction_plot'] = str(aug_plot_path)
            except Exception as plot_exc:
                print(f'[warn] Failed to plot augmented predictions: {plot_exc}')
        except Exception as aug_exc:
            print(f'[warn] Failed to compute augmented predictions: {aug_exc}')
            metrics_summary['augmented_predictions'] = {'count': 0, 'mean': float('nan'), 'min': float('nan'), 'max': float('nan')}
    elif args.positive_augmentation:
        print('[warn] Positive augmentation was requested but no augmented samples were available after preprocessing.')

    metrics_json_path = Path(args.out) / "metrics.json"
    try:
        save_metrics_json(metrics_summary, metrics_json_path)
        print(f"[info] Saved detailed metrics to {metrics_json_path}")
    except Exception as exc:
        print(f"[warn] Failed to write metrics JSON to {metrics_json_path}: {exc}")

    print("[info] Generating prediction maps with MC-Dropout")
    # TODO: Allow stride override?
    # prediction = mc_predict_map(encoder, mlp, stack, window_size=window_size, stride=int(1.00*patch_size), passes=args.passes, show_progress=True, save_prediction=args.save_prediction, save_path=out_dir)
    # prediction = mc_predict_map(encoder, mlp, stack, window_size=window_size, stride=int(1.00*window_size), passes=args.passes, show_progress=True, save_prediction=args.save_prediction, save_path=out_dir)
    prediction = mc_predict_map(
        encoder,
        mlp,
        stack,
        window_size=window_size,
        stride=stride,
        passes=args.passes,
        show_progress=True,
        save_prediction=args.save_prediction,
        save_path=out_dir,
    )

    if getattr(stack, 'is_table', False) and not isinstance(prediction, dict):
        mean_map, std_map = prediction
        mean_vec = mean_map.reshape(-1)
        std_vec = std_map.reshape(-1)
        rows = stack.iter_metadata()
        for idx, meta in enumerate(rows):
            meta['prospectivity_mean'] = float(mean_vec[idx])
            meta['prospectivity_std'] = float(std_vec[idx])
        out_csv = Path(args.out).with_suffix('.csv')
        if not out_csv.parent.exists():
            out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote table predictions to {out_csv}")
        outputs_summary["prediction_table"] = str(out_csv)
    else:
        pos_coord_map = group_positive_coords(pos_coords_final, stack)
        write_prediction_outputs(
            prediction,
            stack,
            out_dir,
            pos_coords_by_region=pos_coord_map,
        )
        outputs_summary["prediction_directory"] = str(out_dir)

    summary_path = Path(args.out) / "training_args_3_cls.json"
    summary_payload = {
        "metrics": metrics_summary,
        "outputs": outputs_summary,
    }
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics_json(summary_payload, summary_path)
        print(f"[info] Saved classifier configuration to {summary_path}")
    except Exception as exc:
        print(f"[warn] Failed to write classifier summary to {summary_path}: {exc}")