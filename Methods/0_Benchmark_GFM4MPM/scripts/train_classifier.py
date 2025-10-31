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
from Common.cls.models.mae_vit import MAEViT
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.training.train_cls import train_classifier, eval_classifier, dataloader_metric_inputORembedding
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics, normalize_metrics, save_metrics_json
from Common.cls.infer.infer_maps import group_coords, mc_predict_map, write_prediction_outputs




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
    ap.add_argument('--mlp-hidden-dims', type=int, nargs='+', default=[256, 128],
                    help='Hidden layer sizes for the classifier MLP (space-separated)')
    ap.add_argument('--mlp-dropout', type=float, default=0.2,
                    help='Dropout probability applied to classifier MLP layers')

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
    metrics_summary["mlp_hidden_dims"] = [int(h) for h in args.mlp_hidden_dims]
    metrics_summary["mlp_dropout"] = float(args.mlp_dropout)

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

    dl_tr, dl_va, metrics_summary_append = dataloader_metric_inputORembedding(Xtr, Xval, ytr, yval, batch_size, positive_augmentation=args.positive_augmentation,
                                                             augmented_patches_all=augmented_patches_all, pos_coord_to_index=pos_coord_to_index,
                                                             window_size=window_size, stack=stack, 
                                                             embedding=None,
                                                             epochs=args.epochs)
    metrics_summary.update(metrics_summary_append)


    # Build models
    encoder = MAEViT(in_chans=stack.count, **mae_kwargs)
    encoder.load_state_dict(state_dict)
    in_dim = encoder.blocks[0].attn.embed_dim

    hidden_dims = tuple(int(h) for h in args.mlp_hidden_dims if int(h) > 0)
    if not hidden_dims:
        raise ValueError("At least one positive hidden dimension must be provided for the classifier MLP.")
    if not (0.0 <= args.mlp_dropout < 1.0):
        raise ValueError("Classifier MLP dropout must be in the range [0.0, 1.0).")
    mlp = MLPDropout(in_dim=in_dim, hidden_dims=hidden_dims, p=float(args.mlp_dropout))

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

    precomputed_embeddings_map: Optional[Dict[str, Tuple[np.ndarray, Dict[Tuple[int, int], int], List[Tuple[int, int]]]]] = None
    embedding_bundle_path = Path(args.step2) / "embeddings.npz"
    if embedding_bundle_path.exists():
        emb_npz = None
        try:
            emb_npz = np.load(embedding_bundle_path, allow_pickle=True)
            embeddings_all = np.asarray(emb_npz["embeddings"], dtype=np.float32)
            metadata_array = emb_npz.get("metadata")
            if embeddings_all.ndim == 2 and metadata_array is not None:
                region_coord_lookup: Dict[str, Dict[Tuple[int, int], int]] = {}
                region_coord_order: Dict[str, List[Tuple[int, int]]] = {}
                missing_metadata = 0
                for idx, meta in enumerate(metadata_array):
                    meta_dict = meta
                    if hasattr(meta, "item") and not isinstance(meta, dict):
                        try:
                            meta_dict = meta.item()
                        except Exception:
                            meta_dict = None
                    if not isinstance(meta_dict, dict):
                        missing_metadata += 1
                        continue
                    row_val = meta_dict.get("row")
                    col_val = meta_dict.get("col")
                    if row_val is None or col_val is None:
                        missing_metadata += 1
                        continue
                    try:
                        row_idx = int(row_val)
                        col_idx = int(col_val)
                    except (TypeError, ValueError):
                        missing_metadata += 1
                        continue
                    region_name = str(meta_dict.get("region") or default_region)
                    coord_key = (row_idx, col_idx)
                    region_lookup = region_coord_lookup.setdefault(region_name, {})
                    if coord_key not in region_lookup:
                        region_coord_order.setdefault(region_name, []).append(coord_key)
                    region_lookup[coord_key] = idx
                if region_coord_lookup:
                    precomputed_embeddings_map = {}
                    total_mapped_unique = 0
                    for region, coord_lookup in region_coord_lookup.items():
                        coord_list = region_coord_order.get(region)
                        if coord_list is None:
                            coord_list = list(coord_lookup.keys())
                        precomputed_embeddings_map[region] = (embeddings_all, coord_lookup, coord_list)
                        total_mapped_unique += len(coord_list)
                    if "GLOBAL" not in precomputed_embeddings_map and region_coord_lookup:
                        global_lookup: Dict[Tuple[int, int], int] = {}
                        global_order: List[Tuple[int, int]] = []
                        for coord_lookup in region_coord_lookup.values():
                            for coord_key, emb_idx in coord_lookup.items():
                                if coord_key not in global_lookup:
                                    global_order.append(coord_key)
                                global_lookup[coord_key] = emb_idx
                        if global_lookup:
                            precomputed_embeddings_map["GLOBAL"] = (embeddings_all, global_lookup, global_order)
                    metrics_summary["precomputed_embeddings"] = {
                        "regions": sorted(region_coord_lookup.keys()),
                        "total_mapped": int(total_mapped_unique),
                        "missing_metadata": int(missing_metadata),
                    }
                    if missing_metadata:
                        print(
                            f"[warn] {missing_metadata} embedding record(s) in {embedding_bundle_path.name} "
                            "were missing region/row/col metadata; falling back to encoder for those coordinates."
                        )
                    else:
                        print(f"[info] Loaded {total_mapped_unique} precomputed embeddings from {embedding_bundle_path}")
                else:
                    print(
                        f"[warn] No usable region metadata found in {embedding_bundle_path}; "
                        "will rely on encoder outputs for inference."
                    )
            else:
                print(
                    f"[warn] embeddings.npz at {embedding_bundle_path} is missing metadata or has unexpected shape; "
                    "will rely on encoder outputs for inference."
                )
        except Exception as exc:
            print(f"[warn] Failed to load precomputed embeddings from {embedding_bundle_path}: {exc}")
        finally:
            if emb_npz is not None:
                try:
                    emb_npz.close()
                except Exception:
                    pass
    else:
        print(f"[info] No embeddings.npz found at {embedding_bundle_path}; using encoder for inference.")

    print("[info] Generating prediction maps with MC-Dropout")
    if precomputed_embeddings_map:
        mapped_total = sum(len(mapping[2]) for key, mapping in precomputed_embeddings_map.items() if key != "GLOBAL")
        print(f"[info] Using {mapped_total} precomputed embedding(s) for prediction inference.")
    prediction = mc_predict_map(
        encoder,
        mlp,
        stack,
        window_size=window_size,
        passes=args.passes,
        show_progress=True,
        save_prediction=args.save_prediction,
        save_path=out_dir,
        precomputed_embeddings=precomputed_embeddings_map,
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
        pos_coord_map = group_coords(pos_coords_final, stack)
        neg_coord_map = group_coords(neg_coords_final, stack)
        write_prediction_outputs(
            prediction,
            stack,
            out_dir,
            pos_coords_by_region=pos_coord_map,
            neg_coords_by_region=neg_coord_map,
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
