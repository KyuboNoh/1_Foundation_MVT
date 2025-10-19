# scripts/train_classifier.py
import argparse
import glob
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

from Common.data_utils import clamp_coords_to_window, filter_valid_raster_coords, load_split_stack, normalize_region_coord
from Common.data_utils import resolve_search_root, load_training_args, load_training_metadata, mae_kwargs_from_training_args, resolve_pretraining_patch, infer_region_from_name, resolve_label_rasters, collect_feature_rasters, read_stack_patch
from Common.debug_visualization import visualize_debug_features
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.models.mlp_dropout import MLPDropout
from src.gfm4mpm.training.train_cls import train_classifier
from src.gfm4mpm.infer.infer_maps import mc_predict_map, group_positive_coords, write_prediction_outputs


def _infer_prediction_region_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[:-12]
    elif stem.endswith("_prediction"):
        stem = stem[:-11]
    if not stem or stem.lower() in {"prediction", "predictions"}:
        return "GLOBAL"
    return stem


def _aggregate_prediction_arrays(arrays: Sequence[np.ndarray], height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    if not arrays:
        shape = (height, width)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    stacked = np.concatenate(arrays, axis=0)
    if stacked.size == 0:
        shape = (height, width)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    coords = stacked[:, :2].astype(np.int64, copy=False)
    mu_vals = stacked[:, 2].astype(np.float64, copy=False)
    var_vals = stacked[:, 3].astype(np.float64, copy=False)

    valid_mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < height)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < width)
    )
    if not np.all(valid_mask):
        coords = coords[valid_mask]
        mu_vals = mu_vals[valid_mask]
        var_vals = var_vals[valid_mask]

    if coords.shape[0] == 0:
        shape = (height, width)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    linear_idx = coords[:, 0] * width + coords[:, 1]
    flat_size = height * width
    counts = np.bincount(linear_idx, minlength=flat_size)
    mu_sum = np.bincount(linear_idx, weights=mu_vals, minlength=flat_size)
    var_sum = np.bincount(linear_idx, weights=var_vals, minlength=flat_size)

    counts = counts.reshape(height, width)
    mu_sum = mu_sum.reshape(height, width)
    var_sum = var_sum.reshape(height, width)

    zero_mask = counts == 0
    safe_counts = counts.astype(np.float32, copy=False)
    if np.any(zero_mask):
        safe_counts = safe_counts.copy()
        safe_counts[zero_mask] = 1.0

    mean_map = (mu_sum / safe_counts).astype(np.float32, copy=False)
    var_avg = (var_sum / safe_counts).astype(np.float32, copy=False)
    std_map = np.sqrt(var_avg, out=var_avg)

    if np.any(zero_mask):
        mean_map[zero_mask] = 0.0
        std_map[zero_mask] = 0.0

    return mean_map, std_map


def _load_saved_prediction_maps(pattern: str, stack: Any) -> Dict[str, Dict[str, Any]]:
    candidates = glob.glob(pattern)
    if not candidates:
        alt_patterns: List[str] = []
        if "_prediction" in pattern:
            alt_patterns.append(pattern.replace("_prediction", "_predictions"))
        if "prediction.npy" in pattern:
            alt_patterns.append(pattern.replace("prediction.npy", "predictions.npy"))
        if not alt_patterns:
            alt_patterns.append(pattern.replace(".npy", "_predictions.npy"))
            alt_patterns.append(pattern.replace(".npy", "_prediction.npy"))
        for alt in alt_patterns:
            candidates.extend(glob.glob(alt))
        if not candidates:
            pattern_path = Path(pattern)
            parent_dir = pattern_path.parent if pattern_path.parent != Path("") else Path(".")
            if parent_dir.exists():
                fallback = [
                    str(p)
                    for p in parent_dir.glob("*.npy")
                    if "pred" in p.name.lower() or "mc" in p.name.lower()
                ]
                candidates.extend(fallback)
    paths = sorted(set(candidates))
    if not paths:
        raise FileNotFoundError(f"No saved prediction files found using pattern {pattern!r}")

    region_lookup: Dict[str, Any] = {}
    if hasattr(stack, "iter_region_stacks"):
        for region_name, region_stack in stack.iter_region_stacks():
            region_lookup[str(region_name)] = region_stack

    aggregated: Dict[str, List[np.ndarray]] = {}
    for path_str in paths:
        arr = np.load(path_str, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] < 4:
            print(f"[warn] Skipping prediction file {path_str} with unexpected shape {arr.shape}")
            continue
        region_key = _infer_prediction_region_name(Path(path_str))
        aggregated.setdefault(region_key, []).append(np.asarray(arr[:, :4], dtype=np.float32))

    if not aggregated:
        raise RuntimeError("No valid prediction arrays were found in the provided files.")

    results: Dict[str, Dict[str, Any]] = {}
    for region_key, arrays in aggregated.items():
        if not arrays:
            continue
        target_stack = region_lookup.get(region_key)
        resolved_key = region_key if region_key else "GLOBAL"
        if target_stack is None:
            if region_key not in {"GLOBAL"} and region_lookup:
                print(f"[warn] Using global stack dimensions for unmatched region '{region_key}'.")
            target_stack = stack
            if not region_lookup:
                resolved_key = "GLOBAL"
        mean_map, std_map = _aggregate_prediction_arrays(arrays, target_stack.height, target_stack.width)
        results[str(resolved_key)] = {"mean": mean_map, "std": std_map, "stack": target_stack}

    if not results:
        raise RuntimeError("Failed to reconstruct any prediction maps from saved arrays.")

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--encoder', required=True)

    ap.add_argument('--prediction-glob', help='Glob pattern to saved *_prediction*.npy files for redraw', default=None)
    ap.add_argument('--out', required=True)

    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands or one of --stac-root/--stac-table (exactly one input source)')

    encoder_path = Path(args.encoder).resolve()
    label_column = 'Training_MVT_Deposit'

    load_result = load_split_stack(
        args.stac_root,
        args.stac_table,
        args.bands,
        encoder_path,
        label_column,
        load_training_args,
        load_training_metadata,
        resolve_label_rasters,
        collect_feature_rasters,
        infer_region_from_name,
    )

    stack = load_result.stack
    training_args_data = load_result.training_args

    patch_size = int(training_args_data['patch'])
    window_size = int(training_args_data['window'])

    if load_result.mode == 'table' and window_size != 1:
        window_size = 1

    state_dict = torch.load(encoder_path, map_location='cpu')
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

    with open(args.splits) as f:
        sp = json.load(f)
    pos_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['pos']]
    neg_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['neg']]
    coords = pos_coords + neg_coords
    labels = [1]*len(pos_coords) + [0]*len(neg_coords)

    pos_coords_by_region = group_positive_coords(pos_coords, stack)

    if args.prediction_glob:
        print(f"[info] Reconstructing prospectivity maps from saved predictions matching {args.prediction_glob}")
        prediction = _load_saved_prediction_maps(args.prediction_glob, stack)


        print(prediction)
        exit()

        
        write_prediction_outputs(
            prediction,
            stack,
            Path(args.out),
            pos_coords_by_region=pos_coords_by_region,
        )
        sys.exit(0)
