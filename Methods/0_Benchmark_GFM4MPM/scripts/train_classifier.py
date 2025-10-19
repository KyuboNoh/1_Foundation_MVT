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

from Common.data_utils import clamp_coords_to_window, filter_valid_raster_coords, load_split_stack, normalize_region_coord
from Common.data_utils import resolve_search_root, load_training_args, load_training_metadata, mae_kwargs_from_training_args, resolve_pretraining_patch, infer_region_from_name, resolve_label_rasters, collect_feature_rasters, read_stack_patch
from Common.debug_visualization import visualize_debug_features
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.models.mlp_dropout import MLPDropout
from src.gfm4mpm.training.train_cls import train_classifier
from src.gfm4mpm.infer.infer_maps import group_positive_coords, mc_predict_map, write_prediction_outputs

class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, window=32):
        self.stack, self.coords, self.labels, self.window = stack, coords, labels, window

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        x = read_stack_patch(self.stack, coord, self.window)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)

    ap.add_argument('--stride', type=int, default=None, help='Stride for sliding window inference (in pixels)')
    ap.add_argument('--passes', type=int, default=10, help='Number of stochastic forward passes for MC-Dropout')
    ap.add_argument('--debug', action='store_true', help='Generate debug plots for labels and feature rasters')

    ap.add_argument('--test-ratio', type=float, default=0.3, help='Fraction of data to use for validation')
    ap.add_argument('--random-seed', type=int, default=42, help='Random seed for data splits and shuffling')

    ap.add_argument('--save-prediction', dest='save_prediction', action='store_true', help='Save prediction results as ascii grid files')
    ap.add_argument('--plot-prediction', dest='plot_prediction', action='store_false', help='Plot prediction results')
    ap.set_defaults(save_prediction=True)
    ap.set_defaults(plot_prediction=False)

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

    stride = args.stride if args.stride is not None else window_size

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

    batch_size = args.batch
    print(f"[info] Using batch size {batch_size} from training_args_1_pretrain.json")

    coords_clamped, clamp_count = clamp_coords_to_window(coords, stack, window_size)
    if clamp_count:
        print(f"[info] Adjusted {clamp_count} split coordinate(s) to stay within window bounds")
    coords = [normalize_region_coord(coord, default_region=default_region) for coord in coords_clamped]
    coords_with_labels = list(zip(coords, labels))

    if load_result.mode == 'raster':
        filtered_coords, dropped_coords = filter_valid_raster_coords(stack, coords, window_size, min_valid_fraction=0.05)
        dropped_count = len(dropped_coords)
        if dropped_count:
            print(f"[info] Dropped {dropped_count} coordinate(s) due to insufficient valid pixels")
        if not filtered_coords:
            raise RuntimeError("All training samples were filtered out; check raster coverage or label coordinates.")
        valid_set = set(filtered_coords)
        coords_with_labels = [(coord, lab) for coord, lab in coords_with_labels if coord in valid_set]

    coords = [coord for coord, _ in coords_with_labels]
    labels = [lab for _, lab in coords_with_labels]
    pos_coords_final = [coord for coord, lab in coords_with_labels if lab == 1]
    neg_coords_final = [coord for coord, lab in coords_with_labels if lab == 0]

    if not coords_with_labels:
        raise RuntimeError("No training samples available after preprocessing.")

    print(f"[info] Training classifier with {len(coords)} samples ({len(pos_coords_final)} positive, {len(neg_coords_final)} negative)")

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
                )
            except Exception as exc:
                print(f"[warn] Failed to generate debug visualisations: {exc}")


    # Split data
    Xtr, Xval, ytr, yval = train_test_split(coords, labels, test_size=args.test_ratio, stratify=labels, random_state=args.random_seed)

    ds_tr = LabeledPatches(stack, Xtr, ytr, window=window_size)
    ds_va = LabeledPatches(stack, Xval, yval, window=window_size)
    worker_count = 0 if getattr(stack, 'kind', None) == 'raster' else 8
    if worker_count == 0:
        print('[info] Using single-process data loading for raster stack')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=worker_count)

    encoder = MAEViT(in_chans=stack.count, **mae_kwargs)
    encoder.load_state_dict(state_dict)
    in_dim = encoder.blocks[0].attn.embed_dim

    # TODO: Add hyperparameters for probability, hidden dims...
    mlp = MLPDropout(in_dim=in_dim)

    # Train classifier
    mlp = train_classifier(encoder, mlp, dl_tr, dl_va, epochs=args.epochs)
    out_dir = Path(args.out) if args.out else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'mlp_classifier.pth'
    torch.save(mlp.state_dict(), out_path)
    print(f"[info] Saved classifier to {out_path}")

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
    else:
        pos_coord_map = group_positive_coords(pos_coords_final, stack)
        write_prediction_outputs(
            prediction,
            stack,
            out_dir,
            pos_coords_by_region=pos_coord_map,
        )
