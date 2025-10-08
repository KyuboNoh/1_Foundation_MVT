# scripts/build_splits.py
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.gfm4mpm.data.geo_stack import GeoStack, load_deposit_pixels
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.sampling.likely_negatives import compute_embeddings, pu_select_negatives


def _resolve_search_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved


def _load_training_args(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates = []
    root = _resolve_search_root(stack_root)
    if root is not None:
        direct = root / 'training_args.json'
        candidates.append(direct)
        try:
            for candidate in root.rglob('training_args.json'):
                candidates.append(candidate)
        except Exception:
            pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_args.json')

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
            return data, resolved
        except Exception as exc:
            print(f"[warn] Failed to load training args from {resolved}: {exc}")
    return None, None


def _mae_kwargs_from_training_args(args_dict: Dict) -> Dict:
    mapping = {
        'encoder_dim': 'embed_dim',
        'encoder_depth': 'depth',
        'encoder_num_heads': 'encoder_num_heads',
        'mlp_ratio': 'mlp_ratio',
        'mlp_ratio_decoder': 'mlp_ratio_dec',
        'decoder_dim': 'dec_dim',
        'decoder_depth': 'dec_depth',
        'decoder_num_heads': 'decoder_num_heads',
        'mask_ratio': 'mask_ratio',
        'patch': 'patch_size',
    }
    mae_kwargs: Dict = {}
    for src_key, dest_key in mapping.items():
        if src_key not in args_dict or args_dict[src_key] is None:
            continue
        value = args_dict[src_key]
        if dest_key in {
            'embed_dim',
            'depth',
            'encoder_num_heads',
            'dec_dim',
            'dec_depth',
            'decoder_num_heads',
            'patch_size',
        }:
            value = int(value)
        elif dest_key in {'mlp_ratio', 'mlp_ratio_dec', 'mask_ratio'}:
            value = float(value)
        mae_kwargs[dest_key] = value
    window = args_dict.get('window')
    if window is not None:
        mae_kwargs['image_size'] = int(window)
    return mae_kwargs


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        return np.pad(embeddings, ((0, 0), (0, max(0, 2 - embeddings.shape[1]))), mode='constant')[:, :2]
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:2].T
    except np.linalg.LinAlgError:
        components = np.eye(embeddings.shape[1], 2)
    return centered @ components


def _save_validation_plot(
    out_dir: Path,
    projection: np.ndarray,
    pos_idx: np.ndarray,
    unk_idx: np.ndarray,
    neg_idx: np.ndarray,
    distances: np.ndarray,
    keep_mask: np.ndarray,
    cutoff: Optional[float],
    filter_top_pct: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Validation plot skipped; matplotlib unavailable: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    proj_pos = projection[pos_idx]
    proj_unknown = projection[unk_idx]
    neg_mask = np.isin(unk_idx, neg_idx)
    proj_neg = proj_unknown[neg_mask]
    proj_unknown_kept = proj_unknown[~neg_mask & keep_mask]
    proj_unknown_filtered = proj_unknown[~keep_mask]

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    if len(proj_unknown_filtered):
        ax_scatter.scatter(
            proj_unknown_filtered[:, 0],
            proj_unknown_filtered[:, 1],
            facecolors='none',
            edgecolors='grey',
            linewidths=0.5,
            s=15,
            alpha=0.4,
            label='Unknown filtered',
        )

    if len(proj_unknown_kept):
        ax_scatter.scatter(
            proj_unknown_kept[:, 0],
            proj_unknown_kept[:, 1],
            facecolors='none',
            edgecolors='black',
            linewidths=0.5,
            s=20,
            label='Unknown kept',
        )

    if len(proj_neg):
        ax_scatter.scatter(
            proj_neg[:, 0],
            proj_neg[:, 1],
            c='#1f77b4',
            s=25,
            alpha=0.85,
            label='Selected negatives',
        )

    if len(proj_pos):
        ax_scatter.scatter(
            proj_pos[:, 0],
            proj_pos[:, 1],
            c='#d62728',
            s=28,
            alpha=0.85,
            label='Positives',
        )

    ax_scatter.set_xlabel('PC 1')
    ax_scatter.set_ylabel('PC 2')
    ax_scatter.set_title('Embedding projection')
    ax_scatter.legend(loc='best')

    if len(distances):
        bins = max(20, int(np.sqrt(len(distances))))
        ax_hist.hist(
            distances[keep_mask],
            bins=bins,
            color='#1f77b4',
            alpha=0.6,
            label='Distances (kept)',
        )
        ax_hist.hist(
            distances[~keep_mask],
            bins=bins,
            color='grey',
            alpha=0.4,
            label='Distances (filtered)',
        )
        if cutoff is not None:
            ax_hist.axvline(cutoff, color='#d62728', linestyle='--', linewidth=1.5, label=f'Cutoff ({filter_top_pct*100:.1f}%)')
        ax_hist.set_xlabel('Min distance to positives')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Distance distribution')
        ax_hist.legend(loc='best')

    fig.tight_layout()
    out_path = out_dir / 'validation_embeddings.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[info] Wrote validation plot to {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--pos_geojson', help='positive deposits (GeoJSON) when using rasters')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--label-column', default='Training_MVT_Deposit', help='label column for STAC table workflow')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC table workflow')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC table workflow')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC table workflow')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--filter_top_pct', type=float, default=0.10)
    ap.add_argument('--negs_per_pos', type=int, default=5)
    ap.add_argument('--validation', action='store_true', help='Generate diagnostic plots for PU negative selection')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands/--pos_geojson or one of --stac-root/--stac-table (exactly one input source)')

    stack_root_path: Optional[Path] = None

    if args.stac_root or args.stac_table:
        stack_source = Path(args.stac_table or args.stac_root).resolve()
        stack_root_path = stack_source.parent if stack_source.is_file() else stack_source
        stack = StacTableStack(
            stack_source,
            label_columns=[args.label_column],
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        labels = stack.label_array(args.label_column)
        pos_indices = np.where(labels == 1)[0]
        unk_indices = np.where(labels == 0)[0]
        pos = [stack.index_to_coord(int(i)) for i in pos_indices]
        unk = [stack.index_to_coord(int(i)) for i in unk_indices]
        patch = 1
        if args.patch != 1:
            print('[info] STAC table detected; overriding patch size to 1')
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        if not args.pos_geojson:
            ap.error('--pos_geojson is required when using --bands')
        band_paths = sorted(glob.glob(args.bands))
        if not band_paths:
            raise RuntimeError(f"No raster bands matched pattern {args.bands}")
        stack = GeoStack(band_paths)
        try:
            common = os.path.commonpath(band_paths)
            stack_root_path = Path(common)
        except ValueError:
            stack_root_path = Path(band_paths[0]).resolve().parent
        pos = load_deposit_pixels(args.pos_geojson, stack)
        grid = set(stack.grid_centers(stride=args.patch))
        pos_set = set(pos)
        unk = list(grid - pos_set)
        patch = args.patch

    coords_all = pos + unk
    if not coords_all:
        raise RuntimeError('No coordinates collected to compute embeddings')
    if not pos:
        raise RuntimeError('No positive samples found; check label column or source data')

    encoder_path = Path(args.encoder).resolve()
    training_args_data, training_args_path = _load_training_args(stack_root_path, encoder_path)
    mae_init_kwargs: Dict = {}
    if training_args_data:
        mae_init_kwargs = _mae_kwargs_from_training_args(training_args_data)
        if 'patch_size' not in mae_init_kwargs:
            mae_init_kwargs['patch_size'] = patch
        patch = int(training_args_data.get('window', patch))
        print(f"[info] Loaded training_args.json from {training_args_path}")
        if args.bands and patch != args.patch:
            grid = set(stack.grid_centers(stride=patch))
            pos_set = set(pos)
            unk = list(grid - pos_set)
            coords_all = pos + unk
            if not coords_all:
                raise RuntimeError('Recomputed coordinates after adjusting window size resulted in an empty set')
    else:
        mae_init_kwargs['patch_size'] = patch
        print('[warn] training_args.json not found; falling back to default MAE configuration')

    X_all = np.stack([stack.read_patch(r, c, patch) for (r, c) in coords_all], axis=0)

    encoder = MAEViT(in_chans=stack.count, **mae_init_kwargs)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    Z_all = compute_embeddings(encoder, X_all)

    pos_idx = np.arange(0, len(pos))
    unk_idx = np.arange(len(pos), len(coords_all))

    if args.validation:
        neg_idx, debug_info = pu_select_negatives(
            Z_all,
            pos_idx,
            unk_idx,
            args.filter_top_pct,
            args.negs_per_pos,
            return_info=True,
        )
    else:
        neg_idx = pu_select_negatives(
            Z_all,
            pos_idx,
            unk_idx,
            args.filter_top_pct,
            args.negs_per_pos,
        )
        debug_info = None

    splits = {
        'pos': [list(map(int, rc)) for rc in pos],
        'neg': [list(map(int, coords_all[i])) for i in neg_idx.tolist()]
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out) / 'splits.json'
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(splits['pos'])} positives and {len(splits['neg'])} negatives to {out_path}")

    if args.validation and debug_info is not None:
        diag_dir = Path(args.out) / 'validation'
        projection = _project_embeddings(Z_all)
        distances = np.asarray(debug_info.get('distances'))
        keep_mask = np.asarray(debug_info.get('keep_mask'), dtype=bool)
        cutoff = debug_info.get('cutoff')
        _save_validation_plot(
            diag_dir,
            projection,
            pos_idx,
            unk_idx,
            np.asarray(neg_idx, dtype=int),
            distances,
            keep_mask,
            cutoff,
            args.filter_top_pct,
        )
