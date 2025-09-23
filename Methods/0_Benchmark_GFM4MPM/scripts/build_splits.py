# scripts/build_splits.py
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from gfm4mpm.data.geo_stack import GeoStack, load_deposit_pixels
from gfm4mpm.data.stac_table import StacTableStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.sampling.likely_negatives import compute_embeddings, pu_select_negatives

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
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands/--pos_geojson or one of --stac-root/--stac-table (exactly one input source)')

    if args.stac_root or args.stac_table:
        stack = StacTableStack(
            Path(args.stac_table or args.stac_root),
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
        stack = GeoStack(sorted(glob.glob(args.bands)))
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

    X_all = np.stack([stack.read_patch(r, c, patch) for (r, c) in coords_all], axis=0)

    encoder = MAEViT(in_chans=stack.count, patch_size=patch)
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    Z_all = compute_embeddings(encoder, X_all)

    pos_idx = np.arange(0, len(pos))
    unk_idx = np.arange(len(pos), len(coords_all))

    neg_idx = pu_select_negatives(Z_all, pos_idx, unk_idx, args.filter_top_pct, args.negs_per_pos)

    splits = {
        'pos': [list(map(int, rc)) for rc in pos],
        'neg': [list(map(int, coords_all[i])) for i in neg_idx.tolist()]
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out) / 'splits.json'
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(splits['pos'])} positives and {len(splits['neg'])} negatives to {out_path}")
