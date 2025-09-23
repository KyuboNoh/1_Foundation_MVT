# scripts/build_splits.py
import argparse, glob, os, json
import numpy as np
import torch
from gfm4mpm.data.geo_stack import GeoStack, load_deposit_pixels
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.sampling.likely_negatives import compute_embeddings, pu_select_negatives

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', required=True)
    ap.add_argument('--pos_geojson', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--filter_top_pct', type=float, default=0.10)
    ap.add_argument('--negs_per_pos', type=int, default=5)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    pos = load_deposit_pixels(args.pos_geojson, stack)
    # unknowns: uniform grid centers minus positives
    grid = set(stack.grid_centers(stride=args.patch))
    pos_set = set(pos)
    unk = list(grid - pos_set)

    # embeddings (encode pos+unk patches)
    coords_all = pos + unk
    X_all = np.stack([stack.read_patch(r,c,args.patch) for (r,c) in coords_all], axis=0)

    encoder = MAEViT(in_chans=stack.count)
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    Z_all = compute_embeddings(encoder, X_all)

    pos_idx = np.arange(0, len(pos))
    unk_idx = np.arange(len(pos), len(pos)+len(unk))

    neg_idx = pu_select_negatives(Z_all, pos_idx, unk_idx, args.filter_top_pct, args.negs_per_pos)

    splits = {
        'pos': [list(map(int, rc)) for rc in pos],
        'neg': [list(map(int, coords_all[i])) for i in neg_idx.tolist()]
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(splits['pos'])} positives and {len(splits['neg'])} negatives to {os.path.join(args.out, 'splits.json')}")
