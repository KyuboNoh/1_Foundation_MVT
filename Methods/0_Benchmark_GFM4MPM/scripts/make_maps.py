# scripts/make_maps.py
import argparse
import csv
import glob
from pathlib import Path
import torch

from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.data.stac_table import StacTableStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.models.mlp_dropout import MLPDropout
from gfm4mpm.infer.infer_maps import mc_predict_map, save_geotiff

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC table workflow')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC table workflow')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC table workflow')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mlp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--passes', type=int, default=30)
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands or one of --stac-root/--stac-table (exactly one input source)')

    if args.stac_root or args.stac_table:
        stack = StacTableStack(
            Path(args.stac_table or args.stac_root),
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        patch = 1
        stride = 1
        if args.patch != 1:
            print('[info] STAC table detected; overriding patch size to 1')
        if args.stride != 1:
            print('[info] STAC table detected; overriding stride to 1')
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        patch = args.patch
        stride = args.stride

    enc = MAEViT(in_chans=stack.count, patch_size=patch)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)
    mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    mean_map, std_map = mc_predict_map(enc, mlp, stack, patch_size=patch, stride=stride, passes=args.passes)

    if getattr(stack, 'is_table', False):
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
        save_geotiff(args.out + '_mean.tif', mean_map, stack.srcs[0])
        save_geotiff(args.out + '_std.tif', std_map, stack.srcs[0])
        print(f"Wrote {args.out}_mean.tif and {args.out}_std.tif")
