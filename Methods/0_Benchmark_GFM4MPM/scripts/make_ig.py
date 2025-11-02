# scripts/make_ig.py
import argparse
import glob
import csv
from pathlib import Path
import torch

from Common.cls.data.geo_stack import GeoStack
from Common.cls.data.stac_table import StacTableStack
from Common.cls.models.mae_vit import MAEViT
from Common.cls.models.mlp_dropout import MLPDropout
from Common.cls.explain.ig_maps import ig_attribution_map

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
    ap.add_argument('--row', type=int, help='row coordinate (rasters only)')
    ap.add_argument('--col', type=int, help='column coordinate (rasters only)')
    ap.add_argument('--index', type=int, help='row index for STAC table entries')
    ap.add_argument('--patch', type=int, default=32)
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands or one of --stac-root/--stac-table (exactly one input source)')

    if args.stac_root or args.stac_table:
        if args.index is None:
            ap.error('--index is required when using --stac-root/--stac-table')
        stack = StacTableStack(
            Path(args.stac_table or args.stac_root),
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        coord = stack.index_to_coord(args.index)
        patch = 1
        if args.patch != 1:
            print('[info] STAC table detected; overriding patch size to 1')
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        if args.row is None or args.col is None:
            ap.error('--row and --col are required when using --bands')
        stack = GeoStack(sorted(glob.glob(args.bands)))
        coord = (args.row, args.col)
        patch = args.patch

    enc = MAEViT(in_chans=stack.count, patch_size=patch)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)
    mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    x = stack.read_patch(coord[0], coord[1], patch)
    x = torch.from_numpy(x[None])
    attrs = ig_attribution_map(enc, mlp, x)
    torch.save(attrs, 'ig_attrs.pt')
    print("Saved IG attributions to ig_attrs.pt")

    if getattr(stack, 'is_table', False):
        flat = attrs[0].reshape(len(stack.feature_columns), -1).mean(axis=1)
        out_csv = Path('ig_attrs_table.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'attribution'])
            for name, val in zip(stack.feature_columns, flat.tolist()):
                writer.writerow([name, float(val)])
        print(f"Saved per-feature IG attributions to {out_csv}")
