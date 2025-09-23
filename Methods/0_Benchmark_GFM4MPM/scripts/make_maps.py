# scripts/make_maps.py
import argparse, glob, torch
from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.models.mlp_dropout import MLPDropout
from gfm4mpm.infer.infer_maps import mc_predict_map, save_geotiff

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mlp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--passes', type=int, default=30)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    enc = MAEViT(in_chans=stack.count)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)
    mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    mean_map, std_map = mc_predict_map(enc, mlp, stack, patch_size=args.patch, stride=args.stride, passes=args.passes)

    # save using first band as ref
    save_geotiff(args.out + '_mean.tif', mean_map, stack.srcs[0])
    save_geotiff(args.out + '_std.tif', std_map, stack.srcs[0])
    print(f"Wrote {args.out}_mean.tif and {args.out}_std.tif")
