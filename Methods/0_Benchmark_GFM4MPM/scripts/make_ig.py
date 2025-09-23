# scripts/make_ig.py
import argparse, glob, torch
from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.models.mlp_dropout import MLPDropout
from gfm4mpm.explain.ig_maps import ig_attribution_map

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mlp', required=True)
    ap.add_argument('--row', type=int, required=True)
    ap.add_argument('--col', type=int, required=True)
    ap.add_argument('--patch', type=int, default=32)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    enc = MAEViT(in_chans=stack.count); enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim); mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    x = stack.read_patch(args.row, args.col, args.patch)
    x = torch.from_numpy(x[None])
    attrs = ig_attribution_map(enc, mlp, x)
    torch.save(attrs, 'ig_attrs.pt')
    print("Saved IG attributions to ig_attrs.pt")
