# scripts/train_classifier.py
import argparse
import glob
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.data.stac_table import StacTableStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.models.mlp_dropout import MLPDropout
from gfm4mpm.training.train_cls import train_classifier

class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, patch=32):
        self.stack, self.coords, self.labels, self.patch = stack, coords, labels, patch
    def __len__(self): return len(self.coords)
    def __getitem__(self, idx):
        r,c = self.coords[idx]
        x = self.stack.read_patch(r,c,self.patch)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC table workflow')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC table workflow')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC table workflow')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
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
        if args.patch != 1:
            print('[info] STAC table detected; overriding patch size to 1')
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        patch = args.patch

    with open(args.splits) as f: sp = json.load(f)
    coords = [tuple(x) for x in sp['pos']] + [tuple(x) for x in sp['neg']]
    labels = [1]*len(sp['pos']) + [0]*len(sp['neg'])

    Xtr, Xval, ytr, yval = train_test_split(coords, labels, test_size=0.2, stratify=labels, random_state=42)

    ds_tr = LabeledPatches(stack, Xtr, ytr, patch=patch)
    ds_va = LabeledPatches(stack, Xval, yval, patch=patch)
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=8)
    dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, num_workers=8)

    encoder = MAEViT(in_chans=stack.count, patch_size=patch)
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = encoder.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)

    mlp = train_classifier(encoder, mlp, dl_tr, dl_va, epochs=args.epochs)
    torch.save(mlp.state_dict(), 'mlp_classifier.pth')
    print("Saved classifier to mlp_classifier.pth")
