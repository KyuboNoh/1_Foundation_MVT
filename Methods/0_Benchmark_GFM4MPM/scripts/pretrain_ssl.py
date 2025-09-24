# scripts/pretrain_ssl.py
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.gfm4mpm.data.geo_stack import GeoStack
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.training.train_ssl import train_ssl


class SSLDataset(Dataset):
    def __init__(self, stack, patch=32, n_samples=200000, seed=1337):
        self.stack, self.patch = stack, patch
        self.rng = np.random.default_rng(seed)
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if hasattr(self.stack, "random_coord"):
            r, c = self.stack.random_coord(self.patch, self.rng)
        else:
            r = self.rng.integers(self.patch // 2, self.stack.height - self.patch // 2)
            c = self.rng.integers(self.patch // 2, self.stack.width - self.patch // 2)
        x = self.stack.read_patch(r, c, self.patch)
        return torch.from_numpy(x)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', type=str, help='glob pattern to bands (e.g., /data/*.tif)')
    ap.add_argument('--stac-root', type=str, help='Path to STAC collection root (table assets)')
    ap.add_argument('--stac-table', type=str, help='Direct path to a STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC tables')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC tables')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC tables')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--patch', type=int, default=16, help='Patch/window size for raster sampling (ignored for STAC tables)')
    ap.add_argument('--mask-ratio', type=float, default=0.75, help='Fraction of patches masked during MAE pretraining')
    ap.add_argument('--encoder-depth', type=int, default=6, help='Number of transformer blocks in the encoder')
    ap.add_argument('--decoder-depth', type=int, default=2, help='Number of transformer blocks in the decoder')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--optimizer', choices=['adamw', 'adam'], default='adamw')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--preview-samples', type=int, default=0, help='If >0, create reconstruction previews for this many samples')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide exactly one of --bands, --stac-root, or --stac-table')

    if args.stac_root or args.stac_table:
        stac_path = Path(args.stac_table or args.stac_root)
        stack = StacTableStack(
            stac_path,
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        patch_size = 1
        if args.patch != 1:
            print("[info] STAC table detected; overriding patch size to 1 for pseudo-patches")
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        patch_size = args.patch

    ds = SSLDataset(stack, patch=patch_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    model = MAEViT(
        in_chans=stack.count,
        patch_size=patch_size,
        depth=args.encoder_depth,
        dec_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
    )
    preview_dir = Path(args.out) / 'previews'
    model, history = train_ssl(
        model,
        dl,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        preview_samples=args.preview_samples,
        preview_dir=preview_dir if args.preview_samples > 0 else None,
        feature_names=getattr(stack, 'feature_columns', None),
    )
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, 'mae_encoder.pth'))
    history_path = Path(args.out) / 'ssl_history.json'
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f"Saved training history to {history_path}")
