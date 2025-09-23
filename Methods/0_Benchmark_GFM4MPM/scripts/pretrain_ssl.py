# scripts/pretrain_ssl.py
import argparse, os, glob, torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.training.train_ssl import train_ssl

class SSLDataset(Dataset):
    def __init__(self, stack: GeoStack, patch=32, n_samples=200000, seed=1337):
        self.stack, self.patch = stack, patch
        self.rng = np.random.default_rng(seed)
        self.n = n_samples
    def __len__(self): return self.n
    def __getitem__(self, idx):
        r = self.rng.integers(self.patch//2, self.stack.height - self.patch//2)
        c = self.rng.integers(self.patch//2, self.stack.width  - self.patch//2)
        x = self.stack.read_patch(r, c, self.patch)
        return torch.from_numpy(x)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', type=str, required=True, help='glob pattern to bands (e.g., /data/*.tif)')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    ds = SSLDataset(stack, patch=args.patch)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    model = MAEViT(in_chans=stack.count)
    model = train_ssl(model, dl, epochs=args.epochs)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, 'mae_encoder.pth'))
