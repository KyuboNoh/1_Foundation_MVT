# scripts/eval_ablation_sparse.py
# Drop 50% of channels at test time to measure robustness
import argparse
import glob
import json
import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from Common.cls.data.geo_stack import GeoStack
from Common.cls.data.stac_table import StacTableStack
from Common.cls.models.mae_vit import MAEViT
from Common.cls.models.mlp_dropout import MLPDropout
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.cls.training.train_cls import eval_classifier
from Common.metrics_logger import DEFAULT_METRIC_ORDER, log_metrics

class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, patch=32):
        self.stack, self.coords, self.labels, self.patch = stack, coords, labels, patch
    def __len__(self): return len(self.coords)
    def __getitem__(self, idx):
        r,c = self.coords[idx]
        x = self.stack.read_patch(r,c,self.patch)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class SparseWrapper(Dataset):
    def __init__(self, base, drop_prob=0.5, seed=1337):
        self.base, self.p, self.rng = base, drop_prob, np.random.default_rng(seed)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        mask = (self.rng.random(x.shape[0]) > self.p).astype(np.float32)
        x = x * torch.from_numpy(mask)[:,None,None]
        return x, y

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC table workflow')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC table workflow')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC table workflow')
    ap.add_argument('--val_json', required=True, help='JSON with {"coords": [[r,c],...], "labels":[0/1,...]}')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mlp', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--drop', type=float, default=0.5)
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

    with open(args.val_json) as f:
        val = json.load(f)
    coords = [tuple(x) for x in val['coords']]
    labels = val['labels']

    base_ds = LabeledPatches(stack, coords, labels, patch=patch)
    ds = SparseWrapper(base_ds, drop_prob=args.drop)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=8)

    enc = MAEViT(in_chans=stack.count, patch_size=patch)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)
    mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    metrics = eval_classifier(enc, mlp, dl)
    log_metrics(f"sparse drop {args.drop}", metrics, order=DEFAULT_METRIC_ORDER)
