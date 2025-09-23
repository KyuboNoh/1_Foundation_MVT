# scripts/eval_ablation_sparse.py
# Drop 50% of channels at test time to measure robustness
import argparse, glob, json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from gfm4mpm.data.geo_stack import GeoStack
from gfm4mpm.models.mae_vit import MAEViT
from gfm4mpm.models.mlp_dropout import MLPDropout
from gfm4mpm.training.train_cls import eval_classifier

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
    ap.add_argument('--bands', required=True)
    ap.add_argument('--val_json', required=True, help='JSON with {"coords": [[r,c],...], "labels":[0/1,...]}')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--mlp', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--drop', type=float, default=0.5)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    with open(args.val_json) as f:
        val = json.load(f)
    coords = [tuple(x) for x in val['coords']]
    labels = val['labels']

    base_ds = LabeledPatches(stack, coords, labels, patch=args.patch)
    ds = SparseWrapper(base_ds, drop_prob=args.drop)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=8)

    enc = MAEViT(in_chans=stack.count)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = enc.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)
    mlp.load_state_dict(torch.load(args.mlp, map_location='cpu'))

    f1, mcc, auprc, auroc = eval_classifier(enc, mlp, dl)
    print(f"[SPARSE DROP {args.drop}] f1={f1:.3f} mcc={mcc:.3f} auprc={auprc:.3f} auroc={auroc:.3f}")
