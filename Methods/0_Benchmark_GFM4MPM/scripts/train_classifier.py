# scripts/train_classifier.py
import argparse, glob, json, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gfm4mpm.data.geo_stack import GeoStack
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
    ap.add_argument('--bands', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
    args = ap.parse_args()

    stack = GeoStack(sorted(glob.glob(args.bands)))
    with open(args.splits) as f: sp = json.load(f)
    coords = [tuple(x) for x in sp['pos']] + [tuple(x) for x in sp['neg']]
    labels = [1]*len(sp['pos']) + [0]*len(sp['neg'])

    Xtr, Xval, ytr, yval = train_test_split(coords, labels, test_size=0.2, stratify=labels, random_state=42)

    ds_tr = LabeledPatches(stack, Xtr, ytr, patch=args.patch)
    ds_va = LabeledPatches(stack, Xval, yval, patch=args.patch)
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=8)
    dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, num_workers=8)

    encoder = MAEViT(in_chans=stack.count)
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    in_dim = encoder.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)

    mlp = train_classifier(encoder, mlp, dl_tr, dl_va, epochs=args.epochs)
    torch.save(mlp.state_dict(), 'mlp_classifier.pth')
    print("Saved classifier to mlp_classifier.pth")
