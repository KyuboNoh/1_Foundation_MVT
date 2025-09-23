# src/gfm4mpm/sampling/likely_negatives.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

@torch.no_grad()
def compute_embeddings(encoder, X, batch_size=1024, device='cuda'):
    encoder.eval().to(device)
    Z = []
    for i in tqdm(range(0, len(X), batch_size), desc="Embeddings"):
        x = torch.from_numpy(X[i:i+batch_size]).to(device)
        z = encoder.encode(x).cpu()
        Z.append(z)
    return torch.cat(Z).numpy()

def pu_select_negatives(Z_all, pos_idx, unk_idx, filter_top_pct=0.1, negatives_per_pos=5, rng=None):
    """Select negatives from unknowns, filtering the top‑similar percent to positives."""
    if rng is None:
        rng = np.random.default_rng(1337)
    Zp = Z_all[pos_idx]
    Zu = Z_all[unk_idx]
    # Compute min distance to any positive for each unknown
    dists = []
    step = 4096
    for i in range(0, len(Zu), step):
        chunk = Zu[i:i+step]
        dm = ((chunk[:,None,:] - Zp[None,:,:])**2).sum(-1)**0.5
        dmin = dm.min(axis=1)
        dists.append(dmin)
    dmin = np.concatenate(dists)
    # filter top‑similar (smallest distance)
    k = int(len(dmin) * filter_top_pct)
    keep_mask = np.ones_like(dmin, dtype=bool)
    if k > 0:
        cutoff = np.partition(dmin, k)[:k].max()
        keep_mask &= dmin > cutoff
    kept_unk = np.array(unk_idx)[keep_mask]
    # sample negatives
    n_neg = min(len(pos_idx) * negatives_per_pos, len(kept_unk))
    neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    return neg_idx
