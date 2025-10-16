# src/gfm4mpm/sampling/likely_negatives.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

@torch.no_grad()
def compute_embeddings(encoder, X, batch_size=1024, device='cuda', show_progress=False):
    encoder.eval().to(device)
    Z = []
    iterator = range(0, len(X), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embeddings")
    for i in iterator:
        x = torch.from_numpy(X[i:i+batch_size]).to(device)
        z = encoder.encode(x).cpu()
        Z.append(z)
    return torch.cat(Z).numpy()

def pu_select_negatives(
    Z_all,
    pos_idx,
    unk_idx,
    filter_top_pct=0.5,
    negatives_per_pos=5,
    rng=None,
    *,
    return_info: bool = False,
):
    """Select negatives from unknowns, filtering the top-similar percent to positives.

    When ``return_info`` is ``True`` an auxiliary dictionary containing the
    distance-to-positive scores and masks used during filtering is returned
    alongside the sampled negative indices.
    """

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
    cutoff = None
    if k > 0:
        cutoff = float(np.partition(dmin, k)[:k].max())
        keep_mask &= dmin > cutoff
    kept_unk = np.array(unk_idx)[keep_mask]

    # sample negatives
    n_neg = min(len(pos_idx) * negatives_per_pos, len(kept_unk))
    neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    if return_info:
        info = {
            "distances": dmin,
            "kept_unknown_indices": kept_unk,
            "cutoff": cutoff,
            "keep_mask": keep_mask,
        }
        return neg_idx, info
    return neg_idx
