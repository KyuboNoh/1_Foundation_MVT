# src/gfm4mpm/infer/infer_maps.py
import numpy as np
import torch
import rasterio

@torch.no_grad()
def mc_predict_map(encoder, mlp, stack, patch_size=32, stride=16, passes=30, device='cuda'):
    encoder.eval().to(device)
    mlp.train().to(device)  # keep dropout ON for MC
    H, W = stack.height, stack.width
    mean_map = np.zeros((H, W), dtype=np.float32)
    var_map  = np.zeros((H, W), dtype=np.float32)
    counts   = np.zeros((H, W), dtype=np.int32)
    centers = stack.grid_centers(stride)
    for r,c in centers:
        x = stack.read_patch(r, c, patch_size)
        x = torch.from_numpy(x[None]).to(device)
        zs = encoder.encode(x)
        preds = []
        for _ in range(passes):
            p = mlp(zs)
            preds.append(p.item())
        mu = float(np.mean(preds)); sig2 = float(np.var(preds))
        r0, c0 = r - patch_size//2, c - patch_size//2
        r1, c1 = r0 + patch_size, c0 + patch_size
        r0, c0 = max(r0,0), max(c0,0)
        r1, c1 = min(r1,H), min(c1,W)
        mean_map[r0:r1, c0:c1] += mu
        var_map[r0:r1, c0:c1]  += sig2
        counts[r0:r1, c0:c1]   += 1
    mean_map /= np.maximum(counts, 1)
    var_map  /= np.maximum(counts, 1)
    return mean_map, np.sqrt(var_map)

def save_geotiff(path, array, ref_src):
    profile = ref_src.profile.copy()
    profile.update(count=1, dtype='float32')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)
