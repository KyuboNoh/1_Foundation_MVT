# src/gfm4mpm/infer/infer_maps.py
import numpy as np
import torch
import torch.nn as nn
import rasterio

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _mc_predict_single_region(
    encoder,
    mlp,
    stack,
    patch_size: int,
    stride: int,
    passes: int,
    device: str,
    show_progress: bool,
    region_name: str = "",
):
    encoder.eval().to(device)
    mlp.eval().to(device)
    for module in mlp.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    H, W = stack.height, stack.width
    mean_map = np.zeros((H, W), dtype=np.float32)
    var_map  = np.zeros((H, W), dtype=np.float32)
    counts   = np.zeros((H, W), dtype=np.int32)
    centers = stack.grid_centers(stride)
    iterator = centers
    if show_progress and tqdm is not None:
        desc = "MC inference"
        if region_name:
            desc = f"{desc} [{region_name}]"
        iterator = tqdm(centers, desc=desc)
    elif show_progress and tqdm is None:
        print("[warn] tqdm not installed; progress bar disabled.")
    for r, c in iterator:
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


@torch.no_grad()
def mc_predict_map(encoder, mlp, stack, patch_size=32, stride=16, passes=30, device=None, show_progress=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hasattr(stack, "resolve_region_stack") and hasattr(stack, "iter_region_stacks"):
        region_results = {}
        for region_name, region_stack in stack.iter_region_stacks():
            region_mean, region_std = _mc_predict_single_region(
                encoder,
                mlp,
                region_stack,
                patch_size=patch_size,
                stride=stride,
                passes=passes,
                device=device,
                show_progress=show_progress,
                region_name=str(region_name),
            )
            region_results[str(region_name)] = {
                "mean": region_mean,
                "std": region_std,
                "stack": region_stack,
            }
        return region_results

    mean_map, std_map = _mc_predict_single_region(
        encoder,
        mlp,
        stack,
        patch_size=patch_size,
        stride=stride,
        passes=passes,
        device=device,
        show_progress=show_progress,
    )
    return mean_map, std_map

def save_geotiff(path, array, ref_src):
    profile = ref_src.profile.copy()
    profile.update(count=1, dtype='float32')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)
