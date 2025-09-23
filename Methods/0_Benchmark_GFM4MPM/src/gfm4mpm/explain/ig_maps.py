# src/gfm4mpm/explain/ig_maps.py
import torch
from captum.attr import IntegratedGradients

@torch.no_grad()
def ig_attribution_map(encoder, mlp, x_batch, baselines=None, device='cuda'):
    encoder.eval().to(device)
    mlp.eval().to(device)
    if baselines is None:
        baselines = torch.zeros_like(x_batch)
    def fwd(inp):
        z = encoder.encode(inp)
        return mlp(z)
    ig = IntegratedGradients(fwd)
    attrs = ig.attribute(x_batch.to(device), baselines=baselines.to(device), target=None, n_steps=32)
    return attrs.cpu()
