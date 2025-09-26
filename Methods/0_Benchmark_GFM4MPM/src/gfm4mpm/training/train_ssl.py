# src/gfm4mpm/training/train_ssl.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm


_LOG = logging.getLogger(__name__)


class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target, reduction="mean")

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape != pred.shape:
            raise ValueError(
                "Mask shape must match predictions; received "
                f"{mask.shape} vs {pred.shape}"
            )

        mask = mask.to(dtype=pred.dtype)
        weighted_sq = (pred - target).pow(2) * mask
        denom = mask.sum()
        if denom <= 0:
            return F.mse_loss(pred, target, reduction="mean")
        return weighted_sq.sum() / denom


def _select_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95))
    raise ValueError(f"Unknown optimizer '{name}'")


def _plot_samples(
    inputs: torch.Tensor,
    masked: torch.Tensor,
    recon: torch.Tensor,
    feature_names: Optional[Iterable[str]],
    feature_metadata: Optional[Iterable[Optional[Dict[str, Optional[str]]]]],
    out_dir: Path,
    prefix: str,
    masks: Optional[torch.Tensor] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    B, C, H, W = inputs.shape
    feat_names = list(feature_names) if feature_names is not None else [f"ch_{i}" for i in range(C)]
    meta = list(feature_metadata) if feature_metadata is not None else [None] * C
    is_scalar = H == 1 and W == 1

    base_cmap = plt.cm.get_cmap("viridis")
    masked_cmap = base_cmap.with_extremes(bad="black")

    def _shared_minmax(tensors: Iterable[torch.Tensor | np.ndarray]) -> tuple[float, float]:
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        for tensor in tensors:
            arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            mn = float(finite.min())
            mx = float(finite.max())
            vmin = mn if vmin is None else min(vmin, mn)
            vmax = mx if vmax is None else max(vmax, mx)
        if vmin is None or vmax is None:
            return 0.0, 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            eps = max(abs(vmin), abs(vmax), 1.0) * 0.01 + 1e-6
            return vmin - eps, vmax + eps
        return vmin, vmax

    def _mask_for(sample_idx: int) -> Optional[np.ndarray]:
        if masks is None:
            return None
        mask_tensor = masks[sample_idx]
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_tensor)
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim < 2:
            return None
        return mask_np.astype(bool)

    for idx in range(B):
        if is_scalar:
            # For table data (1x1 "images"), split previews into numeric vs categorical channels.
            arrays = [t[idx].cpu().view(-1) for t in (inputs, masked, recon)]
            numeric_idx = [i for i, info in enumerate(meta) if not info or not info.get("category")]
            categorical_idx = [i for i, info in enumerate(meta) if info and info.get("category")]

            def _render(indices: List[int], suffix: str, color: str) -> None:
                if not indices:
                    return
                top_k = min(32, len(indices))
                orig = arrays[0][indices]
                order = torch.argsort(orig.abs(), descending=True)[:top_k]
                selected = [indices[int(i)] for i in order]
                labels = [feat_names[i] if i < len(feat_names) else f"ch_{i}" for i in selected]

                fig, axes = plt.subplots(1, 3, figsize=(max(12, top_k * 0.4), 5), sharey=True)
                for ax, data, title in zip(axes, arrays, ("original", "masked", "recon")):
                    vals = data[selected].numpy()
                    ax.bar(range(len(selected)), vals, color=color)
                    ax.set_xticks(range(len(selected)))
                    ax.set_xticklabels(labels, rotation=90)
                    ax.set_title(title)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                fig.suptitle(f"{suffix.title()} features")
                fig.tight_layout()
                fig.savefig(out_dir / f"{prefix}_{suffix}_{idx:03d}.png", dpi=200)
                plt.close(fig)

            _render(numeric_idx, "numeric", "#1f77b4")
            _render(categorical_idx, "categorical", "#ff7f0e")
        else:
            fig, axes = plt.subplots(C, 3, figsize=(9, 3 * C))
            mask_slice = _mask_for(idx)
            for c in range(C):
                name = feat_names[c] if c < len(feat_names) else f"ch_{c}"
                channel_views = [inputs[idx, c], masked[idx, c], recon[idx, c]]
                vmin, vmax = _shared_minmax(channel_views)
                for ax, data, title in zip(
                    (axes[c, j] if C > 1 else axes[j] for j in range(3)),
                    channel_views,
                    ("original", "masked", "recon"),
                ):
                    array = data.detach().cpu().numpy()
                    cmap = base_cmap
                    if title == "masked" and mask_slice is not None:
                        masked_array = np.array(array, copy=True)
                        masked_array[mask_slice] = np.nan
                        array = masked_array
                        cmap = masked_cmap
                    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{name} - {title}")
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_{idx:03d}.png", dpi=150)
            plt.close(fig)

        # Save per-feature window visualisations for all cases (tables and rasters).
        mask_slice = _mask_for(idx)
        for feat_idx in range(C):
            name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"ch_{feat_idx}"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            channel_views = [inputs[idx, feat_idx], masked[idx, feat_idx], recon[idx, feat_idx]]
            vmin, vmax = _shared_minmax(channel_views)
            for ax, data, title in zip(axes, channel_views, ("original", "masked", "recon")):
                array = data.detach().cpu().numpy()
                cmap = base_cmap
                if title == "masked" and mask_slice is not None:
                    masked_array = np.array(array, copy=True)
                    masked_array[mask_slice] = np.nan
                    array = masked_array
                    cmap = masked_cmap
                ax.imshow(array, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
                ax.set_title(f"{name} - {title}")
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_patch_{idx:03d}_{safe_name}.png", dpi=150)
            plt.close(fig)


def train_ssl(
    model,
    dataloader: DataLoader,
    epochs: int = 30,
    lr: float = 2.5e-4,
    optimizer: str = 'adamw',
    device: str | None = None,
    preview_samples: int = 0,
    preview_dir: Optional[Path] = None,
    feature_names: Optional[Iterable[str]] = None,
    feature_metadata: Optional[Iterable[Optional[Dict[str, Optional[str]]]]] = None,
    mask_scope: str = "pixel",
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    if mask_scope not in {"pixel", "patch"}:
        raise ValueError(f"mask_scope must be 'pixel' or 'patch', received '{mask_scope}'")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = _select_optimizer(optimizer, model.parameters(), lr)
    criterion = MAELoss()
    model.train()

    history: List[Dict[str, float]] = []
    warned_ssim = False
    skipped_ssim_scalar = False
    skipped_ssim_small_window = False
    last_spatial_shape: Optional[Tuple[int, int]] = None
    sample_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        running_mae = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        count = 0
        count_ssim = 0

        for batch in tqdm(dataloader, desc=f"SSL epoch {ep}"):
            x = batch.to(device)
            expected_size = getattr(model, "image_size", None)
            if expected_size is not None:
                if isinstance(expected_size, int):
                    expected_size = (expected_size, expected_size)
                if x.dim() == 4 and (x.shape[-2], x.shape[-1]) != expected_size:
                    x = torch.nn.functional.interpolate(x, size=expected_size, mode="nearest")
            elif x.dim() == 4 and x.shape[-2] == 1 and x.shape[-1] == 1:
                target_hw = max(16, getattr(model, "patch_size", 16))
                x = torch.nn.functional.interpolate(x, size=(target_hw, target_hw), mode="nearest")
            if x.dim() >= 4:
                last_spatial_shape = (x.shape[-2], x.shape[-1])
            else:
                last_spatial_shape = None

            pred, mask = model(x)

            expanded_mask: Optional[torch.Tensor] = None
            patch_size = getattr(model, "patch_size", None)
            if isinstance(patch_size, int) and pred.dim() == 4 and mask is not None:
                H, W = pred.shape[-2], pred.shape[-1]
                if H % patch_size == 0 and W % patch_size == 0:
                    H_grid, W_grid = H // patch_size, W // patch_size
                    if mask.numel() == pred.size(0) * H_grid * W_grid:
                        mask_grid = mask.view(pred.size(0), H_grid, W_grid)
                        pixel_mask = mask_grid.unsqueeze(1)
                        pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2)
                        pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=3)
                        expanded_mask = pixel_mask.expand(-1, pred.size(1), -1, -1).to(dtype=pred.dtype)

            loss = criterion(pred, x, mask=expanded_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            if expanded_mask is not None:
                abs_diff = (pred - x).abs() * expanded_mask
                denom = expanded_mask.sum().clamp(min=1.0)
                mae_val = abs_diff.sum() / denom
            else:
                mae_val = torch.nn.functional.l1_loss(pred, x)
            running_mae += mae_val.item() * batch_size

            data_range = (x.max() - x.min()).detach().cpu().item()
            if data_range <= 0:
                data_range = 1.0
            psnr_val = peak_signal_noise_ratio(pred, x, data_range=data_range)
            running_psnr += psnr_val.item() * batch_size
            has_spatial = x.dim() >= 4
            if has_spatial and x.shape[-1] > 1 and x.shape[-2] > 1 and min(x.shape[-2], x.shape[-1]) >= 11:
                # TorchMetrics' SSIM uses an 11x11 window; skip when crops are smaller to avoid invalid padding.
                ssim_val = structural_similarity_index_measure(pred, x, data_range=data_range)
                running_ssim += ssim_val.item() * batch_size
                count_ssim += batch_size
            else:
                if not has_spatial or x.shape[-1] <= 1 or x.shape[-2] <= 1:
                    skipped_ssim_scalar = True
                elif min(x.shape[-2], x.shape[-1]) < 11:
                    skipped_ssim_small_window = True

            if preview_samples > 0 and len(sample_cache) < preview_samples:
                if mask_scope == "patch" and expanded_mask is not None:
                    pixel_mask = expanded_mask[:, :1].bool()
                else:
                    ratio = float(getattr(model, "mask_ratio", 0.0))
                    rand = torch.rand(
                        (x.size(0), 1, x.size(-2), x.size(-1)), device=x.device, dtype=x.dtype
                    )
                    pixel_mask = (rand < ratio)

                masked_img = x.clone()
                preview_mask = pixel_mask.expand(-1, x.size(1), -1, -1)
                masked_img[preview_mask] = 0.0

                for sample_idx in range(x.size(0)):
                    sample_cache.append(
                        (
                            x[sample_idx].detach().cpu(),
                            masked_img[sample_idx].detach().cpu(),
                            pred[sample_idx].detach().cpu(),
                            preview_mask[sample_idx, 0].detach().cpu(),
                        )
                    )
                    if len(sample_cache) >= preview_samples:
                        break

            count += batch_size

        epoch_loss = running_loss / max(1, count)
        epoch_mae = running_mae / max(1, count)
        epoch_psnr = running_psnr / max(1, count)
        epoch_ssim = running_ssim / count_ssim if count_ssim else None

        if epoch_ssim is None and not warned_ssim:
            if skipped_ssim_small_window:
                min_dim = None
                if last_spatial_shape is not None:
                    min_dim = min(last_spatial_shape)
                patch = getattr(model, "patch_size", None)
                if isinstance(patch, (tuple, list)) and patch:
                    patch_info = f"patch size {tuple(patch)}"
                elif patch is not None:
                    patch_info = f"patch size {patch}"
                else:
                    patch_info = "current patch configuration"
                dim_msg = f" (min spatial dim {min_dim})" if min_dim is not None else ""
                print(
                    f"[SSL] SSIM skipped because inputs are smaller than the 11x11 SSIM window{dim_msg}; "
                    f"consider increasing {patch_info}."
                )
            elif skipped_ssim_scalar:
                print(
                    "[SSL] SSIM unavailable because inputs collapse to 1x1 patches; values recorded as null."
                )
            else:
                print(
                    "[SSL] SSIM unavailable (data range likely degenerate); values recorded as null."
                )
            warned_ssim = True

        ssim_str = f"{epoch_ssim:.4f}" if epoch_ssim is not None else "n/a"
        print(
            f"[SSL] epoch {ep} loss={epoch_loss:.4f} mae={epoch_mae:.4f} "
            f"psnr={epoch_psnr:.2f} ssim={ssim_str}"
        )

        history.append(
            {
                "epoch": ep,
                "recon_loss": epoch_loss,
                "mae": epoch_mae,
                "psnr": epoch_psnr,
                "ssim": epoch_ssim,
            }
        )

    if preview_samples > 0 and preview_dir is not None and sample_cache:
        limited = sample_cache[:preview_samples]
        original, masked, recon, mask_stack = zip(*limited)
        original = torch.stack(list(original), dim=0)
        masked = torch.stack(list(masked), dim=0)
        recon = torch.stack(list(recon), dim=0)
        masks = torch.stack(list(mask_stack), dim=0)


        # TODO: Debug this. Why I am getting all NaNs for original, masked and recon? Can I check the original values from the raster?
        print("             [dev] CHECK", original.shape, masked.shape, recon.shape)

        _plot_samples(
            original,
            masked,
            recon,
            feature_names,
            feature_metadata,
            preview_dir,
            prefix="ssl_sample",
            masks=masks,
        )
        msg = f"Saved {len(limited)} reconstruction previews to {preview_dir}"
        _LOG.info(msg)
        print(msg)

    return model, history
