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
    def __init__(
        self,
        patch_size: Optional[int] = None,
        norm_per_patch: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if patch_size is not None and patch_size <= 0:
            raise ValueError("patch_size must be positive when provided")
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.norm_per_patch = bool(norm_per_patch)
        self.eps = float(eps)

    def _check_spatial(self, tensor: torch.Tensor) -> Tuple[int, int, int, int]:
        if self.patch_size is None:
            raise RuntimeError("patch_size must be defined for spatial MAE loss")
        if tensor.dim() != 4:
            raise ValueError("Expected tensor in NCHW format for MAE loss")
        B, C, H, W = tensor.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dims {H}x{W} must be divisible by patch_size {self.patch_size}"
            )
        return B, C, H, W

    def _patchify(self, tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = self._check_spatial(tensor)
        patches = tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            B,
            C,
            H // self.patch_size,
            W // self.patch_size,
            self.patch_size * self.patch_size,
        )
        patches = patches.permute(0, 2, 3, 1, 4).contiguous()
        return patches.view(B, -1, C * self.patch_size * self.patch_size)

    def _unpatchify(self, patches: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        B, C, H, W = shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        patches = patches.view(B, grid_h, grid_w, C, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, C, H, W)

    def _normalize_per_patch(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = tuple(target.shape)
        pred_patches = self._patchify(pred)
        target_patches = self._patchify(target)
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True, unbiased=False)
        denom = torch.sqrt(var + self.eps)
        target_patches = (target_patches - mean) / denom
        pred_patches = (pred_patches - mean) / denom
        pred_norm = self._unpatchify(pred_patches, original_shape)
        target_norm = self._unpatchify(target_patches, original_shape)
        return pred_norm, target_norm

    def _mae_mask_to_pixels(
        self,
        mae_mask: torch.Tensor,
        *,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B, _, H, W = shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        if mae_mask.dim() == 2:
            if mae_mask.shape[1] != grid_h * grid_w:
                raise ValueError("Unexpected MAE mask length for given spatial dimensions")
            mask_grid = mae_mask.view(B, grid_h, grid_w)
            mask_grid = mask_grid.unsqueeze(1)
            mask_grid = mask_grid.repeat_interleave(self.patch_size, dim=2)
            mask_grid = mask_grid.repeat_interleave(self.patch_size, dim=3)
            return mask_grid.to(device=device, dtype=dtype)

        if mae_mask.dim() == 3:
            mask_tensor = mae_mask
            if mask_tensor.size(1) != 1:
                mask_tensor = mask_tensor.unsqueeze(1)
        elif mae_mask.dim() == 4:
            mask_tensor = mae_mask
        else:
            raise ValueError("Unsupported MAE mask dimensionality")

        if mask_tensor.size(1) != 1:
            mask_tensor = mask_tensor[:, :1]
        if mask_tensor.shape[-2:] != (H, W):
            raise ValueError("MAE pixel mask does not match spatial dimensions")
        return mask_tensor.to(device=device, dtype=dtype)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mae_mask: Optional[torch.Tensor] = None,
        pixel_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.patch_size is not None and pred.dim() == 4 and target.dim() == 4:
            B, C, H, W = self._check_spatial(pred)
            pred_norm, target_norm = pred, target
            if self.norm_per_patch:
                pred_norm, target_norm = self._normalize_per_patch(pred, target)

            pixel_mask: Optional[torch.Tensor] = None
            if mae_mask is not None:
                pixel_mask = self._mae_mask_to_pixels(
                    mae_mask,
                    shape=(B, C, H, W),
                    device=pred.device,
                    dtype=pred.dtype,
                )

            if pixel_valid_mask is not None:
                valid = pixel_valid_mask
                if valid.dim() == 3:
                    valid = valid.unsqueeze(1)
                if valid.dim() == 4 and valid.size(1) != 1:
                    valid = valid.any(dim=1, keepdim=True)
                valid = valid.to(device=pred.device, dtype=pred.dtype)
                pixel_mask = valid if pixel_mask is None else pixel_mask * valid

            if pixel_mask is None:
                pixel_mask = torch.ones((B, 1, H, W), device=pred.device, dtype=pred.dtype)

            diff = (pred_norm - target_norm).pow(2) * pixel_mask
            denom = pixel_mask.sum()
            if denom <= 0:
                return diff.mean()
            return diff.sum() / denom

        # Fallback to dense MSE behaviour (e.g., table data).
        if mae_mask is None and pixel_valid_mask is None:
            return F.mse_loss(pred, target, reduction="mean")

        pixel_mask = None
        if mae_mask is not None:
            pixel_mask = mae_mask
            if pixel_mask.dim() == pred.dim() - 1:
                pixel_mask = pixel_mask.unsqueeze(1)
        if pixel_valid_mask is not None:
            valid = pixel_valid_mask
            if valid.dim() == pred.dim() - 1:
                valid = valid.unsqueeze(1)
            pixel_mask = valid if pixel_mask is None else pixel_mask * valid
        if pixel_mask is None:
            return F.mse_loss(pred, target, reduction="mean")

        if pixel_mask.shape != pred.shape:
            raise ValueError("Gating mask must broadcast to prediction shape in fallback path")

        pixel_mask = pixel_mask.to(dtype=pred.dtype)
        weighted_sq = (pred - target).pow(2) * pixel_mask
        denom = pixel_mask.sum()
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
    mae_masks: Optional[Iterable[torch.Tensor]] = None,
    invalid_masks: Optional[Iterable[torch.Tensor]] = None,
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

    def _mask_for(collection: Optional[Iterable[torch.Tensor]], sample_idx: int) -> Optional[np.ndarray]:
        if collection is None:
            return None
        mask_tensor = collection[sample_idx]
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_tensor)
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim < 2:
            return None
        return mask_np.astype(bool)

    for idx in range(B):
        mae_slice = _mask_for(mae_masks, idx)
        invalid_slice = _mask_for(invalid_masks, idx)
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
                    mask_to_apply: Optional[np.ndarray] = None
                    if invalid_slice is not None:
                        mask_to_apply = invalid_slice
                    if title != "original" and mae_slice is not None:
                        mask_to_apply = mae_slice if mask_to_apply is None else (mask_to_apply | mae_slice)
                    if mask_to_apply is not None:
                        masked_array = np.array(array, copy=True)
                        masked_array[mask_to_apply] = np.nan
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
        for feat_idx in range(C):
            name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"ch_{feat_idx}"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            channel_views = [inputs[idx, feat_idx], masked[idx, feat_idx], recon[idx, feat_idx]]
            vmin, vmax = _shared_minmax(channel_views)
            for ax, data, title in zip(axes, channel_views, ("original", "masked", "recon")):
                array = data.detach().cpu().numpy()
                cmap = base_cmap
                mask_to_apply = invalid_slice
                if title != "original" and mae_slice is not None:
                    mask_to_apply = mae_slice if mask_to_apply is None else (mask_to_apply | mae_slice)
                if mask_to_apply is not None:
                    masked_array = np.array(array, copy=True)
                    masked_array[mask_to_apply] = np.nan
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
    norm_per_patch: bool = True,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    if mask_scope not in {"pixel", "patch"}:
        raise ValueError(f"mask_scope must be 'pixel' or 'patch', received '{mask_scope}'")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = _select_optimizer(optimizer, model.parameters(), lr)
    criterion = MAELoss(
        patch_size=getattr(model, "patch_size", None),
        norm_per_patch=norm_per_patch,
    )
    model.train()

    history: List[Dict[str, float]] = []
    warned_ssim = False
    skipped_ssim_scalar = False
    skipped_ssim_small_window = False
    last_spatial_shape: Optional[Tuple[int, int]] = None
    sample_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        running_mae = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        count = 0
        count_ssim = 0

        for batch in tqdm(dataloader, desc=f"SSL epoch {ep}"):
            if isinstance(batch, dict):
                x = batch.get("image")
                if x is None:
                    raise ValueError("Batch dictionary missing 'image' key")
                mask_no_feature = batch.get("mask_pixel_no_feature")
            else:
                x = batch
                mask_no_feature = None

            x = x.to(device)

            mask_no_feature_tensor: Optional[torch.Tensor] = None
            if mask_no_feature is not None:
                mask_no_feature_tensor = mask_no_feature.to(device=device)
                if mask_no_feature_tensor.dtype != torch.bool:
                    mask_no_feature_tensor = mask_no_feature_tensor.bool()
                if mask_no_feature_tensor.dim() == 2:
                    mask_no_feature_tensor = mask_no_feature_tensor.unsqueeze(0).unsqueeze(0)
                elif mask_no_feature_tensor.dim() == 3:
                    mask_no_feature_tensor = mask_no_feature_tensor.unsqueeze(1)
                elif mask_no_feature_tensor.dim() == 4 and mask_no_feature_tensor.size(1) != 1:
                    mask_no_feature_tensor = mask_no_feature_tensor.any(dim=1, keepdim=True)

            expected_size = getattr(model, "image_size", None)
            if expected_size is not None:
                if isinstance(expected_size, int):
                    expected_size = (expected_size, expected_size)
                if x.dim() == 4 and (x.shape[-2], x.shape[-1]) != expected_size:
                    x = torch.nn.functional.interpolate(x, size=expected_size, mode="nearest")
                    if mask_no_feature_tensor is not None:
                        mask_no_feature_tensor = torch.nn.functional.interpolate(
                            mask_no_feature_tensor.float(), size=expected_size, mode="nearest"
                        ) >= 0.5
            elif x.dim() == 4 and x.shape[-2] == 1 and x.shape[-1] == 1:
                target_hw = max(16, getattr(model, "patch_size", 16))
                x = torch.nn.functional.interpolate(x, size=(target_hw, target_hw), mode="nearest")
                if mask_no_feature_tensor is not None:
                    mask_no_feature_tensor = torch.nn.functional.interpolate(
                        mask_no_feature_tensor.float(), size=(target_hw, target_hw), mode="nearest"
                    ) >= 0.5

            if mask_no_feature_tensor is not None:
                mask_no_feature_tensor = mask_no_feature_tensor.bool()

            if x.dim() >= 4:
                last_spatial_shape = (x.shape[-2], x.shape[-1])
            else:
                last_spatial_shape = None

            pred, mae_mask_tokens = model(x)

            patch_size = getattr(model, "patch_size", None)
            pixel_mae_mask: Optional[torch.Tensor] = None
            if (
                isinstance(patch_size, int)
                and pred.dim() == 4
                and mae_mask_tokens is not None
            ):
                H, W = pred.shape[-2], pred.shape[-1]
                if H % patch_size == 0 and W % patch_size == 0:
                    grid_h, grid_w = H // patch_size, W // patch_size
                    if mae_mask_tokens.numel() == pred.size(0) * grid_h * grid_w:
                        mask_grid = mae_mask_tokens.view(pred.size(0), grid_h, grid_w)
                        pixel_mae_mask = mask_grid.unsqueeze(1)
                        pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_size, dim=2)
                        pixel_mae_mask = pixel_mae_mask.repeat_interleave(patch_size, dim=3)
                        pixel_mae_mask = pixel_mae_mask.to(device=pred.device, dtype=pred.dtype)

            if pixel_mae_mask is None:
                pixel_mae_mask = torch.ones(
                    (pred.size(0), 1, pred.shape[-2], pred.shape[-1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )

            pixel_invalid_mask: Optional[torch.Tensor] = None
            pixel_valid_mask_bool: Optional[torch.Tensor] = None
            if mask_no_feature_tensor is not None:
                pixel_invalid_mask = mask_no_feature_tensor.to(device=pred.device)
                pixel_valid_mask_bool = ~pixel_invalid_mask

            loss = criterion(
                pred,
                x,
                mae_mask=pixel_mae_mask,
                pixel_valid_mask=pixel_valid_mask_bool,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size

            supervision_mask = pixel_mae_mask
            if pixel_valid_mask_bool is not None:
                supervision_mask = supervision_mask * pixel_valid_mask_bool.to(dtype=pred.dtype)
            mask_sum = supervision_mask.sum().clamp(min=1.0)
            mae_val = ((pred - x).abs() * supervision_mask).sum() / mask_sum
            running_mae += mae_val.detach().item() * batch_size

            if pixel_valid_mask_bool is not None:
                valid_expand = pixel_valid_mask_bool.expand_as(x)
                invalid_expand = (~pixel_valid_mask_bool).expand_as(x)
                pred_metrics = pred.clone()
                target_metrics = x.clone()
                pred_metrics[invalid_expand] = 0.0
                target_metrics[invalid_expand] = 0.0
                target_valid_vals = x[valid_expand]
                if target_valid_vals.numel() > 0:
                    data_range_val = (target_valid_vals.max() - target_valid_vals.min()).detach().cpu().item()
                else:
                    data_range_val = 1.0
            else:
                pred_metrics = pred
                target_metrics = x
                data_range_val = (target_metrics.max() - target_metrics.min()).detach().cpu().item()

            if data_range_val <= 0:
                data_range_val = 1.0

            psnr_val = peak_signal_noise_ratio(pred_metrics, target_metrics, data_range=data_range_val)
            running_psnr += psnr_val.item() * batch_size
            has_spatial = x.dim() >= 4
            if has_spatial and x.shape[-1] > 1 and x.shape[-2] > 1 and min(x.shape[-2], x.shape[-1]) >= 11:
                ssim_val = structural_similarity_index_measure(pred_metrics, target_metrics, data_range=data_range_val)
                running_ssim += ssim_val.item() * batch_size
                count_ssim += batch_size
            else:
                if not has_spatial or x.shape[-1] <= 1 or x.shape[-2] <= 1:
                    skipped_ssim_scalar = True
                elif min(x.shape[-2], x.shape[-1]) < 11:
                    skipped_ssim_small_window = True

            if preview_samples > 0 and len(sample_cache) < preview_samples:
                if mask_scope == "patch" and pixel_mae_mask is not None:
                    pixel_mask_bool = pixel_mae_mask[:, :1] > 0.5
                else:
                    ratio = float(getattr(model, "mask_ratio", 0.0))
                    rand = torch.rand(
                        (x.size(0), 1, x.size(-2), x.size(-1)), device=x.device, dtype=x.dtype
                    )
                    pixel_mask_bool = (rand < ratio)

                masked_img = x.clone()
                preview_mask = pixel_mask_bool.expand(-1, x.size(1), -1, -1)
                masked_img[preview_mask] = 0.0

                mae_mask_bool = (pixel_mae_mask[:, :1] > 0.5).detach()
                if pixel_invalid_mask is not None:
                    invalid_mask_bool = pixel_invalid_mask.detach().bool()
                else:
                    invalid_mask_bool = torch.zeros_like(mae_mask_bool, dtype=torch.bool)

                for sample_idx in range(x.size(0)):
                    sample_cache.append(
                        (
                            x[sample_idx].detach().cpu(),
                            masked_img[sample_idx].detach().cpu(),
                            pred[sample_idx].detach().cpu(),
                            mae_mask_bool[sample_idx, 0].detach().cpu(),
                            invalid_mask_bool[sample_idx, 0].detach().cpu(),
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
        original, masked, recon, mae_mask_stack, invalid_mask_stack = zip(*limited)
        original = torch.stack(list(original), dim=0)
        masked = torch.stack(list(masked), dim=0)
        recon = torch.stack(list(recon), dim=0)
        mae_masks = torch.stack(list(mae_mask_stack), dim=0)
        invalid_masks = torch.stack(list(invalid_mask_stack), dim=0)


        # TODO: Debug this. Why I am getting all NaNs for original, masked and recon? Can I check the original values from the raster?
        print("             [dev] CHECK", original.shape, masked.shape, recon.shape)

        try:
            orig_flat = original.detach().cpu().view(original.size(0), original.size(1), -1)
            orig_min = orig_flat.min(dim=-1)[0]
            orig_max = orig_flat.max(dim=-1)[0]
            for sample_idx in range(min(3, orig_flat.size(0))):
                channel_stats = []
                for ch in range(orig_flat.size(1)):
                    channel_stats.append(
                        f"ch{ch}:min={orig_min[sample_idx, ch]:.4f},max={orig_max[sample_idx, ch]:.4f}"
                    )
                print(
                    f"[debug] preview sample {sample_idx} value ranges -> " + ", ".join(channel_stats)
                )
        except Exception as exc:  # pragma: no cover - diagnostic helper
            print(f"[debug] Failed to compute preview stats: {exc}")

        _plot_samples(
            original,
            masked,
            recon,
            feature_names,
            feature_metadata,
            preview_dir,
            prefix="ssl_sample",
            mae_masks=mae_masks,
            invalid_masks=invalid_masks,
        )
        msg = f"Saved {len(limited)} reconstruction previews to {preview_dir}"
        _LOG.info(msg)
        print(msg)

    return model, history
