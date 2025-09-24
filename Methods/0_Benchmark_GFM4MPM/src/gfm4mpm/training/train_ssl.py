# src/gfm4mpm/training/train_ssl.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm


class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


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
    out_dir: Path,
    prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    B, C, H, W = inputs.shape
    feat_names = list(feature_names) if feature_names is not None else [f"ch_{i}" for i in range(C)]

    for idx in range(B):
        fig, axes = plt.subplots(C, 3, figsize=(9, 3 * C))
        for c in range(C):
            name = feat_names[c] if c < len(feat_names) else f"ch_{c}"
            for j, (data, title) in enumerate(
                [
                    (inputs[idx, c].cpu(), "original"),
                    (masked[idx, c].cpu(), "masked"),
                    (recon[idx, c].cpu(), "recon"),
                ]
            ):
                ax = axes[c, j] if C > 1 else axes[j]
                im = ax.imshow(data, cmap="viridis")
                ax.set_title(f"{name} - {title}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{idx:03d}.png", dpi=150)
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
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = _select_optimizer(optimizer, model.parameters(), lr)
    criterion = MAELoss()
    model.train()

    history: List[Dict[str, float]] = []
    warned_ssim = False
    sample_cache: List[torch.Tensor] = []

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        running_mae = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        count = 0
        count_ssim = 0

        for batch in tqdm(dataloader, desc=f"SSL epoch {ep}"):
            x = batch.to(device)
            pred, mask = model(x)
            loss = criterion(pred, x)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            mae_val = torch.nn.functional.l1_loss(pred, x)
            running_mae += mae_val.item() * batch_size

            data_range = (x.max() - x.min()).detach().cpu().item()
            if data_range <= 0:
                data_range = 1.0
            psnr_val = peak_signal_noise_ratio(pred, x, data_range=data_range)
            running_psnr += psnr_val.item() * batch_size

            if x.shape[-1] > 1 and x.shape[-2] > 1:
                ssim_val = structural_similarity_index_measure(pred, x, data_range=data_range)
                running_ssim += ssim_val.item() * batch_size
                count_ssim += batch_size

            if preview_samples > 0 and len(sample_cache) < preview_samples:
                with torch.no_grad():
                    _, H_grid, W_grid = model.patch(x)
                mask_grid = mask.view(x.size(0), H_grid, W_grid)
                pixel_mask = mask_grid.unsqueeze(1)
                pixel_mask = pixel_mask.repeat_interleave(model.patch_size, dim=2)
                pixel_mask = pixel_mask.repeat_interleave(model.patch_size, dim=3)
                masked_img = x.clone()
                masked_img[pixel_mask.bool()] = 0.0

                for orig, masked_example, recon_example in zip(x, masked_img, pred):
                    sample_cache.append(
                        torch.stack(
                            [
                                orig.detach().cpu(),
                                masked_example.detach().cpu(),
                                recon_example.detach().cpu(),
                            ],
                            dim=0,
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
            print("[SSL] SSIM unavailable (patches are 1x1 or data range is zero); values recorded as null.")
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
        stacked = torch.stack(sample_cache, dim=0)[:preview_samples]
        original = stacked[:, 0]
        masked = stacked[:, 1]
        recon = stacked[:, 2]
        _plot_samples(original, masked, recon, feature_names, preview_dir, prefix="ssl_sample")
        print(f"Saved {len(stacked)} reconstruction previews to {preview_dir}")

    return model, history
