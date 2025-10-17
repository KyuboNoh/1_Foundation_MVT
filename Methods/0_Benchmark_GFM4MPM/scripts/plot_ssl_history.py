# scripts/plot_ssl_history.py
"""Plot SSL pretraining metrics from ssl_history.json."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("History file must contain a list of epoch records")
    return data


def plot_history(history: List[Dict], out_path: Path | None) -> None:
    epochs = [entry["epoch"] for entry in history]

    def extract_metric(entries: List[Dict], metric: str, split: str) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for entry in entries:
            if split == "train":
                train_data = entry.get("train")
                if isinstance(train_data, dict):
                    values.append(train_data.get(metric))
                else:
                    values.append(entry.get(metric))
            else:
                val_data = entry.get("val")
                if isinstance(val_data, dict):
                    values.append(val_data.get(metric))
                else:
                    values.append(entry.get(f"val_{metric}"))
        return values

    def to_plot(values: List[Optional[float]]) -> List[float]:
        return [float("nan") if v is None else v for v in values]

    train_recon = extract_metric(history, "recon_loss", "train")
    val_recon = extract_metric(history, "recon_loss", "val")
    train_mae = extract_metric(history, "mae", "train")
    val_mae = extract_metric(history, "mae", "val")
    train_psnr = extract_metric(history, "psnr", "train")
    val_psnr = extract_metric(history, "psnr", "val")
    train_ssim = extract_metric(history, "ssim", "train")
    val_ssim = extract_metric(history, "ssim", "val")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    def plot_pair(ax, train_vals, val_vals, title, ylabel=None):
        ax.plot(epochs, to_plot(train_vals), marker="o", label="train", color="tab:blue")
        has_val = any(v is not None for v in val_vals)
        if has_val:
            ax.plot(epochs, to_plot(val_vals), marker="o", label="val", color="tab:orange")
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        if has_val:
            ax.legend()

    plot_pair(axes[0, 0], train_recon, val_recon, "Reconstruction Loss", ylabel="MSE")
    plot_pair(axes[0, 1], train_mae, val_mae, "Mean Absolute Error")
    plot_pair(axes[1, 0], train_psnr, val_psnr, "PSNR", ylabel="dB")

    ax = axes[1, 1]
    train_ssim_plot = to_plot(train_ssim)
    val_ssim_plot = to_plot(val_ssim)
    has_train_ssim = any(v is not None for v in train_ssim)
    has_val_ssim = any(v is not None for v in val_ssim)
    if has_train_ssim:
        ax.plot(epochs, train_ssim_plot, marker="o", label="train", color="tab:red")
    if has_val_ssim:
        ax.plot(epochs, val_ssim_plot, marker="o", label="val", color="tab:purple")
    if not (has_train_ssim or has_val_ssim):
        ax.text(0.5, 0.5, "SSIM unavailable", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend()
    ax.set_title("SSIM")
    ax.set_xlabel("Epoch")

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics stored in ssl_history.json")
    parser.add_argument("history", type=str, help="Path to ssl_history.json")
    parser.add_argument("--out", type=str, help="Optional path to save the plot")
    args = parser.parse_args()

    history = load_history(Path(args.history))
    out_path = Path(args.out) if args.out else None
    plot_history(history, out_path)
