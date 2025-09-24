# scripts/plot_ssl_history.py
"""Plot SSL pretraining metrics from ssl_history.json."""
import argparse
import json
from pathlib import Path
from typing import List, Dict

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
    recon = [entry.get("recon_loss") for entry in history]
    mae = [entry.get("mae") for entry in history]
    psnr = [entry.get("psnr") for entry in history]
    ssim = [entry.get("ssim") for entry in history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax = axes[0, 0]
    ax.plot(epochs, recon, marker="o")
    ax.set_title("Reconstruction Loss")
    ax.set_ylabel("MSE")

    ax = axes[0, 1]
    ax.plot(epochs, mae, marker="o", color="tab:orange")
    ax.set_title("Mean Absolute Error")

    ax = axes[1, 0]
    ax.plot(epochs, psnr, marker="o", color="tab:green")
    ax.set_title("PSNR")
    ax.set_ylabel("dB")
    ax.set_xlabel("Epoch")

    # Filter out None values for SSIM
    valid_pairs = [(e, v) for e, v in zip(epochs, ssim) if v is not None]
    ax = axes[1, 1]
    if valid_pairs:
        xs, ys = zip(*valid_pairs)
        ax.plot(xs, ys, marker="o", color="tab:red")
    else:
        ax.text(0.5, 0.5, "SSIM unavailable", transform=ax.transAxes, ha="center", va="center")
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
