# scripts/inspect_labels.py
"""Utility to inspect potential label columns inside a STAC table export."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.gfm4mpm.data.stac_table import StacTableStack


def summarize_labels(stack: StacTableStack, columns: List[str]) -> List[Dict]:
    summaries: List[Dict] = []
    for name in columns:
        arr = stack.label_array(name)
        total = int(arr.size)
        positives = int(arr.sum())
        negatives = total - positives
        raw = stack.raw_column(name)
        uniques, counts = np.unique(raw, return_counts=True)
        sorted_pairs = sorted(zip(counts, uniques), reverse=True)[:5]
        summaries.append(
            {
                "name": name,
                "total": total,
                "positives": positives,
                "negatives": negatives,
                "top_values": [(str(val), int(cnt)) for cnt, val in sorted_pairs],
            }
        )
    return summaries


def plot_positive_map(stack: StacTableStack, summaries: List[Dict], out_path: Path) -> None:
    if stack.latitudes is None or stack.longitudes is None:
        raise SystemExit("Latitude/Longitude columns are required to plot a map of Present samples.")

    names = [s["name"] for s in summaries]
    n_cols = min(3, len(names)) or 1
    n_rows = math.ceil(len(names) / n_cols) or 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    lat = np.asarray(stack.latitudes, dtype=float)
    lon = np.asarray(stack.longitudes, dtype=float)

    for idx, name in enumerate(names):
        ax = axes[idx // n_cols][idx % n_cols]
        mask = stack.label_array(name) == 1
        ax.scatter(lon[mask], lat[mask], s=6, color="#d62728", alpha=0.7, edgecolors="none")
        ax.set_title(f"{name} (Present={int(mask.sum())})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if mask.any():
            pad = 0.25
            ax.set_xlim(float(lon[mask].min()) - pad, float(lon[mask].max()) + pad)
            ax.set_ylim(float(lat[mask].min()) - pad, float(lat[mask].max()) + pad)
        ax.grid(True, linestyle="--", alpha=0.3)

    # hide unused axes
    for j in range(len(names), n_rows * n_cols):
        ax = axes[j // n_cols][j % n_cols]
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize candidate label columns in a STAC table")
    parser.add_argument("--stac-root", required=True, help="Path to STAC collection root (directory containing collection.json)")
    parser.add_argument("--filter", default="Training_", help="Substring to select candidate label columns (default: Training_)")
    parser.add_argument("--out-json", type=str, help="Optional path to write JSON summary")
    parser.add_argument("--out-plot", type=str, help="Optional path to save a bar chart (PNG)")
    args = parser.parse_args()

    stack = StacTableStack(Path(args.stac_root))
    candidates = [name for name in stack.labels.keys() if args.filter.lower() in name.lower()]
    if not candidates:
        raise SystemExit(f"No label columns matched filter '{args.filter}'")

    summaries = summarize_labels(stack, candidates)

    for summary in summaries:
        print(f"{summary['name']}: positives={summary['positives']} negatives={summary['negatives']} total={summary['total']}")
        if summary["top_values"]:
            tops = ", ".join([f"{val} ({cnt})" for val, cnt in summary["top_values"]])
            print(f"  top values: {tops}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summaries, fh, indent=2)
        print(f"Wrote JSON summary to {out_path}")

    if args.out_plot:
        plot_positive_map(stack, summaries, Path(args.out_plot))
        print(f"Wrote label map to {args.out_plot}")


if __name__ == "__main__":
    main()
