#!/usr/bin/env python3
"""Quick inspector for GSC CSV columns using latitude/longitude grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_grid(
    frame: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    value_column: str,
    grid_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    lat_vals = pd.to_numeric(frame[lat_column], errors="coerce").to_numpy(dtype=np.float64)
    lon_vals = pd.to_numeric(frame[lon_column], errors="coerce").to_numpy(dtype=np.float64)
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=np.float64)

    mask = np.isfinite(lat_vals) & np.isfinite(lon_vals) & np.isfinite(values)
    if not np.any(mask):
        raise ValueError("No finite samples available after filtering.")

    lat_valid = lat_vals[mask]
    lon_valid = lon_vals[mask]
    val_valid = values[mask]

    lat_min = float(np.min(lat_valid))
    lat_max = float(np.max(lat_valid))
    lon_min = float(np.min(lon_valid))
    lon_max = float(np.max(lon_valid))
    if not np.isfinite(lat_min) or not np.isfinite(lat_max) or lat_min == lat_max:
        raise ValueError("Latitude extent is degenerate.")
    if not np.isfinite(lon_min) or not np.isfinite(lon_max) or lon_min == lon_max:
        raise ValueError("Longitude extent is degenerate.")

    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)

    weighted_sum, _, _ = np.histogram2d(
        lon_valid,
        lat_valid,
        bins=[lon_edges, lat_edges],
        weights=val_valid,
    )
    counts, _, _ = np.histogram2d(
        lon_valid,
        lat_valid,
        bins=[lon_edges, lat_edges],
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_grid = np.divide(
            weighted_sum,
            counts,
            out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
            where=counts > 0,
        )

    mean_grid = np.flipud(mean_grid.T)
    extent = (lon_min, lon_max, lat_min, lat_max)
    return mean_grid, extent


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a GSC CSV column with an imshow plot.")
    parser.add_argument(
        "--csv",
        default="/home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube.csv",
        help="Path to the GSC CSV file.",
    )
    parser.add_argument("--value-column", default="Seismic_Moho", help="Target value column to visualize.")
    parser.add_argument(
        "--lat-column",
        default="Latitude_EPSG4326",
        help="Latitude column name (degrees).",
    )
    parser.add_argument(
        "--lon-column",
        default="Longitude_EPSG4326",
        help="Longitude column name (degrees).",
    )
    parser.add_argument("--grid-size", type=int, default=256, help="Resolution of the output grid.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot instead of displaying it.",
    )
    args = parser.parse_args()

    usecols = {args.lat_column, args.lon_column, args.value_column}
    frame = pd.read_csv(args.csv, usecols=list(usecols))

    grid, extent = build_grid(frame, args.lat_column, args.lon_column, args.value_column, args.grid_size)

    import matplotlib

    try:
        matplotlib.use("Agg" if args.output else matplotlib.get_backend())
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="upper", extent=extent, cmap="viridis", aspect="auto")
    ax.set_title(f"{args.value_column} (grid {args.grid_size}x{args.grid_size})")
    ax.set_xlabel(args.lon_column)
    ax.set_ylabel(args.lat_column)
    fig.colorbar(im, ax=ax, label=args.value_column)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
