from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

def _prepare_points(samples: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig = []
    aug = []
    for pt in samples:
        coord = pt.get("coord")
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            continue
        x, y = float(coord[0]), float(coord[1])
        if pt.get("is_augmented"):
            aug.append((x, y))
        else:
            orig.append((x, y))
    orig_arr = np.asarray(orig, dtype=np.float64) if orig else np.zeros((0, 2))
    aug_arr = np.asarray(aug, dtype=np.float64) if aug else np.zeros((0, 2))
    return orig_arr[:, 0] if orig_arr.size else np.asarray([]), orig_arr[:, 1] if orig_arr.size else np.asarray([]), aug_arr[:, 0] if aug_arr.size else np.asarray([]), aug_arr[:, 1] if aug_arr.size else np.asarray([])


def _draw_boundary(ax, geometry: Optional[Dict[str, object]]) -> None:
    if not geometry:
        return
    geom_type = geometry.get("type")
    coords_seq: Iterable
    if geom_type == "Polygon":
        coords_seq = [geometry.get("coordinates", [])]
    elif geom_type == "MultiPolygon":
        coords_seq = geometry.get("coordinates", [])
    else:
        return
    for polygon in coords_seq:
        for ring in polygon:
            ring_arr = np.asarray(ring, dtype=float)
            if ring_arr.ndim == 2 and ring_arr.size:
                ax.plot(ring_arr[:, 0], ring_arr[:, 1], color="black", linewidth=0.8, alpha=0.6)


def save_overlap_debug_plot(
    output_path: Path,
    boundary_geometry: Optional[Dict[str, object]],
    anchor_samples: Sequence[Dict[str, object]],
    target_samples: Sequence[Dict[str, object]],
    *,
    title: str,
    anchor_label: str,
    target_label: str,
    centroid_points: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_boundary(ax, boundary_geometry)

    a_orig_x, a_orig_y, a_aug_x, a_aug_y = _prepare_points(anchor_samples)
    t_orig_x, t_orig_y, t_aug_x, t_aug_y = _prepare_points(target_samples)

    if a_orig_x.size:
        ax.scatter(a_orig_x, a_orig_y, s=35, facecolors="none", edgecolors="#c32f27", linewidths=1.2, label=f"{anchor_label} positive")
    if t_orig_x.size:
        ax.scatter(t_orig_x, t_orig_y, s=35, facecolors="none", edgecolors="#1f77b4", linewidths=1.2, label=f"{target_label} positive")
    if a_aug_x.size:
        ax.scatter(a_aug_x, a_aug_y, s=25, c="#ff7f0e", marker="o", label=f"{anchor_label} positive (aug)")
    if t_aug_x.size:
        ax.scatter(t_aug_x, t_aug_y, s=25, c="#2ca02c", marker="o", label=f"{target_label} positive (aug)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if centroid_points:
        centroid_arr = np.asarray(centroid_points, dtype=np.float64)
        if centroid_arr.size:
            ax.scatter(
                centroid_arr[:, 0],
                centroid_arr[:, 1],
                s=20,
                c="#111111",
                marker="+",
                linewidths=1.2,
                label="Overlap centroid",
            )

    if a_orig_x.size or t_orig_x.size or a_aug_x.size or t_aug_x.size or centroid_points:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
