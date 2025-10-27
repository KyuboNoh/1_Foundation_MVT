from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from rasterio.transform import array_bounds, xy
except ImportError:  # pragma: no cover - optional dependency guard
    array_bounds = None  # type: ignore[assignment]
    xy = None  # type: ignore[assignment]


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)).strip("_") or "feature"


def _iter_region_stacks(stack: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(stack, "iter_region_stacks"):
        for region_name, region_stack in stack.iter_region_stacks():
            yield str(region_name), region_stack
    else:
        default_region = getattr(stack, "default_region", "GLOBAL")
        yield str(default_region), stack


def _coords_to_xy(transform: Any, coord_list: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    if xy is None or not coord_list:
        return np.asarray([]), np.asarray([])
    rows, cols = zip(*coord_list)
    xs, ys = xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def _compute_extent(transform: Any, height: int, width: int) -> Tuple[float, float, float, float]:
    if array_bounds is None:
        raise RuntimeError("rasterio is required to compute spatial extents.")
    left, bottom, right, top = array_bounds(height, width, transform)
    return (left, right, bottom, top)


def visualize_debug_features(
    stack: Any,
    feature_names: Sequence[str],
    pos_coords: Sequence[Tuple[str, int, int]],
    neg_coords: Sequence[Tuple[str, int, int]],
    out_dir: Path,
    *,
    augmented_coords: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> None:
    """Persist debug plots showing label locations and feature rasters per region."""
    if xy is None or array_bounds is None:
        print("[warn] rasterio is required for debug visualisations; skipping.")
        return

    debug_dir = Path(out_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pos_by_region: Dict[str, List[Tuple[int, int]]] = {}
    for region, row, col in pos_coords:
        pos_by_region.setdefault(str(region), []).append((int(row), int(col)))

    neg_by_region: Dict[str, List[Tuple[int, int]]] = {}
    for region, row, col in neg_coords:
        neg_by_region.setdefault(str(region), []).append((int(row), int(col)))

    aug_by_region: Dict[str, List[Tuple[int, int]]] = {}
    if augmented_coords:
        for region, row, col in augmented_coords:
            aug_by_region.setdefault(str(region), []).append((int(row), int(col)))

    feature_labels = list(feature_names) if feature_names else []

    for region_name, region_stack in _iter_region_stacks(stack):
        height = getattr(region_stack, "height", None)
        width = getattr(region_stack, "width", None)
        transform = getattr(region_stack, "transform", None)
        if height is None or width is None or transform is None:
            print(f"[warn] Region '{region_name}' missing geometry metadata; skipping debug visualisation.")
            continue

        try:
            extent = _compute_extent(transform, height, width)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to compute spatial extent for region '{region_name}': {exc}")
            continue

        region_pos = pos_by_region.get(region_name, [])
        region_neg = neg_by_region.get(region_name, [])
        region_aug = aug_by_region.get(region_name, [])

        pos_x, pos_y = _coords_to_xy(transform, region_pos)
        neg_x, neg_y = _coords_to_xy(transform, region_neg)
        aug_x, aug_y = _coords_to_xy(transform, region_aug)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Label Distribution - {region_name}")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if aug_x.size:
            ax.scatter(
                aug_x,
                aug_y,
                s=15,
                # facecolors="none",
                facecolors="orange",
                edgecolors="orange",
                marker="o",
                linewidths=1.0,
                label="Positive (aug)",
            )
        if neg_x.size:
            ax.scatter(neg_x, neg_y, s=10, c="gold", marker="x", label="Negative")
        if pos_x.size:
            ax.scatter(pos_x, pos_y, s=20, facecolors="none", marker="o", label="Positive", edgecolors="red", linewidths=1.0)
        if pos_x.size or neg_x.size or aug_x.size:
            ax.legend(loc="upper right")
        span_x = extent[1] - extent[0]
        span_y = extent[3] - extent[2]
        if span_x > 0 and span_y > 0:
            ax.set_aspect(span_x / span_y * (fig.get_size_inches()[1] / fig.get_size_inches()[0]))
        fig.tight_layout()
        label_path = debug_dir / f"labels_{_sanitize_name(region_name)}.png"
        fig.savefig(label_path, dpi=200)
        plt.close(fig)

        feature_count = getattr(region_stack, "count", 0)
        if feature_count <= 0 or not hasattr(region_stack, "srcs"):
            continue

        for idx in range(feature_count):
            try:
                src = region_stack.srcs[idx]
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Unable to access feature {idx} for region '{region_name}': {exc}")
                continue

            try:
                band = src.read(1, masked=True)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Failed to read feature {idx} for region '{region_name}': {exc}")
                continue

            if np.ma.isMaskedArray(band):
                data = band.filled(np.nan).astype(np.float32, copy=False)
            else:
                data = np.asarray(band, dtype=np.float32)

            finite_mask = np.isfinite(data)
            if finite_mask.any():
                valid = data[finite_mask]
                vmin = float(np.nanpercentile(valid, 2))
                vmax = float(np.nanpercentile(valid, 98))
                if math.isclose(vmin, vmax):
                    vmin = float(np.nanmin(valid))
                    vmax = float(np.nanmax(valid))
            else:
                vmin, vmax = 0.0, 1.0

            feature_name = feature_labels[idx] if idx < len(feature_labels) else f"Feature_{idx}"
            safe_feature = _sanitize_name(feature_name)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                data,
                extent=extent,
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{feature_name} - {region_name}")
            if neg_x.size:
                ax.scatter(neg_x, neg_y, s=10, c="gold", marker="x", label="Negative")
            if pos_x.size:
                ax.scatter(pos_x, pos_y, s=14, c="red", marker="o", edgecolors="white", linewidths=0.3, label="Positive")
            if aug_x.size:
                ax.scatter(
                    aug_x,
                    aug_y,
                    s=15,
                    facecolors="none",
                    edgecolors="orange",
                    marker="o",
                    linewidths=1.0,
                    label="Positive (aug)",
                )
            if pos_x.size or neg_x.size or aug_x.size:
                ax.legend(loc="upper right")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=feature_name)
            fig.tight_layout()
            feature_path = debug_dir / f"{safe_feature}_{_sanitize_name(region_name)}.png"
            fig.savefig(feature_path, dpi=200)
            plt.close(fig)

    print(f"[info] Debug visualisations saved under {debug_dir}")
