from __future__ import annotations
#
# Stage 2 preprocessing for geoscience features:
#   - loads the feature-selected CSV from stage 1
#   - clips outliers, imputes gaps via IDW neighbours, and smooths with local means
#   - standardises numeric features and preserves requested ancillary columns
#   - optionally emits geographic validation maps and distribution plots
#   - writes the cleaned, normalised dataset alongside diagnostics
"""Outlier and Missing value treatment."""

# TODO: Later consider p025 and p975 for hyper-parameter tuning
import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd
from unify.one_d import _read_csv_with_fallback as _csv_read_with_fallback
DEFAULT_NEIGHBOR_COUNT = 6
MAX_IDW_RECURSION = 8
AUTO_EXCLUDED_FEATURES = frozenset({
    "Dict_Sedimentary",
    "Dict_Igneous",
    "Dict_Metamorphic",
})
# Windows
# python .\1_Preproc2_ReGrid_Outlier_Norm_GCS_data.py `
# --csv "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp_selected.csv" `
# --out "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp_selected_Norm.csv" `
# --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" `
# --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Dict_Sedimentary Dict_Igneous Dict_Metamorphic `
# --extra-columns H3_Address H3_Resolution Training_MVT_Deposit Training_MVT_Occurrence`
# --validate

# Linux
# python 1_Preproc2_ReGrid_Outlier_Norm_GCS_data.py --csv "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_temp_selected.csv" --out "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_temp_selected_Norm.csv" --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Dict_Sedimentary Dict_Igneous Dict_Metamorphic --extra-columns H3_Address H3_Resolution Training_MVT_Deposit Training_MVT_Occurrence --validate
# python 1_Preproc2_ReGrid_Outlier_Norm_GCS_data.py --csv "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_selected.csv"      --out "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_selected_Norm.csv"      --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Dict_Sedimentary Dict_Igneous Dict_Metamorphic --extra-columns H3_Address H3_Resolution Training_MVT_Deposit Training_MVT_Occurrence --validate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second stage preprocessing: clip outliers and impute missing values "
            "for features produced by 1_Preproc1_FeatureSelect_GCS_data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to the feature-selected CSV from stage 1.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination for the processed CSV (defaults to alongside the input).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Specific feature columns to process; omit to infer numeric columns automatically.",
    )
    parser.add_argument(
        "--lat-column",
        default="Latitude_EPSG4326",
        help="Latitude column used to infer nearest neighbours when imputing.",
    )
    parser.add_argument(
        "--lon-column",
        default="Longitude_EPSG4326",
        help="Longitude column used to infer nearest neighbours when imputing.",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=DEFAULT_NEIGHBOR_COUNT,
        help="Number of neighbours to average when imputing missing values.",
    )
    parser.add_argument("--extra-columns", nargs="+", default=[],
                        help="Additional column names to keep (e.g., H3 address).",
                        )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Generate diagnostic scatter plots for Australia (lon>100 & lat<0) and North America (lon<100).",
    )
    return parser.parse_args()

def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered

def _determine_feature_columns(frame: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.features:
        frame_columns = set(frame.columns)
        normalized_auto = {str(col).strip().casefold() for col in AUTO_EXCLUDED_FEATURES}
        candidate = []
        missing = []
        skipped = []
        for col in args.features:
            if col not in frame_columns:
                missing.append(col)
                continue
            if str(col).strip().casefold() in normalized_auto:
                continue
            if not pd.api.types.is_numeric_dtype(frame[col]):
                skipped.append(col)
                continue
            candidate.append(col)
        if missing:
            logging.warning("Skipping missing requested features: %s", ", ".join(missing))
        if skipped:
            logging.warning("Skipping non-numeric requested features: %s", ", ".join(skipped))
        return _unique_ordered(candidate)

    exclude = {args.lat_column, args.lon_column}
    exclude.update(AUTO_EXCLUDED_FEATURES)
    normalized_exclude = {str(col).strip().casefold() for col in exclude}

    def _skip_column(name: object) -> bool:
        return str(name).strip().casefold() in normalized_exclude

    numeric_cols = [
        col
        for col in frame.columns
        if not _skip_column(col) and pd.api.types.is_numeric_dtype(frame[col])
    ]

    if not numeric_cols:
        logging.error("No numeric feature columns detected.")
        sys.exit(1)

    return numeric_cols

def _apply_tukey_fence(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=True)
    mask = ~np.isnan(values)
    if not mask.any():
        return series
    clean = values[mask]
    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return series
    upper_fence = q3 + 1.5 * iqr
    p975 = np.percentile(clean, 97.5)
    exceed_mask = mask & (values > upper_fence)
    if exceed_mask.any():
        values[exceed_mask] = p975
    return pd.Series(values, index=series.index)

def _try_import_h3():
    try:
        import h3
        return h3
    except Exception:
        return None

def _neighbour_map_from_latlon(
    frame: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    neighbour_indices = np.full((n, k), -1, dtype=np.int64)
    neighbour_distances = np.full((n, k), np.nan, dtype=float)
    if lat_col not in frame.columns or lon_col not in frame.columns:
        logging.warning(
            "Latitude/Longitude columns missing; cannot compute geographic neighbours.")
        return neighbour_indices, neighbour_distances
    coords = frame[[lat_col, lon_col]].astype(float).to_numpy()
    valid_mask = np.isfinite(coords).all(axis=1)
    positions = np.arange(n, dtype=np.int64)
    valid_positions = positions[valid_mask]
    valid_coords = coords[valid_mask]
    if len(valid_positions) <= 1:
        return neighbour_indices, neighbour_distances
    max_neigh = min(len(valid_positions), k + 1)
    try:
        from sklearn.neighbors import KDTree  # type: ignore
        tree = KDTree(valid_coords)
        distances, neighbours = tree.query(valid_coords, k=max_neigh)
        distances = np.atleast_2d(distances)
        neighbours = np.atleast_2d(neighbours)
        for row_pos, neigh_indices, dist_row in zip(valid_positions, neighbours, distances):
            count = 0
            for nbr_idx, dist in zip(neigh_indices, dist_row):
                neighbour_pos = valid_positions[nbr_idx]
                if neighbour_pos == row_pos:
                    continue
                neighbour_indices[row_pos, count] = neighbour_pos
                neighbour_distances[row_pos, count] = float(dist)
                count += 1
                if count >= k:
                    break
    except Exception:
        for anchor, row_pos in enumerate(valid_positions):
            diff = valid_coords - valid_coords[anchor]
            dist = np.hypot(diff[:, 0], diff[:, 1])
            order = np.argsort(dist)
            count = 0
            for ord_idx in order:
                neighbour_pos = valid_positions[ord_idx]
                if neighbour_pos == row_pos:
                    continue
                neighbour_indices[row_pos, count] = neighbour_pos
                neighbour_distances[row_pos, count] = float(dist[ord_idx])
                count += 1
                if count >= k:
                    break
    return neighbour_indices, neighbour_distances

def _compute_neighbour_map(
    frame: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _neighbour_map_from_latlon(frame, lat_col, lon_col, k)

def _inverse_distance_weighted_impute(
    series: pd.Series,
    neighbour_distances: np.ndarray,
    safe_neighbour_indices: np.ndarray,
    valid_neighbour_mask: np.ndarray,
    *,
    _depth: int = 0,
) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=True)
    missing_mask = np.isnan(values)
    if not missing_mask.any():
        return series
    neighbour_vals = values[safe_neighbour_indices]
    neighbour_vals = np.where(valid_neighbour_mask, neighbour_vals, np.nan)
    distances = np.where(valid_neighbour_mask, neighbour_distances, np.nan)
    eps = 1e-12
    distances = np.where(distances <= eps, eps, distances)
    weights = np.zeros_like(distances, dtype=float)
    valid_distance_mask = ~np.isnan(distances)
    weights[valid_distance_mask] = 1.0 / \
        np.square(distances[valid_distance_mask])
    weights = np.where(~np.isnan(neighbour_vals), weights, 0.0)
    weight_sums = weights.sum(axis=1)
    weighted_values = np.nansum(neighbour_vals * weights, axis=1)
    fill_mask = missing_mask & (weight_sums > 0)
    values[fill_mask] = weighted_values[fill_mask] / weight_sums[fill_mask]
    updated_series = pd.Series(values, index=series.index)
    remaining_mask = np.isnan(values)
    if not remaining_mask.any():
        return updated_series
    if np.array_equal(remaining_mask, missing_mask) or _depth >= MAX_IDW_RECURSION:
        return _impute_with_mean(updated_series)
    return _inverse_distance_weighted_impute(
        updated_series,
        neighbour_distances,
        safe_neighbour_indices,
        valid_neighbour_mask,
        _depth=_depth + 1,
    )

def _smooth_with_neighbours(
    series: pd.Series,
    safe_neighbour_indices: np.ndarray,
    valid_neighbour_mask: np.ndarray,
) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=True)
    if values.size == 0:
        return series
    neighbour_vals = values[safe_neighbour_indices]
    neighbour_vals = np.where(valid_neighbour_mask, neighbour_vals, np.nan)
    neighbour_sum = np.nansum(neighbour_vals, axis=1)
    neighbour_count = np.sum(~np.isnan(neighbour_vals), axis=1)
    valid_self = ~np.isnan(values)
    total_sum = neighbour_sum + np.where(valid_self, values, 0.0)
    total_count = neighbour_count + valid_self.astype(int)
    update_mask = total_count > 0
    values[update_mask] = total_sum[update_mask] / total_count[update_mask]
    return pd.Series(values, index=series.index)

def _impute_with_mean(series: pd.Series) -> pd.Series:
    """Fill missing values with the feature mean."""
    if not series.isna().any():
        return series
    mean_value = series.mean()
    if pd.isna(mean_value):
        return series
    return series.fillna(float(mean_value))

def _standard_scale(frame: pd.DataFrame, columns: list[str]) -> None:
    """Scale selected columns to zero mean and unit variance."""
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        mean_val = values.mean()
        std_val = values.std(ddof=0)
        if pd.isna(mean_val) or pd.isna(std_val):
            continue
        if std_val == 0:
            frame[column] = values.fillna(0.0)
            continue
        standardized = (values - mean_val) / std_val
        frame[column] = standardized

def _generate_validation_plots(
    frame: pd.DataFrame,
    output_path: Path,
    lat_column: str,
    lon_column: str,
    features: list[str],
    subdir_name: str | None = None,
) -> None:
    try:
        import datashader as ds
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception as exc:
        logging.warning(
            "Skipping validation plots; datashader/matplotlib unavailable: %s", exc)
        return
    plt.ioff()
    if subdir_name:
        root_dir = output_path.parent / f"{output_path.stem}_plots"
        plot_dir = root_dir / subdir_name
    else:
        plot_dir = output_path.parent / f"{output_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    region_filters = {
        "AU": (frame[lon_column] > 100) & (frame[lat_column] < 0),
        "NA": (frame[lon_column] < 100),
    }
    for feature in features:
        logging.info(" - Plotting %s feature.", feature)
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        for suffix, mask in region_filters.items():
            selected_mask = mask & values.notna()
            if not selected_mask.any():
                logging.debug(
                    "No data for %s in region %s; skipping plot.", feature, suffix)
                continue
            subset = frame.loc[selected_mask, [lon_column, lat_column]].copy()
            subset[feature] = values.loc[selected_mask]
            data = subset[feature].to_numpy(dtype=float)
            finite_data = data[np.isfinite(data)]
            if finite_data.size == 0:
                logging.debug(
                    "No finite data for %s in region %s; skipping plot.", feature, suffix)
                continue
            vmin = float(np.nanmin(finite_data))
            vmax = float(np.nanmax(finite_data))
            if math.isclose(vmin, vmax):
                vmax = vmin + 1e-9
            x_min = float(subset[lon_column].min())
            x_max = float(subset[lon_column].max())
            y_min = float(subset[lat_column].min())
            y_max = float(subset[lat_column].max())
            if not all(math.isfinite(val) for val in (x_min, x_max, y_min, y_max)):
                logging.debug(
                    "Invalid coordinates for %s in region %s; skipping plot.", feature, suffix)
                continue
            if math.isclose(x_min, x_max):
                delta = max(abs(x_min) * 1e-6, 1e-6)
                x_min -= delta
                x_max += delta
            if math.isclose(y_min, y_max):
                delta = max(abs(y_min) * 1e-6, 1e-6)
                y_min -= delta
                y_max += delta
            span_x = x_max - x_min
            span_y = y_max - y_min
            plot_width = 800
            plot_height = max(
                1, min(800, int(plot_width * span_y / max(span_x, 1e-9))))
            canvas = ds.Canvas(
                plot_width=plot_width,
                plot_height=plot_height,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
            )
            agg = canvas.points(subset, lon_column,
                                lat_column, agg=ds.mean(feature))
            agg_array = agg.to_numpy()
            if not np.isfinite(agg_array).any():
                logging.debug(
                    "Datashader aggregation produced no finite values for %s in region %s.",
                    feature,
                    suffix,
                )
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            cmap = plt.get_cmap("viridis")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            image = ax.imshow(
                np.ma.masked_invalid(agg_array),
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            cbar = fig.colorbar(image, ax=ax)
            cbar.set_label(feature)
            ax.set_xlabel(lon_column)
            ax.set_ylabel(lat_column)
            ax.set_title(f"{feature} ({suffix})")
            ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
            fig.savefig(plot_dir / f"{safe_name}_{suffix}.png", dpi=150)
            plt.close(fig)
    logging.info("Validation plots saved to %s", plot_dir)

def _generate_distribution_plots(
    pre_frame: pd.DataFrame,
    post_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    output_path: Path,
    features: Iterable[str],
    lat_column: str,
    lon_column: str,
) -> None:
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(
            "Skipping distribution plots; matplotlib unavailable: %s", exc)
        return

    def _build_bins(series: pd.Series) -> np.ndarray:
        data = series.to_numpy(dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return np.linspace(-0.5, 0.5, 11)
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if math.isclose(vmin, vmax):
            delta = max(abs(vmin) * 1e-3, 1e-3)
            vmin -= delta
            vmax += delta
        return np.linspace(vmin, vmax, 30)

    plot_dir = output_path.parent / f"{output_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = plot_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    default_series = pd.Series(index=reference_frame.index, dtype=float)
    lat_series = pd.to_numeric(reference_frame.get(lat_column, default_series), errors="coerce")
    lon_series = pd.to_numeric(reference_frame.get(lon_column, default_series), errors="coerce")
    region_filters = {
        "AU": ((lon_series > 100) & (lat_series < 0)).fillna(False),
        "NA": (lon_series < 100).fillna(False),
    }
    region_colors = {
        "AU": "#1f77b4",
        "NA": "#ff7f0e",
    }

    for feature in features:
        if feature not in pre_frame.columns or feature not in post_frame.columns:
            continue
        pre_series = pd.to_numeric(pre_frame[feature], errors="coerce")
        post_series = pd.to_numeric(post_frame[feature], errors="coerce")

        has_data = False
        for mask in region_filters.values():
            if pre_series.where(mask).notna().any() or post_series.where(mask).notna().any():
                has_data = True
                break
        if not has_data:
            continue

        row_definitions = [
            ("All regions", list(region_filters.items())),
            ("North America", [("NA", region_filters["NA"])]),
            ("Australia", [("AU", region_filters["AU"])]),
        ]

        fig, axes = plt.subplots(len(row_definitions), 2, figsize=(11, 10), sharey="row")
        fig.suptitle(feature)

        for row_idx, (row_label, row_regions) in enumerate(row_definitions):
            before_ax = axes[row_idx, 0]
            after_ax = axes[row_idx, 1]

            union_mask = None
            for _, region_mask in row_regions:
                union_mask = region_mask.copy() if union_mask is None else (union_mask | region_mask)
            if union_mask is None:
                union_mask = pd.Series(False, index=pre_series.index)
            union_mask = union_mask.fillna(False)

            pre_bins = _build_bins(pre_series.where(union_mask))
            post_bins = _build_bins(post_series.where(union_mask))

            stats_lines: list[str] = []
            row_pre_plotted = False
            row_post_plotted = False

            for region_name, region_mask in row_regions:
                color = region_colors.get(region_name, "#1f77b4")

                pre_mask = region_mask & pre_series.notna()
                if pre_mask.any():
                    region_pre = pre_series.loc[pre_mask].to_numpy(dtype=float)
                    before_ax.hist(
                        region_pre,
                        bins=pre_bins,
                        density=True,
                        alpha=0.55,
                        label=region_name,
                        color=color,
                    )
                    mean_val = float(np.mean(region_pre)) if region_pre.size else float("nan")
                    std_val = float(np.std(region_pre, ddof=0)) if region_pre.size else float("nan")
                    if np.isfinite(mean_val):
                        before_ax.axvline(mean_val, color=color, linestyle="--", linewidth=1)
                    stats_lines.append(f"{region_name}: mu={mean_val:.2f}, sigma={std_val:.2f}")
                    row_pre_plotted = True

                post_mask = region_mask & post_series.notna()
                if post_mask.any():
                    region_post = post_series.loc[post_mask].to_numpy(dtype=float)
                    after_ax.hist(
                        region_post,
                        bins=post_bins,
                        density=True,
                        alpha=0.55,
                        label=region_name,
                        color=color,
                    )
                    row_post_plotted = True

            if stats_lines:
                before_ax.text(
                    0.02,
                    0.98,
                    "\n".join(stats_lines),
                    transform=before_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
                )

            if not row_pre_plotted:
                before_ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=before_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            if not row_post_plotted:
                after_ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=after_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            before_ax.set_title(f"{row_label} - before scaling")
            after_ax.set_title(f"{row_label} - after scaling")
            before_ax.set_xlabel(feature)
            after_ax.set_xlabel(feature)
            if row_idx == 0:
                before_ax.set_ylabel("Density")
            else:
                before_ax.set_ylabel("")

            handles, labels = before_ax.get_legend_handles_labels()
            if not handles:
                handles, labels = after_ax.get_legend_handles_labels()
            if handles and row_idx == 0:
                before_ax.legend(handles=handles, labels=labels, loc="upper right")
                after_ax.legend(handles=handles, labels=labels, loc="upper right")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        fig.savefig(dist_dir / f"{safe_name}_distribution.png", dpi=150)
        plt.close(fig)

    logging.info("Distribution plots saved to %s", dist_dir)

def _generate_categorical_plots(
    frame: pd.DataFrame,
    output_path: Path,
    columns: Iterable[str],
) -> None:
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(
            "Skipping categorical plots; matplotlib unavailable: %s", exc)
        return
    plot_dir = output_path.parent / f"{output_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        if column not in frame.columns:
            continue
        logging.info(" - Plotting %s distribution.", column)
        series = frame[column]
        counts = series.fillna("<NA>").astype(str).value_counts()
        if counts.empty:
            logging.debug(
                "No data for column %s; skipping categorical plot.", column)
            continue
        top_counts = counts.iloc[:20].copy()
        if len(counts) > 20:
            remainder = counts.iloc[20:].sum()
            if remainder:
                top_counts.loc["<OTHER>"] = remainder
        labels = [str(label) for label in top_counts.index]
        values = top_counts.to_numpy()
        positions = np.arange(len(labels))
        width = max(6.0, min(18.0, 0.6 * len(labels)))
        fig, ax = plt.subplots(figsize=(width, 4.5))
        ax.bar(positions, values, color="#1f77b4")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"{column} distribution")
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", column)
        fig.savefig(plot_dir / f"{safe_name}_categorical.png", dpi=150)
        plt.close(fig)
    logging.info("Categorical plots saved to %s", plot_dir)

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    try:
        frame = _csv_read_with_fallback(args.csv)
    except Exception as exc:
        logging.error("Failed to load CSV %s: %s", args.csv, exc)
        sys.exit(1)
    frame = frame.reset_index(drop=True)
    dtypes = frame.dtypes
    requested_extra_columns = _unique_ordered(args.extra_columns)
    extra_columns_to_keep = [col for col in requested_extra_columns if col in frame.columns]
    missing_extra_columns = [col for col in requested_extra_columns if col not in frame.columns]
    if missing_extra_columns:
        logging.warning(
            "Skipping missing extra columns: %s", ", ".join(missing_extra_columns))
    processed_columns = _determine_feature_columns(frame, args)
    if not processed_columns:
        logging.error("No feature columns selected for processing.")
        sys.exit(1)

    neighbour_count = max(1, args.neighbor_count)
    logging.info("Processing neighbour map (k=%d).", neighbour_count)
    neighbour_indices, neighbour_distances = _compute_neighbour_map(
        frame, args.lat_column, args.lon_column, neighbour_count)
    valid_neighbour_mask = neighbour_indices >= 0
    safe_neighbour_indices = np.where(
        valid_neighbour_mask, neighbour_indices, 0)
    logging.info("Processing %d feature columns.", len(processed_columns))
    for column in processed_columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            logging.warning("Skipping non-numeric feature %s", column)
            continue        
        logging.info(" - Processing %s feature .", column)
        numeric = pd.to_numeric(frame[column], errors="coerce")
        clipped = _apply_tukey_fence(numeric)
        imputed = _inverse_distance_weighted_impute(clipped, neighbour_distances, safe_neighbour_indices, valid_neighbour_mask)
        smoothed = _smooth_with_neighbours(imputed, safe_neighbour_indices, valid_neighbour_mask)
        frame[column] = smoothed
    pre_map_frame = frame.copy(deep=True)
    pre_normalized_values = pre_map_frame[processed_columns].copy()
    _standard_scale(frame, processed_columns)
    post_normalized_values = frame[processed_columns].copy()
    if args.out is None:
        output_path = args.csv.with_name(f"{args.csv.stem}_Processed.csv")
    else:
        output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.validate:
        _generate_validation_plots(
            pre_map_frame,
            output_path,
            args.lat_column,
            args.lon_column,
            processed_columns,
            subdir_name="Plot_Before_Norm",
        )
        _generate_validation_plots(
            frame,
            output_path,
            args.lat_column,
            args.lon_column,
            processed_columns,
            subdir_name="Plot_After_Norm",
        )
        _generate_distribution_plots(
            pre_normalized_values,
            post_normalized_values,
            frame,
            output_path,
            processed_columns,
            args.lat_column,
            args.lon_column,
        )
        excluded_candidates = [col for col in AUTO_EXCLUDED_FEATURES if col in dtypes.index and col not in processed_columns]
        excluded_numeric = [col for col in excluded_candidates if pd.api.types.is_numeric_dtype(dtypes[col])]
        if excluded_numeric:
            _generate_validation_plots(frame, output_path, args.lat_column, args.lon_column, excluded_numeric)
        excluded_categorical = [col for col in excluded_candidates if not pd.api.types.is_numeric_dtype(dtypes[col])]
        skip_for_categorical = set(processed_columns)
        skip_for_categorical.update(excluded_candidates)
        skip_for_categorical.add(args.lat_column)
        skip_for_categorical.add(args.lon_column)
        skip_for_categorical.update(extra_columns_to_keep)
        categorical_columns: list[str] = []
        categorical_columns.extend(excluded_categorical)
        for column in frame.columns:
            if column in skip_for_categorical:
                continue
            if not pd.api.types.is_numeric_dtype(dtypes[column]):
                categorical_columns.append(column)
        if categorical_columns:
            _generate_categorical_plots(frame, output_path, categorical_columns)
    def _append_column(name: str, seen: set[str], target: list[str]) -> None:
        if name in seen or name not in frame.columns:
            return
        target.append(name)
        seen.add(name)

    columns_to_export: list[str] = []
    seen_columns: set[str] = set()
    for column in (args.lat_column, args.lon_column):
        _append_column(column, seen_columns, columns_to_export)
    for column in extra_columns_to_keep:
        _append_column(column, seen_columns, columns_to_export)
    for column in processed_columns:
        _append_column(column, seen_columns, columns_to_export)
    for column in frame.columns:
        _append_column(column, seen_columns, columns_to_export)
    output_frame = frame.loc[:, columns_to_export]
    try:
        output_frame.to_csv(output_path, index=False)
    except Exception as exc:
        logging.error("Failed to write processed CSV to %s: %s", output_path, exc)
        sys.exit(1)
    logging.info("Wrote processed CSV with %d rows and %d columns to %s",
                 len(output_frame.index), len(output_frame.columns), output_path)
    print(output_path)

if __name__ == "__main__":
    main()
