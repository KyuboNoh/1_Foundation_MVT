#!/usr/bin/env python3
"""Select a subset of GSC CSV columns for downstream processing."""

# python .\1_Preproc1_FeatureSelect_GCS_data.py `
# --csv "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp.csv" `
# --out "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp_selected.csv" `
# --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" `
# --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity `
# --special-columns1 Geology_Lithology_Contact Geology_Dictionary_Alkalic Geology_Dictionary_Anatectic Geology_Dictionary_Calcareous Geology_Dictionary_Carbonaceous Geology_Dictionary_Cherty Geology_Dictionary_CoarseClastic Geology_Dictionary_Evaporitic Geology_Dictionary_Felsic Geology_Dictionary_FineClastic Geology_Dictionary_Gneissose Geology_Dictionary_Igneous Geology_Dictionary_Intermediate Geology_Dictionary_Pegmatitic Geology_Dictionary_RedBed Geology_Dictionary_Schistose Geology_Dictionary_Sedimentary Geology_Dictionary_UltramaficMafic `
# --extra-columns H3_Address H3_Resolution H3_Geometry `
# --validate

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import h3
except ImportError:
    h3 = None
from unify.one_d import _read_csv_with_fallback as _csv_read_with_fallback

import pandas as pd

def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Filter the GSC CSV to keep latitude/longitude and selected feature columns, and regrid H3 data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to the input CSV file.")
    parser.add_argument("--out", type=Path, default=None, help="Path for the filtered CSV file (defaults to alongside the input CSV).",)
    parser.add_argument( "--features", nargs="+", required=True, help="Feature column names to retain in the output CSV.", )
    # ✅ NEW: H3 column names with sane defaults
    parser.add_argument("--special-columns1", dest="special_columns1",
                        nargs="+", default=[], help="Dictionary indicator columns to collapse into lithology dictionary layers (following Lawley et al., 2021).",
                        )
    parser.add_argument("--lat-column", default="Latitude_EPSG4326",
                        help="Latitude column name to keep alongside the features.",)
    parser.add_argument("--lon-column", default="Longitude_EPSG4326",
                        help="Longitude column name to keep alongside the features.",)
    parser.add_argument("--extra-columns", nargs="+", default=[],
                        help="Additional column names to keep (e.g., identifiers).",
                        )
    parser.add_argument("--grid-size", type=int, default=7,
                        help="Size of the regular grid (NxN). If not specified, derived from H3 resolution.",
                        )
    parser.add_argument("--grid-method", choices=['mean', 'sum', 'min', 'max'], default='mean',
                        help="Method to aggregate multiple values in a grid cell.",)

    # parser.add_argument("--h3-address-column", default="H3_Address",
    #                     help="Column containing H3 addresses.")
    # parser.add_argument("--h3-resolution-column", default="H3_Resolution",
    #                     help="Column containing H3 resolution (integer).")
    # parser.add_argument("--h3-geometry-column", default="H3_Geometry",
    #                     help="Column containing H3 cell polygon WKT (optional).")    
    
    parser.add_argument("--validate",action="store_true",
                        help="Raise an error if any requested columns are missing.",)

    args, unknown = parser.parse_known_args()
    args.csv = args.csv.expanduser()
    if not args.csv.exists():
        parser.error(f"CSV file not found: {args.csv}")
    args.csv = args.csv.resolve()

    if args.out is not None:
        args.out = args.out.expanduser()

    return args, unknown


SEDIMENTARY_DICT_COLUMNS = {
    "Geology_Dictionary_FineClastic",
    "Geology_Dictionary_Calcareous",
    "Geology_Dictionary_Carbonaceous",
    "Geology_Dictionary_Cherty",
    "Geology_Dictionary_CoarseClastic",
    "Geology_Dictionary_Evaporitic",
    "Geology_Dictionary_RedBed",
    "Geology_Dictionary_Sedimentary",
}

IGNEOUS_DICT_COLUMNS = {
    "Geology_Dictionary_Alkalic",
    "Geology_Dictionary_Felsic",
    "Geology_Dictionary_Igneous",
    "Geology_Dictionary_Intermediate",
    "Geology_Dictionary_Pegmatitic",
    "Geology_Dictionary_UltramaficMafic",
}

METAMORPHIC_DICT_COLUMNS = {
    "Geology_Dictionary_Anatectic",
    "Geology_Dictionary_Gneissose",
    "Geology_Dictionary_Schistose",
}

DICTIONARY_GROUPS = {
    "Dict_Sedimentary": SEDIMENTARY_DICT_COLUMNS,
    "Dict_Igneous": IGNEOUS_DICT_COLUMNS,
    "Dict_Metamorphic": METAMORPHIC_DICT_COLUMNS,
}

def _normalize_column_key(name):
    if not name:
        return ""
    clean = str(name).replace("\ufeff", "").replace("ï»¿", "").strip()
    if clean.startswith('"') and clean.endswith('"') and len(clean) >= 2:
        clean = clean[1:-1]
    if clean.startswith("'") and clean.endswith("'") and len(clean) >= 2:
        clean = clean[1:-1]
    return clean.strip().lower()

def _parse_polygon_wkt(wkt: str) -> Optional[np.ndarray]:
    """Parse WKT polygon string into numpy array of coordinates."""
    text = (wkt or "").strip()
    if not text or not text.upper().startswith("POLYGON"):
        return None
    start = text.find("((")
    end = text.rfind("))")
    if start == -1 or end == -1 or end <= start + 2:
        return None
    coord_text = text[start + 2 : end]
    coords: list[tuple[float, float]] = []
    for token in coord_text.split(','):
        parts = token.strip().split()
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        coords.append((lon, lat))
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return np.asarray(coords, dtype=np.float64)

def detect_h3_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    """Auto-detect H3 address, resolution, and geometry columns from DataFrame header."""
    def norm(s):
        return str(s).replace('\ufeff', '').replace('ï»¿', '').replace('"', '').replace("'", '').strip().lower()
    candidates = {norm(col): col for col in frame.columns}
    addr = res = geom = None
    for key, col in candidates.items():
        if addr is None and 'h3' in key and 'addr' in key:
            addr = col
        if res is None and 'h3' in key and ('res' in key or 'resolution' in key):
            res = col
        if geom is None and 'h3' in key and ('geom' in key or 'wkt' in key):
            geom = col
    # Fallbacks: try partial matches if not found
    if addr is None:
        for key, col in candidates.items():
            if 'h3' in key:
                addr = col; break
    if res is None:
        for key, col in candidates.items():
            if 'res' in key or 'resolution' in key:
                res = col; break
    if geom is None:
        for key, col in candidates.items():
            if 'geom' in key or 'wkt' in key:
                geom = col; break
    if not addr or not res:
        raise ValueError(f"Could not auto-detect H3 address or resolution columns in: {list(frame.columns)}")
    return addr, res, geom

# def regrid_h3_data(
#     frame: pd.DataFrame,
#     feature_columns: list[str],
#     h3_address_col: str = "H3_Address",
#     h3_resolution_col: str = "H3_Resolution",
#     h3_geometry_col: str = "H3_Geometry",
#     lat_column: str = "Latitude_EPSG4326",
#     lon_column: str = "Longitude_EPSG4326",
#     grid_size: Optional[int] = None,
#     method: str = "mean"
# ) -> pd.DataFrame:
#     """Convert H3 hexagonal grid data to regular lat/lon grid while preserving values."""
#     if h3 is None:
#         raise ImportError("h3-py package required for H3 regridding")

#     def _norm(s: str) -> str:
#         return normalize_column_name(str(s)).lower()

#     # Build a single, consistent lookup from normalized -> actual column name
#     norm_map = {_norm(col): col for col in frame.columns}

#     # Resolve actual H3 column names (allow auto-detect fallbacks)
#     addr_key = _norm(h3_address_col)
#     res_key  = _norm(h3_resolution_col)
#     geom_key = _norm(h3_geometry_col)

#     actual_h3_col   = norm_map.get(addr_key)
#     actual_res_col  = norm_map.get(res_key)
#     actual_geom_col = norm_map.get(geom_key)

#     if actual_h3_col is None or actual_res_col is None:
#         # Try auto-detect if names differ in the file
#         det_addr, det_res, det_geom = detect_h3_columns(frame)
#         actual_h3_col = actual_h3_col or det_addr
#         actual_res_col = actual_res_col or det_res
#         actual_geom_col = actual_geom_col or det_geom

#     if actual_h3_col is None:
#         raise ValueError(f"H3 address column '{h3_address_col}' not found in DataFrame columns: {list(frame.columns)}")

#     # Determine grid size from H3 resolution if not specified
#     if grid_size is None and actual_res_col in frame.columns:
#         res_mode = frame[actual_res_col].mode(dropna=True)
#         if res_mode.empty:
#             raise ValueError(f"H3 resolution column '{actual_res_col}' has no valid values; cannot infer grid size.")
#         h3_res_val = res_mode.iloc[0]
#         try:
#             if pd.isna(h3_res_val):
#                 raise ValueError("H3 resolution value is NaN")
#             h3_res = int(float(str(h3_res_val).strip()))
#         except Exception as e:
#             raise ValueError(f"H3 resolution value '{h3_res_val}' in column '{actual_res_col}' is not a valid integer: {e}")
#         edge_km = h3.edge_length(h3_res, unit="km")
#         grid_size = int(np.clip(1.2 * edge_km, 16, 512))
#         logging.info(f"Using grid size {grid_size} based on H3 resolution {h3_res}")
#     elif grid_size is None:
#         grid_size = 256
#         logging.info(f"Using default grid size {grid_size}")

#     # Get coordinates from H3 addresses (fallback to geometry centroid if needed)
#     lats, lons, polygons = [], [], []
#     for h3_addr in frame[actual_h3_col]:
#         try:
#             if pd.isna(h3_addr):
#                 raise ValueError("NaN value")
#             lat, lon = h3.h3_to_geo(str(h3_addr).strip().lower())
#             lats.append(lat); lons.append(lon)
#         except Exception:
#             lats.append(np.nan); lons.append(np.nan)

#     if actual_geom_col in frame.columns:
#         for idx, geom in enumerate(frame[actual_geom_col]):
#             poly = _parse_polygon_wkt(geom)
#             polygons.append(poly)
#             if poly is not None and (not np.isfinite(lats[idx]) or not np.isfinite(lons[idx])):
#                 lats[idx] = float(np.mean(poly[:, 1]))
#                 lons[idx] = float(np.mean(poly[:, 0]))
#     else:
#         polygons = [None] * len(frame)

#     frame = frame.copy()
#     frame['lat'] = lats
#     frame['lon'] = lons

#     valid = np.isfinite(frame['lat']) & np.isfinite(frame['lon'])
#     if not valid.any():
#         raise ValueError("No valid coordinates found from H3 addresses or geometry")

#     lat_min, lat_max = frame.loc[valid, 'lat'].min(), frame.loc[valid, 'lat'].max()
#     lon_min, lon_max = frame.loc[valid, 'lon'].min(), frame.loc[valid, 'lon'].max()

#     lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
#     lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
#     lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
#     lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

#     # Get all columns to process (both feature columns and aggregated ones)
#     columns_to_process = list(feature_columns)
#     columns_to_process.extend(col for col in frame.columns if col.startswith('Dict_'))

#     # Pre-compute cell assignments for efficiency
#     cell_assignments = {}  # (i, j) -> [row indices]
#     for idx in range(len(frame)):
#         if not valid.iloc[idx]:
#             continue

#         poly = polygons[idx]
#         if poly is not None:
#             from shapely.geometry import Point, Polygon
#             shape_poly = Polygon(poly)
#             lat_min_idx = max(0, np.searchsorted(lat_edges, poly[:, 1].min()) - 1)
#             lat_max_idx = min(grid_size, np.searchsorted(lat_edges, poly[:, 1].max()) + 1)
#             lon_min_idx = max(0, np.searchsorted(lon_edges, poly[:, 0].min()) - 1)
#             lon_max_idx = min(grid_size, np.searchsorted(lon_edges, poly[:, 0].max()) + 1)
#             for i in range(lat_min_idx, lat_max_idx):
#                 for j in range(lon_min_idx, lon_max_idx):
#                     if shape_poly.contains(Point(lon_centers[j], lat_centers[i])):
#                         cell_assignments.setdefault((i, j), []).append(idx)
#         else:
#             i = np.clip(np.searchsorted(lat_edges, frame['lat'].iloc[idx]) - 1, 0, grid_size - 1)
#             j = np.clip(np.searchsorted(lon_edges, frame['lon'].iloc[idx]) - 1, 0, grid_size - 1)
#             cell_assignments.setdefault((i, j), []).append(idx)

#     # Process each column and create gridded data
#     all_data = []  # Will contain one row per grid cell

#     # Get grid cell indices
#     cell_indices = sorted(cell_assignments.keys())
    
#     # Process each grid cell
#     for i, j in cell_indices:
#         cell_row = {
#             lat_column: lat_centers[i],
#             lon_column: lon_centers[j]
#         }
        
#         indices = cell_assignments[(i, j)]
        
#         # Process each feature
#         for feature in columns_to_process:
#             if feature not in frame.columns:
#                 continue

#             # Get values for this cell
#             values = frame[feature].iloc[indices]
#             values = values.dropna()
            
#             if values.empty:
#                 continue

#             # Determine feature type and process accordingly
#             is_binary = feature.startswith('Dict_')
#             is_categorical = not pd.api.types.is_numeric_dtype(frame[feature]) or is_binary

#             if is_categorical:
#                 # For categorical/binary features, use mode (most frequent value)
#                 mode_val = values.mode().iloc[0]
#                 if is_binary:
#                     # Ensure binary values are stored as integers
#                     cell_row[feature] = int(mode_val)
#                 else:
#                     cell_row[feature] = str(mode_val)
#             else:
#                 # For numerical features, use specified aggregation method
#                 try:
#                     num_values = pd.to_numeric(values, errors='coerce')
#                     finite_values = num_values[np.isfinite(num_values)]
#                     if not finite_values.empty:
#                         if method == 'mean':
#                             cell_row[feature] = finite_values.mean()
#                         elif method == 'sum':
#                             cell_row[feature] = finite_values.sum()
#                         elif method == 'min':
#                             cell_row[feature] = finite_values.min()
#                         elif method == 'max':
#                             cell_row[feature] = finite_values.max()
#                 except Exception as e:
#                     logging.warning(f"Could not process {feature} in grid cell ({i},{j}): {e}")
        
#         # Only add cells that have data (more than just lat/lon)
#         if len(cell_row) > 2:
#             all_data.append(cell_row)

#     if not all_data:
#         raise ValueError("No features were successfully regridded")

#     # Create DataFrame with all grid cells
#     result_frame = pd.DataFrame(all_data)
    
#     # Add grid metadata as attributes
#     result_frame.attrs['grid_size'] = grid_size
#     result_frame.attrs['lat_min'] = lat_min
#     result_frame.attrs['lat_max'] = lat_max
#     result_frame.attrs['lon_min'] = lon_min
#     result_frame.attrs['lon_max'] = lon_max

#     # Ensure consistent column order: lat, lon, features
#     column_order = [lat_column, lon_column]
#     feature_columns = [col for col in result_frame.columns if col not in column_order]
#     column_order.extend(sorted(feature_columns))
    
#     return result_frame[column_order]


def normalize_column_name(name: str) -> str:
    """Normalize column name by removing BOM markers and cleaning up encoding issues."""
    if not name:
        return ""
    # Remove BOM and other special characters
    clean = (name.replace('\ufeff', '')
                .replace('ï»¿', '')
                .replace('\x00', '')
                .strip())
    # Remove quotes if present
    if clean.startswith('"') and clean.endswith('"') and len(clean) >= 2:
        clean = clean[1:-1]
    if clean.startswith('\'') and clean.endswith('\'') and len(clean) >= 2:
        clean = clean[1:-1]
    return clean.strip()

def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _collect_columns(args: argparse.Namespace) -> list[str]:
    requested = [args.lat_column, args.lon_column]
    requested.extend(args.features)
    requested.extend(args.extra_columns)

    # # ✅ Always include the H3 columns
    # for h3_col in [args.h3_address_column, args.h3_resolution_column, args.h3_geometry_column]:
    #     if h3_col and h3_col not in requested:
    #         requested.append(h3_col)

    return _unique_ordered(requested)


def _present_absent_to_int(series: pd.Series) -> pd.Series:
    """Normalize dictionary Present/Absent flags to integer indicators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).clip(lower=0, upper=1).astype(int)
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
    mapped = normalized.map({"present": 1, "absent": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    return mapped.fillna(0).astype(int)

def _collapse_dictionary_layers(frame: pd.DataFrame, special_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Collapse dictionary columns into grouped lithology indicators."""
    if not special_columns:
        return frame, []

    present = [col for col in special_columns if col in frame.columns]
    missing = [col for col in special_columns if col not in frame.columns]
    if missing:
        logging.warning("Skipping missing special columns: %s", ", ".join(missing))

    aggregated: dict[str, pd.Series] = {}
    for group_name, columns in DICTIONARY_GROUPS.items():
        active_columns = [col for col in columns if col in frame.columns]
        if not active_columns:
            aggregated[group_name] = pd.Series(0, index=frame.index, dtype=int)
            continue
        stacked = pd.DataFrame(
            {col: _present_absent_to_int(frame[col]) for col in active_columns},
            index=frame.index,
        )
        aggregated[group_name] = stacked.max(axis=1).astype(int)

    if aggregated:
        frame = frame.assign(**aggregated)

    drop_columns = [col for col in present if col in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    return frame, list(aggregated.keys())


def _generate_validation_plots(
    frame: pd.DataFrame,
    output_path: Path,
    selected_columns: list[str],
    aggregated_columns: list[str],
    lat_column: str = "Latitude_EPSG4326",
    lon_column: str = "Longitude_EPSG4326",
) -> None:
    """Save quick-look scatter plots for validation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("Skipping validation plots; matplotlib unavailable: %s", exc)
        return

    plt.ioff()
    plot_dir = output_path.parent / f"{output_path.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    columns_to_plot = _unique_ordered(selected_columns + aggregated_columns)
    columns_to_plot = [
        column
        for column in columns_to_plot
        if column not in {lat_column, lon_column} and column in frame.columns
    ]

    if not columns_to_plot:
        logging.info("Validation plots requested but no feature columns available.")
        return

    import re

    for column in columns_to_plot:
        subset = frame[[lon_column, lat_column, column]].dropna()
        if subset.empty:
            logging.debug(
                "Skipping plot for %s because data is empty after dropping NaNs.",
                column,
            )
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        x = subset[lon_column]
        y = subset[lat_column]
        values = subset[column]

        if pd.api.types.is_numeric_dtype(values):
            scatter = ax.scatter(x, y, c=values, cmap="viridis", s=15, alpha=0.8)
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label(column)
        else:
            categories = values.astype(str)
            codes, labels = pd.factorize(categories, sort=True)
            scatter = ax.scatter(x, y, c=codes, cmap="tab20", s=20, alpha=0.8)
            unique_codes = sorted(set(codes))
            handles = []
            for code in unique_codes:
                label = labels[code]
                color = scatter.cmap(scatter.norm(code))
                handles.append(mpatches.Patch(color=color, label=str(label)))
            ax.legend(handles=handles, title=column, loc="best", fontsize="small")

        ax.set_xlabel(lon_column)
        ax.set_ylabel(lat_column)
        ax.set_title(f"{column} over {lat_column}/{lon_column}")

        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", column)
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    logging.info("Validation plots saved to %s", plot_dir)


def main() -> None:
    # Use INFO by default to avoid verbose debug output during normal runs
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args, unknown = parse_args()

    if unknown:
        logging.warning("Ignoring unknown arguments: %s", " ".join(unknown))

    requested_columns = _collect_columns(args)
    special_columns = _unique_ordered(args.special_columns1)

    try:
        header = _csv_read_with_fallback(args.csv, {"nrows": 0})
    except Exception as exc:
        logging.error("Failed to read CSV header from %s: %s", args.csv, exc)
        sys.exit(1)

    # Normalize column names
    header.columns = [normalize_column_name(col) for col in header.columns]
    available_columns = list(header.columns)
    available_set = set(available_columns)
    # Create lookup from normalized to original names
    column_lookup = {normalize_column_name(col): col for col in header.columns}
    
    # Debug: print available columns
    logging.debug("Available columns in CSV: %s", available_columns)
    logging.debug("Column lookup after normalization: %s", column_lookup)

    read_candidates = _unique_ordered(requested_columns + special_columns)
    logging.debug("Requested columns: %s", read_candidates)
    missing = [col for col in read_candidates if col not in available_set]
    if missing and args.validate:
        logging.error("Missing required columns: %s", ", ".join(missing))
        sys.exit(1)
    if missing:
        logging.warning("Skipping missing columns: %s", ", ".join(missing))

    selected_columns = [col for col in requested_columns if col in available_set]
    selected_special_columns = [col for col in special_columns if col in available_set]
    if not selected_columns:
        logging.error("No valid columns left after filtering; cannot write output CSV.")
        sys.exit(1)

    use_columns = _unique_ordered(selected_columns + selected_special_columns)

    try:
        frame = _csv_read_with_fallback(args.csv, {"usecols": use_columns})
    except Exception as exc:
        logging.error("Pandas failed to load requested columns: %s", exc)
        sys.exit(1)

    frame = frame.reindex(columns=use_columns)

    aggregated_columns: list[str] = []
    if selected_special_columns:
        frame, aggregated_columns = _collapse_dictionary_layers(frame, selected_special_columns)

    final_columns = _unique_ordered(selected_columns + aggregated_columns)
    frame = frame.reindex(columns=[col for col in final_columns if col in frame.columns])

    # print(frame)
    # # Regrid immediately after column selection
    # try:
    #     frame = regrid_h3_data(
    #         frame,
    #         feature_columns=args.features,
    #         h3_address_col=args.h3_address_column,
    #         h3_resolution_col=args.h3_resolution_column,
    #         h3_geometry_col=args.h3_geometry_column,
    #         lat_column=args.lat_column,
    #         lon_column=args.lon_column,
    #         grid_size=args.grid_size,
    #         method=args.grid_method
    #     )
    # except Exception as exc:
    #     logging.error("Failed to regrid H3 data: %s", exc)
    #     sys.exit(1)
    # print(frame)

    if args.out is None:
        output_path = args.csv.with_name(f"{args.csv.stem}_Selected.csv")
    else:
        output_path = args.out

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate validation plots
    import logging as _logging
    _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
    _generate_validation_plots(
            frame,
            output_path,
            selected_columns,
            aggregated_columns,
            lat_column=args.lat_column,
            lon_column=args.lon_column,
    )

    try:
        frame.to_csv(output_path, index=False)
    except Exception as exc:
        logging.error("Failed to write output CSV to %s: %s", output_path, exc)
        sys.exit(1)

    logging.info("Wrote %d columns to %s", len(frame.columns), output_path)
    print(output_path)


if __name__ == "__main__":
    main()

