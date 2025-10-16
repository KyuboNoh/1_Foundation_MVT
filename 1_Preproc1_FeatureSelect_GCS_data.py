#!/usr/bin/env python3
"""Select a subset of GSC CSV columns for downstream processing."""

# python 1_Preproc1_FeatureSelect_GCS_data.py --csv "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube.csv" --out "/home/qubuntu25/Desktop/Research/Data/2021_Table04_Datacube_selected.csv" --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity --special-columns1 Geology_Lithology_Contact Geology_Dictionary_Alkalic Geology_Dictionary_Anatectic Geology_Dictionary_Calcareous Geology_Dictionary_Carbonaceous Geology_Dictionary_Cherty Geology_Dictionary_CoarseClastic Geology_Dictionary_Evaporitic Geology_Dictionary_Felsic Geology_Dictionary_FineClastic Geology_Dictionary_Gneissose Geology_Dictionary_Igneous Geology_Dictionary_Intermediate Geology_Dictionary_Pegmatitic Geology_Dictionary_RedBed Geology_Dictionary_Schistose Geology_Dictionary_Sedimentary Geology_Dictionary_UltramaficMafic --extra-columns H3_Address H3_Resolution H3_Geometry Training_MVT_Deposit Training_MVT_Occurrence --validate

# python .\1_Preproc1_FeatureSelect_GCS_data.py `
# --csv "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp.csv" `
# --out "C:\Users\kyubo\Desktop\Research\Data\2021_Table04_Datacube_temp_selected.csv" `
# --lat-column "Latitude_EPSG4326" --lon-column "Longitude_EPSG4326" `
# --features Terrane_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity `
# --special-columns1 Geology_Lithology_Contact Geology_Dictionary_Alkalic Geology_Dictionary_Anatectic Geology_Dictionary_Calcareous Geology_Dictionary_Carbonaceous Geology_Dictionary_Cherty Geology_Dictionary_CoarseClastic Geology_Dictionary_Evaporitic Geology_Dictionary_Felsic Geology_Dictionary_FineClastic Geology_Dictionary_Gneissose Geology_Dictionary_Igneous Geology_Dictionary_Intermediate Geology_Dictionary_Pegmatitic Geology_Dictionary_RedBed Geology_Dictionary_Schistose Geology_Dictionary_Sedimentary Geology_Dictionary_UltramaficMafic `
# --extra-columns H3_Address H3_Resolution H3_Geometry Training_MVT_Deposit Training_MVT_Occurrence`
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

    if args.out is None:
        output_path = args.csv.with_name(f"{args.csv.stem}_Selected.csv")
    else:
        output_path = args.out

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate validation plots
    import logging as _logging
    _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
    if args.validate:
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

