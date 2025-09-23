from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pystac

from .cube import build_netcdf_cube, build_zarr_cube, harmonize_rasters_to_grid
from .labels import build_label_metadata, normalize_chip_labels, normalize_label_vectors
from .one_d import csv_to_parquet
from .stacify import add_datacube_item, add_label_item, add_raster_item, add_table_item
from .vectors import infer_table_columns, to_geoparquet

_LOG = logging.getLogger("unify.cli")


def _parse_json_maybe(value: Optional[str]) -> Optional[Dict]:
    if not value:
        return None
    return json.loads(value)


def _temp_collection() -> pystac.Collection:
    spatial = pystac.SpatialExtent([[-180, -90, 180, 90]])
    temporal = pystac.TemporalExtent([[None, None]])
    extent = pystac.Extent(spatial, temporal)
    return pystac.Collection(
        id="unify-temp",
        description="Temporary collection for CLI outputs",
        license="proprietary",
        extent=extent,
    )


def cmd_harmonize(args: argparse.Namespace) -> None:
    rasters = [Path(p) for p in args.rasters]
    outputs = harmonize_rasters_to_grid(
        rasters,
        target_epsg=args.target_crs,
        pixel_size=args.pixel_size,
        resampling=args.resampling,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        cogify=args.cogify,
    )
    for out_path in outputs:
        print(out_path)


def cmd_cube(args: argparse.Namespace) -> None:
    harmonized = [Path(p) for p in args.harmonized]
    registry = Path(args.registry)
    out_path = Path(args.out)
    chunks = _parse_json_maybe(args.chunks)

    if args.format == "zarr":
        result = build_zarr_cube(harmonized, registry, out_path, chunks=chunks)
    else:
        result = build_netcdf_cube(harmonized, registry, out_path, chunks=chunks)
    print(result)


def cmd_vector(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for vector_path in args.vectors:
        dst = out_dir / (Path(vector_path).stem + ".parquet")
        result = to_geoparquet(Path(vector_path), dst, args.target_crs)
        cols = infer_table_columns(result)
        print(json.dumps({"path": str(result), "columns": cols}))


def cmd_one_d(args: argparse.Namespace) -> None:
    schema = _parse_json_maybe(args.schema)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table in args.tables:
        dst = out_dir / (Path(table).stem + ".parquet")
        result = csv_to_parquet(Path(table), dst, schema_hints=schema)
        print(result)


def cmd_labels_vector(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for vector_path in args.vectors:
        dst = out_dir / (Path(vector_path).stem + ".parquet")
        result = normalize_label_vectors(Path(vector_path), dst, args.target_crs, args.class_property)
        print(result)


def cmd_labels_chip(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _parse_json_maybe(args.schema)
    for table in args.tables:
        dst = out_dir / (Path(table).stem + ".parquet")
        result = normalize_chip_labels(Path(table), dst, schema_hints=schema)
        print(result)


def _save_item(item: pystac.Item, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    href = dest_dir / f"{item.id}.json"
    item.set_self_href(str(href))
    item.save_object(include_self_link=False)
    return href


def cmd_stac_datacube(args: argparse.Namespace) -> None:
    collection = _temp_collection()
    dims = _parse_json_maybe(args.dims) or {}
    variables = _parse_json_maybe(args.variables) or {}
    chunks = _parse_json_maybe(args.chunks)
    item = add_datacube_item(
        collection,
        Path(args.cube),
        Path(args.registry),
        dims=dims,
        chunks=chunks,
        variables_meta=variables,
        derived_from_items=args.derived_from,
    )
    href = _save_item(item, Path(args.item_dir))
    print(href)


def cmd_stac_table(args: argparse.Namespace) -> None:
    collection = _temp_collection()
    columns = infer_table_columns(Path(args.table)) if args.infer_columns else None
    item = add_table_item(collection, Path(args.table), primary_geom=args.primary_geom, columns=columns)
    href = _save_item(item, Path(args.item_dir))
    print(href)


def cmd_stac_raster(args: argparse.Namespace) -> None:
    collection = _temp_collection()
    item = add_raster_item(collection, Path(args.cog), derived_from=args.derived_from)
    href = _save_item(item, Path(args.item_dir))
    print(href)


def cmd_stac_label(args: argparse.Namespace) -> None:
    collection = _temp_collection()
    metadata = build_label_metadata(
        args.class_property,
        classes=args.classes,
        tasks=args.tasks,
        label_properties=args.label_properties,
    )
    item = add_label_item(collection, Path(args.labels), metadata, args.target_item)
    href = _save_item(item, Path(args.item_dir))
    print(href)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified data harmonization and STAC registration utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_harm = sub.add_parser("harmonize", help="Harmonize rasters to a shared grid")
    p_harm.add_argument("rasters", nargs="+", help="Input raster paths")
    p_harm.add_argument("--target-crs", required=True, help="Target CRS EPSG code, e.g. EPSG:3005")
    p_harm.add_argument("--pixel-size", type=float, required=True, help="Target pixel size in target CRS units")
    p_harm.add_argument("--resampling", default="cubic", help="GDAL resampling method")
    p_harm.add_argument("--out-dir", help="Output directory for harmonized rasters")
    p_harm.add_argument("--cogify", action="store_true", help="Convert outputs to COGs")
    p_harm.set_defaults(func=cmd_harmonize)

    p_cube = sub.add_parser("cube", help="Build a Zarr or NetCDF cube from harmonized rasters")
    p_cube.add_argument("harmonized", nargs="+", help="Harmonized raster paths")
    p_cube.add_argument("--registry", required=True, help="Path to feature registry YAML")
    p_cube.add_argument("--out", required=True, help="Output path for cube")
    p_cube.add_argument("--format", choices=["zarr", "netcdf"], default="zarr")
    p_cube.add_argument("--chunks", help="JSON dictionary of chunk sizes")
    p_cube.set_defaults(func=cmd_cube)

    p_vec = sub.add_parser("vector", help="Normalize vectors to GeoParquet")
    p_vec.add_argument("vectors", nargs="+", help="Input vector datasets")
    p_vec.add_argument("--target-crs", required=True, help="Target CRS EPSG code")
    p_vec.add_argument("--out-dir", required=True, help="Directory for converted outputs")
    p_vec.set_defaults(func=cmd_vector)

    p_one = sub.add_parser("one-d", help="Convert CSV tables to Parquet")
    p_one.add_argument("tables", nargs="+", help="CSV files")
    p_one.add_argument("--out-dir", required=True, help="Directory for Parquet outputs")
    p_one.add_argument("--schema", help="Optional JSON schema hints (col:type)")
    p_one.set_defaults(func=cmd_one_d)

    p_lab_vec = sub.add_parser("labels-vector", help="Normalize vector labels and emit GeoParquet")
    p_lab_vec.add_argument("vectors", nargs="+", help="Label vector datasets")
    p_lab_vec.add_argument("--target-crs", required=True)
    p_lab_vec.add_argument("--class-property", required=True, help="Property containing class labels")
    p_lab_vec.add_argument("--out-dir", required=True)
    p_lab_vec.set_defaults(func=cmd_labels_vector)

    p_lab_chip = sub.add_parser("labels-chip", help="Normalize chip/patch label tables to Parquet")
    p_lab_chip.add_argument("tables", nargs="+", help="Label tables (CSV)")
    p_lab_chip.add_argument("--out-dir", required=True)
    p_lab_chip.add_argument("--schema", help="Optional JSON schema hints")
    p_lab_chip.set_defaults(func=cmd_labels_chip)

    p_stac_cube = sub.add_parser("stac-datacube", help="Create a STAC Item for a datacube")
    p_stac_cube.add_argument("--cube", required=True, help="Path to Zarr directory or NetCDF file")
    p_stac_cube.add_argument("--registry", required=True, help="Feature registry YAML used to build the cube")
    p_stac_cube.add_argument("--dims", help="JSON description of cube:dimensions")
    p_stac_cube.add_argument("--chunks", help="JSON description of cube:chunks")
    p_stac_cube.add_argument("--variables", help="JSON description of cube:variables overrides")
    p_stac_cube.add_argument("--derived-from", nargs="*", default=[], help="Optional source HREFs")
    p_stac_cube.add_argument("--item-dir", required=True, help="Directory to write the STAC item")
    p_stac_cube.set_defaults(func=cmd_stac_datacube)

    p_stac_table = sub.add_parser("stac-table", help="Create a STAC Item for a table dataset")
    p_stac_table.add_argument("--table", required=True, help="Path to table (Parquet/GPKG)")
    p_stac_table.add_argument("--primary-geom", default="geometry", help="Primary geometry column name")
    p_stac_table.add_argument("--infer-columns", action="store_true", help="Infer table:columns metadata")
    p_stac_table.add_argument("--item-dir", required=True)
    p_stac_table.set_defaults(func=cmd_stac_table)

    p_stac_raster = sub.add_parser("stac-raster", help="Create a STAC Item for a COG")
    p_stac_raster.add_argument("--cog", required=True, help="Path to COG")
    p_stac_raster.add_argument("--derived-from", help="Optional source HREF")
    p_stac_raster.add_argument("--item-dir", required=True)
    p_stac_raster.set_defaults(func=cmd_stac_raster)

    p_stac_label = sub.add_parser("stac-label", help="Create a STAC Label Item")
    p_stac_label.add_argument("--labels", required=True, help="Path to labels parquet")
    p_stac_label.add_argument("--class-property", required=True)
    p_stac_label.add_argument("--classes", nargs="+", required=True)
    p_stac_label.add_argument("--tasks", nargs="+", default=["classification"])
    p_stac_label.add_argument("--label-properties", nargs="*", default=None)
    p_stac_label.add_argument("--target-item", required=True, help="Target STAC item href")
    p_stac_label.add_argument("--item-dir", required=True)
    p_stac_label.set_defaults(func=cmd_stac_label)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

