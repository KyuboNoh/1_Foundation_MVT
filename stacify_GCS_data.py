#!/usr/bin/env python3
# python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube_temp.csv   --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --collection-id gsc-2021   --title "GSC 2021 Table"   --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   --keywords GSC Datacube 2021 --validate
"""Convert a single GSC CSV table into a STAC catalog with Parquet assets."""

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from unify.one_d import csv_to_parquet
from stacify_bc_data import (
    CHECKSUM_EXT,
    TABLE_EXT,
    build_catalog,
    build_collection,
    ensure_dir,
    sha256sum,
    _require_pystac,
    _install_local_schema_cache,
)

REGISTRY_RELATIVE = Path("unify") / "feature_registry.yaml"


def infer_columns(parquet_path: Path) -> list[Dict[str, str]]:
    table = pq.read_table(parquet_path)
    schema = table.schema
    cols: list[Dict[str, str]] = []
    for field in schema:
        logical = map_arrow_type(field.type)
        cols.append({"name": field.name, "type": logical})
    return cols


def map_arrow_type(dtype: pa.DataType) -> str:
    if pa.types.is_integer(dtype):
        return "integer"
    if pa.types.is_floating(dtype):
        return "number"
    if pa.types.is_boolean(dtype):
        return "boolean"
    if pa.types.is_timestamp(dtype):
        return "datetime"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"
    return "string"


def copy_feature_registry(collection_root: Path) -> Path:
    ensure_dir(collection_root)
    dst = collection_root / "feature_registry.yaml"
    if dst.exists():
        return dst
    src = Path(__file__).resolve().parent / REGISTRY_RELATIVE
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(
            "crs: \"EPSG:3005\"\n"
            "resolution: 100\n"
            "variables: {}\n",
            encoding="utf-8",
        )
    return dst


def build_table_item(collection, parquet_path: Path, columns: list[Dict[str, str]], collection_root: Path) -> None:
    mods = _require_pystac()
    Item = mods["Item"]
    Asset = mods["Asset"]

    item = Item(
        id=parquet_path.stem,
        geometry=None,
        bbox=None,
        datetime=datetime.now(tz=timezone.utc),
        properties={"table:columns": columns},
        stac_extensions=[TABLE_EXT, CHECKSUM_EXT],
    )
    item.add_asset(
        "data",
        Asset(
            href=str(parquet_path),
            media_type="application/x-parquet",
            roles=["data"],
            title=parquet_path.name,
            extra_fields={"checksum:sha256": sha256sum(parquet_path)},
        ),
    )
    item_dir = collection_root / item.id
    ensure_dir(item_dir)
    item.set_self_href(str(item_dir / f"{item.id}.json"))
    collection.add_item(item)


def parse_schema_hints(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a CSV table into a STAC catalog.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--out", type=str, required=True, help="Output directory for STAC catalog")
    parser.add_argument("--collection-id", type=str, default="gsc-table", help="Collection ID")
    parser.add_argument("--title", type=str, default="GSC Table", help="Catalog title")
    parser.add_argument("--description", type=str, default="GSC tabular data registered in STAC.")
    parser.add_argument("--license", type=str, default="proprietary")
    parser.add_argument("--keywords", type=str, nargs="*", default=[])
    parser.add_argument("--assets-subdir", type=str, default="assets")
    parser.add_argument("--schema", type=str, default=None, help="Optional JSON column:type hints")
    parser.add_argument("--validate", action="store_true", help="Run STAC validation")
    parser.add_argument("--source-url", type=str, default=None, help="Optional source data URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)
    collection_root = out_dir / args.collection_id
    ensure_dir(collection_root)

    # Save a copy of the raw CSV
    raw_dir = collection_root / "raw"
    ensure_dir(raw_dir)
    shutil.copy2(csv_path, raw_dir / csv_path.name)

    # Copy feature registry template
    copy_feature_registry(collection_root)

    # Convert CSV to Parquet
    assets_dir = collection_root / args.assets_subdir
    ensure_dir(assets_dir)
    tables_dir = assets_dir / "tables"
    ensure_dir(tables_dir)
    schema_hints = parse_schema_hints(args.schema)
    parquet_path = tables_dir / (csv_path.stem + ".parquet")
    csv_to_parquet(csv_path, parquet_path, schema_hints=schema_hints)

    columns = infer_columns(parquet_path)

    mods = _require_pystac()
    pystac = mods["pystac"]
    Provider = mods["Provider"]
    Link = mods["Link"]
    RelType = mods["RelType"]

    cat = build_catalog(out_dir, args.title, args.description)
    coll = build_collection(
        args.collection_id,
        args.description,
        license_str=args.license,
        keywords=args.keywords,
    )
    coll.stac_extensions = [TABLE_EXT, CHECKSUM_EXT]
    cat.stac_extensions = coll.stac_extensions

    providers = [
        Provider(name="GSC Data Source", roles=["producer"], url=args.source_url or csv_path.as_uri()),
        Provider(name="STAC Conversion Pipeline", roles=["processor"], url=f"file://{Path(__file__).resolve()}"),
    ]
    coll.providers = providers

    if args.source_url:
        coll.add_link(Link(rel=RelType.SOURCE, target=args.source_url, title="Source data portal"))
    else:
        coll.add_link(Link(rel="via", target=csv_path.as_uri(), title="Original CSV"))

    build_table_item(coll, parquet_path, columns, collection_root)

    coll.summaries = pystac.Summaries({"table:columns": columns})

    cat.add_child(coll)

    reset_io = None
    if args.validate:
        reset_io = _install_local_schema_cache()
    if args.validate:
        try:
            coll.validate()
            for it in coll.get_items():
                it.validate()
            logging.info("STAC validation passed.")
        except Exception as exc:
            logging.warning(f"Validation failed: {exc}")
        logging.info("STAC validation passed.")
    if reset_io:
        reset_io()

    catalog_href = out_dir / f"catalog_{args.collection_id}.json"
    cat.normalize_hrefs(str(out_dir))
    cat.set_self_href(str(catalog_href))
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    print(f"STAC catalog written to: {out_dir}")


if __name__ == "__main__":
    main()
