from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
    return proc


def to_geoparquet(vector_path: Path, out_path: Path, target_epsg: str) -> Path:
    """Reproject `vector_path` to `target_epsg` and emit GeoParquet via ogr2ogr."""
    vector_path = Path(vector_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ogr2ogr = shutil.which("ogr2ogr")
    if not ogr2ogr:
        raise RuntimeError("ogr2ogr not found on PATH")

    try:
        _run(
            [
                ogr2ogr,
                "-f",
                "Parquet",
                str(out_path),
                str(vector_path),
                "-t_srs",
                target_epsg,
            ]
        )
        return out_path
    except Exception as exc:
        _LOG.warning("ogr2ogr Parquet write failed (%s); falling back to GeoPackage", exc)

    gpkg_path = out_path.with_suffix(".gpkg")
    _run(
        [
            ogr2ogr,
            "-f",
            "GPKG",
            str(gpkg_path),
            str(vector_path),
            "-t_srs",
            target_epsg,
        ]
    )
    return gpkg_path


def infer_table_columns(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        table = pq.read_table(path, columns=None)
        schema = table.schema
        cols = []
        for field in schema:
            logical = _map_arrow_type(field.type)
            cols.append({"name": field.name, "type": logical})
        return cols

    ogrinfo = shutil.which("ogrinfo")
    if not ogrinfo:
        raise RuntimeError("ogrinfo not found; required to inspect GeoPackage schema")

    proc = _run([ogrinfo, "-so", str(path), path.stem])
    cols: List[Dict[str, str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("geometry"):
            cols.append({"name": "geometry", "type": "geometry"})
        elif line.startswith(path.stem):
            continue
        elif line.startswith("FID"):
            continue
        elif line.endswith(")"):
            name, type_part = line.split("(")
            logical = _map_ogr_type(type_part.rstrip(")"))
            cols.append({"name": name.strip(), "type": logical})
    return cols


def _map_arrow_type(dtype: pa.DataType) -> str:
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


def _map_ogr_type(type_name: str) -> str:
    type_name = type_name.lower()
    if any(token in type_name for token in ["int", "integer"]):
        return "integer"
    if any(token in type_name for token in ["real", "double", "float", "numeric"]):
        return "number"
    if "date" in type_name or "time" in type_name:
        return "datetime"
    if "bool" in type_name:
        return "boolean"
    return "string"
