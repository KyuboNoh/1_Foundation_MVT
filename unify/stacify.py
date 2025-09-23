from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pystac
import xarray as xr
import yaml

from stacify_bc_data import raster_item_from_tif, sha256sum

_LOG = logging.getLogger(__name__)

DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.1.0/schema.json"
LABEL_EXT = "https://stac-extensions.github.io/label/v1.0.1/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
TABLE_EXT = "https://stac-extensions.github.io/table/v1.2.0/schema.json"
CHECKSUM_EXT = "https://stac-extensions.github.io/checksum/v1.0.0/schema.json"


def _ogr_extent(path: Path) -> Optional[List[float]]:
    ogrinfo = shutil.which("ogrinfo")
    if not ogrinfo:
        return None
    proc = subprocess.run([ogrinfo, "-so", str(path), path.stem], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Extent:"):
            # Extent: (minX, maxX) - (minY, maxY)
            try:
                left = line.split(":")[1]
                first, second = left.split("-")
                minx, miny = [float(tok) for tok in first.strip(" ()").split(",")]
                maxx, maxy = [float(tok) for tok in second.strip(" ()").split(",")]
                return [minx, miny, maxx, maxy]
            except Exception:
                return None
    return None


def _geometry_from_bbox(bbox: List[float]) -> Dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[0], bbox[3]],
            [bbox[2], bbox[3]],
            [bbox[2], bbox[1]],
            [bbox[0], bbox[1]],
        ]],
    }


def add_datacube_item(
    collection: pystac.Collection,
    cube_path: Path,
    registry_yaml: Path,
    dims: Dict[str, Dict],
    chunks: Optional[Dict[str, int]],
    variables_meta: Dict[str, Dict],
    derived_from_items: Optional[Iterable[str]] = None,
) -> pystac.Item:
    cube_path = Path(cube_path)
    registry_yaml = Path(registry_yaml)
    registry = yaml.safe_load(registry_yaml.read_text())
    proj_code = registry.get("crs")

    if cube_path.suffix == ".zarr" or cube_path.is_dir():
        ds = xr.open_zarr(cube_path)
    else:
        ds = xr.open_dataset(cube_path)

    x = ds.coords.get("x")
    y = ds.coords.get("y")
    if x is None or y is None:
        raise ValueError("Cube dataset must expose x/y coordinates")
    bbox = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
    geom = _geometry_from_bbox(bbox)

    item = pystac.Item(
        id=cube_path.stem,
        geometry=geom,
        bbox=bbox,
        datetime=datetime.now(tz=timezone.utc),
        properties={},
        stac_extensions=[DATACUBE_EXT, PROJ_EXT, CHECKSUM_EXT],
    )
    item.properties["cube:dimensions"] = dims
    item.properties["cube:variables"] = variables_meta
    if chunks:
        item.properties["cube:chunks"] = chunks
    if proj_code:
        item.properties["proj:code"] = proj_code

    extra_fields = {}
    if cube_path.is_file():
        extra_fields["checksum:sha256"] = sha256sum(cube_path)

    media_type = "application/vnd+zarr" if cube_path.is_dir() else "application/x-netcdf"
    asset = pystac.Asset(
        href=str(cube_path),
        media_type=media_type,
        roles=["data"],
        extra_fields=extra_fields,
    )

    item.add_asset("data", asset)

    for href in derived_from_items or []:
        item.add_link(pystac.Link(rel="derived_from", target=href))

    collection.add_item(item)
    return item


def add_table_item(
    collection: pystac.Collection,
    table_path: Path,
    primary_geom: str = "geometry",
    columns: Optional[List[Dict[str, str]]] = None,
) -> pystac.Item:
    table_path = Path(table_path)
    bbox = _ogr_extent(table_path)
    geom = _geometry_from_bbox(bbox) if bbox else None

    item = pystac.Item(
        id=table_path.stem,
        geometry=geom,
        bbox=bbox,
        datetime=datetime.now(tz=timezone.utc),
        properties={},
        stac_extensions=[TABLE_EXT, CHECKSUM_EXT],
    )
    if columns:
        item.properties["table:columns"] = columns
    item.add_asset(
        "data",
        pystac.Asset(
            href=str(table_path),
            media_type="application/x-parquet" if table_path.suffix.lower() == ".parquet" else "application/geopackage+sqlite3",
            roles=["data"],
            extra_fields={
                "table:primary_geometry": primary_geom,
                "checksum:sha256": sha256sum(table_path),
            },
        ),
    )
    collection.add_item(item)
    return item


def add_raster_item(collection: pystac.Collection, cog_path: Path, derived_from: Optional[str] = None) -> pystac.Item:
    item = raster_item_from_tif(Path(cog_path))
    if derived_from:
        item.add_link(pystac.Link(rel="derived_from", target=derived_from))
    collection.add_item(item)
    return item


def add_label_item(
    collection: pystac.Collection,
    labels_parquet: Path,
    label_metadata: Dict,
    target_item_href: Union[str, pystac.Item],
) -> pystac.Item:
    if isinstance(target_item_href, pystac.Item):
        target_item = target_item_href
    else:
        target_item = pystac.Item.from_file(target_item_href)
    item = pystac.Item(
        id=f"labels-{Path(labels_parquet).stem}",
        geometry=target_item.geometry,
        bbox=target_item.bbox,
        datetime=datetime.now(tz=timezone.utc),
        properties=label_metadata,
        stac_extensions=[LABEL_EXT, CHECKSUM_EXT],
    )
    item.add_asset(
        "labels",
        pystac.Asset(
            href=str(labels_parquet),
            media_type="application/x-parquet",
            roles=["labels"],
            extra_fields={"checksum:sha256": sha256sum(labels_parquet)},
        ),
    )
    item.add_link(pystac.Link(rel="target", target=target_item.get_self_href() or target_item_href))
    item.add_link(pystac.Link(rel="derived_from", target=target_item.get_self_href() or target_item_href))

    collection.add_item(item)
    return item
