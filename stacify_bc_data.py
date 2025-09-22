
#!/usr/bin/env python3
# stacify_bc.py
# Robust utilities to convert BC raster/vector data into a STAC catalog.
#
# Requirements:
#   pip install pystac rasterio rio-cogeo geopandas pyogrio shapely pyarrow
#
# Example:
#   python stacify_bc.py \
#       --raw-dir /mnt/data \
#       --out /mnt/data/stac_catalog \
#       --collection-id bc-raw \
#       --license "CC-BY-4.0" \
#       --keywords MVT SEDEX British_Columbia \
#       --cogify \
#       --target-crs EPSG:3005
#
# Author: ChatGPT
# Date: 2025-09-22

import argparse
import os
import sys
import hashlib
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Defer heavy imports until functions run to improve robustness when listing help, etc.
def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except Exception as e:
        print("ERROR: rasterio is required (pip install rasterio).", file=sys.stderr)
        raise

def _require_geopandas():
    try:
        import geopandas as gpd
        return gpd
    except Exception as e:
        print("ERROR: geopandas is required (pip install geopandas pyogrio pyarrow shapely).", file=sys.stderr)
        raise

def _require_pystac():
    try:
        import pystac
        from pystac import Catalog, Collection, Item, Asset, Extent
        from pystac.extensions.projection import ProjectionExtension
        # TODO (ImportError: cannot import name 'Band' from 'pystac.extensions.raster') Check STAC version...
        from pystac.extensions.raster import RasterExtension, Band
        return pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band
    except Exception as e:
        print("ERROR: pystac and extensions are required (pip install pystac).", file=sys.stderr)
        raise

def _has_rio_cogeo():
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        return True
    except Exception:
        return False

TABLE_EXT = "https://stac-extensions.github.io/table/v1.2.0/schema.json"
CHECKSUM_EXT = "https://stac-extensions.github.io/checksum/v1.0.0/schema.json"

def sha256sum(path: Path, blocksize: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def guess_mimetype(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".tif", ".tiff"]:
        return "image/tiff; application=geotiff"
    if suf in [".parquet"]:
        return "application/x-parquet"
    if suf in [".gpkg"]:
        return "application/geopackage+sqlite3"
    if suf in [".geojson", ".json"]:
        return "application/geo+json"
    return "application/octet-stream"

def to_cog(src_tif: Path, dst_tif: Path, overview_resampling: str = "nearest") -> None:
    """Convert a GeoTIFF to a COG. Prefer rio-cogeo; fallback to gdal_translate. Else copy."""
    if src_tif.resolve() == dst_tif.resolve():
        return
    ensure_dir(dst_tif.parent)

    if _has_rio_cogeo():
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        profile = cog_profiles.get("deflate")
        profile.update(dict(BIGTIFF="IF_SAFER"))
        with rasterio.open(src_tif) as src:
            cog_translate(
                src,
                dst_tif,
                profile,
                in_memory=False,
                config={"GDAL_TIFF_OVR_BLOCKSIZE": "512"},
                overview_level=None,
                overview_resampling=overview_resampling,
                web_optimized=False,
            )
        return

    gdal_translate = shutil.which("gdal_translate")
    if gdal_translate:
        cmd = [
            gdal_translate,
            "-of", "COG",
            "-co", "COMPRESS=DEFLATE",
            "-co", "BIGTIFF=IF_SAFER",
            str(src_tif),
            str(dst_tif)
        ]
        rc = os.system(" ".join(cmd))
        if rc != 0:
            logging.warning("gdal_translate failed; copying original TIFF instead.")
            shutil.copy2(src_tif, dst_tif)
        return

    logging.warning("Neither rio-cogeo nor GDAL COG tools found; copying TIFF without COG optimization.")
    shutil.copy2(src_tif, dst_tif)

def raster_item_from_tif(tif_path: Path, item_id: Optional[str] = None, asset_href: Optional[str] = None):
    rasterio = _require_rasterio()
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    with rasterio.open(tif_path) as ds:
        bounds = ds.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        transform = list(ds.transform)
        height, width = ds.height, ds.width
        crs = ds.crs
        epsg = crs.to_epsg() if crs else None
        dt = ds.dtypes[0] if ds.count >= 1 else "uint8"
        nodata = ds.nodata

        geom = {
            "type": "Polygon",
            "coordinates": [[
                [bounds.left, bounds.bottom],
                [bounds.left, bounds.top],
                [bounds.right, bounds.top],
                [bounds.right, bounds.bottom],
                [bounds.left, bounds.bottom]
            ]]
        }

        item = Item(
            id=item_id or tif_path.stem,
            geometry=geom,
            bbox=bbox,
            datetime=None,
            properties={},
        )

        ProjectionExtension.add_to(item)
        proj = ProjectionExtension.ext(item)
        proj.epsg = epsg
        proj.shape = [height, width]
        proj.transform = transform

        asset = Asset(
            href=asset_href or str(tif_path),
            media_type=guess_mimetype(tif_path),
            roles=["data"],
            title=tif_path.name
        )
        item.add_asset("data", asset)

        RasterExtension.add_to(item)
        rext = RasterExtension.ext(asset, add_if_missing=True)
        band = Band.create(data_type=dt, nodata=nodata)
        rext.bands = [band]

        # checksum
        try:
            asset.extra_fields = asset.extra_fields or {}
            asset.extra_fields["checksum:sha256"] = sha256sum(Path(asset.href))
        except Exception:
            pass

        return item

def shp_to_geoparquet(shp_path: Path, out_dir: Path, target_crs: Optional[str] = None) -> Path:
    gpd = _require_geopandas()
    ensure_dir(out_dir)
    gpq_path = out_dir / (shp_path.stem + ".parquet")
    gdf = gpd.read_file(shp_path)
    if target_crs:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            logging.warning(f"CRS reprojection to {target_crs} failed; keeping original. Error: {e}")
    if "geometry" not in gdf.columns:
        raise RuntimeError(f"No geometry column found in {shp_path}")
    gdf.to_parquet(gpq_path, index=False)
    return gpq_path

def table_columns_from_geoparquet(parquet_path: Path) -> List[Dict]:
    gpd = _require_geopandas()
    gdf = gpd.read_parquet(parquet_path)
    cols = []
    for name, dtype in zip(gdf.columns, gdf.dtypes):
        if name == "geometry":
            t = "geometry"
        else:
            if np.issubdtype(dtype, np.integer):
                t = "integer"
            elif np.issubdtype(dtype, np.floating):
                t = "number"
            elif dtype == "bool" or dtype == bool:
                t = "boolean"
            else:
                t = "string"
        cols.append({"name": name, "type": t})
    return cols

def vector_item_from_geoparquet(parquet_path: Path, item_id: Optional[str] = None, asset_href: Optional[str] = None):
    gpd = _require_geopandas()
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    gdf = gpd.read_parquet(parquet_path)
    # Geometry/bounds in WGS84 for STAC item bbox
    try:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            g84 = gdf.to_crs(4326)
        else:
            g84 = gdf
    except Exception:
        g84 = gdf

    minx, miny, maxx, maxy = g84.total_bounds
    bbox = [float(minx), float(miny), float(maxx), float(maxy)]
    geom = {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[0], bbox[3]],
            [bbox[2], bbox[3]],
            [bbox[2], bbox[1]],
            [bbox[0], bbox[1]]
        ]]
    }

    item = Item(
        id=item_id or Path(parquet_path).stem,
        geometry=geom,
        bbox=bbox,
        datetime=None,
        properties={"stac_extensions": [TABLE_EXT]},
    )

    ProjectionExtension.add_to(item)
    proj = ProjectionExtension.ext(item)
    try:
        epsg = gdf.crs.to_epsg() if gdf.crs else None
    except Exception:
        epsg = None
    proj.epsg = epsg

    asset = Asset(
        href=asset_href or str(parquet_path),
        media_type=guess_mimetype(Path(parquet_path)),
        roles=["data"],
        title=Path(parquet_path).name,
        extra_fields={"table:primary_geometry": "geometry"}
    )
    # checksum
    try:
        asset.extra_fields["checksum:sha256"] = sha256sum(Path(parquet_path))
    except Exception:
        pass

    item.add_asset("data", asset)

    # Add table:columns
    item.properties["table:columns"] = table_columns_from_geoparquet(Path(parquet_path))

    return item

def build_catalog(out_dir: Path, title: str, description: str):
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    cat = Catalog(
        id=title.lower().replace(" ", "-"),
        description=description,
        title=title,
        catalog_type=pystac.catalog.CatalogType.SELF_CONTAINED
    )
    cat.normalize_hrefs(str(out_dir))
    return cat

def build_collection(collection_id: str,
                     description: str,
                     license_str: str = "proprietary",
                     keywords: Optional[List[str]] = None):
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    spatial_extent = pystac.SpatialExtent([[-180, -90, 180, 90]])
    temporal_extent = pystac.TemporalExtent([[None, None]])
    extent = Extent(spatial_extent, temporal_extent)

    coll = Collection(
        id=collection_id,
        description=description,
        license=license_str,
        extent=extent,
        keywords=keywords or []
    )
    return coll

def add_raster_assets(collection,
                      files: List[Path],
                      out_asset_dir: Path,
                      cogify: bool = True) -> List:
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    ensure_dir(out_asset_dir)
    items = []
    for tif in files:
        try:
            asset_path = out_asset_dir / (tif.stem + "_cog.tif") if cogify else out_asset_dir / tif.name
            if cogify:
                to_cog(tif, asset_path)
            else:
                if tif.resolve() != asset_path.resolve():
                    shutil.copy2(tif, asset_path)
            item = raster_item_from_tif(asset_path)
            collection.add_item(item)
            items.append(item)
            logging.info(f"Added raster item: {item.id}")
        except Exception as e:
            logging.exception(f"Failed to add raster {tif}: {e}")
    return items

def add_vector_assets(collection,
                      shp_files: List[Path],
                      out_asset_dir: Path,
                      target_crs: Optional[str] = None) -> List:
    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    ensure_dir(out_asset_dir)
    items = []
    for shp in shp_files:
        try:
            gpq = shp_to_geoparquet(shp, out_asset_dir, target_crs=target_crs)
            item = vector_item_from_geoparquet(gpq)
            collection.add_item(item)
            items.append(item)
            logging.info(f"Added vector item: {item.id}")
        except Exception as e:
            logging.exception(f"Failed to add vector {shp}: {e}")
    return items

def scan_files(raw_dir: Path) -> Tuple[List[Path], List[Path]]:
    rasters, shapefiles = [], []
    for p in raw_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in [".tif", ".tiff"]:
                rasters.append(p)
            elif p.suffix.lower() == ".shp":
                shapefiles.append(p)
    return rasters, shapefiles

def main():
    parser = argparse.ArgumentParser(description="Convert BC rasters/vectors into a STAC catalog.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing .tif/.tiff and .shp files.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for STAC catalog.")
    parser.add_argument("--collection-id", type=str, default="bc-raw", help="STAC Collection ID to create.")
    parser.add_argument("--title", type=str, default="BC Raw Data", help="Catalog title.")
    parser.add_argument("--description", type=str, default="BC rasters and vectors registered in STAC.", help="Catalog description.")
    parser.add_argument("--license", type=str, default="proprietary", help="License string, e.g., CC-BY-4.0.")
    parser.add_argument("--keywords", type=str, nargs="*", default=[], help="Optional keywords for the collection.")
    parser.add_argument("--cogify", action="store_true", help="Convert rasters to Cloud-Optimized GeoTIFFs.")
    parser.add_argument("--target-crs", type=str, default=None, help="Optional target CRS for vectors (e.g., EPSG:3005).")
    parser.add_argument("--assets-subdir", type=str, default="assets", help="Subdirectory (under OUT) for asset files.")
    parser.add_argument("--validate", action="store_true", help="Run STAC validation before saving.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()

    cat = build_catalog(out_dir, args.title, args.description)
    coll = build_collection(args.collection_id, args.description, license_str=args.license, keywords=args.keywords)

    rasters, shapefiles = scan_files(raw_dir)
    logging.info(f"Found {len(rasters)} rasters and {len(shapefiles)} shapefiles in {raw_dir}")

    asset_dir = out_dir / args.assets_subdir
    raster_items = add_raster_assets(coll, rasters, asset_dir / "rasters", cogify=args.cogify)
    vector_items = add_vector_assets(coll, shapefiles, asset_dir / "vectors", target_crs=args.target_crs)

    cat.add_child(coll)

    if args.validate:
        try:
            import pystac.validation as pv
            pv.validate_collection(coll)
            for it in coll.get_items():
                pv.validate_item(it)
            logging.info("STAC validation passed.")
        except Exception as e:
            logging.warning(f"Validation failed or validator unavailable: {e}")

    cat.normalize_hrefs(str(out_dir))
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    print(f"STAC catalog written to: {out_dir}")

if __name__ == "__main__":
    main()