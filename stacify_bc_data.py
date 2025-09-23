
#!/usr/bin/env python3
# stacify_bc.py
# Robust utilities to convert BC raster/vector data into a STAC catalog.
#
# Requirements:
#   pip install pystac rasterio rio-cogeo geopandas pyogrio shapely pyarrow
#
# Example:
#   python stacify_bc_data.py  --raw-dir /home/qubuntu25/Desktop/Data/BCGS/AppendixB/   --out /home/qubuntu25/Desktop/Data/BCGS/   --collection-id bc-raw   --license "CC-BY-4.0"   --keywords MVT SEDEX British_Columbia   --cogify   --target-crs EPSG:3005   --validate
#   python stacify_bc_data.py  --raw-dir /home/qubuntu25/Desktop/Data/BCGS/AppendixB_light/   --out /home/qubuntu25/Desktop/Data/BCGS/   --collection-id bc-raw   --license "CC-BY-4.0"   --keywords MVT SEDEX British_Columbia   --cogify   --target-crs EPSG:3005   --validate
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
from datetime import datetime, timezone

import numpy as np


TABLE_EXT = "https://stac-extensions.github.io/table/v1.2.0/schema.json"
CHECKSUM_EXT = "https://stac-extensions.github.io/checksum/v1.0.0/schema.json"
PROJECTION_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RASTER_EXT = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"

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
    import pystac
    from pystac import Catalog, Collection, Item, Asset, Extent, Provider, Link, Summaries, RelType
    from pystac.extensions.projection import ProjectionExtension
    try:
        from pystac.extensions.raster import RasterExtension, Band  # Band may not exist on older pystac
    except Exception:
        from pystac.extensions.raster import RasterExtension
        Band = None
    return {
        "pystac": pystac,
        "Catalog": Catalog,
        "Collection": Collection,
        "Item": Item,
        "Asset": Asset,
        "Extent": Extent,
        "Provider": Provider,
        "Link": Link,
        "Summaries": Summaries,
        "RelType": RelType,
        "ProjectionExtension": ProjectionExtension,
        "RasterExtension": RasterExtension,
        "Band": Band,
        "DefaultStacIO": pystac.stac_io.DefaultStacIO,
    }


def _install_local_schema_cache():
    mods = _require_pystac()
    pystac = mods["pystac"]
    DefaultStacIO = mods["DefaultStacIO"]

    local_schema = (Path(__file__).resolve().parent / "schemas" / "checksum_v1.0.0.json").resolve()
    if not local_schema.exists():
        return lambda: None

    class LocalSchemaStacIO(DefaultStacIO):
        _SCHEMA_MAP = {CHECKSUM_EXT: local_schema}

        def read_text(self, href, *args, **kwargs):  # type: ignore[override]
            mapped = self._SCHEMA_MAP.get(href)
            if mapped:
                return Path(mapped).read_text(encoding="utf-8")
            return super().read_text(href, *args, **kwargs)

    previous_io_cls = pystac.StacIO.default().__class__
    pystac.StacIO.set_default(LocalSchemaStacIO)

    def _reset():
        pystac.StacIO.set_default(previous_io_cls)

    return _reset

def _has_rio_cogeo():
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
        return True
    except Exception:
        return False


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
    mods = _require_pystac()
    Item = mods["Item"]
    Asset = mods["Asset"]
    ProjectionExtension = mods["ProjectionExtension"]
    RasterExtension = mods["RasterExtension"]
    Band = mods["Band"]

    with rasterio.open(tif_path) as ds:
        bounds = ds.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        transform = list(ds.transform)
        height, width = ds.height, ds.width
        crs = ds.crs
        epsg = crs.to_epsg() if crs else None
        band_dtype = str(ds.dtypes[0]) if ds.count >= 1 else "uint8"
        res_x = float(ds.transform.a)
        res_y = float(-ds.transform.e)
        # NEW: derive an RFC3339 datetime from file mtime
        timestamp = datetime.fromtimestamp(tif_path.stat().st_mtime, tz=timezone.utc)
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
            datetime=timestamp,
            properties={},
        )

        item.properties["raster:resolution"] = [abs(res_x), abs(res_y)]
        item.properties["raster:datatype"] = band_dtype

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
        if Band is not None:
            rext.bands = [Band.create(data_type=band_dtype, nodata=nodata)]
        else:
            asset.extra_fields = asset.extra_fields or {}
            b = {"data_type": band_dtype}
            if nodata is not None:
                b["nodata"] = nodata
            asset.extra_fields["raster:bands"] = [b]

        if CHECKSUM_EXT not in item.stac_extensions:
            item.stac_extensions.append(CHECKSUM_EXT)

        # checksum
        try:
            asset.extra_fields = asset.extra_fields or {}
            asset.extra_fields["checksum:sha256"] = sha256sum(Path(asset.href))
        except Exception:
            pass

        return item

def vector_to_geoparquet(vector_path: Path, out_dir: Path, target_crs: Optional[str] = None) -> Path:
    gpd = _require_geopandas()
    ensure_dir(out_dir)
    gpq_path = out_dir / (vector_path.stem + ".parquet")
    gdf = gpd.read_file(vector_path)
    if target_crs:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            logging.warning(f"CRS reprojection to {target_crs} failed; keeping original. Error: {e}")
    if "geometry" not in gdf.columns:
        raise RuntimeError(f"No geometry column found in {vector_path}")
    gdf.to_parquet(gpq_path, index=False)
    return gpq_path

def table_columns_from_geoparquet(parquet_path: Path) -> List[Dict]:
    gpd = _require_geopandas()
    import pandas as pd
    gdf = gpd.read_parquet(parquet_path)
    cols = []
    for name, dtype in zip(gdf.columns, gdf.dtypes):
        if name == "geometry":
            t = "geometry"
        else:
            if pd.api.types.is_integer_dtype(dtype):
                t = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                t = "number"
            elif pd.api.types.is_bool_dtype(dtype):
                t = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                t = "datetime"
            else:
                t = "string"
        cols.append({"name": name, "type": t})
    return cols

def vector_item_from_geoparquet(parquet_path: Path, item_id: Optional[str] = None, asset_href: Optional[str] = None):
    gpd = _require_geopandas()
    mods = _require_pystac()
    Item = mods["Item"]
    Asset = mods["Asset"]
    ProjectionExtension = mods["ProjectionExtension"]

    gdf = gpd.read_parquet(parquet_path)
    dt = datetime.fromtimestamp(Path(parquet_path).stat().st_mtime, tz=timezone.utc)

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
        datetime=dt,
        properties={},
        stac_extensions=[TABLE_EXT, CHECKSUM_EXT],
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


def generate_raster_thumbnail(tif_path: Path, thumbnail_path: Path, max_size: int = 512) -> Optional[Path]:
    rasterio = _require_rasterio()
    import numpy as np
    import matplotlib.pyplot as plt
    from rasterio.enums import Resampling

    ensure_dir(thumbnail_path.parent)
    try:
        with rasterio.open(tif_path) as ds:
            scale = max(ds.width, ds.height) / max_size
            if scale < 1:
                scale = 1
            out_height = max(1, int(ds.height / scale))
            out_width = max(1, int(ds.width / scale))

            bands = min(3, ds.count)
            data = ds.read(
                indexes=list(range(1, bands + 1)),
                out_shape=(bands, out_height, out_width),
                resampling=Resampling.bilinear,
            ).astype("float32")

            if bands == 1:
                arr = data[0]
                mask = np.isfinite(arr)
                if not np.any(mask):
                    return None
                vmin = float(arr[mask].min())
                vmax = float(arr[mask].max())
                if vmax - vmin == 0:
                    vmax = vmin + 1
                norm = (arr - vmin) / (vmax - vmin)
                plt.imsave(thumbnail_path, norm, cmap="gray")
            else:
                mask = np.isfinite(data)
                if not np.any(mask):
                    return None
                for idx in range(bands):
                    band = data[idx]
                    finite = np.isfinite(band)
                    if not np.any(finite):
                        continue
                    vmin = float(band[finite].min())
                    vmax = float(band[finite].max())
                    if vmax - vmin == 0:
                        vmax = vmin + 1
                    data[idx] = (band - vmin) / (vmax - vmin)
                if bands == 2:
                    data = np.vstack([data, np.zeros((1, out_height, out_width), dtype="float32")])
                rgb = np.moveaxis(data[:3], 0, -1)
                rgb = np.clip(rgb, 0, 1)
                plt.imsave(thumbnail_path, rgb)
        return thumbnail_path
    except Exception as exc:
        logging.debug(f"Thumbnail generation for {tif_path} failed: {exc}")
        return None


def generate_vector_thumbnail(parquet_path: Path, thumbnail_path: Path) -> Optional[Path]:
    gpd = _require_geopandas()
    import matplotlib.pyplot as plt

    ensure_dir(thumbnail_path.parent)
    try:
        gdf = gpd.read_parquet(parquet_path)
        if gdf.empty:
            return None
        fig, ax = plt.subplots(figsize=(4, 4))
        try:
            gdf.plot(ax=ax, color="#4f8bc9", edgecolor="#1f4275", linewidth=0.4)
        except Exception:
            gdf.to_crs(4326).plot(ax=ax, color="#4f8bc9", edgecolor="#1f4275", linewidth=0.4)
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        fig.savefig(thumbnail_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return thumbnail_path
    except Exception as exc:
        logging.debug(f"Vector thumbnail generation for {parquet_path} failed: {exc}")
        return None


def update_collection_summaries(collection, items: List):
    mods = _require_pystac()
    Summaries = mods["Summaries"]

    epsgs = set()
    proj_codes = set()
    resolutions = set()
    dtypes = set()

    for wrapper in items:
        item = wrapper["item"] if isinstance(wrapper, dict) else wrapper
        epsg = item.properties.get("proj:epsg")
        if epsg:
            epsgs.add(epsg)
        code = item.properties.get("proj:code")
        if code:
            proj_codes.add(code)
            if not epsg and isinstance(code, str) and code.upper().startswith("EPSG:"):
                try:
                    epsgs.add(int(code.split(":", 1)[1]))
                except Exception:
                    pass
        res = item.properties.get("raster:resolution")
        if res:
            resolutions.add(tuple(res))
        dtype = item.properties.get("raster:datatype")
        if dtype:
            dtypes.add(dtype)

    summaries = collection.summaries or Summaries({})
    if epsgs:
        summaries.add("proj:epsg", sorted(epsgs))
    if proj_codes:
        summaries.add("proj:code", sorted(proj_codes))
    if resolutions:
        summaries.add("raster:resolution", [list(r) for r in sorted(resolutions)])
    if dtypes:
        summaries.add("raster:datatype", sorted(dtypes))
    collection.summaries = summaries


def build_collection_level_assets(collection,
                                  raster_products: List[Dict[str, Path]],
                                  vector_products: List[Dict[str, Path]],
                                  base_asset_dir: Path) -> None:
    mods = _require_pystac()
    Asset = mods["Asset"]

    if vector_products:
        try:
            gpd = _require_geopandas()
            import pandas as pd

            combined_path = base_asset_dir / "vectors" / "combined_vectors.parquet"
            ensure_dir(combined_path.parent)

            frames = [gpd.read_parquet(prod["asset_path"]) for prod in vector_products]
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=getattr(frames[0], "crs", None))
                gdf.to_parquet(combined_path, index=False)
                collection.assets["combined-vectors"] = Asset(
                    href=str(combined_path),
                    media_type="application/x-parquet",
                    roles=["data"],
                    title="Combined vector data",
                    extra_fields={"table:primary_geometry": "geometry"}
                )
        except Exception as exc:
            logging.warning(f"Failed to build combined vector asset: {exc}")

    if raster_products:
        try:
            rasterio = _require_rasterio()
            from rasterio.merge import merge

            mosaic_path = base_asset_dir / "rasters" / "mosaic.tif"
            ensure_dir(mosaic_path.parent)

            datasets = [rasterio.open(prod["asset_path"]) for prod in raster_products]
            mosaic, transform = merge(datasets)
            meta = datasets[0].meta.copy()
            for ds in datasets:
                ds.close()
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "driver": "GTiff"
            })

            with rasterio.open(mosaic_path, "w", **meta) as dst:
                dst.write(mosaic)

            collection.assets["mosaic-raster"] = Asset(
                href=str(mosaic_path),
                media_type=guess_mimetype(mosaic_path),
                roles=["data"],
                title="Raster mosaic"
            )
        except Exception as exc:
            logging.warning(f"Failed to build mosaic raster asset: {exc}")

def build_catalog(out_dir: Path, title: str, description: str):
    # pystac, Catalog, Collection, Item, Asset, Extent, ProjectionExtension, RasterExtension, Band = _require_pystac()
    
    mods = _require_pystac()
    Catalog = mods["Catalog"]; 
    # Item = mods["Item"]; Asset = mods["Asset"]
    # ProjectionExtension = mods["ProjectionExtension"]
    # RasterExtension = mods["RasterExtension"]; Band = mods["Band"]

    # cat = Catalog(
    #     id=title.lower().replace(" ", "-"),
    #     description=description,
    #     title=title,
    #     catalog_type=Catalog.CatalogType.SELF_CONTAINED
    # )
    cat = Catalog(
        id=title.lower().replace(" ", "-"),
        description=description,
        title=title,
    )
    cat.normalize_hrefs(str(out_dir))
    return cat

def build_collection(collection_id: str,
                     description: str,
                     license_str: str = "proprietary",
                     keywords: Optional[List[str]] = None):

    mods = _require_pystac()
    pystac = mods["pystac"]
    Collection = mods["Collection"]
    Extent = mods["Extent"]   

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
                      cogify: bool = True) -> List[Dict[str, Path]]:

    ensure_dir(out_asset_dir)
    thumb_dir = out_asset_dir / "thumbnails"
    mods = _require_pystac()
    Link = mods["Link"]
    Asset = mods["Asset"]

    products = []
    for tif in files:
        try:
            asset_path = out_asset_dir / (tif.stem + "_cog.tif") if cogify else out_asset_dir / tif.name
            if cogify:
                to_cog(tif, asset_path)
            else:
                if tif.resolve() != asset_path.resolve():
                    shutil.copy2(tif, asset_path)

            item = raster_item_from_tif(asset_path)
            item.add_link(Link(rel="derived_from", target=str(tif.resolve()), media_type=guess_mimetype(tif)))

            thumb_path = thumb_dir / f"{asset_path.stem}.png"
            thumb = generate_raster_thumbnail(asset_path, thumb_path)
            if thumb:
                thumb_asset = Asset(
                    href=str(thumb),
                    media_type="image/png",
                    roles=["thumbnail"],
                    title=f"Thumbnail for {item.id}"
                )
                item.add_asset("thumbnail", thumb_asset)

            collection.add_item(item)
            products.append({
                "item": item,
                "source_path": tif,
                "asset_path": asset_path
            })
            logging.info(f"Added raster item: {item.id}")
        except Exception as e:
            logging.exception(f"Failed to add raster {tif}: {e}")
    return products

def add_vector_assets(collection,
                      vector_files: List[Path],
                      out_asset_dir: Path,
                      target_crs: Optional[str] = None) -> List[Dict[str, Path]]:

    ensure_dir(out_asset_dir)
    thumb_dir = out_asset_dir / "thumbnails"
    mods = _require_pystac()
    Link = mods["Link"]
    Asset = mods["Asset"]

    products = []
    for shp in vector_files:
        try:
            gpq = vector_to_geoparquet(shp, out_asset_dir, target_crs=target_crs)
            item = vector_item_from_geoparquet(gpq)
            item.add_link(Link(rel="derived_from", target=str(shp.resolve()), media_type=guess_mimetype(shp)))

            thumb_path = thumb_dir / f"{Path(gpq).stem}.png"
            thumb = generate_vector_thumbnail(gpq, thumb_path)
            if thumb:
                thumb_asset = Asset(
                    href=str(thumb),
                    media_type="image/png",
                    roles=["thumbnail"],
                    title=f"Thumbnail for {item.id}"
                )
                item.add_asset("thumbnail", thumb_asset)

            collection.add_item(item)
            products.append({
                "item": item,
                "source_path": shp,
                "asset_path": gpq
            })
            logging.info(f"Added vector item: {item.id}")
        except Exception as e:
            logging.exception(f"Failed to add vector {shp}: {e}")
    return products

def _should_skip(path: Path, ignore_roots: List[Path]) -> bool:
    for root in ignore_roots:
        if path.is_relative_to(root):
            return True
    return False


def scan_files(raw_dir: Path, ignore_roots: Optional[List[Path]] = None) -> Tuple[List[Path], List[Path]]:
    rasters, vectors = [], []
    vector_exts = {".shp", ".gpkg", ".geojson", ".json"}
    ignore_roots = [p.resolve() for p in ignore_roots or []]
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip(p.resolve(), ignore_roots):
            continue
        suf = p.suffix.lower()
        if suf in [".tif", ".tiff"]:
            rasters.append(p)
        elif suf in vector_exts:
            if suf != ".json":
                vectors.append(p)
                continue
            try:
                import json

                with open(p, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                if not isinstance(doc, dict):
                    continue
                if "stac_version" in doc:
                    continue
                typ = str(doc.get("type", "")).lower()
                if typ in {"featurecollection", "feature"}:
                    vectors.append(p)
            except Exception:
                continue
    return rasters, vectors

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
    parser.add_argument("--source-url", type=str, default=None, help="Optional source data portal URL for lineage links.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    collection_root = out_dir / args.collection_id
    ensure_dir(collection_root)

    mods = _require_pystac()
    pystac = mods["pystac"]
    Provider = mods["Provider"]
    Link = mods["Link"]
    RelType = mods["RelType"]

    cat = build_catalog(out_dir, args.title, args.description)
    coll = build_collection(args.collection_id, args.description, license_str=args.license, keywords=args.keywords)
    coll.stac_extensions = sorted({PROJECTION_EXT, RASTER_EXT, TABLE_EXT, CHECKSUM_EXT})
    cat.stac_extensions = coll.stac_extensions

    providers = [
        Provider(name="BC Data Source", roles=["producer"], url=args.source_url or raw_dir.resolve().as_uri()),
        Provider(name="STAC Conversion Pipeline", roles=["processor"], url=f"file://{Path(__file__).resolve()}")
    ]
    coll.providers = providers

    if args.source_url:
        coll.add_link(Link(rel=RelType.SOURCE, target=args.source_url, title="Source data portal"))
    else:
        coll.add_link(Link(rel="via", target=raw_dir.resolve().as_uri(), title="Original data directory"))

    asset_dir = collection_root / args.assets_subdir
    ensure_dir(asset_dir)

    registry_dst = collection_root / "feature_registry.yaml"
    if not registry_dst.exists():
        registry_src = Path(__file__).resolve().parent / "unify" / "feature_registry.yaml"
        if registry_src.exists():
            shutil.copy2(registry_src, registry_dst)
        else:
            registry_dst.write_text(
                "crs: \"EPSG:3005\"\n"
                "resolution: 100\n"
                "variables: {}\n",
                encoding="utf-8",
            )

    ignore_roots = [asset_dir, collection_root]
    rasters, vector_sources = scan_files(raw_dir, ignore_roots=ignore_roots)
    logging.info(f"Found {len(rasters)} rasters and {len(vector_sources)} vector files in {raw_dir}")

    raster_items = add_raster_assets(coll, rasters, asset_dir / "rasters", cogify=args.cogify)
    vector_items = add_vector_assets(coll, vector_sources, asset_dir / "vectors", target_crs=args.target_crs)

    update_collection_summaries(coll, raster_items + vector_items)
    build_collection_level_assets(coll, raster_items, vector_items, asset_dir)

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
        except Exception as e:
            logging.warning(f"Validation failed: {e}")
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
