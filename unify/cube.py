from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import rasterio
import xarray as xr
import yaml


def _load_registry(registry_path: Path) -> Dict:
    with open(registry_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):  # basic sanity
        raise ValueError("feature registry must be a mapping")
    return data


def _guess_variable_name(raster_path: Path, variables: Iterable[str]) -> str:
    stem = raster_path.stem.lower()
    for var in variables:
        if var.lower() in stem:
            return var
    return raster_path.stem.replace(" ", "_")


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({cmd}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")


def harmonize_rasters_to_grid(
    in_tifs: Iterable[Path],
    target_epsg: str,
    pixel_size: float,
    resampling: str = "cubic",
    out_dir: Optional[Path] = None,
    cogify: bool = False,
) -> List[Path]:
    """Reproject rasters to a common grid (gdalwarp) and optionally COG-ify."""
    if out_dir is None:
        out_dir = Path.cwd() / "harmonized"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gdalwarp = shutil.which("gdalwarp")
    gdal_translate = shutil.which("gdal_translate")
    if gdalwarp is None:
        raise RuntimeError("gdalwarp is required but not available on PATH")
    if cogify and gdal_translate is None:
        raise RuntimeError("gdal_translate is required for COG conversion but not available on PATH")

    results: List[Path] = []
    for tif in in_tifs:
        tif = Path(tif)
        warped = out_dir / f"{tif.stem}_epsg{target_epsg.split(':')[-1]}.tif"
        cmd = [
            gdalwarp,
            "-t_srs",
            target_epsg,
            "-tr",
            str(pixel_size),
            str(pixel_size),
            "-r",
            resampling,
            "-of",
            "GTiff",
            str(tif),
            str(warped),
        ]
        _run(cmd)

        final_path = warped
        if cogify:
            cog_path = out_dir / f"{warped.stem}_cog.tif"
            cmd = [
                gdal_translate,
                "-of",
                "COG",
                "-co",
                "COMPRESS=DEFLATE",
                str(warped),
                str(cog_path),
            ]
            _run(cmd)
            final_path = cog_path
            warped.unlink(missing_ok=True)

        results.append(final_path)

    return results


def _read_raster(path: Path) -> xr.DataArray:
    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        y_coords = transform.f + transform.e * (np.arange(src.height) + 0.5)
        x_coords = transform.c + transform.a * (np.arange(src.width) + 0.5)
        data = np.squeeze(data)
        if data.ndim == 2:
            dims = ("y", "x")
            coords = {"y": y_coords, "x": x_coords}
        elif data.ndim == 3:
            dims = ("band", "y", "x")
            coords = {"band": np.arange(1, data.shape[0] + 1), "y": y_coords, "x": x_coords}
        else:
            raise ValueError(f"Unsupported raster dimensionality for {path}: {data.shape}")
        attrs = {
            "transform": json.dumps(transform.to_gdal()),
            "crs": src.crs.to_wkt() if src.crs else "",
            "nodata": src.nodata,
        }
    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs, name=path.stem)


def build_zarr_cube(
    harmonized_cogs: Iterable[Path],
    feature_registry_yaml: Path,
    out_path: Path,
    chunks: Optional[Dict[str, int]] = None,
) -> Path:
    registry = _load_registry(Path(feature_registry_yaml))
    variables_meta = registry.get("variables", {})

    arrays = {}
    sample_da: Optional[xr.DataArray] = None
    for cog in harmonized_cogs:
        da = _read_raster(Path(cog))
        name = _guess_variable_name(Path(cog), variables_meta.keys())
        meta = variables_meta.get(name, {})
        da = da.astype(meta.get("dtype", da.dtype))
        da.attrs.update(meta)
        arrays[name] = da
        if sample_da is None:
            sample_da = da

    if not arrays:
        raise ValueError("No rasters provided to build_zarr_cube")

    dataset = xr.Dataset(arrays)
    dataset.attrs["crs"] = registry.get("crs")
    dataset.attrs["resolution"] = registry.get("resolution")

    if chunks:
        dataset = dataset.chunk(chunks)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(out_path, mode="w")
    return out_path


def build_netcdf_cube(
    harmonized_cogs: Iterable[Path],
    feature_registry_yaml: Path,
    out_path: Path,
    chunks: Optional[Dict[str, int]] = None,
    compression: str = "zlib",
    complevel: int = 4,
) -> Path:
    registry = _load_registry(Path(feature_registry_yaml))
    encoding = {}
    data_vars = {}

    for cog in harmonized_cogs:
        da = _read_raster(Path(cog))
        name = _guess_variable_name(Path(cog), registry.get("variables", {}).keys())
        meta = registry.get("variables", {}).get(name, {})
        da = da.astype(meta.get("dtype", da.dtype))
        da.attrs.update(meta)
        data_vars[name] = da
        encoding[name] = {"zlib": compression == "zlib", "complevel": complevel}

    if not data_vars:
        raise ValueError("No rasters provided to build_netcdf_cube")

    dataset = xr.Dataset(data_vars)
    dataset.attrs["crs"] = registry.get("crs")
    dataset.attrs["resolution"] = registry.get("resolution")

    if chunks:
        dataset = dataset.chunk(chunks)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(out_path, encoding=encoding)
    return out_path
