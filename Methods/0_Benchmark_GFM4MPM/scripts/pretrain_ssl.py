# scripts/pretrain_ssl.py
import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.gfm4mpm.data.geo_stack import GeoStack
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.training.train_ssl import train_ssl

# TODO: later check it for AWS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# TODO: skip_nan is implemented to play with boundary problems. But latter will be updated for the case where the boundary "tif" is provided.
class SSLDataset(Dataset):
    def __init__(
        self,
        stack,
        window: int = 32,
        n_samples: int = 200000,
        seed: int = 1337,
        skip_nan: bool = True,
        max_resample_attempts: int = 32,
    ):
        self.stack, self.window = stack, window
        self.rng = np.random.default_rng(seed)
        self.n = n_samples
        self.skip_nan = bool(skip_nan)
        self.max_resample_attempts = max(1, int(max_resample_attempts))

    def __len__(self):
        return self.n

    # TODO: Make this code to only consider samples that generates full window*window images?
    def __getitem__(self, idx):
        attempts = self.max_resample_attempts if self.skip_nan else 1
        last_patch = None
        for _ in range(attempts):
            if hasattr(self.stack, "random_coord"):
                r, c = self.stack.random_coord(self.window, self.rng)
            else:
                half = max(1, self.window // 2)
                max_row = max(half + 1, self.stack.height - half)
                max_col = max(half + 1, self.stack.width - half)
                r = int(self.rng.integers(half, max_row))
                c = int(self.rng.integers(half, max_col))
            x = self.stack.read_patch(r, c, self.window)
            last_patch = x
            if not self.skip_nan or np.isfinite(x).all():
                return torch.from_numpy(x)

        if self.skip_nan and last_patch is not None:
            raise RuntimeError(
                f"Failed to sample a finite patch after {self.max_resample_attempts} attempts; "
                "consider disabling --skip-nan or checking raster nodata handling."
            )

        return torch.from_numpy(last_patch)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    return slug or "feature"


def _load_training_metadata(stac_root: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        stac_root / "training_metadata.json",
        stac_root / "assetization" / "training_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fh:
                try:
                    return json.load(fh)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse training metadata JSON at {candidate}: {exc}") from exc
    return None


def _find_feature_entry(entries: Dict[str, Any], feature_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if feature_name in entries:
        return feature_name, entries[feature_name]
    lowered = feature_name.lower()
    for key, value in entries.items():
        if key.lower() == lowered:
            return key, value
    return None


def _collect_stac_raster_paths(stac_root: Path, features: Optional[Sequence[str]]) -> Tuple[list[str], Optional[Dict[str, Any]]]:
    metadata = _load_training_metadata(stac_root)
    if metadata:
        features_section = metadata.get("features", {})
        entries: Dict[str, Any] = features_section.get("entries", {})
        total_features = features_section.get("total_features", len(entries))
        print(f"[info] training_metadata.json reports {total_features} feature(s).")

        if not entries:
            print("[warn] training_metadata.json contained no feature entries; falling back to filesystem scan.")
            metadata = None
        else:
            if features:
                selected_features: Iterable[str] = features
            else:
                selected_features = sorted(entries.keys())
                summary_names = ", ".join(selected_features)
                print(
                    f"[info] No explicit --features provided; using all {len(selected_features)} features"
                    f" from metadata: {summary_names}"
                )

            raster_paths: list[str] = []
            for feature_name in selected_features:
                lookup = _find_feature_entry(entries, feature_name)
                if lookup is None:
                    raise FileNotFoundError(
                        f"Feature '{feature_name}' not present in training_metadata.json; available keys: {', '.join(entries.keys())}"
                    )
                actual_name, entry = lookup
                tif_records = entry.get("tifs", [])
                if not tif_records:
                    raise FileNotFoundError(f"No TIFF records listed for feature '{actual_name}' in training_metadata.json")

                num_tifs = entry.get("num_tifs", len(tif_records))
                print(f"[info] Feature '{actual_name}' has {num_tifs} map(s):")
                for record in tif_records:
                    path_value = record.get("path")
                    if not path_value:
                        continue
                    tif_path = (stac_root / path_value).resolve()
                    region = record.get("region")
                    print(f"        - {tif_path} (region={region})")

                preferred = None
                for record in tif_records:
                    if str(record.get("region", "")).upper() == "NA":
                        preferred = record
                        break
                if preferred is None:
                    preferred = tif_records[0]

                path_value = preferred.get("path")
                if not path_value:
                    raise FileNotFoundError(
                        f"Preferred TIFF record for feature '{actual_name}' lacked a path entry in training_metadata.json"
                    )
                tif_path = (stac_root / path_value).resolve()
                raster_paths.append(str(tif_path))

            return raster_paths, metadata

    # Fallback: file system scan
    assets_dir = (stac_root / "assets" / "rasters").resolve()
    if not assets_dir.exists():
        raise FileNotFoundError(f"No raster assets directory found at {assets_dir}")

    candidates = [
        path for path in assets_dir.rglob("*_cog.tif") if "thumbnails" not in {p.lower() for p in path.parts}
    ]
    if not candidates:
        candidates = [
            path for path in assets_dir.rglob("*.tif") if "thumbnails" not in {p.lower() for p in path.parts}
        ]

    if not candidates:
        raise FileNotFoundError(f"No raster GeoTIFFs found under {assets_dir}")

    if features:
        ordered: list[Path] = []
        remaining = candidates.copy()
        for feature in features:
            slug = _slugify(feature)
            match = None
            for path in remaining:
                stem_lower = path.stem.lower()
                stem_lower = stem_lower[:-4] if stem_lower.endswith("_cog") else stem_lower
                if slug in stem_lower:
                    match = path
                    break
            if match is None:
                raise FileNotFoundError(f"No raster asset matching feature '{feature}' (slug '{slug}') under {assets_dir}")
            ordered.append(match)
            remaining.remove(match)
        candidates = ordered
    else:
        candidates = sorted(candidates)

    return [str(path.resolve()) for path in candidates], None


def _maybe_check_image_preproc(
    stack,
    feature_name: str,
    window: int,
    out_dir: Path,
    rng_seed: int = 0,
) -> None:
    """Render a diagnostic plot comparing the full feature grid vs. sampled patches."""

    grid: Optional[np.ndarray] = None
    lat_axis: Optional[np.ndarray] = None
    lon_axis: Optional[np.ndarray] = None
    candidate_mask: Optional[np.ndarray] = None

    if getattr(stack, "is_table", False):
        latitudes = getattr(stack, "latitudes", None)
        longitudes = getattr(stack, "longitudes", None)
        if latitudes is None or longitudes is None:
            print("[check] Latitude/longitude columns unavailable; skipping image preprocessing diagnostic plot.")
            return

        try:
            raw_vals = stack.raw_column(feature_name)
        except KeyError:
            cols = list(getattr(stack, "feature_columns", []))
            fallback_idx: Optional[int] = None
            for idx, col in enumerate(cols):
                if col.lower() == feature_name.lower():
                    fallback_idx = idx
                    break
            if fallback_idx is None:
                print(
                    f"[check] Feature '{feature_name}' not found in STAC table; available columns include: {', '.join(cols[:10])}"
                )
                return
            raw_vals = stack.features[:, fallback_idx]

        raw_array = np.asarray(raw_vals)
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)

        def _safe_float(arr):
            result = np.empty(len(arr), dtype=np.float32)
            for idx, val in enumerate(arr):
                try:
                    result[idx] = float(val)
                except (TypeError, ValueError):
                    result[idx] = np.nan
            return result

        feature_vals = _safe_float(raw_array)
        lat_vals = _safe_float(latitudes)
        lon_vals = _safe_float(longitudes)

        valid_mask = np.isfinite(feature_vals) & np.isfinite(lat_vals) & np.isfinite(lon_vals)
        if valid_mask.sum() < window:
            print("[check] Not enough valid samples to render diagnostic plots.")
            return

        feat_valid = feature_vals[valid_mask]
        lat_valid = lat_vals[valid_mask]
        lon_valid = lon_vals[valid_mask]

        lat_keys, lat_inverse = np.unique(np.round(lat_valid, 6), return_inverse=True)
        lon_keys, lon_inverse = np.unique(np.round(lon_valid, 6), return_inverse=True)

        grid_shape = (lat_keys.size, lon_keys.size)
        grid_sum = np.zeros(grid_shape, dtype=np.float64)
        grid_count = np.zeros(grid_shape, dtype=np.int32)
        lat_sum = np.zeros(lat_keys.size, dtype=np.float64)
        lon_sum = np.zeros(lon_keys.size, dtype=np.float64)
        lat_count = np.zeros(lat_keys.size, dtype=np.int32)
        lon_count = np.zeros(lon_keys.size, dtype=np.int32)

        for idx, value in enumerate(feat_valid):
            r = lat_inverse[idx]
            c = lon_inverse[idx]
            grid_sum[r, c] += value
            grid_count[r, c] += 1
            lat_sum[r] += lat_valid[idx]
            lon_sum[c] += lon_valid[idx]
            lat_count[r] += 1
            lon_count[c] += 1

        grid = np.divide(
            grid_sum,
            grid_count,
            out=np.full_like(grid_sum, np.nan, dtype=np.float32),
            where=grid_count > 0,
        ).astype(np.float32, copy=False)

        lat_axis = np.divide(lat_sum, lat_count, out=lat_keys.astype(np.float64), where=lat_count > 0)
        lon_axis = np.divide(lon_sum, lon_count, out=lon_keys.astype(np.float64), where=lon_count > 0)
        candidate_mask = grid_count > 0
    else:
        srcs = list(getattr(stack, "srcs", []))
        if not srcs:
            print("[check] No raster sources available; skipping image preprocessing diagnostic plot.")
            return

        band_idx = 0
        feature_key = (feature_name or "").strip()
        band_paths = list(getattr(stack, "band_paths", []))
        if feature_key:
            try:
                idx = int(feature_key)
                if -len(srcs) <= idx < len(srcs):
                    band_idx = idx % len(srcs)
            except ValueError:
                lowered = feature_key.lower()
                match = None
                for idx, path in enumerate(band_paths):
                    stem = Path(path).stem.lower()
                    if lowered in stem:
                        match = idx
                        break
                if match is None:
                    for idx, path in enumerate(band_paths):
                        name = Path(path).name.lower()
                        if lowered in name:
                            match = idx
                            break
                if match is not None:
                    band_idx = match
                else:
                    print(f"[check] Feature '{feature_key}' not matched to raster band; defaulting to band 0.")

        try:
            reader = srcs[band_idx]
        except IndexError:
            print(f"[check] Raster band index {band_idx} is out of range; skipping diagnostics.")
            return

        try:
            band = reader.read(1, masked=True)
        except Exception as exc:
            print(f"[check] Unable to read raster band {band_idx}: {exc}")
            return

        if np.ma.isMaskedArray(band):
            candidate_mask = np.asarray(~band.mask, dtype=bool)
            grid = band.filled(np.nan).astype(np.float32, copy=False)
        else:
            grid = np.asarray(band, dtype=np.float32)
            candidate_mask = np.ones_like(grid, dtype=bool)

        nodata = getattr(reader, "nodata", None)
        if nodata is not None:
            invalid = np.isclose(grid, nodata, equal_nan=True)
            grid[invalid] = np.nan
            candidate_mask &= ~invalid

        candidate_mask &= np.isfinite(grid)

        transform = getattr(stack, "transform", None)
        if transform is not None:
            try:
                from rasterio.transform import xy

                rows = np.arange(grid.shape[0], dtype=np.int64)
                cols = np.arange(grid.shape[1], dtype=np.int64)
                lon_axis, _ = xy(transform, np.zeros_like(cols), cols, offset="center")
                _, lat_axis = xy(transform, rows, np.zeros_like(rows), offset="center")
                lon_axis = np.asarray(lon_axis, dtype=np.float64)
                lat_axis = np.asarray(lat_axis, dtype=np.float64)
            except Exception:
                lon_axis = np.arange(grid.shape[1], dtype=np.float64)
                lat_axis = np.arange(grid.shape[0], dtype=np.float64)
        else:
            lon_axis = np.arange(grid.shape[1], dtype=np.float64)
            lat_axis = np.arange(grid.shape[0], dtype=np.float64)

    if grid is None or candidate_mask is None or lat_axis is None or lon_axis is None:
        return

    finite_vals = grid[np.isfinite(grid)]
    if finite_vals.size == 0:
        print("[check] All grid cells are NaN for feature diagnostics; skipping plot.")
        return

    vmin = float(np.nanpercentile(finite_vals, 2))
    vmax = float(np.nanpercentile(finite_vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))

    total_valid = int(candidate_mask.sum())
    if total_valid == 0:
        print("[check] No populated grid cells available for patch sampling; skipping plot.")
        return

    rng = np.random.default_rng(rng_seed)
    max_samples = min(6, total_valid)

    if total_valid <= 1_000_000:
        candidate_centers = np.argwhere(candidate_mask)
        sampled_idx = rng.choice(candidate_centers.shape[0], size=max_samples, replace=False)
        selected = candidate_centers[sampled_idx]
    else:
        sampled: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        rows, cols = candidate_mask.shape
        max_attempts = max(1000, max_samples * 200)
        while len(sampled) < max_samples and attempts < max_attempts:
            attempts += 1
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            key = (r, c)
            if not candidate_mask[r, c] or key in seen:
                continue
            sampled.append(key)
            seen.add(key)
        if len(sampled) < max_samples:
            fallback = np.argwhere(candidate_mask)
            if fallback.size > 0:
                remaining = max_samples - len(sampled)
                fallback_idx = rng.choice(fallback.shape[0], size=min(remaining, fallback.shape[0]), replace=False)
                for idx in np.atleast_1d(fallback_idx):
                    r, c = fallback[int(idx)]
                    key = (int(r), int(c))
                    if key in seen:
                        continue
                    sampled.append(key)
                    if len(sampled) == max_samples:
                        break
        selected = np.asarray(sampled, dtype=np.int64)

    num_patches = selected.shape[0]
    if num_patches == 0:
        print("[check] Unable to select valid patches for diagnostics; skipping plot.")
        return

    patches = []
    lat_ranges = []
    lon_ranges = []
    half = window // 2
    n_lat, n_lon = grid.shape
    for r, c in selected:
        row_idx = np.clip(np.arange(r - half, r - half + window), 0, n_lat - 1)
        col_idx = np.clip(np.arange(c - half, c - half + window), 0, n_lon - 1)
        patch = grid[np.ix_(row_idx, col_idx)]
        patches.append(patch)
        lat_slice = lat_axis[row_idx]
        lon_slice = lon_axis[col_idx]
        lat_ranges.append((float(np.nanmin(lat_slice)), float(np.nanmax(lat_slice))))
        lon_ranges.append((float(np.nanmin(lon_slice)), float(np.nanmax(lon_slice))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_cols = 3
    fig_rows = int(np.ceil(num_patches / fig_cols))
    height = 4 + fig_rows * 3
    width = 6 + fig_cols * 3
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(fig_rows + 1, fig_cols, height_ratios=[4] + [1] * fig_rows)
    ax_main = fig.add_subplot(gs[0, :])

    extent = [float(np.nanmin(lon_axis)), float(np.nanmax(lon_axis)), float(np.nanmin(lat_axis)), float(np.nanmax(lat_axis))]
    im = ax_main.imshow(grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    ax_main.set_title(f"Original '{feature_name}' values")
    ax_main.set_xlabel("Longitude" if getattr(stack, "is_table", False) else "X coordinate")
    ax_main.set_ylabel("Latitude" if getattr(stack, "is_table", False) else "Y coordinate")
    fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label=feature_name)

    for idx, patch in enumerate(patches):
        row = idx // fig_cols
        col = idx % fig_cols
        ax = fig.add_subplot(gs[row + 1, col])
        masked_patch = np.ma.masked_invalid(patch)
        lon_min, lon_max = lon_ranges[idx]
        lat_min, lat_max = lat_ranges[idx]
        extent_patch = [lon_min, lon_max, lat_min, lat_max]
        ax.imshow(masked_patch, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent_patch, aspect="auto")
        ax.set_title(f"Patch {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    for empty_idx in range(num_patches, fig_rows * fig_cols):
        row = empty_idx // fig_cols
        col = empty_idx % fig_cols
        fig.add_subplot(gs[row + 1, col]).axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preproc_check_{feature_name.replace(' ', '_')}.png"
    print("out_path: ", out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[check] Saved preprocessing diagnostic plot to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', type=str, help='glob pattern to bands (e.g., /data/*.tif)')
    ap.add_argument('--stac-root', type=str, help='Path to STAC collection root (table assets)')
    ap.add_argument('--stac-table', type=str, help='Direct path to a STAC Parquet table asset')
    ap.add_argument('--features', nargs='+', help='Feature columns to use for STAC tables')
    ap.add_argument('--lat-column', type=str, help='Latitude column name for STAC tables')
    ap.add_argument('--lon-column', type=str, help='Longitude column name for STAC tables')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--patch', type=int, default=4, help='Patch size for MAE patch embedding (must divide window)')
    ap.add_argument('--window', type=int, default=16, help='Square crop size (pixels) for SSL inputs')
    ap.add_argument('--mask-ratio', type=float, default=0.75, help='Fraction of patches masked during MAE pretraining')
    ap.add_argument('--mask-scope', choices=['pixel', 'patch'], default='patch',
                    help='Choose masking granularity for debug previews (pixel zeros individual pixels, patch zeros whole patches)')
    ap.add_argument('--encoder-depth', type=int, default=6, help='Number of transformer blocks in the encoder')
    ap.add_argument('--decoder-depth', type=int, default=2, help='Number of transformer blocks in the decoder')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--optimizer', choices=['adamw', 'adam'], default='adamw')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--preview-samples', type=int, default=0, help='If >0, create reconstruction previews for this many samples')
    ap.add_argument('--check-image-preproc', action='store_true', help='Render diagnostic plots for patch windowing')
    ap.add_argument('--check-feature', type=str,
                    default='Gravity_Bouguer_HGM_Worms_Proximity', help='Feature name to visualize when running image preprocessing diagnostics',
                    )
    ap.add_argument('--skip-nan', dest='skip_nan', action='store_true',
                    help='Resample until patches contain only finite values (default behaviour)')
    ap.add_argument('--allow-nan', dest='skip_nan', action='store_false',
                    help='Allow NaNs in sampled patches (disables resampling)')
    ap.set_defaults(skip_nan=True)
    ap.add_argument('--skip-nan-attempts', type=int, default=64,
                    help='Maximum resampling attempts when skipping NaN-containing patches')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide exactly one of --bands, --stac-root, or --stac-table')

    if args.stac_table:
        stac_path = Path(args.stac_table)
        stack = StacTableStack(
            stac_path,
            feature_columns=args.features,
            latitude_column=args.lat_column,
            longitude_column=args.lon_column,
        )
        window_size = args.window
        if args.features:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    elif args.stac_root:
        stac_root_path = Path(args.stac_root).resolve()
        raster_paths, metadata = _collect_stac_raster_paths(stac_root_path, args.features)
        print(f"[info] Using {len(raster_paths)} raster asset(s) from {stac_root_path}")
        stack = GeoStack(raster_paths)
        window_size = args.window
    else:
        stack = GeoStack(sorted(glob.glob(args.bands)))
        window_size = args.window

    if args.window <= 0:
        ap.error('--window must be positive')
    if args.patch <= 0:
        ap.error('--patch must be positive')
    if args.window % args.patch != 0:
        ap.error('--window must be divisible by --patch to form an integer patch grid')

    preview_dir = Path(args.out) / 'previews'

    if args.check_image_preproc:
        _maybe_check_image_preproc(stack, args.check_feature, window_size, preview_dir)

    ds = SSLDataset(
        stack,
        window=window_size,
        skip_nan=args.skip_nan,
        max_resample_attempts=args.skip_nan_attempts,
    )

    # print('[dev] stack:', stack)
    # if hasattr(stack, '__len__'):
    #     print('[dev] stack length:', len(stack))
    # print('[dev] dataset:', ds)
    # try:
    #     first_item = ds[0]
    #     print('[dev] first item type:', type(first_item))
    #     if isinstance(first_item, dict):
    #         for key, value in first_item.items():
    #             value_info = getattr(value, 'shape', None) or getattr(value, 'size', None)
    #             print(f"[dev] ds[0]['{key}'] ->", value_info if value_info is not None else type(value))
    #     elif hasattr(first_item, 'shape'):
    #         print('[dev] first item shape:', first_item.shape, first_item)
    # except Exception as exc:
    #     print('[dev] failed to inspect first item:', exc)
    # exit()


    # TODO: Make it generalized for future development...
    worker_count = 8
    if getattr(stack, "kind", None) == "raster":
        worker_count = 0
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=worker_count, pin_memory=True)

    model = MAEViT(
        in_chans=stack.count,
        patch_size=args.patch,
        depth=args.encoder_depth,
        dec_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
        image_size=window_size,
    )
    model, history = train_ssl(
        model,
        dl,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        preview_samples=args.preview_samples,
        preview_dir=preview_dir if args.preview_samples > 0 else None,
        feature_names=getattr(stack, 'feature_columns', None),
        feature_metadata=getattr(stack, 'feature_metadata', None),
        mask_scope=args.mask_scope,
    )
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, 'mae_encoder.pth'))
    history_path = Path(args.out) / 'ssl_history.json'
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f"Saved training history to {history_path}")
