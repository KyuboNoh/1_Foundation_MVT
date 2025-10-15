# scripts/build_splits.py
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Sequence

import numpy as np
import torch
import rasterio

# ensure repo-root execution can resolve the local src package
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.gfm4mpm.data.geo_stack import GeoStack, load_deposit_pixels
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.sampling.likely_negatives import compute_embeddings, pu_select_negatives

def _resolve_search_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved

def _load_training_args(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates = []
    root = _resolve_search_root(stack_root)
    if root is not None:
        metadata_candidates = [
            root / 'training_metadata.json',
            root / 'assetization' / 'training_metadata.json',
        ]
        for meta_candidate in metadata_candidates:
            try:
                resolved_meta = meta_candidate.resolve()
            except Exception:
                continue
            if not resolved_meta.exists():
                continue
            try:
                with resolved_meta.open('r', encoding='utf-8') as fh:
                    meta_data = json.load(fh)
                pretraining_entry = meta_data.get('pretraining')
                if isinstance(pretraining_entry, dict):
                    args_data = pretraining_entry.get('args')
                    if isinstance(args_data, dict):
                        return args_data, resolved_meta
            except Exception as exc:
                print(f"[warn] Failed to read pretraining args from {resolved_meta}: {exc}")
        
        direct = root / 'training_args.json'
        candidates.append(direct)
        try:
            for candidate in root.rglob('training_args.json'):
                candidates.append(candidate)
        except Exception:
            pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_args.json')

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
            return data, resolved
        except Exception as exc:
            print(f"[warn] Failed to load training args from {resolved}: {exc}")
    return None, None


def _load_training_metadata(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates: List[Path] = []
    root = _resolve_search_root(stack_root)
    if root is not None:
        for candidate in [
            root / 'training_metadata.json',
            root / 'assetization' / 'training_metadata.json',
        ]:
            candidates.append(candidate)
        try:
            for candidate in root.rglob('training_metadata.json'):
                candidates.append(candidate)
        except Exception:
            pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_metadata.json')

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as fh:
                return json.load(fh), resolved
        except Exception as exc:
            print(f"[warn] Failed to load training metadata from {resolved}: {exc}")
    return None, None


def _mae_kwargs_from_training_args(args_dict: Dict) -> Dict:
    mapping = {
        'encoder_dim': 'embed_dim',
        'encoder_depth': 'depth',
        'encoder_num_heads': 'encoder_num_heads',
        'mlp_ratio': 'mlp_ratio',
        'mlp_ratio_decoder': 'mlp_ratio_dec',
        'decoder_dim': 'dec_dim',
        'decoder_depth': 'dec_depth',
        'decoder_num_heads': 'decoder_num_heads',
        'mask_ratio': 'mask_ratio',
        'patch': 'patch_size',
    }
    mae_kwargs: Dict = {}
    for src_key, dest_key in mapping.items():
        if src_key not in args_dict or args_dict[src_key] is None:
            continue
        value = args_dict[src_key]
        if dest_key in {
            'embed_dim',
            'depth',
            'encoder_num_heads',
            'dec_dim',
            'dec_depth',
            'decoder_num_heads',
            'patch_size',
        }:
            value = int(value)
        elif dest_key in {'mlp_ratio', 'mlp_ratio_dec', 'mask_ratio'}:
            value = float(value)
        mae_kwargs[dest_key] = value
    window = args_dict.get('window')
    if window is not None:
        mae_kwargs['image_size'] = int(window)
    return mae_kwargs


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        return np.pad(embeddings, ((0, 0), (0, max(0, 2 - embeddings.shape[1]))), mode='constant')[:, :2]
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:2].T
    except np.linalg.LinAlgError:
        components = np.eye(embeddings.shape[1], 2)
    return centered @ components


def _save_validation_plot(
    out_dir: Path,
    projection: np.ndarray,
    pos_idx: np.ndarray,
    unk_idx: np.ndarray,
    neg_idx: np.ndarray,
    distances: np.ndarray,
    keep_mask: np.ndarray,
    cutoff: Optional[float],
    filter_top_pct: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Validation plot skipped; matplotlib unavailable: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    proj_pos = projection[pos_idx]
    proj_unknown = projection[unk_idx]
    neg_mask = np.isin(unk_idx, neg_idx)
    proj_neg = proj_unknown[neg_mask]
    proj_unknown_kept = proj_unknown[~neg_mask & keep_mask]
    proj_unknown_filtered = proj_unknown[~keep_mask]

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    if len(proj_unknown_filtered):
        ax_scatter.scatter(
            proj_unknown_filtered[:, 0],
            proj_unknown_filtered[:, 1],
            facecolors='none',
            edgecolors='grey',
            linewidths=0.5,
            s=15,
            alpha=0.4,
            label='Unknown filtered',
        )

    if len(proj_unknown_kept):
        ax_scatter.scatter(
            proj_unknown_kept[:, 0],
            proj_unknown_kept[:, 1],
            facecolors='none',
            edgecolors='black',
            linewidths=0.5,
            s=20,
            label='Unknown kept',
        )

    if len(proj_neg):
        ax_scatter.scatter(
            proj_neg[:, 0],
            proj_neg[:, 1],
            c='#1f77b4',
            s=25,
            alpha=0.85,
            label='Selected negatives',
        )

    if len(proj_pos):
        ax_scatter.scatter(
            proj_pos[:, 0],
            proj_pos[:, 1],
            c='#d62728',
            s=28,
            alpha=0.85,
            label='Positives',
        )

    ax_scatter.set_xlabel('PC 1')
    ax_scatter.set_ylabel('PC 2')
    ax_scatter.set_title('Embedding projection')
    ax_scatter.legend(loc='best')

    if len(distances):
        bins = max(20, int(np.sqrt(len(distances))))
        ax_hist.hist(
            distances[keep_mask],
            bins=bins,
            color='#1f77b4',
            alpha=0.6,
            label='Distances (kept)',
        )
        ax_hist.hist(
            distances[~keep_mask],
            bins=bins,
            color='grey',
            alpha=0.4,
            label='Distances (filtered)',
        )
        if cutoff is not None:
            ax_hist.axvline(cutoff, color='#d62728', linestyle='--', linewidth=1.5, label=f'Cutoff ({filter_top_pct*100:.1f}%)')
        ax_hist.set_xlabel('Min distance to positives')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Distance distribution')
        ax_hist.legend(loc='best')

    fig.tight_layout()
    out_path = out_dir / 'validation_embeddings.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[info] Wrote validation plot to {out_path}")


def _resolve_pretraining_patch(args_dict: Optional[Dict], fallback: int) -> int:
    if not args_dict:
        return fallback
    for key in ('patch', 'window'):
        value = args_dict.get(key)
        if value is None:
            continue
        try:
            patch_val = int(value)
        except (TypeError, ValueError):
            continue
        if patch_val > 0:
            return patch_val
    return fallback


def _infer_region_from_name(path: Path) -> Optional[str]:
    stem_parts = path.stem.split('_')
    for token in reversed(stem_parts):
        token = token.upper()
        if token in {'GLOBAL', 'WORLD'}:
            return 'GLOBAL'
        if token.isalpha() and 1 <= len(token) <= 4:
            return token
    return None


def _resolve_label_rasters(
    metadata: Optional[Dict],
    metadata_path: Optional[Path],
    label_column: str,
) -> List[Dict[str, Any]]:
    if not metadata or not metadata_path:
        return []
    labels_section = metadata.get('labels')
    if not isinstance(labels_section, dict):
        return []
    entries = labels_section.get('entries')
    if not isinstance(entries, dict):
        return []
    entry = entries.get(label_column)
    if not isinstance(entry, dict):
        return []
    tif_entries = entry.get('tifs')
    if not isinstance(tif_entries, list):
        return []
    base_dir = metadata_path.parent
    resolved: List[Dict[str, Any]] = []
    for tif_info in tif_entries:
        if not isinstance(tif_info, dict):
            continue
        path_value = tif_info.get('path') or tif_info.get('filename')
        if not path_value:
            continue
        tif_path = Path(path_value)
        if not tif_path.is_absolute():
            tif_path = (base_dir / tif_path).resolve()
        if not tif_path.exists():
            print(f"[warn] Label raster not found: {tif_path}")
            continue
        region = tif_info.get('region')
        region_key = str(region or _infer_region_from_name(tif_path) or 'GLOBAL').upper()
        resolved.append({"path": tif_path, "region": region_key})
    return resolved


def _collect_feature_rasters(
    metadata: Optional[Dict],
    metadata_path: Optional[Path],
    feature_names: Optional[Sequence[str]],
) -> Dict[str, List[Path]]:
    if not metadata or not metadata_path:
        return {}
    features_section = metadata.get('features')
    if not isinstance(features_section, dict):
        return {}
    entries = features_section.get('entries')
    if not isinstance(entries, dict):
        return {}
    base_dir = metadata_path.parent

    def _lookup(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if name in entries:
            return name, entries[name]
        lowered = name.lower()
        for key, value in entries.items():
            if key.lower() == lowered:
                return key, value
        return None

    ordered_features: List[str]
    if feature_names:
        ordered_features = list(feature_names)
    else:
        ordered_features = sorted(entries.keys(), key=str.lower)

    region_map: Dict[str, Dict[str, Path]] = {}
    for feature_name in ordered_features:
        match = _lookup(feature_name)
        if match is None:
            print(f"[warn] Feature '{feature_name}' missing from training_metadata.json entries.")
            continue
        canonical_name, entry = match
        tif_records = entry.get('tifs', [])
        if not isinstance(tif_records, list):
            continue
        for record in tif_records:
            if not isinstance(record, dict):
                continue
            path_value = record.get('path') or record.get('filename')
            if not path_value:
                continue
            tif_path = Path(path_value)
            if not tif_path.is_absolute():
                tif_path = (base_dir / tif_path).resolve()
            else:
                tif_path = tif_path.resolve()
            if not tif_path.exists():
                print(f"[warn] Feature raster not found: {tif_path}")
                continue
            region = record.get('region')
            region_key = str(region or _infer_region_from_name(tif_path) or 'GLOBAL').upper()
            region_map.setdefault(region_key, {})[canonical_name] = tif_path

    usable: Dict[str, List[Path]] = {}
    for region, feature_map in region_map.items():
        missing = [name for name in ordered_features if name not in feature_map]
        if missing:
            continue
        usable[region] = [feature_map[name] for name in ordered_features]
    return usable


def _load_label_pixels(raster_paths: Sequence[Path]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for path in raster_paths:
        try:
            with rasterio.open(path) as src:
                data = src.read(1, masked=True)
        except Exception as exc:
            print(f"[warn] Failed to read label raster {path}: {exc}")
            continue
        if np.ma.isMaskedArray(data):
            mask = (~data.mask) & (data.data > 0.5)
        else:
            valid = np.isfinite(data)
            mask = valid & (data > 0.5)
        if not mask.any():
            continue
        rows, cols = np.where(mask)
        coords.extend(zip(rows.tolist(), cols.tolist()))
    return coords

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--pos_geojson', help='positive deposits (GeoJSON) when using rasters')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--filter_top_pct', type=float, default=0.10)
    ap.add_argument('--negs_per_pos', type=int, default=5)
    ap.add_argument('--validation', action='store_true', help='Generate diagnostic plots for PU negative selection')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands/--pos_geojson or one of --stac-root/--stac-table (exactly one input source)')

    stack_root_path: Optional[Path] = None
    training_args_data: Optional[Dict] = None
    training_args_path: Optional[Path] = None
    training_metadata: Optional[Dict] = None
    training_metadata_path: Optional[Path] = None
    encoder_path = Path(args.encoder).resolve()

    label_column = 'Training_MVT_Deposit'

    if args.stac_root or args.stac_table:
        stack_source = Path(args.stac_table or args.stac_root).resolve()
        stack_root_path = stack_source.parent if stack_source.is_file() else stack_source
        training_args_data, training_args_path = _load_training_args(stack_root_path, encoder_path)
        training_metadata, training_metadata_path = _load_training_metadata(stack_root_path, encoder_path)
        feature_columns = None
        lat_column = None
        lon_column = None
        if training_args_data:
            feature_columns = training_args_data.get('features')
            lat_column = training_args_data.get('lat_column')
            lon_column = training_args_data.get('lon_column')
            label_column = training_args_data.get('label_column', label_column)

        # print("feature_columns", feature_columns); 
        # print(training_args_data)
        # exit()

        stack = StacTableStack(
            stack_source,
            label_columns=[label_column],
            feature_columns=feature_columns,
            latitude_column=lat_column,
            longitude_column=lon_column,
        )
        labels = stack.label_array(label_column)
        pos_indices = np.where(labels == 1)[0]
        unk_indices = np.where(labels == 0)[0]
        pos = [stack.index_to_coord(int(i)) for i in pos_indices]
        unk = [stack.index_to_coord(int(i)) for i in unk_indices]
        patch = _resolve_pretraining_patch(training_args_data, 1)
        if feature_columns:
            print(f"[info] Using {len(stack.feature_columns)} feature columns from STAC table")
    else:
        band_paths_glob = sorted(glob.glob(args.bands))
        if not band_paths_glob:
            raise RuntimeError(f"No raster bands matched pattern {args.bands}")
        try:
            common = os.path.commonpath(band_paths_glob)
            stack_root_path = Path(common)
        except ValueError:
            stack_root_path = Path(band_paths_glob[0]).resolve().parent

        training_args_data, training_args_path = _load_training_args(stack_root_path, encoder_path)
        training_metadata, training_metadata_path = _load_training_metadata(stack_root_path, encoder_path)
        feature_columns = None
        if training_args_data:
            feature_columns = training_args_data.get('features')
            label_column = training_args_data.get('label_column', label_column)

        label_records = _resolve_label_rasters(training_metadata, training_metadata_path, label_column)
        label_by_region: Dict[str, List[Dict[str, Any]]] = {}
        for record in label_records:
            region_key = str(record.get('region') or 'GLOBAL').upper()
            label_by_region.setdefault(region_key, []).append(record)

        if not label_by_region and stack_root_path:
            search_dir = stack_root_path if stack_root_path.is_dir() else stack_root_path.parent
            fallback_paths = sorted(Path(search_dir).glob(f"*{label_column.lower()}*.tif"))
            if fallback_paths:
                for path in fallback_paths:
                    region_hint = _infer_region_from_name(path) or 'GLOBAL'
                    label_by_region.setdefault(region_hint.upper(), []).append({"path": path.resolve(), "region": region_hint})
                print(f"[info] Using fallback label raster discovery for '{label_column}'.")

        selected_region: Optional[str] = None
        if label_by_region:
            selected_region = max(label_by_region.items(), key=lambda kv: len(kv[1]))[0]
            if len(label_by_region) > 1:
                detail = ", ".join(f"{key}:{len(val)}" for key, val in label_by_region.items())
                print(f"[info] Multiple label regions detected ({detail}); using region '{selected_region}'.")
            else:
                print(f"[info] Label rasters located for region '{selected_region}'.")

        feature_regions = _collect_feature_rasters(training_metadata, training_metadata_path, feature_columns)

        band_paths: List[str] = []
        if feature_regions:
            candidate_region = selected_region
            if candidate_region and candidate_region in feature_regions:
                selected_feature_paths = feature_regions[candidate_region]
            elif 'GLOBAL' in feature_regions:
                candidate_region = 'GLOBAL'
                selected_feature_paths = feature_regions['GLOBAL']
            else:
                candidate_region, selected_feature_paths = next(iter(feature_regions.items()))
            if candidate_region:
                print(f"[info] Using feature rasters for region '{candidate_region}'.")
            band_paths: List[str] = [str(path) for path in selected_feature_paths]
        else:
            label_slug = label_column.lower()
            filtered_band_paths: List[str] = []
            for p in band_paths_glob:
                name = Path(p).name.lower()
                if label_slug in name:
                    continue
                region_hint = _infer_region_from_name(Path(p)) if selected_region else None
                if selected_region and region_hint and region_hint != selected_region:
                    continue
                filtered_band_paths.append(p)
            if not filtered_band_paths:
                raise RuntimeError(f"No feature rasters remain after filtering label column '{label_column}'")
            removed = len(band_paths_glob) - len(filtered_band_paths)
            if removed:
                print(f"[info] Removed {removed} label or mismatched region raster(s) from band list.")
            band_paths = filtered_band_paths

        if feature_columns and band_paths and len(band_paths) != len(feature_columns):
            print(
                f"[warn] Feature raster count ({len(band_paths)}) does not match training feature list "
                f"({len(feature_columns)}); verify assetization outputs."
            )

        stack = GeoStack(band_paths)
        patch = _resolve_pretraining_patch(training_args_data, 32)

        pos: List[Tuple[int, int]] = []
        if label_by_region and selected_region:
            label_paths_selected = [record["path"] for record in label_by_region[selected_region]]
            pos = _load_label_pixels(label_paths_selected)
            if pos:
                print(f"[info] Loaded {len(pos)} positive samples from label rasters.")

        if not pos:
            if args.pos_geojson:
                pos = load_deposit_pixels(args.pos_geojson, stack)
            else:
                raise RuntimeError('No label rasters found in training_metadata.json and --pos_geojson not provided')

        if pos:
            pos = list(dict.fromkeys((int(r), int(c)) for r, c in pos))
        grid = set(stack.grid_centers(stride=patch))
        pos_set = set(pos)
        unk = list(grid - pos_set)

    coords_all = pos + unk
    if not coords_all:
        raise RuntimeError('No coordinates collected to compute embeddings')
    if not pos:
        raise RuntimeError('No positive samples found; check label column or source data')

    mae_init_kwargs: Dict = {}
    if training_args_data:
        mae_init_kwargs = _mae_kwargs_from_training_args(training_args_data)
        if training_args_path:
            print(f"[info] Loaded training_args.json from {training_args_path}")
    else:
        print('[warn] training_args.json not found; falling back to default MAE configuration')

    if getattr(stack, 'kind', None) == 'table' and hasattr(stack, '_normalized'):
        mae_init_kwargs.setdefault('patch_size', 1)
        patch_size = 1
        normalized = stack._normalized.astype(np.float32)
        def _patch_batches(batch_coords: list[Tuple[int, int]], batch_size: int = 2048) -> np.ndarray:
            indices = [coord[0] for coord in batch_coords]
            batch = normalized[indices]
            return batch[:, :, None, None]
    else:
        default_patch = _resolve_pretraining_patch(training_args_data, patch)
        mae_init_kwargs.setdefault('patch_size', default_patch)
        patch_size = default_patch
        def _patch_batches(batch_coords: list[Tuple[int, int]], batch_size: int = 256) -> np.ndarray:
            patches = [stack.read_patch(r, c, patch_size) for r, c in batch_coords]
            return np.stack(patches, axis=0)

    encoder = MAEViT(in_chans=stack.count, **mae_init_kwargs)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))

    batch_size = 2048 if getattr(stack, 'kind', None) == 'table' else 256
    embedding_chunks: list[np.ndarray] = []
    for start in range(0, len(coords_all), batch_size):
        chunk_coords = coords_all[start:start + batch_size]
        batch_array = _patch_batches(chunk_coords)
        embeddings = compute_embeddings(encoder, batch_array)
        embedding_chunks.append(embeddings)
    Z_all = np.concatenate(embedding_chunks, axis=0)

    pos_idx = np.arange(0, len(pos))
    unk_idx = np.arange(len(pos), len(coords_all))

    if args.validation:
        neg_idx, debug_info = pu_select_negatives(
            Z_all,
            pos_idx,
            unk_idx,
            args.filter_top_pct,
            args.negs_per_pos,
            return_info=True,
        )
    else:
        neg_idx = pu_select_negatives(
            Z_all,
            pos_idx,
            unk_idx,
            args.filter_top_pct,
            args.negs_per_pos,
        )
        debug_info = None

    splits = {
        'pos': [list(map(int, rc)) for rc in pos],
        'neg': [list(map(int, coords_all[i])) for i in neg_idx.tolist()]
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out) / 'splits.json'
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(splits['pos'])} positives and {len(splits['neg'])} negatives to {out_path}")

    if args.validation and debug_info is not None:
        diag_dir = Path(args.out) / 'validation'
        projection = _project_embeddings(Z_all)
        distances = np.asarray(debug_info.get('distances'))
        keep_mask = np.asarray(debug_info.get('keep_mask'), dtype=bool)
        cutoff = debug_info.get('cutoff')
        _save_validation_plot(
            diag_dir,
            projection,
            pos_idx,
            unk_idx,
            np.asarray(neg_idx, dtype=int),
            distances,
            keep_mask,
            cutoff,
            args.filter_top_pct,
        )
