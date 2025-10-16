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
from tqdm import tqdm

# ensure repo-root execution can resolve the local src package
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.data_utils import (
    SplitDataLoad,
    clamp_coords_to_window,
    load_split_stack,
    normalize_region_coord,
    region_coord_to_dict,
)
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
        search_dirs = [root] + list(root.parents)
        for base in search_dirs:
            metadata_candidates = [
                base / 'training_metadata.json',
                base / 'assetization' / 'training_metadata.json',
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
                            merged_args = dict(args_data)
                            features_override = pretraining_entry.get('features')
                            if isinstance(features_override, list):
                                merged_args['features'] = list(dict.fromkeys(str(f) for f in features_override))
                            output_dir_val = pretraining_entry.get('output_dir')
                            if output_dir_val:
                                try:
                                    out_path = Path(output_dir_val)
                                    if not out_path.is_absolute():
                                        out_path = (resolved_meta.parent / out_path).resolve()
                                    ta_candidate = out_path / 'training_args_1_pretrain.json'
                                    if ta_candidate.exists():
                                        with ta_candidate.open('r', encoding='utf-8') as ta_fh:
                                            ta_data = json.load(ta_fh)
                                        if isinstance(ta_data, dict):
                                            merged_args.update(
                                                {k: v for k, v in ta_data.items() if k not in merged_args or k == 'features'}
                                            )
                                            features_ta = ta_data.get('features')
                                            if isinstance(features_ta, list):
                                                merged_args['features'] = list(dict.fromkeys(str(f) for f in features_ta))
                                except Exception as exc:
                                    print(f"[warn] Failed to read training_args_1_pretrain.json from metadata output_dir: {exc}")
                            return merged_args, resolved_meta
                except Exception as exc:
                    print(f"[warn] Failed to read pretraining args from {resolved_meta}: {exc}")

        direct_pretrain = root / 'training_args_1_pretrain.json'
        direct_standard = root / 'training_args.json'
        candidates.append(direct_pretrain)
        candidates.append(direct_standard)
        try:
            for candidate in root.rglob('training_args_1_pretrain.json'):
                candidates.append(candidate)
            for candidate in root.rglob('training_args.json'):
                candidates.append(candidate)
        except Exception:
            pass

    encoder_dir = encoder_path.resolve().parent
    candidates.append(encoder_dir / 'training_args_1_pretrain.json')
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
        search_dirs = [root] + list(root.parents)
        for base in search_dirs:
            for candidate in [
                base / 'training_metadata.json',
                base / 'assetization' / 'training_metadata.json',
            ]:
                candidates.append(candidate)
            try:
                for candidate in base.rglob('training_metadata.json'):
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
        distances_arr = np.asarray(distances, dtype=float)
        bin_edges = np.histogram_bin_edges(distances_arr, bins=bins)
        kept_counts, _ = np.histogram(distances[keep_mask], bins=bin_edges)
        ax_hist.hist(
            distances[keep_mask],
            bins=bin_edges,
            color='#1f77b4',
            alpha=0.6,
            label='Distances (kept)',
        )
        filtered_counts: Optional[np.ndarray] = None
        if (~keep_mask).any():
            filtered_counts, _ = np.histogram(distances[~keep_mask], bins=bin_edges)
            ax_hist.hist(
                distances[~keep_mask],
                bins=bin_edges,
                color='grey',
                alpha=0.4,
                label='Distances (filtered)',
            )
        if cutoff is not None:
            ax_hist.axvline(cutoff, color='#d62728', linestyle='--', linewidth=1.5, label=f'Cutoff ({filter_top_pct*100:.1f}%)')

        # max_count = float(kept_counts.max()) if kept_counts.size else 0.0
        # if filtered_counts is not None and filtered_counts.size:
        #     max_count = max(max_count, float(filtered_counts.max()))
        max_count = float(filtered_counts.max())

        ax_hist.set_xlabel('Min distance to positives')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Distance distribution')
        if max_count > 0:
            ax_hist.set_ylim(0, max(max_count, 1.0))
        ax_hist.legend(loc='best')

    fig.tight_layout()
    out_path = out_dir / 'embeddings_labeling.png'
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
    priority_codes = {'NA', 'AU', 'SA', 'EU', 'AS', 'AF', 'OC', 'GL', 'CA', 'US'}
    for token in reversed(stem_parts):
        up = token.upper()
        if up in priority_codes:
            return up
    for token in reversed(stem_parts):
        up = token.upper()
        if up in {'GLOBAL', 'WORLD'}:
            return 'GLOBAL'
        if up.isalpha() and 1 <= len(up) <= 4:
            return up
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

    region_map: Dict[str, Dict[str, List[Path]]] = {}
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
            region_map.setdefault(region_key, {}).setdefault(canonical_name, []).append(tif_path)

    usable: Dict[str, List[Path]] = {}
    for region, feature_map in region_map.items():
        missing = [name for name in ordered_features if name not in feature_map]
        if missing:
            continue
        ordered_paths: List[Path] = []
        for name in ordered_features:
            paths = feature_map.get(name)
            if not paths:
                break
            ordered_paths.extend(sorted(paths, key=lambda p: p.name))
        if len(ordered_paths) == sum(len(feature_map[name]) for name in ordered_features if name in feature_map):
            usable[region] = ordered_paths
    return usable


def _read_stack_patch(stack: Any, coord: Any, window: int) -> np.ndarray:
    if hasattr(stack, "resolve_region_stack"):
        default_region = getattr(stack, "default_region", None)
        if default_region is None:
            regions = getattr(stack, "regions", [])
            default_region = regions[0] if regions else "GLOBAL"
        region, row, col = normalize_region_coord(coord, default_region=default_region)
        region_stack = stack.resolve_region_stack(region)
        return region_stack.read_patch(row, col, window)

    if isinstance(coord, dict):
        row = coord.get("row")
        col = coord.get("col")
        if row is None or col is None:
            raise ValueError(f"Coordinate dictionary missing row/col: {coord}")
        return stack.read_patch(int(row), int(col), window)

    if isinstance(coord, (tuple, list)) and len(coord) == 3:
        _, row, col = coord
        return stack.read_patch(int(row), int(col), window)

    row, col = coord
    return stack.read_patch(int(row), int(col), window)


def _load_label_pixels(raster_paths: Dict[str, Sequence[Path]]) -> List[Tuple[str, int, int]]:
    regional_coords: List[Tuple[str, int, int]] = []
    for region, paths in raster_paths.items():
        for path in paths:
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
            regional_coords.extend((region, int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist()))
    return regional_coords


def _load_deposit_pixels_with_regions(geojson_path: str, stack: Any) -> List[Tuple[str, int, int]]:
    if hasattr(stack, "resolve_region_stack"):
        coords: List[Tuple[str, int, int]] = []
        for region_name, region_stack in stack.iter_region_stacks():
            region_coords = load_deposit_pixels(geojson_path, region_stack)
            coords.extend((region_name, int(r), int(c)) for r, c in region_coords)
        return coords

    coords = load_deposit_pixels(geojson_path, stack)
    return [("GLOBAL", int(r), int(c)) for r, c in coords]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--pos_geojson', help='positive deposits (GeoJSON) when using rasters')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--filter_top_pct', type=float, default=0.50)
    ap.add_argument('--negs_per_pos', type=int, default=5)
    ap.add_argument('--readembedding', action='store_true', help='Load embeddings from cached embeddings.npy in the output directory')
    ap.add_argument('--validation', action='store_true', help='Generate diagnostic plots for PU negative selection')
    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands/--pos_geojson or one of --stac-root/--stac-table (exactly one input source)')

    stack_root_path: Optional[Path] = None
    training_args_data: Optional[Dict] = None
    training_args_path: Optional[Path] = None
    encoder_path = Path(args.encoder).resolve()

    label_column = 'Training_MVT_Deposit'
    feature_columns: Optional[List[str]] = None

    load_result = load_split_stack(
        args.stac_root,
        args.stac_table,
        args.bands,
        encoder_path,
        label_column,
        _load_training_args,
        _load_training_metadata,
        _resolve_label_rasters,
        _collect_feature_rasters,
        _infer_region_from_name,
    )

    stack = load_result.stack
    stack_root_path = load_result.stack_root
    label_column = load_result.label_column
    feature_columns = load_result.feature_columns
    training_args_data = load_result.training_args
    training_args_path = load_result.training_args_path

    if load_result.mode == 'table':
        patch = _resolve_pretraining_patch(training_args_data, 1)
        labels = stack.label_array(label_column)
        pos_indices = np.where(labels == 1)[0]
        unk_indices = np.where(labels == 0)[0]
        pos = [stack.index_to_coord(int(i)) for i in pos_indices]
        unk = [stack.index_to_coord(int(i)) for i in unk_indices]
    else:
        patch = _resolve_pretraining_patch(training_args_data, 32)
        pos_raw: List[Tuple[str, int, int]] = []
        if load_result.label_paths:
            pos_raw = _load_label_pixels(load_result.label_paths)
            if pos_raw:
                print(f"[info] Loaded {len(pos_raw)} positive samples from label rasters.")
        if not pos_raw:
            if args.pos_geojson:
                pos_raw = _load_deposit_pixels_with_regions(args.pos_geojson, stack)
            else:
                raise RuntimeError('No label rasters found in training_metadata.json and --pos_geojson not provided')

        pos_raw = list(dict.fromkeys(pos_raw))
        pos_clamped, clamp_count_pos = clamp_coords_to_window(pos_raw, stack, patch)
        if clamp_count_pos:
            print(f"[info] Adjusted {clamp_count_pos} positive coordinate(s) to stay within window bounds")

        default_region = getattr(stack, "default_region", None)
        if default_region is None:
            default_region = "GLOBAL"
            if hasattr(stack, "resolve_region_stack"):
                regions = getattr(stack, "regions", [])
                if regions:
                    default_region = regions[0]

        pos = [
            normalize_region_coord(coord, default_region=default_region)
            for coord in pos_clamped
        ]
        pos = list(dict.fromkeys(pos))

        grid_coords = [
            normalize_region_coord(coord, default_region=default_region)
            for coord in stack.grid_centers(stride=patch)
        ]
        grid_set = set(grid_coords)
        pos_set = set(pos)
        unk_raw = list(grid_set - pos_set)
        unk_clamped, clamp_count_unk = clamp_coords_to_window(unk_raw, stack, patch)
        if clamp_count_unk:
            print(f"[info] Adjusted {clamp_count_unk} unlabeled coordinate(s) to stay within window bounds")
        unk = [
            normalize_region_coord(coord, default_region=default_region)
            for coord in unk_clamped
        ]
        unk = list(dict.fromkeys(unk))

    coords_all = pos + unk
    if not coords_all:
        raise RuntimeError('No coordinates collected to compute embeddings')
    if not pos:
        raise RuntimeError('No positive samples found; check label column or source data')

    if hasattr(stack, "resolve_region_stack"):
        serialize_default_region = getattr(stack, "default_region", None)
        if serialize_default_region is None:
            regions = getattr(stack, "regions", [])
            serialize_default_region = regions[0] if regions else "GLOBAL"
    else:
        serialize_default_region = "GLOBAL"

    def _serialize_coord_entry(coord: Any) -> Dict[str, int]:
        region, row, col = normalize_region_coord(coord, default_region=serialize_default_region)
        return region_coord_to_dict((region, row, col))

    state_dict = torch.load(encoder_path, map_location='cpu')
    expected_channels: Optional[int] = None
    patch_proj_weight = state_dict.get('patch.proj.weight')
    if isinstance(patch_proj_weight, torch.Tensor):
        expected_channels = patch_proj_weight.shape[1]
    if expected_channels is not None and stack.count != expected_channels:
        raise RuntimeError(
            f"MAE encoder expects {expected_channels} input channel(s) but raster stack provides {stack.count}; "
            "verify feature selection and metadata alignment."
        )

    mae_init_kwargs: Dict = {}
    if training_args_data:
        mae_init_kwargs = _mae_kwargs_from_training_args(training_args_data)
        if training_args_path:
            print(f"[info] Loaded training_args_1_pretrain.json for feature selection from {training_args_path}")
    else:
        print('[warn] training_args_1_pretrain.json not found; falling back to default MAE configuration')

    if getattr(stack, 'kind', None) == 'table' and hasattr(stack, '_normalized'):
        mae_init_kwargs.setdefault('patch_size', 1)
        patch_size = 1
        normalized = stack._normalized.astype(np.float32)
        def _patch_batches(batch_coords: list[Tuple[int, int]], batch_size: int = 256) -> np.ndarray:
            indices = [coord[0] for coord in batch_coords]
            batch = normalized[indices]
            return batch[:, :, None, None]
    else:
        default_patch = _resolve_pretraining_patch(training_args_data, patch)
        mae_init_kwargs.setdefault('patch_size', default_patch)
        patch_size = default_patch
        mae_window = mae_init_kwargs.get('image_size')
        if isinstance(mae_window, (tuple, list)):
            mae_window = int(mae_window[0])
        elif mae_window is not None:
            mae_window = int(mae_window)
        else:
            mae_window = patch_size
        def _patch_batches(batch_coords: List[Any], batch_size: int = 256) -> np.ndarray:
            patches = [_read_stack_patch(stack, coord, mae_window) for coord in batch_coords]
            return np.stack(patches, axis=0)

    encoder = MAEViT(in_chans=stack.count, **mae_init_kwargs)
    encoder.load_state_dict(state_dict)

    # default_batch_size = 256
    # batch_size = default_batch_size
    if training_args_data:
        for batch_key in (
            'batch',
        ):
            candidate = training_args_data.get(batch_key)
            if candidate is None:
                continue
            try:
                resolved_size = int(candidate)
            except (TypeError, ValueError):
                try:
                    resolved_size = int(float(candidate))
                except (TypeError, ValueError):
                    continue
            if resolved_size > 0:
                batch_size = resolved_size
                break
    # if batch_size != default_batch_size:
    print(f"[info] Using batch size {batch_size} from training_args_1_pretrain.json")
    
    out_dir = Path(args.out)
    embedding_cache_path = out_dir / 'embeddings.npy'

    if not args.readembedding:
        embedding_chunks: list[np.ndarray] = []
        total_batches = (len(coords_all) + batch_size - 1) // batch_size
        batch_iter = tqdm(
            range(0, len(coords_all), batch_size),
            total=total_batches,
            desc="Embedding patches",
        )
        for start in batch_iter:
            chunk_coords = coords_all[start:start + batch_size]
            batch_array = _patch_batches(chunk_coords)
            embeddings = compute_embeddings(encoder, batch_array)
            embedding_chunks.append(embeddings)

        if not embedding_chunks:
            raise RuntimeError('No coordinates collected to compute embeddings')
        Z_all = np.concatenate(embedding_chunks, axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(embedding_cache_path, Z_all)
            print(f"[info] Saved embeddings to {embedding_cache_path}")
        except Exception as exc:
            print(f"[warn] Failed to save embeddings cache to {embedding_cache_path}: {exc}")
    else:
        if not embedding_cache_path.exists():
            raise FileNotFoundError(
                f"Cached embeddings not found at {embedding_cache_path}; "
                "run without --readembedding to generate them."
            )
        Z_all = np.load(embedding_cache_path)
        if Z_all.shape[0] != len(coords_all):
            raise ValueError(
                f"Cached embeddings at {embedding_cache_path} have {Z_all.shape[0]} rows; "
                f"expected {len(coords_all)} based on coordinates."
            )
        print(f"[info] Loaded embeddings from {embedding_cache_path}")

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
        'pos': [_serialize_coord_entry(rc) for rc in pos],
        'neg': [_serialize_coord_entry(coords_all[i]) for i in neg_idx.tolist()],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'splits.json'
    with out_path.open('w') as f:
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
