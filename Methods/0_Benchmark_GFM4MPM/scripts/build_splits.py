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

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency
    CRS = None  # type: ignore[assignment]
    Transformer = None  # type: ignore[assignment]

# ensure repo-root execution can resolve the local src package
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.data_utils import (
    SplitDataLoad,
    clamp_coords_to_window,
    filter_valid_raster_coords,
    load_split_stack,
    normalize_region_coord,
    prefilter_valid_window_coords,
    region_coord_to_dict,
)
from Common.debug_visualization import visualize_debug_features
from src.gfm4mpm.data.geo_stack import GeoStack, load_deposit_pixels
from src.gfm4mpm.data.stac_table import StacTableStack
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.sampling.likely_negatives import compute_embeddings, pu_select_negatives
from Common.data_utils import resolve_search_root, load_training_args, load_training_metadata, mae_kwargs_from_training_args, resolve_pretraining_patch, infer_region_from_name, resolve_label_rasters, collect_feature_rasters, read_stack_patch

TARGET_CRS_EPSG = 8857
TARGET_CRS = CRS.from_epsg(TARGET_CRS_EPSG) if CRS is not None else None
_TRANSFORMER_CACHE: Dict[str, Transformer] = {}


def _project_to_target_crs(x: float, y: float, src_crs) -> Tuple[float, float]:
    if TARGET_CRS is None or Transformer is None or src_crs is None:
        return float(x), float(y)
    try:
        src = CRS.from_user_input(src_crs)
        key = src.to_string()
        transformer = _TRANSFORMER_CACHE.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(src, TARGET_CRS, always_xy=True)
            _TRANSFORMER_CACHE[key] = transformer
        x_t, y_t = transformer.transform(x, y)
        return float(x_t), float(y_t)
    except Exception:
        return float(x), float(y)


def _prepare_region_stack_lookup(stack: Any) -> Dict[str, Any]:
    lookup: Dict[str, Any] = {}
    if hasattr(stack, "iter_region_stacks"):
        try:
            for region_name, region_stack in stack.iter_region_stacks():
                lookup[str(region_name)] = region_stack
        except Exception:
            lookup = {}
    if not lookup:
        lookup["GLOBAL"] = stack
    return lookup


def _metadata_for_raster_coord(
    region_lookup: Dict[str, Any],
    coord: Tuple[str, int, int],
    label_value: int,
) -> Tuple[str, Dict[str, Any], Tuple[float, float]]:
    region, row, col = coord
    entry: Dict[str, Any] = {
        "region": region,
        "row": int(row),
        "col": int(col),
        "label": int(label_value),
        "label_type": "positive" if label_value == 1 else "unlabeled",
    }
    tile_id = f"{region}_{int(row)}_{int(col)}"
    region_stack = region_lookup.get(str(region)) or region_lookup.get("GLOBAL")
    coords_xy = (float("nan"), float("nan"))
    if region_stack is not None:
        transform = getattr(region_stack, "transform", None)
        crs = getattr(region_stack, "crs", None)
        if transform is not None:
            try:
                x, y = rasterio.transform.xy(transform, int(row), int(col), offset="center")
                coords_xy = _project_to_target_crs(float(x), float(y), crs)
            except Exception:
                pass
        if crs is not None:
            try:
                entry["crs"] = str(crs)
            except Exception:
                entry["crs"] = None
    return tile_id, entry, coords_xy


def _metadata_for_table_coord(stack: Any, coord: Tuple[int, int], label_value: int) -> Tuple[str, Dict[str, Any], Tuple[float, float]]:
    index = int(coord[0]) if isinstance(coord, (tuple, list)) else int(coord)
    entry: Dict[str, Any] = {
        "index": index,
        "label": int(label_value),
        "label_type": "positive" if label_value == 1 else "unlabeled",
    }
    coords_xy = (float("nan"), float("nan"))
    try:
        meta_row = stack.metadata_row(index)
        if isinstance(meta_row, dict):
            entry.update(meta_row)
    except Exception:
        pass
    tile_id = str(entry.get("h3_address") or entry.get("index") or f"row_{index}")
    lon = entry.get("longitude")
    lat = entry.get("latitude")
    if lon is not None and lat is not None:
        coords_xy = _project_to_target_crs(float(lon), float(lat), "EPSG:4326")
    return tile_id, entry, coords_xy


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


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: _to_serializable(val) for key, val in vars(args).items()}

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
    ap.add_argument('--debug', action='store_true', help='Generate debug visualisations of labels and feature rasters')
    args = ap.parse_args()

    args_record = _namespace_to_dict(args)
    metrics_summary: Dict[str, Any] = {}

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
        load_training_args,
        load_training_metadata,
        resolve_label_rasters,
        collect_feature_rasters,
        infer_region_from_name,
    )

    stack = load_result.stack
    stack_root_path = load_result.stack_root
    label_column = load_result.label_column
    feature_columns = load_result.feature_columns
    training_args_data = load_result.training_args
    training_args_path = load_result.training_args_path
    metrics_summary["input_mode"] = load_result.mode
    if training_args_path is not None:
        metrics_summary["pretrain_args_path"] = str(training_args_path)
    if stack_root_path is not None:
        metrics_summary["stack_root"] = str(stack_root_path)

    if load_result.mode == 'table':
        patch = _resolve_pretraining_patch(training_args_data, 1)
        labels = stack.label_array(label_column)
        pos_indices = np.where(labels == 1)[0]
        unk_indices = np.where(labels == 0)[0]
        pos = [stack.index_to_coord(int(i)) for i in pos_indices]
        unk = [stack.index_to_coord(int(i)) for i in unk_indices]
        metrics_summary.update(
            initial_pos=len(pos),
            initial_unknown=len(unk),
        )
    else:
        patch = _resolve_pretraining_patch(training_args_data, 32)
        pos_raw: List[Tuple[str, int, int]] = []
        if load_result.label_paths:
            pos_raw = _load_label_pixels(load_result.label_paths)
            if pos_raw:
                print(f"[info] Loaded {len(pos_raw)} positive samples from label rasters.")
        metrics_summary["initial_pos_raw"] = len(pos_raw)
        if not pos_raw:
            if args.pos_geojson:
                pos_raw = _load_deposit_pixels_with_regions(args.pos_geojson, stack)
            else:
                raise RuntimeError('No label rasters found in training_metadata.json and --pos_geojson not provided')

        pos_raw = list(dict.fromkeys(pos_raw))
        print("[info] Clamping coordinates for positives to valid patch windows...")
        pos_clamped, clamp_count_pos = clamp_coords_to_window(pos_raw, stack, patch)
        if clamp_count_pos:
            print(f"[info] Adjusted {clamp_count_pos} positive coordinate(s) to stay within window bounds")
        metrics_summary["clamp_count_pos"] = int(clamp_count_pos)

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

        pos, dropped_pos = filter_valid_raster_coords(stack, pos, patch, min_valid_fraction=0.05)
        if dropped_pos:
            print(f"[info] Dropped {len(dropped_pos)} positive coordinate(s) lacking sufficient valid pixels")
        metrics_summary["dropped_pos_invalid"] = len(dropped_pos)
        if not pos:
            raise RuntimeError("No usable positive samples remain after filtering; check label coverage.")

        grid_coords = [
            normalize_region_coord(coord, default_region=default_region)
            for coord in stack.grid_centers(stride=patch)
        ]
        grid_set = set(grid_coords)
        pos_set = set(pos)
        unk_raw = list(grid_set - pos_set)
        metrics_summary["initial_unknown_raw"] = len(unk_raw)
        print("[info] Clamping coordinates for unknowns to valid patch windows...")
        unk_clamped, clamp_count_unk = clamp_coords_to_window(unk_raw, stack, patch)
        if clamp_count_unk:
            print(f"[info] Adjusted {clamp_count_unk} unlabeled coordinate(s) to stay within window bounds")
        metrics_summary["clamp_count_unknown"] = int(clamp_count_unk)
        unk = [
            normalize_region_coord(coord, default_region=default_region)
            for coord in unk_clamped
        ]
        unk = list(dict.fromkeys(unk))

        prefiltered_unk = prefilter_valid_window_coords(stack, unk, patch)
        dropped_prefilter = len(unk) - len(prefiltered_unk)
        if dropped_prefilter:
            print(f"[info] Prefiltered {dropped_prefilter} unlabeled coordinate(s) with insufficient window coverage")
        metrics_summary["prefilter_dropped_unknown"] = int(dropped_prefilter)
        print(f"[info] Filtering invalid raster coordinate(s) starts...")
        unk, dropped_unk = filter_valid_raster_coords(stack, prefiltered_unk, patch, min_valid_fraction=0.05)
        if dropped_unk:
            print(f"[info] Dropped {len(dropped_unk)} candidate negative coordinate(s) lacking sufficient valid pixels")
        metrics_summary["dropped_unknown_invalid"] = len(dropped_unk)

    coords_all = pos + unk
    if not coords_all:
        raise RuntimeError('No coordinates collected to compute embeddings')
    if not pos:
        raise RuntimeError('No positive samples found; check label column or source data')
    metrics_summary["total_coords"] = len(coords_all)
    metrics_summary["final_positive_candidates"] = len(pos)
    metrics_summary["final_unknown_candidates"] = len(unk)

    labels_array = np.zeros(len(coords_all), dtype=np.int8)
    labels_array[:len(pos)] = 1

    region_lookup = _prepare_region_stack_lookup(stack) if load_result.mode != "table" else {}
    metadata_records: List[Dict[str, Any]] = []
    tile_ids: List[str] = []
    coords_xy = np.full((len(coords_all), 2), np.nan, dtype=np.float64)

    if load_result.mode == "table":
        for idx, coord in enumerate(coords_all):
            tile_id, meta_entry, xy = _metadata_for_table_coord(stack, coord, int(labels_array[idx]))
            tile_ids.append(tile_id)
            metadata_records.append(meta_entry)
            coords_xy[idx] = xy
    else:
        for idx, coord in enumerate(coords_all):
            tile_id, meta_entry, xy = _metadata_for_raster_coord(region_lookup, coord, int(labels_array[idx]))
            tile_ids.append(tile_id)
            metadata_records.append(meta_entry)
            coords_xy[idx] = xy

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

    multi_region = hasattr(stack, "resolve_region_stack")
    coords_by_region: Dict[str, List[Tuple[int, Tuple[str, int, int]]]] = {}
    pos_idx_by_region: Dict[str, List[int]] = {}
    unk_idx_by_region: Dict[str, List[int]] = {}

    for idx, coord in enumerate(pos):
        region = coord[0]
        pos_idx_by_region.setdefault(region, []).append(idx)
        coords_by_region.setdefault(region, []).append((idx, coord))

    for offset, coord in enumerate(unk):
        global_idx = len(pos) + offset
        region = coord[0]
        unk_idx_by_region.setdefault(region, []).append(global_idx)
        coords_by_region.setdefault(region, []).append((global_idx, coord))

    if not coords_by_region:
        coords_by_region["GLOBAL"] = list(enumerate(coords_all))

    region_indices_map: Dict[str, List[int]] = {
        region: [idx for idx, _ in entries]
        for region, entries in coords_by_region.items()
    }
    region_coords_map: Dict[str, List[Tuple[str, int, int]]] = {
        region: [coord for _, coord in entries]
        for region, entries in coords_by_region.items()
    }

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
    print(f"[info] Using batch size {batch_size} from training_args_1_pretrain.json")
    
    out_dir = Path(args.out)
    embedding_cache_path = out_dir / 'embeddings.npy'

    region_embeddings_map: Dict[str, np.ndarray] = {}

    if not args.readembedding:
        Z_all: Optional[np.ndarray] = None
        out_dir.mkdir(parents=True, exist_ok=True)
        total_regions = len(region_coords_map)
        for region, region_coords in region_coords_map.items():
            region_indices = region_indices_map.get(region, [])
            if not region_coords:
                continue
            embedding_chunks: List[np.ndarray] = []
            total_batches = (len(region_coords) + batch_size - 1) // batch_size
            iterator = range(0, len(region_coords), batch_size)
            if total_regions > 1:
                iterator = tqdm(
                    iterator,
                    total=total_batches,
                    desc=f"Embedding patches [{region}]",
                    leave=False,
                )
            else:
                iterator = tqdm(
                    iterator,
                    total=total_batches,
                    desc="Embedding patches",
                )
            for start in iterator:
                chunk_coords = region_coords[start:start + batch_size]
                batch_array = _patch_batches(chunk_coords)
                embeddings = compute_embeddings(encoder, batch_array)
                embedding_chunks.append(embeddings)

            if not embedding_chunks:
                continue
            region_embeddings = np.concatenate(embedding_chunks, axis=0)
            region_embeddings_map[region] = region_embeddings
            if Z_all is None:
                Z_all = np.empty((len(coords_all), region_embeddings.shape[1]), dtype=region_embeddings.dtype)
            Z_all[region_indices, :] = region_embeddings

        if Z_all is None:
            raise RuntimeError('No coordinates collected to compute embeddings')
        bundle_path = embedding_cache_path.with_suffix('.npz')
        try:
            np.save(embedding_cache_path, Z_all)
            print(f"[info] Saved embeddings to {embedding_cache_path}")
        except Exception as exc:
            print(f"[warn] Failed to save embeddings cache to {embedding_cache_path}: {exc}")
        try:
            np.savez_compressed(
                bundle_path,
                embeddings=Z_all,
                labels=labels_array,
                tile_ids=np.asarray(tile_ids, dtype=object),
                coords=coords_xy.astype(np.float32, copy=False),
                metadata=np.asarray(metadata_records, dtype=object),
            )
            print(f"[info] Saved embedding bundle to {bundle_path}")
        except Exception as exc:
            print(f"[warn] Failed to save embedding bundle to {bundle_path}: {exc}")
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
        for region, region_indices in region_indices_map.items():
            region_embeddings_map[region] = Z_all[region_indices]
        print(f"[info] Loaded embeddings from {embedding_cache_path}")

    neg_idx_list: List[np.ndarray] = []
    debug_info_by_region: Dict[str, Dict[str, Any]] = {}

    for region, pos_idx_list in pos_idx_by_region.items():
        unk_idx_list = unk_idx_by_region.get(region, [])
        if not pos_idx_list:
            continue
        if not unk_idx_list:
            print(f"[warn] Region '{region}' has no candidate unlabeled points; skipping negative selection for this region.")
            continue

        pos_idx_arr = np.asarray(pos_idx_list, dtype=int)
        unk_idx_arr = np.asarray(unk_idx_list, dtype=int)

        if args.validation:
            neg_idx_region, info = pu_select_negatives(
                Z_all,
                pos_idx_arr,
                unk_idx_arr,
                args.filter_top_pct,
                args.negs_per_pos,
                return_info=True,
            )
        else:
            neg_idx_region = pu_select_negatives(
                Z_all,
                pos_idx_arr,
                unk_idx_arr,
                args.filter_top_pct,
                args.negs_per_pos,
            )
            info = None

        neg_idx_region = np.asarray(neg_idx_region, dtype=int)
        if neg_idx_region.size == 0:
            print(f"[warn] Negative selection for region '{region}' returned 0 samples; consider adjusting parameters.")
        neg_idx_list.append(neg_idx_region)

        if info is not None:
            region_indices = region_indices_map.get(region, [])
            lookup = {global_idx: local_idx for local_idx, global_idx in enumerate(region_indices)}
            pos_local = np.asarray([lookup[idx] for idx in pos_idx_list], dtype=int)
            unk_local = np.asarray([lookup[idx] for idx in unk_idx_list], dtype=int)
            neg_local = np.asarray([lookup[idx] for idx in neg_idx_region if idx in lookup], dtype=int)
            info.update(
                pos_local=pos_local,
                unk_local=unk_local,
                neg_local=neg_local,
                region_embeddings=region_embeddings_map.get(region),
            )
            debug_info_by_region[region] = info

    if not neg_idx_list:
        raise RuntimeError("Failed to select any negatives; ensure each region has positives and candidate negatives.")

    neg_idx = np.concatenate([arr for arr in neg_idx_list if arr.size])
    metrics_summary["selected_negatives"] = int(neg_idx.size)

    splits = {
        'pos': [_serialize_coord_entry(rc) for rc in pos],
        'neg': [_serialize_coord_entry(coords_all[i]) for i in neg_idx.tolist()],
    }

    if args.debug and load_result.mode != 'table':
        feature_list = feature_columns or getattr(stack, "feature_columns", None) or []
        neg_coords_selected = [coords_all[i] for i in neg_idx.tolist()]
        try:
            visualize_debug_features(
                stack,
                feature_list,
                pos,
                neg_coords_selected,
                Path(args.out),
            )
        except Exception as exc:
            print(f"[warn] Failed to generate debug visualisations: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'splits.json'
    with out_path.open('w') as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(splits['pos'])} positives and {len(splits['neg'])} negatives to {out_path}")
    metrics_summary["final_pos"] = len(splits['pos'])
    metrics_summary["final_neg"] = len(splits['neg'])

    if args.validation and debug_info_by_region:
        diag_root = Path(args.out) / 'validation'
        for region, info in debug_info_by_region.items():
            region_embeddings = info.get('region_embeddings')
            if region_embeddings is None or region_embeddings.size == 0:
                continue
            projection = _project_embeddings(region_embeddings)
            distances = np.asarray(info.get('distances'))
            keep_mask = np.asarray(info.get('keep_mask'), dtype=bool)
            cutoff = info.get('cutoff')
            pos_local = np.asarray(info.get('pos_local'), dtype=int)
            unk_local = np.asarray(info.get('unk_local'), dtype=int)
            neg_local = np.asarray(info.get('neg_local'), dtype=int)
            diag_dir = diag_root / str(region)
            _save_validation_plot(
                diag_dir,
                projection,
                pos_local,
                unk_local,
                neg_local,
                distances,
                keep_mask,
                cutoff,
                args.filter_top_pct,
            )

    summary_path = Path(args.out) / "training_args_2_buildsplits.json"
    summary_payload = {
        "arguments": args_record,
        "metrics": _to_serializable(metrics_summary),
    }
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)
        print(f"[info] Saved build-splits configuration to {summary_path}")
    except Exception as exc:
        print(f"[warn] Failed to write build-splits summary to {summary_path}: {exc}")
