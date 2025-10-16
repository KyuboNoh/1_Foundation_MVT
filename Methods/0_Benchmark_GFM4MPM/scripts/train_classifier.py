# scripts/train_classifier.py
import argparse
import json
import os
import sys
import csv
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # points to Methods/0_Benchmark_GFM4MPM
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Common.data_utils import clamp_coords_to_window, load_split_stack, normalize_region_coord
from src.gfm4mpm.models.mae_vit import MAEViT
from src.gfm4mpm.models.mlp_dropout import MLPDropout
from src.gfm4mpm.training.train_cls import train_classifier
from src.gfm4mpm.infer.infer_maps import mc_predict_map, save_geotiff


def _resolve_search_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved


def _load_training_args(stack_root: Optional[Path], encoder_path: Path) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates: List[Path] = []
    root = _resolve_search_root(stack_root)
    if root is not None:
        search_dirs = [root] + list(root.parents)
        for base in search_dirs:
            metadata_candidates = [base / 'training_metadata.json', base / 'assetization' / 'training_metadata.json']
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
        candidates.extend([direct_pretrain, direct_standard])
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
            for candidate in [base / 'training_metadata.json', base / 'assetization' / 'training_metadata.json']:
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
class LabeledPatches(Dataset):
    def __init__(self, stack, coords, labels, window=32):
        self.stack, self.coords, self.labels, self.window = stack, coords, labels, window

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        x = _read_stack_patch(self.stack, coord, self.window)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', help='glob to raster bands (e.g. /data/*.tif)')
    ap.add_argument('--stac-root', help='STAC collection root (table workflow)')
    ap.add_argument('--stac-table', help='Direct path to STAC Parquet table asset')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
    # ap.add_argument('--stride', type=int, default=2, help='Stride for sliding window inference (in pixels)')
    ap.add_argument('--passes', type=int, default=2, help='Number of stochastic forward passes for MC-Dropout')

    args = ap.parse_args()

    modes = [bool(args.bands), bool(args.stac_root), bool(args.stac_table)]
    if sum(modes) != 1:
        ap.error('Provide either --bands or one of --stac-root/--stac-table (exactly one input source)')

    encoder_path = Path(args.encoder).resolve()
    label_column = 'Training_MVT_Deposit'

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
    training_args_data = load_result.training_args

    # TODO: What is special about patch size for table mode?
    if load_result.mode == 'table':
        default_patch = 1
    else:
        default_patch = args.patch

    patch_size = _resolve_pretraining_patch(training_args_data, default_patch)
    if training_args_data and training_args_data.get('window'):
        window_size = int(training_args_data['window'])
    else:
        window_size = default_patch if load_result.mode == 'table' else args.patch

    if load_result.mode == 'table' and window_size != 1:
        window_size = 1

    state_dict = torch.load(encoder_path, map_location='cpu')
    patch_proj = state_dict.get('patch.proj.weight')
    if isinstance(patch_proj, torch.Tensor):
        expected_channels = patch_proj.shape[1]
        if stack.count != expected_channels:
            raise RuntimeError(
                f"MAE encoder expects {expected_channels} input channel(s) but raster stack provides {stack.count}; "
                "verify feature selection and metadata alignment."
                )

    mae_kwargs: Dict[str, Any] = {}
    if training_args_data:
        mae_kwargs = _mae_kwargs_from_training_args(training_args_data)
    mae_kwargs.setdefault('patch_size', patch_size)
    mae_kwargs.setdefault('image_size', window_size)

    if hasattr(stack, "resolve_region_stack"):
        default_region = getattr(stack, "default_region", None)
        if default_region is None:
            regions = getattr(stack, "regions", [])
            default_region = regions[0] if regions else "GLOBAL"
    else:
        default_region = "GLOBAL"

    with open(args.splits) as f:
        sp = json.load(f)
    pos_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['pos']]
    neg_coords = [normalize_region_coord(entry, default_region=default_region) for entry in sp['neg']]
    coords = pos_coords + neg_coords
    labels = [1]*len(pos_coords) + [0]*len(neg_coords)

    batch_size = 512
    if training_args_data:
        for batch_key in ('batch',):
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

    print(f"[info] Training classifier with {len(coords)} samples ({len(pos_coords)} positive, {len(neg_coords)} negative)")
    coords_clamped, clamp_count = clamp_coords_to_window(coords, stack, window_size)
    if clamp_count:
        print(f"[info] Adjusted {clamp_count} split coordinate(s) to stay within window bounds")
    coords = [normalize_region_coord(coord, default_region=default_region) for coord in coords_clamped]

    Xtr, Xval, ytr, yval = train_test_split(coords, labels, test_size=0.2, stratify=labels, random_state=42)

    ds_tr = LabeledPatches(stack, Xtr, ytr, window=window_size)
    ds_va = LabeledPatches(stack, Xval, yval, window=window_size)
    worker_count = 0 if getattr(stack, 'kind', None) == 'raster' else 8
    if worker_count == 0:
        print('[info] Using single-process data loading for raster stack')
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=worker_count)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=worker_count)

    encoder = MAEViT(in_chans=stack.count, **mae_kwargs)
    encoder.load_state_dict(state_dict)
    in_dim = encoder.blocks[0].attn.embed_dim
    mlp = MLPDropout(in_dim=in_dim)

    mlp = train_classifier(encoder, mlp, dl_tr, dl_va, epochs=args.epochs)
    out_dir = Path(args.out) if args.out else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'mlp_classifier.pth'
    torch.save(mlp.state_dict(), out_path)
    print(f"[info] Saved classifier to {out_path}")

    print("[info] Generating prediction maps with MC-Dropout")
    # TODO: Allow stride override?
    prediction = mc_predict_map(
        encoder,
        mlp,
        stack,
        patch_size=window_size,
        stride=2 * patch_size,
        passes=args.passes,
        show_progress=True,
    )

    if isinstance(prediction, dict):
        for region_name, outputs in prediction.items():
            mean_map = outputs.get("mean")
            std_map = outputs.get("std")
            region_stack = outputs.get("stack", stack)
            if mean_map is None or std_map is None:
                continue
            ref_src = region_stack.srcs[0] if getattr(region_stack, "srcs", None) else stack.srcs[0]
            mean_path = f"{args.out}_{region_name}_mean.tif"
            std_path = f"{args.out}_{region_name}_std.tif"
            save_geotiff(mean_path, mean_map, ref_src)
            save_geotiff(std_path, std_map, ref_src)
            print(f"Wrote {mean_path} and {std_path}")
    else:
        mean_map, std_map = prediction

        if getattr(stack, 'is_table', False):
            mean_vec = mean_map.reshape(-1)
            std_vec = std_map.reshape(-1)
            rows = stack.iter_metadata()
            for idx, meta in enumerate(rows):
                meta['prospectivity_mean'] = float(mean_vec[idx])
                meta['prospectivity_std'] = float(std_vec[idx])
            out_csv = Path(args.out).with_suffix('.csv')
            if not out_csv.parent.exists():
                out_csv.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote table predictions to {out_csv}")
        else:
            save_geotiff(args.out + '_mean.tif', mean_map, stack.srcs[0])
            save_geotiff(args.out + '_std.tif', std_map, stack.srcs[0])
            print(f"Wrote {args.out}_mean.tif and {args.out}_std.tif")
