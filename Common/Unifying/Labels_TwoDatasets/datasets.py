from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from Common.cls.training.train_cls import dataloader_metric_inputORembedding
from Common.data_utils import EmbeddingRecord

DatasetConfig = Any

if TYPE_CHECKING:
    from Common.Unifying.Labels_TwoDatasets.fusion_utils.workspace import OverlapAlignmentWorkspace

def auto_coord_error(workspace: "OverlapAlignmentWorkspace", anchor_name: str, target_name: str) -> Optional[float]:
    meta = getattr(workspace, "integration_dataset_meta", {}) or {}
    anchor_meta = meta.get(anchor_name, {})
    target_meta = meta.get(target_name, {})

    def _numbers(dictionary: Dict[str, object], *keys: str) -> List[float]:
        values: List[float] = []
        for key in keys:
            value = dictionary.get(key)
            if isinstance(value, (int, float)) and value > 0:
                values.append(float(value))
        return values

    anchor_vals = _numbers(anchor_meta, "pixel_resolution", "min_resolution", "window_spacing")
    target_vals = _numbers(target_meta, "pixel_resolution", "min_resolution", "window_spacing")
    candidates = anchor_vals + target_vals
    if not candidates:
        return None

    coarse = max(candidates)
    fine = min(candidates)
    # Heuristic: allow deviations up to half of the coarser tile, but never smaller than the finer tile spacing.
    return max(coarse * 0.5, fine)

def _load_json(path: Path) -> Optional[Dict]:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse metadata JSON {path}: {exc}") from exc


def _build_region_lookup(doc: Optional[Dict]) -> Dict[str, str]:
    if not isinstance(doc, dict):
        return {}
    lookup: Dict[str, str] = {}
    # Primary structure used by integrate_stac.py
    regions = doc.get("regions")
    if isinstance(regions, dict):
        for region_name, region_payload in regions.items():
            if not isinstance(region_payload, dict):
                continue
            feature_paths = region_payload.get("feature_paths")
            if isinstance(feature_paths, dict):
                for _, paths in feature_paths.items():
                    if not isinstance(paths, list):
                        continue
                    for path in paths:
                        try:
                            tile_id = Path(path).stem
                        except Exception:
                            continue
                        lookup[tile_id] = str(region_name)
    # Some metadata bundles expose `tiles` with region annotations
    tiles = doc.get("tiles")
    if isinstance(tiles, list):
        for entry in tiles:
            if not isinstance(entry, dict):
                continue
            tile_id = entry.get("tile_id") or entry.get("id") or entry.get("name")
            region = entry.get("region")
            if tile_id is None or region is None:
                continue
            lookup[str(tile_id)] = str(region)
    return lookup


def _extract_window(metadata_entry: Optional[Dict], window_array: Optional[np.ndarray], index: int) -> Optional[Tuple[int, int]]:
    """Attempt to derive the sampling window (rows, cols) for the embedding."""

    if metadata_entry and isinstance(metadata_entry, dict):
        # Common keys written by stacify scripts
        for key in ("window_size", "window_shape", "window"):
            value = metadata_entry.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    return int(value[0]), int(value[1])
                except Exception:
                    continue
        if "height" in metadata_entry and "width" in metadata_entry:
            try:
                return int(metadata_entry["height"]), int(metadata_entry["width"])
            except Exception:
                pass

    if window_array is None:
        return None

    try:
        raw = window_array[index]
    except Exception:
        return None

    # window arrays sometimes stored as e.g. [rows, cols] or [[row0, col0], ...]
    raw_arr = np.asarray(raw)
    if raw_arr.size < 2:
        return None
    if raw_arr.ndim == 1:
        row_val, col_val = raw_arr[:2]
    else:
        row_val, col_val = raw_arr[0], raw_arr[1]
    try:
        return int(row_val), int(col_val)
    except Exception:
        return None


def load_embedding_records(config: DatasetConfig, dataset_meta: Optional[Dict[str, object]] = None) -> List[EmbeddingRecord]:
    """
    Load embedding records for a dataset, mirroring the behaviour from v0 while
    exposing additional fields (region/pixel resolution/window).
    """

    npz_path = config.embedding_path
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported embedding bundle {npz_path}. Expected .npz.")
    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding bundle {npz_path} not found.")

    with np.load(npz_path, allow_pickle=True) as bundle:
        try:
            embeddings = bundle["embeddings"]
            labels = bundle["labels"].astype(int)
        except KeyError as exc:
            raise KeyError(f"Missing required array {exc.args[0]!r} in {npz_path}.") from exc

        tile_ids = bundle.get("tile_ids")
        if tile_ids is None:
            tile_ids = np.asarray([f"{config.name}_tile_{idx}" for idx in range(len(embeddings))], dtype=str)
        else:
            tile_ids = np.asarray(tile_ids).astype(str)

        coords = bundle.get("coords")
        coord_array = None
        if coords is not None:
            coord_array = np.asarray(coords, dtype=np.float64)
            if coord_array.ndim != 2 or coord_array.shape[1] < 2:
                coord_array = None

        metadata_array = bundle.get("metadata")
        if metadata_array is not None:
            metadata_array = np.asarray(metadata_array, dtype=object)

        window_array = None
        for key in ("windows", "window_size", "window_shape"):
            if key in bundle:
                window_array = np.asarray(bundle[key], dtype=object)
                break

        pixel_res_array = None
        for key in ("pixel_resolution", "resolution", "pixel_resolutions"):
            if key in bundle:
                pixel_res_array = np.asarray(bundle[key], dtype=float)
                break

        metadata_doc = _load_json(config.metadata_path) if config.metadata_path else None
        region_lookup = _build_region_lookup(metadata_doc)

        records: List[EmbeddingRecord] = []
        upper_region_filter = {str(region).upper() for region in config.region_filter} if config.region_filter else None

        default_pixel_resolution = None
        if dataset_meta:
            value = dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
            try:
                default_pixel_resolution = float(value) if value is not None else None
            except Exception:
                default_pixel_resolution = None

        for idx, embedding in enumerate(embeddings):
            label_raw = int(labels[idx])
            label = 1 if label_raw == config.positive_label else 0
            tile_id = str(tile_ids[idx])

            coord_tuple: Optional[Tuple[float, float]] = None
            if coord_array is not None:
                row = coord_array[idx]
                if row.size >= 2 and np.all(np.isfinite(row[:2])):
                    coord_tuple = (float(row[0]), float(row[1]))

            meta_entry = None
            if metadata_array is not None:
                meta_entry = metadata_array[idx]
                if isinstance(meta_entry, np.ndarray) and meta_entry.dtype == object and meta_entry.ndim == 0:
                    meta_entry = meta_entry.item()
                elif hasattr(meta_entry, "item") and not isinstance(meta_entry, dict):
                    try:
                        meta_entry = meta_entry.item()
                    except Exception:
                        pass
                if not isinstance(meta_entry, dict):
                    meta_entry = None

            row_val = None
            col_val = None
            if isinstance(meta_entry, dict):
                row_val = meta_entry.get("row")
                col_val = meta_entry.get("col")
            try:
                row_int = int(row_val) if row_val is not None else None
            except Exception:
                row_int = None
            try:
                col_int = int(col_val) if col_val is not None else None
            except Exception:
                col_int = None
            # Fallback: parse from tile ID (often formatted REGION_row_col)
            region_from_id = None
            if tile_id:
                parts = tile_id.split("_")
                if len(parts) >= 2:
                    region_from_id = "_".join(parts[:-2]) if len(parts) >= 3 else None
                    try:
                        col_candidate = int(parts[-1])
                        row_candidate = int(parts[-2])
                        row_int = row_int if row_int is not None else row_candidate
                        col_int = col_int if col_int is not None else col_candidate
                    except Exception:
                        pass
            row_col = (row_int, col_int) if row_int is not None and col_int is not None else None

            if isinstance(meta_entry, dict):
                region_candidate = meta_entry.get("region") or meta_entry.get("Region")
            else:
                region_candidate = None
            if region_candidate is None and region_from_id is not None:
                region_candidate = region_from_id
            if region_candidate is None:
                region_candidate = region_lookup.get(tile_id)
            region = str(region_candidate or "GLOBAL")

            if upper_region_filter is not None:
                if region.upper() not in upper_region_filter:
                    continue

            pixel_resolution = None
            if pixel_res_array is not None and pixel_res_array.size > idx:
                value = pixel_res_array[idx]
                if np.isfinite(value):
                    pixel_resolution = float(value)
            elif isinstance(meta_entry, dict):
                value = meta_entry.get("pixel_resolution") or meta_entry.get("resolution")
                try:
                    pixel_resolution = float(value) if value is not None else None
                except Exception:
                    pixel_resolution = None
            if pixel_resolution is None and default_pixel_resolution is not None:
                pixel_resolution = default_pixel_resolution

            window_size = _extract_window(meta_entry, window_array, idx)

            record = EmbeddingRecord(
                index=idx,
                embedding=np.asarray(embedding),
                label=label,
                tile_id=tile_id,
                coord=coord_tuple,
                row_col=row_col,
                region=region,
                pixel_resolution=pixel_resolution,
                window_size=window_size,
                metadata=meta_entry,
            )
            records.append(record)

    if not records:
        raise ValueError(f"No embedding records were loaded for dataset {config.name}.")
    return records


def stack_embeddings(records: Sequence[EmbeddingRecord]) -> torch.Tensor:
    """Stack record embeddings into a torch tensor."""

    if not records:
        raise ValueError("stack_embeddings expects at least one record.")
    if torch is None:
        raise ImportError("stack_embeddings requires PyTorch; install torch before invoking this helper.")
    tensors = [torch.from_numpy(record.embedding).float() for record in records]
    return torch.stack(tensors)


def summarise_records(records: Sequence[EmbeddingRecord]) -> Dict[str, object]:
    """Produce lightweight statistics for logging/debugging."""

    total = len(records)
    label_counts = {0: 0, 1: 0}
    regions: Dict[str, int] = {}
    resolutions: List[float] = []
    window_sizes: List[Tuple[int, int]] = []

    for record in records:
        label_counts[record.label] = label_counts.get(record.label, 0) + 1
        regions[record.region] = regions.get(record.region, 0) + 1
        if record.pixel_resolution is not None:
            resolutions.append(record.pixel_resolution)
        if record.window_size is not None:
            window_sizes.append(record.window_size)

    def _summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    window_summary = None
    if window_sizes:
        rows = [float(size[0]) for size in window_sizes]
        cols = [float(size[1]) for size in window_sizes]
        window_summary = {
            "rows": _summary(rows),
            "cols": _summary(cols),
        }

    return {
        "count": total,
        "labels": label_counts,
        "regions": regions,
        "pixel_resolution": _summary(resolutions),
        "window_size": window_summary,
    }


def _load_pn_lookup(path: Optional[Path]) -> Optional[Dict[str, set[Tuple[str, int, int]]]]:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    pos_entries = payload.get("pos") or []
    neg_entries = payload.get("neg") or []

    def _to_key(entry: Dict[str, object]) -> Optional[Tuple[str, int, int]]:
        region = entry.get("region")
        row = entry.get("row")
        col = entry.get("col")
        if region is None or row is None or col is None:
            return None
        try:
            return (str(region).upper(), int(row), int(col))
        except Exception:
            return None

    pos_set: set[Tuple[str, int, int]] = set()
    neg_set: set[Tuple[str, int, int]] = set()
    for collection, target in ((pos_entries, pos_set), (neg_entries, neg_set)):
        if not isinstance(collection, list):
            continue
        for entry in collection:
            if isinstance(entry, dict):
                key = _to_key(entry)
                if key is not None:
                    target.add(key)
    return {"pos": pos_set, "neg": neg_set}

def _count_pn_lookup(
    pn_lookup: Optional[Dict[str, set[Tuple[str, int, int]]]],
    region_filter: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    if not pn_lookup:
        return {"positive": 0, "negative": 0}

    def _normalise_region(value: Optional[object]) -> str:
        if value is None:
            return "NONE"
        return str(value).upper()

    allowed_regions: Optional[set[str]] = None
    if region_filter:
        allowed_regions = {str(region).upper() for region in region_filter if region is not None}

    def _count(entries: Iterable[Tuple[str, int, int]]) -> int:
        if entries is None:
            return 0
        if not allowed_regions:
            return sum(1 for _ in entries)

        total = 0
        for region_value, _, _ in entries:
            region_key = _normalise_region(region_value)
            if region_key in allowed_regions:
                total += 1
            elif region_key == "NONE" and ("GLOBAL" in allowed_regions or "ALL" in allowed_regions):
                total += 1
        return total

    pos = _count(pn_lookup.get("pos", ()))
    neg = _count(pn_lookup.get("neg", ()))
    return {"positive": int(pos), "negative": int(neg)}

def _compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    interior = mask.copy()
    interior[:-1, :] &= mask[1:, :]
    interior[1:, :] &= mask[:-1, :]
    interior[:, :-1] &= mask[:, 1:]
    interior[:, 1:] &= mask[:, :-1]
    boundary &= ~interior
    return boundary


class _MaskStack:
    def __init__(self, mask_info: Dict[str, object]):
        shape = mask_info.get("shape")
        self.height = int(shape[0]) if shape is not None else 0
        self.width = int(shape[1]) if shape is not None else 0
        self.transform = mask_info.get("transform")
        self.crs = None

    def grid_centers(self, stride: int) -> List[Tuple[int, int]]:
        centers: List[Tuple[int, int]] = []
        for r in range(0, self.height, stride):
            for c in range(0, self.width, stride):
                centers.append((r, c))
        return centers


def _make_mask_reference(mask_info: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if mask_info is None:
        return None
    mask_array = mask_info.get("array")
    mask_transform = mask_info.get("transform")
    mask_shape = mask_info.get("shape")
    mask_nodata = mask_info.get("nodata")
    if mask_array is None or mask_transform is None or mask_shape is None:
        return None
    mask_arr = np.asarray(mask_array)
    valid_mask = np.isfinite(mask_arr)
    if mask_nodata is not None and np.isfinite(mask_nodata):
        valid_mask &= ~np.isclose(mask_arr, mask_nodata)
    valid_mask &= mask_arr != 0
    height = int(mask_shape[0])
    width = int(mask_shape[1])
    if valid_mask.shape != (height, width):
        valid_mask = valid_mask.reshape((height, width))
    boundary_mask = _compute_boundary_mask(valid_mask) if valid_mask.any() else None
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "transform": mask_transform,
    }
    return {
        "profile": profile,
        "transform": mask_transform,
        "crs": None,
        "valid_mask": valid_mask if valid_mask.any() else None,
        "boundary_mask": boundary_mask,
    }


def _build_dataloaders(
    dataset: Dict[str, object],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Any,
) -> Tuple[Any, Any, Dict[str, object]]:
    features = dataset["features"]
    labels = dataset["labels"]
    ytr = labels[train_idx]
    yval = labels[val_idx] if val_idx.size else np.empty(0, dtype=int)
    dl_tr, dl_val, metrics_summary = dataloader_metric_inputORembedding(
        train_idx,
        val_idx,
        ytr,
        yval,
        cfg.cls_training.batch_size,
        positive_augmentation=False,
        embedding=features,
        epochs=cfg.cls_training.epochs,
    )
    return dl_tr, dl_val, metrics_summary
