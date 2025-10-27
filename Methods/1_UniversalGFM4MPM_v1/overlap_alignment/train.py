from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = TensorDataset = None  # type: ignore[assignment]

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:  # optional plotting for debug mode
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from .config import AlignmentConfig, load_config
from .workspace import OverlapAlignmentPair, OverlapAlignmentWorkspace
from .datasets import auto_coord_error

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 overlap alignment trainer (positive-only)")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    parser.add_argument("--use-positive-only", action="store_true", help="Restrict training pairs to positive tiles.")
    parser.add_argument("--use-positive-augmentation", action="store_true", help="Enable positive augmentation vectors if provided.")
    parser.add_argument("--objective", choices=["dcca", "barlow"], default=None, help="Alignment objective to optimise (default: dcca).")
    parser.add_argument("--projection-dim", type=int, default=None, help="Override projection head output dimension.")

    parser.add_argument("--aggregator", choices=["weighted_pool"], default=None, help="Aggregation strategy for fine-grained tiles (default: weighted_pool).")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics and save overlap figures.")
    return parser.parse_args()


def _load_augmented_embeddings(path: Path, region_filter: Optional[Sequence[str]]) -> Tuple[Dict[str, List[np.ndarray]], int]:
    mapping: Dict[str, List[np.ndarray]] = {}
    count = 0
    region_filter_set: Optional[set[str]] = None
    if region_filter:
        region_filter_set = {str(entry).upper() for entry in region_filter}
    try:
        with np.load(path, allow_pickle=True) as bundle:
            embeddings = bundle.get("embeddings")
            metadata = bundle.get("metadata")
            flags = bundle.get("is_augmented")
            tile_ids = bundle.get("tile_ids")
            if embeddings is None or metadata is None or flags is None or tile_ids is None:
                print(f"[warn] positive augmentation bundle {path} is missing required arrays; ignoring.")
                return {}, 0
            for emb, meta, flag, tile_id in zip(embeddings, metadata, flags, tile_ids):
                if not bool(flag):
                    continue
                meta_dict: Dict[str, object] = {}
                if isinstance(meta, dict):
                    meta_dict = meta
                else:
                    try:
                        meta_dict = meta.item()
                    except Exception:
                        meta_dict = {}
                region_value = str(meta_dict.get("region") or meta_dict.get("Region") or "").upper()
                if region_filter_set and region_value and region_value not in region_filter_set:
                    continue
                source_tile_id = meta_dict.get("source_tile_id")
                if source_tile_id is None and isinstance(tile_id, str):
                    source_tile_id = tile_id.split("__")[0]
                if source_tile_id is None:
                    continue
                emb_arr = np.asarray(emb, dtype=np.float32)
                mapping.setdefault(str(source_tile_id), []).append(emb_arr)
                count += 1
    except FileNotFoundError:
        print(f"[warn] positive augmentation file {path} not found.")
        return {}, 0
    except Exception as exc:
        print(f"[warn] Failed to load positive augmentation file {path}: {exc}")
        return {}, 0
    return mapping, count


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if torch is None or nn is None or DataLoader is None:
        raise ImportError("Overlap alignment training requires PyTorch; install torch before running the trainer.")

    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.training.epochs = max(1, int(args.epochs))
    if args.batch_size is not None:
        cfg.training.batch_size = max(1, int(args.batch_size))
    if args.lr is not None:
        cfg.training.lr = float(args.lr)
    if args.projection_dim is not None:
        cfg.projection_dim = max(1, int(args.projection_dim))
    if args.objective is not None:
        cfg.alignment_objective = args.objective.lower()
    if args.aggregator is not None:
        cfg.aggregator = args.aggregator.lower()
    if args.use_positive_only:
        cfg.use_positive_only = True
    if args.use_positive_augmentation:
        cfg.use_positive_augmentation = True
    debug_mode = bool(args.debug)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    workspace = OverlapAlignmentWorkspace(cfg)
    anchor_name, target_name = _dataset_pair(cfg)
    max_coord_error = auto_coord_error(workspace, anchor_name, target_name)

    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    if debug_mode:
        overlap_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
        positive_tiles: Dict[str, set] = {anchor_name: set(), target_name: set()}
        overlapping_positive_pairs = 0
        for pair in pairs:
            for dataset_name, record in (
                (pair.anchor_dataset, pair.anchor_record),
                (pair.target_dataset, pair.target_record),
            ):
                marker = record.tile_id if record.tile_id is not None else record.index
                overlap_tiles.setdefault(dataset_name, set()).add(marker)
                if int(record.label) == 1:
                    positive_tiles.setdefault(dataset_name, set()).add(marker)
            if int(pair.anchor_record.label) == 1 and int(pair.target_record.label) == 1:
                overlapping_positive_pairs += 1
        print(
            "[debug] overlap positives: "
            f"{anchor_name}={len(positive_tiles.get(anchor_name, set()))} / {len(overlap_tiles.get(anchor_name, set()))}, "
            f"{target_name}={len(positive_tiles.get(target_name, set()))} / {len(overlap_tiles.get(target_name, set()))}, "
            f"positive-overlap-pairs={overlapping_positive_pairs}"
        )
    if not pairs:
        raise RuntimeError("No overlap pairs were resolved; cannot start training.")

    augmentation_by_dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
    if cfg.use_positive_augmentation:
        for dataset_cfg in cfg.datasets:
            if dataset_cfg.pos_aug_path is None:
                continue
            aug_map, aug_count = _load_augmented_embeddings(dataset_cfg.pos_aug_path, dataset_cfg.region_filter)
            if aug_map:
                augmentation_by_dataset[dataset_cfg.name] = aug_map
                print(
                    f"[info] Loaded {aug_count} augmented embeddings for dataset {dataset_cfg.name} "
                    f"from {dataset_cfg.pos_aug_path}"
                )
            else:
                print(
                    f"[warn] No augmented embeddings retained for dataset {dataset_cfg.name}; "
                    "region filter or bundle contents may have excluded all entries."
                )

    dataset_meta_map = getattr(workspace, "integration_dataset_meta", {}) or {}
    anchor_vecs, target_vecs, label_hist, debug_data = _build_positive_aligned_pairs(
        pairs,
        anchor_name=anchor_name,
        target_name=target_name,
        use_positive_only=cfg.use_positive_only,
        aggregator=cfg.aggregator,
        gaussian_sigma=cfg.gaussian_sigma,
        debug=debug_mode,
        dataset_meta_map=dataset_meta_map,
        anchor_augment_map=augmentation_by_dataset.get(anchor_name),
        target_augment_map=augmentation_by_dataset.get(target_name),
    )

    if not anchor_vecs:
        print("[warn] No qualifying overlap groups were found for training.")
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
        return
    if cfg.alignment_objective.lower() == "dcca" and len(anchor_vecs) < 2:
        print(
            "[warn] Not enough positive overlap pairs to optimise DCCA "
            f"(need at least 2, found {len(anchor_vecs)}). Aborting training."
        )
        if debug_mode:
            _maybe_save_debug_figures(cfg, debug_data)
            _persist_metrics(cfg, {"objective": cfg.alignment_objective.lower(), "max_coord_error": max_coord_error}, [], debug_data)
        return

    anchor_tensor = torch.stack(anchor_vecs)
    target_tensor = torch.stack(target_vecs)
    dataset = TensorDataset(anchor_tensor, target_tensor)
    batch_size = min(cfg.training.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > 1)

    in_dim_anchor = anchor_tensor.size(1)
    in_dim_target = target_tensor.size(1)
    proj_out = cfg.projection_dim
    projector_a = ProjectionHead(in_dim_anchor, proj_out).to(device)
    projector_b = ProjectionHead(in_dim_target, proj_out).to(device)

    optimizer = torch.optim.Adam(
        list(projector_a.parameters()) + list(projector_b.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    objective = cfg.alignment_objective.lower()
    if objective not in {"dcca", "barlow"}:
        raise ValueError(f"Unsupported alignment objective: {objective}")

    epoch_history: List[Dict[str, float]] = []

    for epoch in range(cfg.training.epochs):
        running_loss = 0.0
        running_corr = 0.0
        batches = 0
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.training.epochs}", leave=False)
        for anchor_batch, target_batch in iterator:
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            if objective == "dcca" and anchor_batch.size(0) < 2:
                if tqdm is not None:
                    iterator.set_postfix({"skip": "batch<2"})
                continue
            optimizer.zero_grad()
            z_a = projector_a(anchor_batch)
            z_b = projector_b(target_batch)
            if objective == "dcca":
                loss, singular_vals = dcca_loss(z_a, z_b)
                mean_corr = float(singular_vals.mean().item()) if singular_vals.numel() else 0.0
            else:
                raise NotImplementedError("Barlow Twins objective is not implemented yet.")
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if objective == "dcca":
                running_corr += mean_corr
            batches += 1
            if tqdm is not None:
                iterator.set_postfix({"loss": loss.item(), "corr": mean_corr})
        avg_loss = running_loss / max(batches, 1)
        avg_corr = running_corr / max(batches, 1) if objective == "dcca" else float("nan")
        if batches == 0:
            print(f"[warn] No batches processed in epoch {epoch+1}; stopping early.")
            break
        epoch_history.append({"epoch": epoch + 1, "loss": avg_loss, "mean_correlation": avg_corr})
        print(f"[epoch {epoch+1}/{cfg.training.epochs}] loss={avg_loss:.6f} corr={avg_corr:.6f} batches={batches}")
        if objective == "dcca" and abs(avg_loss) < 1e-12 and abs(avg_corr) < 1e-12:
            print(f"[info] DCCA updates collapsed (loss≈0, corr≈0) at epoch {epoch+1}; stopping early.")
            break

    debug_anchor_count = len(debug_data.get("anchor_positive", [])) if debug_mode and debug_data else 0
    debug_target_count = len(debug_data.get("target_positive", [])) if debug_mode and debug_data else 0
    debug_pair_count = len(debug_data.get("selected_pairs", [])) if debug_mode and debug_data else 0

    last_history = epoch_history[-1] if epoch_history else {"loss": None, "mean_correlation": None}

    summary = {
        "objective": objective,
        "aggregator": cfg.aggregator,
        "use_positive_only": cfg.use_positive_only,
        "projection_dim": proj_out,
        "label_histogram": dict(label_hist),
        "num_pairs": len(anchor_vecs),
        "epochs": cfg.training.epochs,
        "batch_size": batch_size,
        "lr": cfg.training.lr,
        "max_coord_error": max_coord_error,
        "final_loss": last_history.get("loss"),
        "final_mean_correlation": last_history.get("mean_correlation"),
        "debug_positive_anchor": debug_anchor_count,
        "debug_positive_target": debug_target_count,
        "debug_positive_pair_links": debug_pair_count,
    }
    print("[info] training summary:", summary)

    _persist_state(cfg, projector_a, projector_b, summary)
    _persist_metrics(cfg, summary, epoch_history, debug_data if debug_mode else None)
    if debug_mode:
        _maybe_save_debug_figures(cfg, debug_data)


def _dataset_pair(cfg: AlignmentConfig) -> Tuple[str, str]:
    if len(cfg.datasets) < 2:
        raise ValueError("Overlap alignment requires at least two datasets.")
    return cfg.datasets[0].name, cfg.datasets[1].name

def _build_positive_aligned_pairs(
    pairs: Sequence[OverlapAlignmentPair],
    *,
    anchor_name: str,
    target_name: str,
    use_positive_only: bool,
    aggregator: str,
    gaussian_sigma: Optional[float],
    debug: bool,
    dataset_meta_map: Dict[str, Dict[str, object]],
    anchor_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
    target_augment_map: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Counter,
    Optional[Dict[str, List[Dict[str, object]]]],
]:
    anchor_augment_map = anchor_augment_map or {}
    target_augment_map = target_augment_map or {}
    grouped: Dict[str, Dict[str, object]] = {}
    anchor_aug_added = 0
    target_aug_added = 0
    label_hist: Counter = Counter()
    debug_data: Optional[Dict[str, object]] = None
    if debug:
        debug_data = {
            "anchor_positive": {},
            "target_positive": {},
            "selected_pairs": [],
            "anchor_name": anchor_name,
            "target_name": target_name,
        }

    for pair in pairs:
        anchor_record, target_record = pair.anchor_record, pair.target_record
        anchor_ds, target_ds = pair.anchor_dataset, pair.target_dataset

        if anchor_ds != anchor_name or target_ds != target_name:
            if pair.target_dataset == anchor_name and pair.anchor_dataset == target_name:
                anchor_record, target_record = pair.target_record, pair.anchor_record
                anchor_ds, target_ds = pair.target_dataset, pair.anchor_dataset
            else:
                continue

        anchor_label = int(anchor_record.label)
        target_label = int(target_record.label)

        if debug_data is not None:
            if anchor_label == 1:
                _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
            if target_label == 1:
                _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))

        if use_positive_only and anchor_label != 1:
            continue
        if use_positive_only and target_label != 1:
            continue

        label_key = _label_key(anchor_label, target_label, anchor_name, target_name)
        label_hist[label_key] += 1

        if debug_data is not None:
            if anchor_label == 1 and target_label == 1:
                anchor_sample = _add_debug_sample(debug_data["anchor_positive"], anchor_record, anchor_name, dataset_meta_map.get(anchor_name, {}))
                target_sample = _add_debug_sample(debug_data["target_positive"], target_record, target_name, dataset_meta_map.get(target_name, {}))
                if anchor_sample is not None and target_sample is not None:
                    debug_data["selected_pairs"].append({"anchor": anchor_sample, "target": target_sample})

        tile_id = anchor_record.tile_id
        grouped.setdefault(tile_id, {
            "anchor": anchor_record,
            "targets": [],
            "weights": [],
        })
        grouped[tile_id]["targets"].append(target_record)
        grouped[tile_id]["weights"].append(_pair_weight(anchor_record, target_record, gaussian_sigma))

    anchor_vecs: List[torch.Tensor] = []
    target_vecs: List[torch.Tensor] = []

    for entry in grouped.values():
        targets: List = entry["targets"]
        if not targets:
            continue
        aggregated, normalized_weights, target_stack = _aggregate_targets(entry['anchor'], targets, entry['weights'], aggregator)
        if aggregated is None:
            continue
        anchor_tensor = torch.from_numpy(entry['anchor'].embedding).float()
        anchor_vecs.append(anchor_tensor)
        target_vecs.append(aggregated.clone())
        anchor_aug_list = anchor_augment_map.get(entry['anchor'].tile_id, [])
        for aug_emb in anchor_aug_list:
            aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
            if aug_tensor.shape != anchor_tensor.shape:
                continue
            anchor_vecs.append(aug_tensor)
            target_vecs.append(aggregated.clone())
            label_hist[label_key] += 1
            anchor_aug_added += 1

        for idx_target, target_record in enumerate(targets):
            aug_list = target_augment_map.get(target_record.tile_id, [])
            if not aug_list:
                continue
            for aug_emb in aug_list:
                aug_tensor = torch.from_numpy(np.asarray(aug_emb, dtype=np.float32)).float()
                if aug_tensor.shape != target_stack[idx_target].shape:
                    continue
                modified_stack = target_stack.clone()
                modified_stack[idx_target] = aug_tensor
                aggregated_aug = torch.matmul(normalized_weights.unsqueeze(0), modified_stack).squeeze(0)
                anchor_vecs.append(anchor_tensor.clone())
                target_vecs.append(aggregated_aug)
                label_hist[label_key] += 1
                target_aug_added += 1


    if debug_data is not None:
        anchor_serialised = [_serialise_sample(sample) for sample in debug_data["anchor_positive"].values()]
        target_serialised = [_serialise_sample(sample) for sample in debug_data["target_positive"].values()]
        pair_serialised = [
            {
                "anchor": _serialise_sample(pair["anchor"]),
                "target": _serialise_sample(pair["target"]),
            }
            for pair in debug_data["selected_pairs"]
        ]
        debug_payload = {
            "anchor_positive": anchor_serialised,
            "target_positive": target_serialised,
            "selected_pairs": pair_serialised,
            "anchor_name": anchor_name,
            "target_name": target_name,
            'anchor_augmented_pairs': int(anchor_aug_added),
            'target_augmented_pairs': int(target_aug_added),
        }
    else:
        debug_payload = None

    return anchor_vecs, target_vecs, label_hist, debug_payload

def _add_debug_sample(
    store: Dict[str, Dict[str, object]],
    record,
    dataset_name: str,
    dataset_meta: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if record.coord is None or any(math.isnan(c) for c in record.coord[:2]):
        return None
    coord = (float(record.coord[0]), float(record.coord[1]))
    sample = store.get(record.tile_id)
    if sample is None:
        sample = {
            "tile_id": record.tile_id,
            "coord": coord,
            "dataset": dataset_name,
        }
        window = record.window_size
        pixel_res = record.pixel_resolution
        if window is None and dataset_meta:
            spacing = dataset_meta.get("window_spacing") or dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
            if spacing is not None:
                pixel_res = float(spacing)
                window = (1, 1)
        if window is not None:
            sample["window_size"] = [int(window[0]), int(window[1])]
        if pixel_res is None and dataset_meta:
            pixel_res = dataset_meta.get("pixel_resolution") or dataset_meta.get("min_resolution")
        if pixel_res is not None:
            sample["pixel_resolution"] = float(pixel_res)
        if window is not None and pixel_res is not None:
            width = window[1] * float(pixel_res)
            height = window[0] * float(pixel_res)
            sample["footprint"] = [width, height]
        store[record.tile_id] = sample
    return sample

def _serialise_sample(sample: Dict[str, object]) -> Dict[str, object]:
    coord = sample.get("coord")
    coord_list = [float(coord[0]), float(coord[1])] if coord is not None else None
    data: Dict[str, object] = {
        "tile_id": sample.get("tile_id"),
    }
    if coord_list is not None:
        data["coord"] = coord_list
    if sample.get("dataset") is not None:
        data["dataset"] = sample["dataset"]
    if "window_size" in sample and sample["window_size"] is not None:
        data["window_size"] = [int(sample["window_size"][0]), int(sample["window_size"][1])]
    if "pixel_resolution" in sample and sample["pixel_resolution"] is not None:
        data["pixel_resolution"] = float(sample["pixel_resolution"])
    if sample.get("footprint") is not None:
        data["footprint"] = [float(sample["footprint"][0]), float(sample["footprint"][1])]
    return data

def _label_key(anchor_label: int, target_label: int, anchor_name: str, target_name: str) -> str:
    if anchor_label == 1 and target_label == 1:
        return "positive_common"
    if anchor_label == 1 and target_label != 1:
        return f"positive_{anchor_name}"
    if anchor_label != 1 and target_label == 1:
        return f"positive_{target_name}"
    return "unlabelled"


def _pair_weight(anchor_record, target_record, gaussian_sigma: Optional[float]) -> float:
    if gaussian_sigma is None:
        return 1.0
    if anchor_record.coord is None or target_record.coord is None:
        return 1.0
    anchor_xy = np.asarray(anchor_record.coord, dtype=float)
    target_xy = np.asarray(target_record.coord, dtype=float)
    if anchor_xy.size < 2 or target_xy.size < 2:
        return 1.0
    dist = float(np.linalg.norm(anchor_xy[:2] - target_xy[:2]))
    if dist <= 0.0:
        return 1.0
    weight = math.exp(- (dist ** 2) / (2.0 * (gaussian_sigma ** 2)))
    return float(weight + 1e-8)


def _aggregate_targets(anchor_record, targets: Sequence, weights: Sequence[float], aggregator: str) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if aggregator != "weighted_pool":
        raise NotImplementedError(f"Aggregator '{aggregator}' is not implemented yet.")

    embeddings = [torch.from_numpy(target.embedding).float() for target in targets]
    if not embeddings:
        return None, torch.empty(0), torch.empty(0)
    stacked = torch.stack(embeddings)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.isnan(weight_tensor).any() or float(weight_tensor.sum()) <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    aggregated = torch.matmul(weight_tensor.unsqueeze(0), stacked).squeeze(0)
    return aggregated, weight_tensor, stacked


if torch is not None:

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden = max(out_dim, min(in_dim, 512))
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class ProjectionHead:  # type: ignore[too-many-ancestors]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ProjectionHead requires PyTorch. Install torch before training.")


def dcca_loss(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        dummy = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        return dummy, torch.zeros(0, device=z_a.device)

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    n = z_a.size(0)
    cov_aa = (z_a.T @ z_a) / (n - 1) + eps * torch.eye(z_a.size(1), device=z_a.device)
    cov_bb = (z_b.T @ z_b) / (n - 1) + eps * torch.eye(z_b.size(1), device=z_b.device)
    cov_ab = (z_a.T @ z_b) / (n - 1)

    inv_sqrt_aa = _matrix_inverse_sqrt(cov_aa, eps)
    inv_sqrt_bb = _matrix_inverse_sqrt(cov_bb, eps)
    t_mat = inv_sqrt_aa @ cov_ab @ inv_sqrt_bb
    try:
        singular_values = torch.linalg.svdvals(t_mat)
    except RuntimeError:
        zero = torch.tensor(0.0, device=z_a.device, requires_grad=True)
        return zero, torch.zeros(0, device=z_a.device)
    loss = -singular_values.sum()
    return loss, singular_values


def _matrix_inverse_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        eigvals, eigvecs = torch.linalg.eigh(mat)
    except RuntimeError:
        dim = mat.size(0)
        return torch.eye(dim, device=mat.device, dtype=mat.dtype)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return inv_sqrt


def _prepare_output_dir(primary: Optional[Path], fallback_relative: str) -> Optional[Path]:
    if primary is not None:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except Exception as exc:
            print(f"[warn] Unable to create directory {primary}: {exc}. Falling back to local path.")
    fallback = Path.cwd() / fallback_relative
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception as exc:
        print(f"[warn] Unable to create fallback directory {fallback}: {exc}")
        return None


def _persist_state(cfg: AlignmentConfig, proj_a: nn.Module, proj_b: nn.Module, summary: Dict[str, object]) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return

    state_path = target_dir / "overlap_alignment_stage1.pt"
    payload = {
        "projection_head_a": proj_a.state_dict(),
        "projection_head_b": proj_b.state_dict(),
        "summary": summary,
    }
    try:
        torch.save(payload, state_path)
        print(f"[info] saved stage-1 weights to {state_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist training state: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            alt_path = fallback_dir / state_path.name
            try:
                torch.save(payload, alt_path)
                print(f"[info] saved stage-1 weights to fallback location {alt_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist training state to fallback location: {exc_fallback}")


def _persist_metrics(
    cfg: AlignmentConfig,
    summary: Dict[str, object],
    epoch_history: Sequence[Dict[str, float]],
    debug_data: Optional[Dict[str, List[Dict[str, object]]]] = None,
) -> None:
    primary = None
    if cfg.output_dir is not None:
        primary = cfg.output_dir
    elif cfg.log_dir is not None:
        primary = cfg.log_dir
    target_dir = _prepare_output_dir(primary, "overlap_alignment_outputs")
    if target_dir is None:
        return
    metrics_path = target_dir / "overlap_alignment_stage1_metrics.json"
    payload = {
        "summary": summary,
        "epoch_history": list(epoch_history),
    }
    if debug_data:
        filtered_debug = {key: value for key, value in debug_data.items() if value}
        if filtered_debug:
            payload["debug"] = filtered_debug
    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] saved metrics to {metrics_path}")
    except Exception as exc:
        print(f"[warn] Unable to persist metrics: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_outputs")
        if fallback_dir is not None:
            fallback_path = fallback_dir / metrics_path.name
            try:
                with fallback_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                print(f"[info] saved metrics to fallback location {fallback_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to persist metrics to fallback location: {exc_fallback}")


def _maybe_save_debug_figures(cfg: AlignmentConfig, debug_data: Optional[Dict[str, List[Dict[str, object]]]]) -> None:
    if debug_data is None:
        print("[warn] Debug data unavailable; skipping figure generation.")
        return
    if plt is None:
        print("[warn] Matplotlib is not installed; cannot save debug figures.")
        return

    viz_primary = None
    if cfg.output_dir is not None:
        viz_primary = cfg.output_dir / "bridge_visualizations" / "overlap"
    elif cfg.log_dir is not None:
        viz_primary = cfg.log_dir / "bridge_visualizations" / "overlap"
    viz_dir = _prepare_output_dir(viz_primary, "overlap_alignment_debug/bridge_visualizations/overlap")
    if viz_dir is None:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw overlap boundary if available
    geometry = None
    try:
        with Path(cfg.overlap_pairs_path).open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        geometry = doc.get("overlap", {}).get("geometry")
    except Exception:
        geometry = None

    if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
        for ring in geometry.get("coordinates", []):
            ring_arr = np.asarray(ring, dtype=float)
            if ring_arr.ndim == 2 and ring_arr.size:
                ax.plot(ring_arr[:, 0], ring_arr[:, 1], color="black", linewidth=1.0, alpha=0.5)

    anchor_samples = debug_data.get("anchor_positive", []) or []
    target_samples = debug_data.get("target_positive", []) or []
    selected_pairs = debug_data.get("selected_pairs", []) or []
    anchor_label = (
        anchor_samples[0].get("dataset")
        if anchor_samples and anchor_samples[0].get("dataset")
        else debug_data.get("anchor_name", "anchor")
    )
    target_label = (
        target_samples[0].get("dataset")
        if target_samples and target_samples[0].get("dataset")
        else debug_data.get("target_name", "target")
    )

    if anchor_samples:
        anchor_coords = np.asarray([sample["coord"] for sample in anchor_samples if sample.get("coord")], dtype=float)
        if anchor_coords.size:
            ax.scatter(
                anchor_coords[:, 0],
                anchor_coords[:, 1],
                s=40,
                facecolors="none",
                edgecolors="#1f77b4",
                linewidths=1.4,
                label=f"positive_{anchor_label}",
            )
    if target_samples:
        target_coords = np.asarray([sample["coord"] for sample in target_samples if sample.get("coord")], dtype=float)
        if target_coords.size:
            ax.scatter(
                target_coords[:, 0],
                target_coords[:, 1],
                s=36,
                facecolors="none",
                edgecolors="#2ca02c",
                linewidths=1.4,
                label=f"positive_{target_label}",
            )

    if selected_pairs:
        for idx, pair in enumerate(selected_pairs):
            anchor_sample = pair["anchor"]
            target_sample = pair["target"]
            if not anchor_sample.get("coord") or not target_sample.get("coord"):
                continue
            ax.scatter(
                anchor_sample["coord"][0],
                anchor_sample["coord"][1],
                s=80,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.0,
                label="selected positives" if idx == 0 else None,
                zorder=5,
            )
            ax.scatter(
                target_sample["coord"][0],
                target_sample["coord"][1],
                s=80,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.0,
                zorder=5,
            )
            ax.plot(
                [anchor_sample["coord"][0], target_sample["coord"][0]],
                [anchor_sample["coord"][1], target_sample["coord"][1]],
                color="#d62728",
                alpha=0.45,
                linewidth=1.2,
                linestyle="--",
            )

    # Draw approximate windows as rectangles
    try:
        from matplotlib import patches
    except ImportError:
        patches = None

    if patches is not None:
        for sample in (anchor_samples + target_samples):
            coord = sample.get("coord")
            footprint = sample.get("footprint")
            if coord is None or footprint is None:
                continue
            width, height = footprint
            if width is None or height is None or width <= 0 or height <= 0:
                continue
            rect = patches.Rectangle(
                (coord[0] - width / 2.0, coord[1] - height / 2.0),
                width,
                height,
                linewidth=0.8,
                edgecolor="#ff7f0e",
                facecolor="none",
                alpha=0.35,
            )
            ax.add_patch(rect)

    ax.set_title("Overlap positive samples and selected pairs")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    fig_path = viz_dir / "debug_positive_overlap.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[info] saved debug figure to {fig_path}")
    except Exception as exc:
        print(f"[warn] Unable to save debug figure {fig_path}: {exc}")
        fallback_dir = _prepare_output_dir(None, "overlap_alignment_debug/bridge_visualizations/overlap")
        if fallback_dir is not None:
            fallback_path = fallback_dir / fig_path.name
            try:
                fig.savefig(fallback_path, dpi=200)
                print(f"[info] saved debug figure to fallback location {fallback_path}")
            except Exception as exc_fallback:
                print(f"[warn] Unable to save debug figure to fallback location {fallback_path}: {exc_fallback}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
