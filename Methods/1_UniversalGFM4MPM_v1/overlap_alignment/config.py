from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_path(value: Optional[str | Path], base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    path_obj = Path(value)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


@dataclass
class DatasetConfig:
    """
    Configuration for a single embedding dataset.

    Parameters
    ----------
    name:
        Identifier used when referencing this dataset throughout the alignment pipeline.
    embedding_path:
        `.npz` bundle containing `embeddings`, `labels`, and optional metadata arrays.
    metadata_path:
        Optional JSON metadata exported via `integrate_stac.py`.
    pn_split_path:
        Optional JSON/NPY path describing positive/negative splits (kept for parity with v0, unused for now).
    region_filter:
        Optional list of region names; when provided only tiles from the matching regions are loaded.
    class_prior:
        Optional prior probability for positive class; inferred from labels when omitted.
    positive_label:
        Label in `labels` array treated as the positive class.
    weight:
        Optional weighting factor for downstream optimisation.
    """

    name: str
    embedding_path: Path
    metadata_path: Optional[Path] = None
    pn_split_path: Optional[Path] = None
    region_filter: Optional[List[str]] = None
    class_prior: Optional[float] = None
    positive_label: int = 1
    weight: float = 1.0
    pos_aug_path: Optional[Path] = None

    def resolve(self, base_dir: Path) -> "DatasetConfig":
        resolved = DatasetConfig(
            name=self.name,
            embedding_path=_as_path(self.embedding_path, base_dir) or self.embedding_path,
            metadata_path=_as_path(self.metadata_path, base_dir),
            pn_split_path=_as_path(self.pn_split_path, base_dir),
            region_filter=list(self.region_filter) if self.region_filter else None,
            class_prior=self.class_prior,
            positive_label=self.positive_label,
            weight=self.weight,
            pos_aug_path=_as_path(self.pos_aug_path, base_dir),
        )
        return resolved


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class AlignmentConfig:
    """
    High-level configuration driving the overlap alignment sandbox.

    The aim for `v1` is to modularise data ingestion and overlap reasoning.
    """

    datasets: List[DatasetConfig] = field(default_factory=list)
    integration_metadata_path: Optional[Path] = None
    overlap_pairs_path: Optional[Path] = None
    overlap_pairs_augmented_path: Optional[Path] = None
    overlap_mask_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    seed: int = 17
    device: str = "cuda"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    projection_dim: int = 256
    alignment_objective: str = "dcca"
    aggregator: str = "weighted_pool"
    gaussian_sigma: Optional[float] = None
    use_positive_only: bool = False
    use_positive_augmentation: bool = False

    def resolve_paths(self, base_dir: Path) -> "AlignmentConfig":
        resolved = AlignmentConfig(
            datasets=[dataset.resolve(base_dir) for dataset in self.datasets],
            integration_metadata_path=_as_path(self.integration_metadata_path, base_dir),
            overlap_pairs_path=_as_path(self.overlap_pairs_path, base_dir),
            overlap_pairs_augmented_path=_as_path(self.overlap_pairs_augmented_path, base_dir),
            overlap_mask_path=_as_path(self.overlap_mask_path, base_dir),
            output_dir=_as_path(self.output_dir, base_dir),
            log_dir=_as_path(self.log_dir, base_dir),
            seed=self.seed,
            device=self.device,
            training=self.training,
            projection_dim=self.projection_dim,
            alignment_objective=self.alignment_objective,
            aggregator=self.aggregator,
            gaussian_sigma=self.gaussian_sigma,
            use_positive_only=self.use_positive_only,
            use_positive_augmentation=self.use_positive_augmentation,
        )
        return resolved


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {path} does not exist.") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON configuration {path}: {exc}") from exc


def load_config(path: Path) -> AlignmentConfig:
    """
    Load an overlap-alignment configuration from JSON.

    The schema is intentionally backwards-compatible with `v0` configs so that
    existing experiment descriptors can be reused with minor edits.
    """

    base_dir = path.parent.resolve()
    payload = _read_json(path)

    dataset_entries = payload.get("datasets")
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError("Configuration must provide a non-empty 'datasets' list.")

    datasets = [
        DatasetConfig(
            name=str(entry["name"]),
            embedding_path=Path(entry["embedding_path"]),
            metadata_path=Path(entry["metadata_path"]) if entry.get("metadata_path") else None,
            pn_split_path=Path(entry["pn_split_path"]) if entry.get("pn_split_path") else None,
            region_filter=list(entry.get("region_filter") or []) or None,
            class_prior=entry.get("class_prior"),
            positive_label=int(entry.get("positive_label", 1)),
            weight=float(entry.get("weight", 1.0)),
            pos_aug_path=Path(entry["pos_aug_path"]) if entry.get("pos_aug_path") else None,
        )
        for entry in dataset_entries
        if isinstance(entry, dict)
    ]

    optimization_payload = payload.get("optimization")
    projection_override = None
    if isinstance(optimization_payload, dict):
        training_payload = dict(optimization_payload)
        projection_override = training_payload.pop("projection_dim", None)
    else:
        maybe_training = payload.get("training", {})
        training_payload = dict(maybe_training) if isinstance(maybe_training, dict) else {}
    training_cfg = TrainingConfig(**training_payload) if isinstance(training_payload, dict) else TrainingConfig()
    projection_value = payload.get("projection_dim")
    if projection_value is None:
        projection_value = projection_override
    if projection_value is None:
        projection_value = 256

    config = AlignmentConfig(
        datasets=datasets,
        integration_metadata_path=Path(payload["integration_metadata_path"]) if payload.get("integration_metadata_path") else None,
        overlap_pairs_path=Path(payload["overlap_pairs_path"]) if payload.get("overlap_pairs_path") else None,
        overlap_pairs_augmented_path=Path(payload["overlap_pairs_augmented_path"]) if payload.get("overlap_pairs_augmented_path") else None,
        overlap_mask_path=Path(payload["overlap_mask_path"]) if payload.get("overlap_mask_path") else None,
        output_dir=Path(payload["output_dir"]) if payload.get("output_dir") else None,
        log_dir=Path(payload["log_dir"]) if payload.get("log_dir") else None,
        seed=int(payload.get("seed", 17)),
        device=str(payload.get("device", "cuda")),
        training=training_cfg,
        projection_dim=int(projection_value),
        alignment_objective=str(payload.get("alignment_objective", "dcca")).lower(),
        aggregator=str(payload.get("aggregator", "weighted_pool")).lower(),
        gaussian_sigma=float(payload.get("gaussian_sigma")) if payload.get("gaussian_sigma") is not None else None,
        use_positive_only=bool(payload.get("use_positive_only", False)),
        use_positive_augmentation=bool(payload.get("use_positive_augmentation", False)),
    )
    return config.resolve_paths(base_dir)
