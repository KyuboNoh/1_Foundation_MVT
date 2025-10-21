from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset to be integrated."""

    name: str
    embedding_path: Path
    metadata_path: Optional[Path] = None
    region_filter: Optional[List[str]] = None
    class_prior: Optional[float] = None
    weight: float = 1.0

    def resolve(self, base_dir: Optional[Path] = None) -> "DatasetConfig":
        base = base_dir or Path.cwd()
        embedding_path = Path(self.embedding_path)
        if not embedding_path.is_absolute():
            embedding_path = (base / embedding_path).resolve()
        metadata_path = None
        if self.metadata_path is not None:
            meta_path = Path(self.metadata_path)
            metadata_path = meta_path if meta_path.is_absolute() else (base / meta_path).resolve()
        return DatasetConfig(
            name=self.name,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
            region_filter=list(self.region_filter) if self.region_filter else None,
            class_prior=self.class_prior,
            weight=self.weight,
        )


@dataclass
class OptimizationConfig:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 50


@dataclass
class LossWeights:
    supcon: float = 1.0
    prototype: float = 0.2
    nnpu: float = 1.0
    info_nce: float = 0.5
    kl_align: float = 0.5
    dann: float = 0.2
    mmd: float = 0.0
    film: float = 0.0


@dataclass
class IntegrateConfig:
    datasets: List[DatasetConfig]
    output_dir: Path
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    overlap_pairs_path: Optional[Path] = None
    overlap_mask_path: Optional[Path] = None
    log_dir: Optional[Path] = None
    seed: int = 17
    device: str = "cuda"

    def resolve_paths(self, base_dir: Optional[Path] = None) -> "IntegrateConfig":
        base = base_dir or Path.cwd()
        resolved_datasets = [cfg.resolve(base) for cfg in self.datasets]
        output_dir = (base / self.output_dir).resolve() if not self.output_dir.is_absolute() else self.output_dir
        overlap_pairs_path = None
        if self.overlap_pairs_path is not None:
            overlap_pairs_path = (base / self.overlap_pairs_path).resolve() if not self.overlap_pairs_path.is_absolute() else self.overlap_pairs_path
        overlap_mask_path = None
        if self.overlap_mask_path is not None:
            overlap_mask_path = (base / self.overlap_mask_path).resolve() if not self.overlap_mask_path.is_absolute() else self.overlap_mask_path
        log_dir = None
        if self.log_dir is not None:
            log_dir = (base / self.log_dir).resolve() if not self.log_dir.is_absolute() else self.log_dir
        return IntegrateConfig(
            datasets=resolved_datasets,
            output_dir=output_dir,
            optimization=self.optimization,
            loss_weights=self.loss_weights,
            overlap_pairs_path=overlap_pairs_path,
            overlap_mask_path=overlap_mask_path,
            log_dir=log_dir,
            seed=self.seed,
            device=self.device,
        )


def load_config(path: Path) -> IntegrateConfig:
    data = _load_json(path)
    datasets = [DatasetConfig(**ds) for ds in data["datasets"]]
    optimization = OptimizationConfig(**data.get("optimization", {}))
    loss_weights = LossWeights(**data.get("loss_weights", {}))
    cfg = IntegrateConfig(
        datasets=datasets,
        output_dir=Path(data["output_dir"]),
        optimization=optimization,
        loss_weights=loss_weights,
        overlap_pairs_path=Path(data["overlap_pairs_path"]) if data.get("overlap_pairs_path") else None,
        overlap_mask_path=Path(data["overlap_mask_path"]) if data.get("overlap_mask_path") else None,
        log_dir=Path(data["log_dir"]) if data.get("log_dir") else None,
        seed=data.get("seed", 17),
        device=data.get("device", "cuda"),
    )
    return cfg.resolve_paths(path.parent)
