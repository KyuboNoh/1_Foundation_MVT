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
    pn_split_path: Optional[Path] = None
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
        pn_split_path = None
        if self.pn_split_path is not None:
            pn_path_obj = Path(self.pn_split_path)
            pn_split_path = pn_path_obj if pn_path_obj.is_absolute() else (base / pn_path_obj).resolve()
        return DatasetConfig(
            name=self.name,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
            pn_split_path=pn_split_path,
            region_filter=list(self.region_filter) if self.region_filter else None,
            class_prior=self.class_prior,
            weight=self.weight,
        )


@dataclass
class OptimizationConfig:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
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
class InferenceConfig:
    export_geotiff: bool = True
    export_png: bool = True
    export_npz: bool = True


@dataclass
class IntegrateConfig:
    datasets: List[DatasetConfig]
    output_dir: Path
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    use_previous_negatives: bool = True
    generate_inference: bool = False
    inference: Optional[InferenceConfig] = None
    debug: bool = False
    debug_log_every: int = 1
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
            use_previous_negatives=self.use_previous_negatives,
            generate_inference=self.generate_inference,
            inference=self._resolve_inference(base),
            debug=self.debug,
            debug_log_every=self.debug_log_every,
        )

    def _resolve_inference(self, base: Path) -> Optional[InferenceConfig]:
        if self.inference is None:
            return None
        return InferenceConfig(
            export_geotiff=self.inference.export_geotiff,
            export_png=self.inference.export_png,
            export_npz=self.inference.export_npz,
        )


def load_config(path: Path) -> IntegrateConfig:
    data = _load_json(path)
    datasets = [DatasetConfig(**ds) for ds in data["datasets"]]
    optimization = OptimizationConfig(**data.get("optimization", {}))
    loss_weights_data = dict(data.get("loss_weights", {}))
    if "pn_or_nnpu" in loss_weights_data and "nnpu" not in loss_weights_data:
        loss_weights_data["nnpu"] = loss_weights_data.pop("pn_or_nnpu")
    loss_weights = LossWeights(**loss_weights_data)
    inference_block = data.get("inference")
    inference_cfg = InferenceConfig(**inference_block) if isinstance(inference_block, dict) else None
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
        use_previous_negatives=data.get("use_previous_negatives", True),
        generate_inference=bool(data.get("generate_inference", False)),
        inference=inference_cfg,
        debug=data.get("debug", False),
        debug_log_every=data.get("debug_log_every", 1),
    )
    return cfg.resolve_paths(path.parent)
