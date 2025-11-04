from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _as_path(value: Optional[str | Path], base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    path_obj = Path(value)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def _normalise_path_value(
    value: Optional[str | Path],
    project_dir: Optional[Path],
    *,
    allow_root_relative: bool = False,
) -> Optional[Path]:
    if value is None:
        return None
    path_obj = Path(str(value))
    if project_dir is not None:
        try:
            path_obj.relative_to(project_dir)
            return path_obj
        except ValueError:
            pass
        if path_obj.is_absolute():
            if allow_root_relative and path_obj.anchor in ("/", "\\"):
                stripped = str(path_obj).lstrip("/\\")
                return Path(stripped) if stripped else Path(".")
            return path_obj.resolve()
        if allow_root_relative and isinstance(value, str) and (value.startswith("/") or value.startswith("\\")):
            return Path(value.lstrip("/\\"))
    return path_obj.resolve() if path_obj.is_absolute() else path_obj


def _normalise_pipeline_presteps(section: Any) -> Dict[str, Any]:
    if isinstance(section, dict):
        return dict(section)
    if isinstance(section, list):
        merged: Dict[str, Any] = {}
        for item in section:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def _first_match(directory: Path, patterns: List[str]) -> Optional[Path]:
    if not directory.exists():
        return None
    for pattern in patterns:
        for candidate in sorted(directory.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _parse_hidden_dims(value: Any) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        dims: List[int] = []
        for item in value:
            try:
                dim = int(item)
            except (TypeError, ValueError):
                continue
            if dim > 0:
                dims.append(dim)
        return tuple(dims)
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return tuple()
    return (dim,) if dim > 0 else tuple()



def _coerce_config_section(
    section: Optional[Dict[str, Any]],
    config_type: type[ConfigBase],
    *,
    alias_map: Optional[Dict[str, str]] = None,
    base: Optional[ConfigBase] = None,
) -> ConfigBase:
    seed_cfg = base if base is not None else config_type()
    data = _config_to_kwargs(seed_cfg)
    extras = data.pop("extras", {})
    if isinstance(seed_cfg, ConfigDCCA):
        recognised_float = {"lr", "weight_decay", "validation_fraction", "dcca_eps", "singular_value_drop_ratio", "tcc_ratio"}
    else:
        recognised_float = {"lr", "weight_decay", "validation_fraction"}
    recognised_int = {"batch_size", "epochs", "mc_dropout_passes", "projection_dim"}
    recognised_str = {"overlapregion_label"} if issubclass(config_type, ConfigCLS) else set()
    recognised_keys = set(recognised_float) | set(recognised_int) | {"mlp_hidden_dims"} | recognised_str
    alias_map = alias_map or {}
    if not isinstance(section, dict):
        cfg = config_type(**data)
        cfg.extras.update(extras)
        return cfg
    for key, value in section.items():
        canonical = alias_map.get(key, key)
        if canonical == "mlp_hidden_dims":
            dims = _parse_hidden_dims(value)
            if dims:
                data["mlp_hidden_dims"] = dims
            continue
        if canonical in recognised_int:
            try:
                data[canonical] = int(value)
            except (TypeError, ValueError):
                continue
            continue
        if canonical in recognised_float:
            try:
                data[canonical] = float(value)
            except (TypeError, ValueError):
                continue
            continue
        if canonical in recognised_str:
            data[canonical] = str(value)
            continue
        extras[canonical] = value
    cfg = config_type(**data)
    if extras:
        cfg.extras.update(extras)
    return cfg


def _resolve_under_project(project_dir: Path, value: str | Path, *, allow_external: bool = True) -> Path:
    path_obj = Path(str(value))
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
        if allow_external:
            try:
                resolved.relative_to(project_dir)
                return resolved
            except ValueError:
                if path_obj.drive:
                    return resolved
        stripped = str(path_obj).lstrip("/\\")
        return (project_dir / stripped).resolve()
    stripped = str(path_obj).lstrip("/\\")
    return (project_dir / stripped).resolve()


def _combine_with_base(base: Path, value: Optional[str | Path]) -> Path:
    if value is None:
        return base
    path_obj = Path(str(value))
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            if path_obj.drive:
                return resolved
        stripped = str(path_obj).lstrip("/\\")
        return (base / stripped).resolve() if stripped else base
    stripped = str(path_obj).lstrip("/\\")
    return (base / stripped).resolve()


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
    encoder_path: Optional[Path] = None
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
            encoder_path=_as_path(self.encoder_path, base_dir) if self.encoder_path else None,
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
class ConfigBase:
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigDCCA(ConfigBase):
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    validation_fraction: float = 0.3
    projection_dim: Optional[int] = None
    dcca_eps: Optional[float] = None
    singular_value_drop_ratio: Optional[float] = None
    tcc_ratio: Optional[float] = None
    mlp_hidden_dims: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ConfigCLS(ConfigBase):
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    validation_fraction: float = 0.3
    mc_dropout_passes: Optional[int] = None
    mlp_hidden_dims: Tuple[int, ...] = field(default_factory=tuple)
    overlapregion_label: str = "None"


def _config_to_kwargs(cfg: ConfigBase) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        
        "extras": dict(cfg.extras),
    }
    if isinstance(cfg, ConfigDCCA):
        data.update(
            {
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "projection_dim": cfg.projection_dim,
                "mlp_hidden_dims": cfg.mlp_hidden_dims,
                "validation_fraction": cfg.validation_fraction,
                "dcca_eps": cfg.dcca_eps,
                "singular_value_drop_ratio": cfg.singular_value_drop_ratio,
                "tcc_ratio": cfg.tcc_ratio,
            }
        )
    if isinstance(cfg, ConfigCLS):
        data.update(
            {
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "mlp_hidden_dims": cfg.mc_dropout_passes,
                "validation_fraction": cfg.validation_fraction,
                "overlapregion_label": cfg.overlapregion_label
            }
        )
    return data

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
    training: ConfigBase = field(default_factory=ConfigBase)
    dcca_training: ConfigDCCA = field(default_factory=ConfigDCCA)
    cls_training: Optional[ConfigCLS] = None
    projection_dim: int = 256
    alignment_objective: str = "dcca"
    pairing_mode: str = "one_to_one"
    aggregator: str = "weighted_pool"
    gaussian_sigma: Optional[float] = None
    use_positive_only: bool = False
    use_positive_augmentation: bool = False
    project_main_dir: Optional[Path] = None
    pipeline_presteps: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cls_training is None:
            base_kwargs = _config_to_kwargs(self.training)
            extras = base_kwargs.pop("extras", {})
            cls_cfg = ConfigCLS(**base_kwargs)
            if extras:
                cls_cfg.extras.update(extras)
            self.cls_training = cls_cfg

    def resolve_paths(self, base_dir: Path) -> "AlignmentConfig":
        project_dir = _as_path(self.project_main_dir, base_dir) if self.project_main_dir else base_dir
        resolved = AlignmentConfig(
            datasets=[dataset.resolve(project_dir) for dataset in self.datasets],
            integration_metadata_path=_as_path(self.integration_metadata_path, project_dir) if self.integration_metadata_path else None,
            overlap_pairs_path=_as_path(self.overlap_pairs_path, project_dir) if self.overlap_pairs_path else None,
            overlap_pairs_augmented_path=_as_path(self.overlap_pairs_augmented_path, project_dir) if self.overlap_pairs_augmented_path else None,
            overlap_mask_path=_as_path(self.overlap_mask_path, project_dir) if self.overlap_mask_path else None,
            output_dir=_as_path(self.output_dir, project_dir) if self.output_dir else None,
            log_dir=_as_path(self.log_dir, project_dir) if self.log_dir else None,
            seed=self.seed,
            device=self.device,
            training=self.training,
            dcca_training=self.dcca_training,
            cls_training=self.cls_training,
            projection_dim=self.projection_dim,
            alignment_objective=self.alignment_objective,
            pairing_mode=self.pairing_mode,
            aggregator=self.aggregator,
            gaussian_sigma=self.gaussian_sigma,
            use_positive_only=self.use_positive_only,
            use_positive_augmentation=self.use_positive_augmentation,
            project_main_dir=project_dir,
            pipeline_presteps=dict(self.pipeline_presteps),
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

    config_dir = path.parent.resolve()
    payload = _read_json(path)

    project_main_dir_raw = payload.get("project_main_dir") or payload.get("Project_main_dir")
    project_main_dir = Path(str(project_main_dir_raw)).expanduser().resolve() if project_main_dir_raw else config_dir

    pipeline_section = payload.get("Pipeline_presteps") or payload.get("pipeline_presteps")
    pipeline_presteps = _normalise_pipeline_presteps(pipeline_section)
    dataset_entries = payload.get("datasets")
    if not dataset_entries:
        dataset_entries = pipeline_presteps.get("datasets")
    if not dataset_entries:
        dataset_entries = pipeline_presteps.get("step1_datasets")
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError("Configuration must provide a non-empty 'datasets' list.")

    datasets: List[DatasetConfig] = []
    for entry in dataset_entries:
        if not isinstance(entry, dict):
            continue
        if "name" not in entry:
            raise ValueError("Each dataset entry must define a 'name'.")
        name = str(entry["name"])
        if "main_path" in entry:
            main_path_value = entry.get("main_path")
            if main_path_value is None:
                raise ValueError(f"Dataset '{name}' missing 'main_path'.")
            base_dir = _resolve_under_project(project_main_dir, main_path_value)
            encoder_path = base_dir
            metadata_path = (base_dir / "training_metadata.json").resolve()
            embedding_dir = _combine_with_base(base_dir, entry.get("embedding_path"))
            embedding_path = (embedding_dir / "embeddings.npz").resolve()
            pn_dir_value = entry.get("pn_split_path")
            pn_dir = _combine_with_base(base_dir, pn_dir_value if pn_dir_value is not None else entry.get("embedding_path"))
            pn_split_path = (pn_dir / "splits.json").resolve()
            region_spec = entry.get("region_filter")
            if region_spec is None:
                region_filter = ["GLOBAL"]
            elif isinstance(region_spec, str):
                region_filter = [region_spec]
            else:
                region_filter = list(region_spec)
            dataset_cfg = DatasetConfig(
                name=name,
                embedding_path=embedding_path,
                encoder_path=encoder_path,
                metadata_path=metadata_path,
                pn_split_path=pn_split_path,
                region_filter=region_filter,
                class_prior=entry.get("class_prior"),
                positive_label=int(entry.get("positive_label", 1)),
                weight=float(entry.get("weight", 1.0)),
                pos_aug_path=_normalise_path_value(entry.get("pos_aug_path"), project_main_dir),
            )
            datasets.append(dataset_cfg)
        else:
            if "embedding_path" not in entry:
                raise ValueError(f"Dataset '{name}' missing 'embedding_path'.")
            region_spec = entry.get("region_filter")
            if region_spec is None:
                region_filter = None
            elif isinstance(region_spec, str):
                region_filter = [region_spec]
            else:
                region_filter = list(region_spec)
            datasets.append(
                DatasetConfig(
                    name=name,
                    embedding_path=_normalise_path_value(entry["embedding_path"], project_main_dir) or Path(entry["embedding_path"]),
                    encoder_path=_normalise_path_value(entry.get("encoder_path"), project_main_dir),
                    metadata_path=_normalise_path_value(entry.get("metadata_path"), project_main_dir),
                    pn_split_path=_normalise_path_value(entry.get("pn_split_path"), project_main_dir),
                    region_filter=region_filter,
                    class_prior=entry.get("class_prior"),
                    positive_label=int(entry.get("positive_label", 1)),
                    weight=float(entry.get("weight", 1.0)),
                    pos_aug_path=_normalise_path_value(entry.get("pos_aug_path"), project_main_dir),
                )
            )

    data_integration_project = (
        payload.get("2_Data_integration_projectname")
        or payload.get("data_integration_project")
        or pipeline_presteps.get("2_Data_integration_projectname")
        or pipeline_presteps.get("data_integration_project")
        or pipeline_presteps.get("step2_data_integration_project")
    )
    integration_dir = project_main_dir / data_integration_project if data_integration_project else None

    optimization_payload = payload.get("optimization")
    projection_override = None
    training_cfg = ConfigBase()
    dcca_training_cfg = ConfigDCCA()
    cls_training_cfg = ConfigCLS()

    if isinstance(optimization_payload, dict):
        legacy_payload = dict(optimization_payload)
        step1_payload = legacy_payload.pop("step1_DCCA", None)
        step2_payload = legacy_payload.pop("step2_CLS", None)

        if isinstance(step1_payload, dict):
            dcca_training_cfg = _coerce_config_section(step1_payload, ConfigDCCA)
            if step1_payload.get("projection_dim") is not None:
                projection_override = step1_payload.get("projection_dim")

        if isinstance(step2_payload, dict):
            cls_training_cfg = _coerce_config_section(
                step2_payload,
                ConfigCLS,
                alias_map={"dcca_mlp_hidden_dims": "mlp_hidden_dims"},
                base=cls_training_cfg,
            )

        if legacy_payload:
            legacy_projection = legacy_payload.pop("projection_dim", None)
            legacy_cfg = _coerce_config_section(legacy_payload, ConfigBase)
            default_cfg = ConfigBase()
            if legacy_cfg != default_cfg:
                training_cfg = legacy_cfg
                dcca_training_cfg = _coerce_config_section(legacy_payload, ConfigDCCA, base=dcca_training_cfg)
                cls_training_cfg = _coerce_config_section(legacy_payload, ConfigCLS, base=cls_training_cfg)
            if legacy_projection is not None:
                projection_override = legacy_projection
    else:
        maybe_training = payload.get("training", {})
        if isinstance(maybe_training, dict):
            training_cfg = _coerce_config_section(maybe_training, ConfigBase)
            dcca_training_cfg = _coerce_config_section(maybe_training, ConfigDCCA)
            cls_training_cfg = _coerce_config_section(maybe_training, ConfigCLS)
    projection_value = payload.get("projection_dim")
    if projection_value is None:
        projection_value = projection_override
    if projection_value is None:
        projection_value = 256

    integration_metadata_value = payload.get("integration_metadata_path")
    if integration_metadata_value is None and integration_dir is not None:
        integration_metadata_value = str(integration_dir / "combined_metadata.json")

    overlap_pairs_value = payload.get("overlap_pairs_path")
    overlap_pairs_aug_value = payload.get("overlap_pairs_augmented_path")
    overlap_mask_value = payload.get("overlap_mask_path")

    if integration_dir is not None:
        if len(datasets) != 2:
            raise ValueError("Integration config requires at least two datasets to resolve overlap pairs path")
        pair_dir = integration_dir / "data" / "Pairs"
        pair_key = f"{datasets[0].name}_{datasets[1].name}"
        overlap_pairs_value = str(pair_dir / f"{pair_key}_overlap_pairs.json")
        overlap_pairs_aug_value = str(pair_dir / f"{pair_key}_overlap_pairs_positive.json")
        if overlap_mask_value is None:
            mask_dirs = [
                integration_dir / "data" / "Boundary_UnifiedCRS" / "Overlap",
                integration_dir / "Boundary_UnifiedCRS" / "Overlap",
                integration_dir,
            ]
            for mask_dir in mask_dirs:
                mask_match = _first_match(mask_dir, ["*study_area_overlap*.tif", "*study_area_overlap*.tiff", "*overlap*.tif", "*overlap*.tiff"])
                if mask_match is not None:
                    overlap_mask_value = str(mask_match)
                    break

    output_dir_value = payload.get("output_dir")
    log_dir_value = payload.get("log_dir")

    config = AlignmentConfig(
        datasets=datasets,
        integration_metadata_path=_normalise_path_value(integration_metadata_value, project_main_dir),
        overlap_pairs_path=_normalise_path_value(overlap_pairs_value, project_main_dir),
        overlap_pairs_augmented_path=_normalise_path_value(overlap_pairs_aug_value, project_main_dir),
        overlap_mask_path=_normalise_path_value(overlap_mask_value, project_main_dir),
        output_dir=_normalise_path_value(output_dir_value, project_main_dir, allow_root_relative=True),
        log_dir=_normalise_path_value(log_dir_value, project_main_dir, allow_root_relative=True),
        seed=int(payload.get("seed", 17)),
        device=str(payload.get("device", "cuda")),
        training=training_cfg,
        dcca_training=dcca_training_cfg,
        cls_training=cls_training_cfg,
        projection_dim=int(projection_value),
        alignment_objective=str(payload.get("alignment_objective", "dcca")).lower(),
        pairing_mode=str(payload.get("pairing_mode", "one_to_one")).lower(),
        aggregator=str(payload.get("aggregator", "weighted_pool")).lower(),
        gaussian_sigma=float(payload.get("gaussian_sigma")) if payload.get("gaussian_sigma") is not None else None,
        use_positive_only=bool(payload.get("use_positive_only", False)),
        use_positive_augmentation=bool(payload.get("use_positive_augmentation", False)),
        project_main_dir=project_main_dir,
        pipeline_presteps=pipeline_presteps,
    )
    return config.resolve_paths(project_main_dir)
