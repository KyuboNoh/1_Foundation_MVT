from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from .config import IntegrateConfig
    from .data import EmbeddingDataset, load_records
    from .losses import (
        GradientReversal,
        InfoNCELoss,
        MMDLoss,
        NNPU,
        PrototypeLoss,
        SupConLoss,
        SymmetricKLLoss,
    )
    from .models import ClassificationHead, DomainDiscriminator, ProjectionHead
except ImportError:  # pragma: no cover - support direct execution via importlib
    from config import IntegrateConfig  # type: ignore
    from data import EmbeddingDataset, load_records  # type: ignore
    from losses import (  # type: ignore
        GradientReversal,
        InfoNCELoss,
        MMDLoss,
        NNPU,
        PrototypeLoss,
        SupConLoss,
        SymmetricKLLoss,
    )
    from models import ClassificationHead, DomainDiscriminator, ProjectionHead  # type: ignore


@dataclass
class DatasetBundle:
    name: str
    dataset: EmbeddingDataset
    loader: DataLoader
    head: ClassificationHead
    projector: ProjectionHead
    prototype: torch.nn.Parameter
    class_prior: float
    metadata_path: Optional[Path] = None
    reference_rasters: Dict[str, Path] = field(default_factory=dict)


class FMIntegrator:
    def __init__(self, config: IntegrateConfig):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.supcon = SupConLoss().to(self.device)
        self.prototype_loss = PrototypeLoss().to(self.device)
        self.info_nce = InfoNCELoss().to(self.device)
        self.kl_loss = SymmetricKLLoss().to(self.device)
        self.mmd = MMDLoss().to(self.device)
        self.output_dir = self.cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.log_dir:
            self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        self.bundles: Dict[str, DatasetBundle] = {}
        self.domain_disc: Optional[DomainDiscriminator] = None
        self.grl = GradientReversal().to(self.device)
        self.history: List[Dict[str, float]] = []

        self._prepare_datasets()
        self._prepare_overlap_pairs()

    def _prepare_datasets(self) -> None:
        for ds_cfg in self.cfg.datasets:
            records = load_records(ds_cfg.embedding_path, ds_cfg.metadata_path, ds_cfg.region_filter)
            dataset = EmbeddingDataset(records, class_prior=ds_cfg.class_prior)
            class_prior = ds_cfg.class_prior
            if class_prior is None:
                labels = np.array([rec.label for rec in records])
                class_prior = float(labels.mean()) if labels.size > 0 else 0.1
            embeddings = torch.stack([torch.from_numpy(rec.embedding).float() for rec in records])
            labels = torch.tensor([rec.label for rec in records], dtype=torch.long)
            ds = TensorDataset(embeddings, labels)
            loader = DataLoader(ds, batch_size=self.cfg.optimization.batch_size, shuffle=True, drop_last=False)
            proj = ProjectionHead(dataset.embedding_dim, output_dim=dataset.embedding_dim).to(self.device)
            head = ClassificationHead(dataset.embedding_dim).to(self.device)
            prototype = torch.nn.Parameter(torch.zeros(dataset.embedding_dim, device=self.device))
            bundle = DatasetBundle(
                name=ds_cfg.name,
                dataset=dataset,
                loader=loader,
                head=head,
                projector=proj,
                prototype=prototype,
                class_prior=class_prior,
                metadata_path=ds_cfg.metadata_path,
            )
            self.bundles[ds_cfg.name] = bundle

        if len(self.bundles) >= 2:
            emb_dim = next(iter(self.bundles.values())).dataset.embedding_dim
            self.domain_disc = DomainDiscriminator(emb_dim).to(self.device)

    def _prepare_overlap_pairs(self) -> None:
        self.overlap_pairs: List[Tuple[str, str]] = []
        pair_summary: Dict[str, int] = {}
        if self.cfg.overlap_pairs_path and self.cfg.overlap_pairs_path.exists():
            try:
                with self.cfg.overlap_pairs_path.open("r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                def _emit_row(entry: Dict[str, object], *, dataset_names: Optional[Iterable[str]] = None) -> None:
                    ds_a = entry.get("dataset_a")
                    ds_b = entry.get("dataset_b")
                    if ds_a is None or ds_b is None:
                        if dataset_names is not None:
                            ds_a, ds_b = list(dataset_names)[:2]
                    tile_a = entry.get("dataset_a_tile") or entry.get("dataset_a_lookup") or entry.get("dataset_0_lookup")
                    tile_b = entry.get("dataset_b_tile") or entry.get("dataset_b_lookup") or entry.get("dataset_1_lookup")
                    centroid = (
                        entry.get("overlap_centroid")
                        or entry.get("overlap_centerloid")
                        or entry.get("centroid")
                    )
                    if centroid is None:
                        native_a = entry.get("dataset_a_native_point") or entry.get("dataset_0_native_point")
                        native_b = entry.get("dataset_b_native_point") or entry.get("dataset_1_native_point")
                        if isinstance(native_a, (list, tuple)) and isinstance(native_b, (list, tuple)) and len(native_a) == 2 and len(native_b) == 2:
                            centroid = [
                                float(native_a[0] + native_b[0]) / 2.0,
                                float(native_a[1] + native_b[1]) / 2.0,
                            ]
                    row_data = {
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "tile_a": tile_a,
                        "tile_b": tile_b,
                        "centroid": centroid,
                        "dataset_a_row_col": entry.get("dataset_a_row_col") or entry.get("dataset_0_row_col"),
                        "dataset_b_row_col": entry.get("dataset_b_row_col") or entry.get("dataset_1_row_col"),
                    }
                    if ds_a and ds_b:
                        key = f"{ds_a}->{ds_b}"
                        pair_summary[key] = pair_summary.get(key, 0) + 1
                    self.overlap_pairs.append(row_data)

                if isinstance(doc, list):
                    for row in doc:
                        if isinstance(row, dict):
                            _emit_row(row)
                elif isinstance(doc, dict):
                    dataset_names = None
                    if isinstance(doc.get("dataset_pair"), (list, tuple)) and len(doc["dataset_pair"]) >= 2:
                        dataset_names = [str(doc["dataset_pair"][0]), str(doc["dataset_pair"][1])]
                    pairs_payload = doc.get("pairs")
                    if isinstance(pairs_payload, list):
                        for entry in pairs_payload:
                            if isinstance(entry, dict):
                                _emit_row(entry, dataset_names=dataset_names)
                    else:
                        for value in doc.values():
                            if isinstance(value, list):
                                for entry in value:
                                    if isinstance(entry, dict):
                                        _emit_row(entry, dataset_names=dataset_names)
                else:
                    print(f"[warn] Unrecognised overlap pairs format in {self.cfg.overlap_pairs_path}")
            except Exception as exc:
                print(f"[warn] Failed to read overlap pairs: {exc}")
                self.overlap_pairs = []

        self.tile_lookup: Dict[str, Tuple[str, torch.Tensor, int]] = {}
        self.dataset_coord_arrays: Dict[str, np.ndarray] = {}
        self.dataset_coord_indices: Dict[str, List[int]] = {}
        for bundle in self.bundles.values():
            coords_list: List[Tuple[float, float]] = []
            indices: List[int] = []
            for idx, rec in enumerate(bundle.dataset.records):
                emb = torch.from_numpy(rec.embedding).float()
                self.tile_lookup[rec.tile_id] = (bundle.name, emb, rec.label)
                if rec.coord is not None and not any(np.isnan(rec.coord)):
                    coords_list.append(tuple(rec.coord))
                    indices.append(idx)
            if coords_list:
                self.dataset_coord_arrays[bundle.name] = np.asarray(coords_list, dtype=np.float64)
                self.dataset_coord_indices[bundle.name] = indices

        if self.overlap_pairs:
            pair_total = len(self.overlap_pairs)
            print(f"[info] Detected {pair_total} overlapping tile pair(s) for cross-view alignment")
            if pair_summary:
                for key, count in sorted(pair_summary.items()):
                    print(f"  [info]   {key}: {count}")
        else:
            print("[info] No overlap pairs detected; Phase 2 alignment will be skipped")

    def _optimizers(self):
        params = []
        for bundle in self.bundles.values():
            params += list(bundle.head.parameters())
            params += list(bundle.projector.parameters())
            params += [bundle.prototype]
        opt = torch.optim.AdamW(params, lr=self.cfg.optimization.lr, weight_decay=self.cfg.optimization.weight_decay)
        disc_opt = None
        if self.domain_disc is not None:
            disc_opt = torch.optim.AdamW(self.domain_disc.parameters(), lr=self.cfg.optimization.lr * 0.5)
        return opt, disc_opt

    def _nearest_embedding(self, dataset_name: str, centroid: Sequence[float]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        bundle = self.bundles.get(dataset_name)
        if bundle is None:
            return None
        coord_array = self.dataset_coord_arrays.get(dataset_name)
        coord_indices = self.dataset_coord_indices.get(dataset_name)
        if coord_array is None or coord_indices is None or centroid is None:
            return None
        centroid_np = np.asarray(centroid, dtype=np.float64)
        if centroid_np.size != 2 or np.any(np.isnan(centroid_np)):
            return None
        diffs = coord_array - centroid_np
        dist_sq = np.sum(diffs * diffs, axis=1)
        best_idx = int(np.argmin(dist_sq))
        record_idx = coord_indices[best_idx]
        rec = bundle.dataset.records[record_idx]
        emb = torch.from_numpy(rec.embedding).float().to(self.device)
        proj = bundle.projector(emb)
        logits = bundle.head(proj)
        return proj, logits

    def _gather_overlap_batch(self, batch_size: int = 128):
        if not self.overlap_pairs:
            return None
        selected = random.sample(self.overlap_pairs, min(batch_size, len(self.overlap_pairs)))
        anchor_embeddings = []
        target_embeddings = []
        anchor_logits = []
        target_logits = []
        for pair in selected:
            ds_a = pair.get("dataset_a")
            ds_b = pair.get("dataset_b")
            centroid = pair.get("centroid")
            result_a = self._nearest_embedding(ds_a, centroid)
            result_b = self._nearest_embedding(ds_b, centroid)
            if result_a is None or result_b is None:
                tile_a = pair.get("tile_a")
                tile_b = pair.get("tile_b")
                if tile_a in self.tile_lookup and tile_b in self.tile_lookup:
                    ds_a_lookup, emb_a, _ = self.tile_lookup[tile_a]
                    ds_b_lookup, emb_b, _ = self.tile_lookup[tile_b]
                    bundle_a = self.bundles.get(ds_a_lookup)
                    bundle_b = self.bundles.get(ds_b_lookup)
                    if bundle_a is None or bundle_b is None:
                        continue
                    proj_a = bundle_a.projector(emb_a.to(self.device))
                    proj_b = bundle_b.projector(emb_b.to(self.device))
                    logits_a = bundle_a.head(proj_a)
                    logits_b = bundle_b.head(proj_b)
                    result_a = (proj_a, logits_a)
                    result_b = (proj_b, logits_b)
            if result_a is None or result_b is None:
                continue
            proj_a, logits_a = result_a
            proj_b, logits_b = result_b
            anchor_embeddings.append(proj_b)
            target_embeddings.append(proj_a)
            anchor_logits.append(logits_b)
            target_logits.append(logits_a)
        if not anchor_embeddings:
            return None
        return (
            torch.stack(anchor_embeddings),
            torch.stack(target_embeddings),
            torch.stack(anchor_logits),
            torch.stack(target_logits),
        )

    def _domain_batch(self, batch_size: int = 512):
        if self.domain_disc is None:
            return None
        samples = []
        labels = []
        for idx, bundle in enumerate(self.bundles.values()):
            indices = np.random.choice(len(bundle.dataset.records), size=min(batch_size // len(self.bundles), len(bundle.dataset.records)), replace=False)
            emb = torch.stack([torch.from_numpy(bundle.dataset.records[i].embedding).float() for i in indices])
            samples.append(bundle.projector(emb.to(self.device)))
            domain_label = torch.full((emb.size(0),), idx, dtype=torch.float32, device=self.device)
            labels.append(domain_label)
        if not samples:
            return None
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def train(self):
        optimizer, disc_optimizer = self._optimizers()
        lambda_cfg = self.cfg.loss_weights
        def _format_loss(value: float) -> str:
            return f"{value:.2e}" if abs(value) >= 1e-6 else f"{value:.2e}"

        if tqdm is not None:
            epoch_iter = tqdm(range(self.cfg.optimization.epochs), total=self.cfg.optimization.epochs, desc="Epochs")
        else:
            epoch_iter = range(self.cfg.optimization.epochs)

        for epoch in epoch_iter:
            phase1_stats = defaultdict(float)
            phase1_stats["batches"] = 0.0
            overlap_stats = defaultdict(float)
            domain_stats = defaultdict(float)
            optimizer.zero_grad()
            phase1_iter = None
            if tqdm is not None:
                total_batches = sum(len(bundle.loader) for bundle in self.bundles.values())
                phase1_iter = tqdm(total=total_batches, desc=f"Phase1 [{epoch+1}]", leave=False)
            for bundle in self.bundles.values():
                for batch in bundle.loader:
                    embeddings, labels = [t.to(self.device) for t in batch]
                    proj = bundle.projector(embeddings)
                    logits = bundle.head(proj)
                    # Loss components
                    loss = torch.tensor(0.0, device=self.device)
                    pos_mask = labels == 1
                    if pos_mask.sum() > 1 and lambda_cfg.supcon > 0:
                        loss_sup = self.supcon(proj[pos_mask], labels[pos_mask])
                        loss = loss + lambda_cfg.supcon * loss_sup
                        phase1_stats["supcon"] += float(loss_sup.item())
                    if pos_mask.sum() > 0 and lambda_cfg.prototype > 0:
                        loss_proto = self.prototype_loss(proj[pos_mask], bundle.prototype)
                        loss = loss + lambda_cfg.prototype * loss_proto
                        phase1_stats["prototype"] += float(loss_proto.item())
                    if lambda_cfg.nnpu > 0:
                        nnpu = NNPU(bundle.class_prior).to(self.device)
                        loss_nnpu = nnpu(logits, labels)
                        loss = loss + lambda_cfg.nnpu * loss_nnpu
                        phase1_stats["nnpu"] += float(loss_nnpu.item())
                    phase1_stats["total"] += float(loss.item())
                    phase1_stats["batches"] += 1.0
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if phase1_iter is not None:
                        phase1_iter.set_postfix({"sup": _format_loss(phase1_stats["supcon"] / max(phase1_stats["batches"], 1.0))})
                        phase1_iter.update(1)
            if phase1_iter is not None:
                phase1_iter.close()

            # Phase 2 alignment
            overlap_batch = self._gather_overlap_batch()
            if overlap_batch is not None and lambda_cfg.info_nce > 0:
                anchor_z, target_z, anchor_logits, target_logits = overlap_batch
                loss_overlap = torch.tensor(0.0, device=self.device)
                if lambda_cfg.info_nce > 0:
                    info_term = self.info_nce(anchor_z, target_z)
                    loss_overlap = loss_overlap + lambda_cfg.info_nce * info_term
                    overlap_stats["info_nce"] += float(info_term.item())
                if lambda_cfg.kl_align > 0:
                    kl_term = self.kl_loss(anchor_logits, target_logits)
                    loss_overlap = loss_overlap + lambda_cfg.kl_align * kl_term
                    overlap_stats["kl_align"] += float(kl_term.item())
                optimizer.zero_grad()
                loss_overlap.backward()
                optimizer.step()
                overlap_stats["total"] += float(loss_overlap.item())
                overlap_stats["batches"] += 1.0
            if tqdm is not None and overlap_stats["batches"]:
                tqdm.write(
                    f"  Phase2[{epoch+1}] infoNCE={_format_loss(overlap_stats['info_nce'] / overlap_stats['batches'])} "
                    f"KL={_format_loss(overlap_stats['kl_align'] / overlap_stats['batches'])}"
                )

            # Phase 3 domain adaptation
            domain_batch = self._domain_batch()
            if domain_batch is not None and lambda_cfg.dann > 0 and self.domain_disc is not None:
                features, domain_labels = domain_batch
                optimizer.zero_grad()
                self.domain_disc.train()
                rev_features = self.grl(features)
                disc_logits = self.domain_disc(rev_features)
                domain_targets = (domain_labels > 0).float()
                loss_domain = F.binary_cross_entropy_with_logits(disc_logits, domain_targets)
                if lambda_cfg.mmd > 0 and len(self.bundles) >= 2:
                    bundle_list = list(self.bundles.values())
                    src = bundle_list[0].projector(torch.stack([torch.from_numpy(rec.embedding).float().to(self.device) for rec in bundle_list[0].dataset.records[:128]]) )
                    tgt = bundle_list[1].projector(torch.stack([torch.from_numpy(rec.embedding).float().to(self.device) for rec in bundle_list[1].dataset.records[:128]]) )
                    mmd_term = self.mmd(src, tgt)
                    loss_domain = loss_domain + lambda_cfg.mmd * mmd_term
                    domain_stats["mmd"] += float(mmd_term.item())
                loss_domain = lambda_cfg.dann * loss_domain
                loss_domain.backward()
                optimizer.step()
                domain_stats["total"] += float(loss_domain.item())
                domain_stats["batches"] += 1.0
            if tqdm is not None and domain_stats["batches"]:
                tqdm.write(
                    f"  Phase3[{epoch+1}] domain={_format_loss(domain_stats['total']/domain_stats['batches'])} "
                    f"mmd={_format_loss(domain_stats['mmd']/domain_stats['batches'])}"
                )

            summary = {
                "epoch": epoch + 1,
                "phase1_loss": phase1_stats["total"] / max(phase1_stats["batches"], 1.0),
                "supcon_loss": phase1_stats["supcon"] / max(phase1_stats["batches"], 1.0),
                "prototype_loss": phase1_stats["prototype"] / max(phase1_stats["batches"], 1.0),
                "nnpu_loss": phase1_stats["nnpu"] / max(phase1_stats["batches"], 1.0),
                "overlap_loss": overlap_stats["total"] / max(overlap_stats["batches"], 1.0),
                "info_nce_loss": overlap_stats["info_nce"] / max(overlap_stats["batches"], 1.0),
                "kl_align_loss": overlap_stats["kl_align"] / max(overlap_stats["batches"], 1.0),
                "domain_loss": domain_stats["total"] / max(domain_stats["batches"], 1.0),
                "mmd_loss": domain_stats["mmd"] / max(domain_stats["batches"], 1.0),
            }
            self.history.append(summary)

            if tqdm is not None:
                epoch_iter.set_postfix({
                    "phase1": _format_loss(summary['phase1_loss']),
                    "overlap": _format_loss(summary['overlap_loss']),
                    "domain": _format_loss(summary['domain_loss']),
                })
            elif (epoch + 1) % self.cfg.optimization.log_every == 0 or epoch == self.cfg.optimization.epochs - 1:
                print(
                    f"Epoch {epoch+1}/{self.cfg.optimization.epochs} | phase1 {summary['phase1_loss']:.4f} "
                    f"overlap {summary['overlap_loss']:.4f} domain {summary['domain_loss']:.4f}"
                )

        self._save_artifacts()
        self._write_history()
        self._generate_inference_outputs()

    def _save_artifacts(self):
        artifact_dir = self.output_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        def _to_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        cfg_dict = _to_serializable(asdict(self.cfg))

        state = {
            "config": cfg_dict,
            "datasets": {},
        }
        for name, bundle in self.bundles.items():
            bundle_dir = artifact_dir / name
            bundle_dir.mkdir(exist_ok=True)
            torch.save(bundle.head.state_dict(), bundle_dir / "classifier_head.pt")
            torch.save(bundle.projector.state_dict(), bundle_dir / "projection_head.pt")
            torch.save(bundle.prototype.detach().cpu(), bundle_dir / "prototype.pt")
            state["datasets"][name] = {
                "class_prior": bundle.class_prior,
                "records": len(bundle.dataset.records),
            }
        with (artifact_dir / "state.json").open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def _write_history(self):
        if not self.history:
            return
        history_path = self.output_dir / "training_history.json"
        try:
            with history_path.open("w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=2)
            print(f"[info] Training history saved to {history_path}")
        except Exception as exc:
            print(f"[warn] Failed to write training history: {exc}")

    def _generate_inference_outputs(self) -> None:
        if not getattr(self.cfg, "generate_inference", False):
            return
        inference_root = self.output_dir / "inference"
        inference_root.mkdir(parents=True, exist_ok=True)
        batch_size = min(2048, max(1, self.cfg.optimization.batch_size))
        prediction_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        index_lookup: Dict[str, Dict[str, int]] = {}

        for name, bundle in self.bundles.items():
            def _get_item(container, index, default=None):
                try:
                    if isinstance(container, torch.Tensor):
                        if container.size(0) > index:
                            return container[index]
                        return default
                    if isinstance(container, np.ndarray):
                        if container.shape[0] > index:
                            return container[index]
                        return default
                    if isinstance(container, (list, tuple)):
                        if len(container) > index:
                            return container[index]
                        return default
                except Exception:
                    return default
                return default

            dataset_dir = inference_root / name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            total = len(bundle.dataset.records)
            if total == 0:
                continue
            tile_ids = np.empty(total, dtype=object)
            coords = np.full((total, 2), np.nan, dtype=np.float32)
            rows = np.full(total, -1, dtype=np.int32)
            cols = np.full(total, -1, dtype=np.int32)
            regions = np.empty(total, dtype=object)
            regions[:] = ""
            logits = np.empty(total, dtype=np.float32)
            probs = np.empty(total, dtype=np.float32)
            labels = np.empty(total, dtype=np.int8)

            loader = DataLoader(bundle.dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            bundle.projector.eval()
            bundle.head.eval()
            offset = 0
            with torch.no_grad():
                for emb_batch, label_batch, meta_batch in loader:
                    batch_len = emb_batch.size(0)
                    emb_device = emb_batch.to(self.device)
                    proj = bundle.projector(emb_device)
                    logit_batch = bundle.head(proj)
                    prob_batch = torch.sigmoid(logit_batch)
                    logits[offset:offset + batch_len] = logit_batch.view(-1).cpu().numpy().astype(np.float32, copy=False)
                    probs[offset:offset + batch_len] = prob_batch.view(-1).cpu().numpy().astype(np.float32, copy=False)
                    labels[offset:offset + batch_len] = label_batch.view(-1).cpu().numpy().astype(np.int8, copy=False)

                    tile_batch = meta_batch.get("tile_id")
                    coord_batch = meta_batch.get("coord")
                    if tile_batch is None:
                        tile_batch = [None] * batch_len
                    if coord_batch is None:
                        coord_batch = [None] * batch_len
                    info_batch = meta_batch.get("metadata")
                    for idx in range(batch_len):
                        target = offset + idx
                        tile_value = _get_item(tile_batch, idx, "")
                        tile_ids[target] = str(tile_value)
                        coord_value = _get_item(coord_batch, idx, None)
                        if isinstance(coord_value, torch.Tensor):
                            coord_np = coord_value.detach().cpu().numpy()
                        else:
                            coord_np = np.asarray(coord_value) if isinstance(coord_value, (list, tuple, np.ndarray)) else None
                        if coord_np is not None and coord_np.size == 2:
                            coords[target] = coord_np.astype(np.float32, copy=False)
                        if isinstance(info_batch, list):
                            meta_value = _get_item(info_batch, idx, {})
                        elif isinstance(info_batch, dict):
                            meta_value = {}
                            for key, values in info_batch.items():
                                if isinstance(values, (list, tuple)):
                                    meta_value[key] = values[idx] if len(values) > idx else None
                                elif isinstance(values, torch.Tensor):
                                    if values.size(0) > idx:
                                        meta_value[key] = values[idx].item() if values.dim() == 1 else values[idx].detach().cpu().numpy()
                                elif isinstance(values, np.ndarray):
                                    if values.shape[0] > idx:
                                        meta_value[key] = values[idx].item() if values.ndim == 1 else values[idx]
                                else:
                                    try:
                                        meta_value[key] = values[idx]
                                    except Exception:
                                        continue
                        else:
                            meta_value = info_batch
                        if not isinstance(meta_value, dict):
                            meta_value = {}
                        if meta_value:
                            row_val = meta_value.get("row")
                            col_val = meta_value.get("col")
                            if row_val is not None:
                                rows[target] = int(row_val)
                            if col_val is not None:
                                cols[target] = int(col_val)
                            region_val = meta_value.get("region")
                            if region_val is not None:
                                regions[target] = str(region_val)
                        # leave defaults if metadata missing
                    offset += batch_len

            np.savez_compressed(
                dataset_dir / "predictions.npz",
                tile_ids=tile_ids,
                coords=coords,
                logits=logits,
                probs=probs,
                labels=labels,
                rows=rows,
                cols=cols,
                regions=regions,
            )
            prediction_arrays[name] = {
                "tile_ids": tile_ids,
                "coords": coords,
                "logits": logits,
                "probs": probs,
                "labels": labels,
                "rows": rows,
                "cols": cols,
                "regions": regions,
            }
            index_lookup[name] = {str(tile_ids[i]): i for i in range(tile_ids.size)}
            self._render_probability_rasters(
                dataset_name=name,
                rows=rows,
                cols=cols,
                probs=probs,
                regions=regions,
                output_directory=dataset_dir,
                suffix=None,
            )

        if not self.overlap_pairs or not prediction_arrays:
            return

        overlap_dir = inference_root / "overlap"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for entry in self.overlap_pairs:
            ds_a = entry.get("dataset_a")
            ds_b = entry.get("dataset_b")
            if not ds_a or not ds_b:
                continue
            if ds_a not in prediction_arrays or ds_b not in prediction_arrays:
                continue
            grouped[(ds_a, ds_b)].append(entry)

        for (ds_a, ds_b), entries in grouped.items():
            arr_a = prediction_arrays[ds_a]
            arr_b = prediction_arrays[ds_b]
            idx_a = index_lookup[ds_a]
            idx_b = index_lookup[ds_b]

            count = len(entries)
            if count == 0:
                continue
            tiles_a = np.empty(count, dtype=object)
            tiles_b = np.empty(count, dtype=object)
            probs_a = np.full(count, np.nan, dtype=np.float32)
            probs_b = np.full(count, np.nan, dtype=np.float32)
            logits_a = np.full(count, np.nan, dtype=np.float32)
            logits_b = np.full(count, np.nan, dtype=np.float32)
            coords_a = np.full((count, 2), np.nan, dtype=np.float32)
            coords_b = np.full((count, 2), np.nan, dtype=np.float32)
            centroid = np.full((count, 2), np.nan, dtype=np.float32)
            rows_a = np.full(count, -1, dtype=np.int32)
            cols_a = np.full(count, -1, dtype=np.int32)
            rows_b = np.full(count, -1, dtype=np.int32)
            cols_b = np.full(count, -1, dtype=np.int32)
            regions_a = np.empty(count, dtype=object)
            regions_b = np.empty(count, dtype=object)
            regions_a[:] = ""
            regions_b[:] = ""

            for i, entry in enumerate(entries):
                tile_a = entry.get("tile_a")
                tile_b = entry.get("tile_b")
                tiles_a[i] = tile_a
                tiles_b[i] = tile_b
                idx0 = idx_a.get(str(tile_a))
                idx1 = idx_b.get(str(tile_b))
                if idx0 is not None:
                    probs_a[i] = arr_a["probs"][idx0]
                    logits_a[i] = arr_a["logits"][idx0]
                    coords_a[i] = arr_a["coords"][idx0]
                    rows_a[i] = arr_a["rows"][idx0]
                    cols_a[i] = arr_a["cols"][idx0]
                    region_val = arr_a["regions"][idx0]
                    if isinstance(region_val, str):
                        regions_a[i] = region_val
                    elif hasattr(region_val, "item"):
                        regions_a[i] = str(region_val.item())
                if idx1 is not None:
                    probs_b[i] = arr_b["probs"][idx1]
                    logits_b[i] = arr_b["logits"][idx1]
                    coords_b[i] = arr_b["coords"][idx1]
                    rows_b[i] = arr_b["rows"][idx1]
                    cols_b[i] = arr_b["cols"][idx1]
                    region_val = arr_b["regions"][idx1]
                    if isinstance(region_val, str):
                        regions_b[i] = region_val
                    elif hasattr(region_val, "item"):
                        regions_b[i] = str(region_val.item())
                centroid_val = entry.get("centroid")
                if isinstance(centroid_val, (list, tuple, np.ndarray)) and len(centroid_val) == 2:
                    centroid[i] = np.asarray(centroid_val, dtype=np.float32)

            np.savez_compressed(
                overlap_dir / f"{ds_a}_{ds_b}_overlap.npz",
                dataset_a=np.array([ds_a] * count, dtype=object),
                dataset_b=np.array([ds_b] * count, dtype=object),
                tiles_a=tiles_a,
                tiles_b=tiles_b,
                probs_a=probs_a,
                probs_b=probs_b,
                logits_a=logits_a,
                logits_b=logits_b,
                coords_a=coords_a,
                coords_b=coords_b,
                overlap_centroid=centroid,
                rows_a=rows_a,
                cols_a=cols_a,
                rows_b=rows_b,
                cols_b=cols_b,
                regions_a=regions_a,
                regions_b=regions_b,
            )

            self._render_probability_rasters(
                dataset_name=ds_a,
                rows=rows_a,
                cols=cols_a,
                probs=probs_a,
                regions=regions_a,
                output_directory=overlap_dir,
                suffix=f"{ds_b.lower()}_overlap",
            )
            self._render_probability_rasters(
                dataset_name=ds_b,
                rows=rows_b,
                cols=cols_b,
                probs=probs_b,
                regions=regions_b,
                output_directory=overlap_dir,
                suffix=f"{ds_a.lower()}_overlap",
            )

    def _resolve_reference_rasters(self, bundle: DatasetBundle) -> Dict[str, Path]:
        if bundle.reference_rasters:
            return bundle.reference_rasters
        meta_path = bundle.metadata_path
        if meta_path is None or not meta_path.exists():
            return {}
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta_doc = json.load(fh)
        except Exception as exc:
            print(f"[warn] Unable to read metadata for {bundle.name}: {exc}")
            return {}
        dataset_root = meta_path.parent
        features = meta_doc.get("features", {})
        entries = features.get("entries", {})
        for entry in entries.values():
            tifs = entry.get("tifs") or []
            for record in tifs:
                path_val = record.get("path") or record.get("href")
                if not path_val:
                    continue
                tif_path = Path(path_val)
                if not tif_path.is_absolute():
                    tif_path = (dataset_root / tif_path).resolve()
                if not tif_path.exists():
                    continue
                region = str(record.get("region") or "GLOBAL").upper()
                if region not in bundle.reference_rasters:
                    bundle.reference_rasters[region] = tif_path
        return bundle.reference_rasters

    def _render_probability_rasters(
        self,
        dataset_name: str,
        rows: np.ndarray,
        cols: np.ndarray,
        probs: np.ndarray,
        regions: np.ndarray,
        output_directory: Path,
        suffix: Optional[str],
    ) -> None:
        bundle = self.bundles.get(dataset_name)
        if bundle is None:
            return
        try:
            import rasterio
        except Exception as exc:
            print(f"[warn] rasterio not available; skipping GeoTIFF export for {dataset_name}: {exc}")
            return
        references = self._resolve_reference_rasters(bundle)
        if not references:
            print(f"[warn] No reference rasters located for {dataset_name}; skipping GeoTIFF export")
            return

        if regions is None or len(regions) == 0:
            region_codes = np.array(["GLOBAL"] * len(rows), dtype=object)
        else:
            region_codes = []
            for value in regions:
                if isinstance(value, str) and value.strip():
                    region_codes.append(value.strip().upper())
                else:
                    region_codes.append("GLOBAL")
            region_codes = np.asarray(region_codes, dtype=object)

        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
        probs = probs.astype(np.float32, copy=False)

        for region in np.unique(region_codes):
            ref_path = references.get(region) or references.get("GLOBAL")
            if ref_path is None and references:
                ref_path = next(iter(references.values()))
            if ref_path is None:
                print(f"[warn] {dataset_name}: no reference raster available for region {region}")
                continue
            try:
                with rasterio.open(ref_path) as src:
                    profile = src.profile.copy()
                    width = src.width
                    height = src.height
                    array = np.full((height, width), np.nan, dtype=np.float32)
            except Exception as exc:
                print(f"[warn] {dataset_name}: failed to open reference raster {ref_path}: {exc}")
                continue

            mask = region_codes == region
            if not np.any(mask):
                continue
            r_idx = rows[mask]
            c_idx = cols[mask]
            values = probs[mask]
            valid = (
                (r_idx >= 0)
                & (r_idx < height)
                & (c_idx >= 0)
                & (c_idx < width)
            )
            if values.ndim == 1:
                valid = valid & np.isfinite(values)
            if not np.any(valid):
                print(f"[warn] {dataset_name}: no valid pixels within bounds for region {region}")
                continue
            r_idx = r_idx[valid]
            c_idx = c_idx[valid]
            values = values[valid]

            width_int = int(width)
            flat_idx = r_idx * width_int + c_idx
            unique_flat, inverse = np.unique(flat_idx, return_inverse=True)
            sums = np.zeros(unique_flat.shape, dtype=np.float64)
            counts = np.zeros(unique_flat.shape, dtype=np.int64)
            np.add.at(sums, inverse, values)
            np.add.at(counts, inverse, 1)
            averaged = sums / np.maximum(counts, 1)
            row_unique = (unique_flat // width_int).astype(np.int64, copy=False)
            col_unique = (unique_flat % width_int).astype(np.int64, copy=False)
            array[row_unique, col_unique] = averaged.astype(np.float32, copy=False)

            profile.update({
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "driver": "GTiff",
            })
            if suffix:
                output_filename = f"probabilities_{region.lower()}_{suffix}.tif"
            else:
                output_filename = f"probabilities_{region.lower()}.tif"
            output_path = output_directory / output_filename
            try:
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(array, 1)
                print(f"[info] Wrote inference GeoTIFF for {dataset_name} ({region}) -> {output_path}")
            except Exception as exc:
                print(f"[warn] {dataset_name}: failed to write GeoTIFF {output_path}: {exc}")


def run_training(config_path: Path):
    try:
        from .config import load_config_1_int_FMs
    except ImportError:  # pragma: no cover
        from config import load_config_1_int_FMs  # type: ignore

    cfg = load_config_1_int_FMs(config_path)
    trainer = FMIntegrator(cfg)
    trainer.train()
