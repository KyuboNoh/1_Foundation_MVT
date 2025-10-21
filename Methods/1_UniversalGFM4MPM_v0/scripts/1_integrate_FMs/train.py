from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
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
            )
            self.bundles[ds_cfg.name] = bundle

        if len(self.bundles) >= 2:
            emb_dim = next(iter(self.bundles.values())).dataset.embedding_dim
            self.domain_disc = DomainDiscriminator(emb_dim).to(self.device)

    def _prepare_overlap_pairs(self) -> None:
        self.overlap_pairs: List[Tuple[str, str]] = []
        if self.cfg.overlap_pairs_path and self.cfg.overlap_pairs_path.exists():
            try:
                with self.cfg.overlap_pairs_path.open("r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                for row in doc:
                    a = row.get("dataset_a_tile")
                    b = row.get("dataset_b_tile")
                    if a and b:
                        self.overlap_pairs.append((a, b))
            except Exception as exc:
                print(f"[warn] Failed to read overlap pairs: {exc}")
                self.overlap_pairs = []

        self.tile_lookup: Dict[str, Tuple[str, torch.Tensor, int]] = {}
        for bundle in self.bundles.values():
            for idx, rec in enumerate(bundle.dataset.records):
                emb = torch.from_numpy(rec.embedding).float()
                self.tile_lookup[rec.tile_id] = (bundle.name, emb, rec.label)

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

    def _gather_overlap_batch(self, batch_size: int = 128):
        if not self.overlap_pairs:
            return None
        pairs = random.sample(self.overlap_pairs, min(batch_size, len(self.overlap_pairs)))
        anchor_embeddings = []
        target_embeddings = []
        anchor_logits = []
        target_logits = []
        for tile_a, tile_b in pairs:
            if tile_a not in self.tile_lookup or tile_b not in self.tile_lookup:
                continue
            ds_a, emb_a, label_a = self.tile_lookup[tile_a]
            ds_b, emb_b, label_b = self.tile_lookup[tile_b]
            bundle_a = self.bundles.get(ds_a)
            bundle_b = self.bundles.get(ds_b)
            if bundle_a is None or bundle_b is None:
                continue
            z_a = bundle_a.projector(emb_a.to(self.device))
            z_b = bundle_b.projector(emb_b.to(self.device))
            anchor_embeddings.append(z_b)
            target_embeddings.append(z_a)
            anchor_logits.append(bundle_b.head(z_b))
            target_logits.append(bundle_a.head(z_a))
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


def run_training(config_path: Path):
    try:
        from .config import load_config
    except ImportError:  # pragma: no cover
        from config import load_config  # type: ignore

    cfg = load_config(config_path)
    trainer = FMIntegrator(cfg)
    trainer.train()
