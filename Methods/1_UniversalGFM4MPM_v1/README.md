# Universal GFM4MPM v1 (overlap alignment sandbox)

The `v1` refresh introduces a modular overlap-alignment workspace that separates
data ingestion, overlap reasoning, and (future) optimisation. The immediate goal
is to understand how pre-computed embeddings overlap across datasets before
implementing the training loop.

## Layout

- `overlap_alignment/config.py` — JSON schema loader (backwards-compatible with `v0`).
- `overlap_alignment/datasets.py` — embedding loader exposing region / window / resolution metadata.
- `overlap_alignment/overlaps.py` — parser for `*_overlap_pairs.json` with robust handling of schema variants.
- `overlap_alignment/workspace.py` — workspace that pairs embeddings for co-located tiles.
- `overlap_alignment/cli.py` — quick diagnostics over the loaded workspace.

## Quick start

```bash
python -m Methods.1_UniversalGFM4MPM_v1.overlap_alignment.cli --config /home/qubuntu25/Desktop/Research/Data/2_UFM_v1/config_ufm_v1_debug.json
```

The command prints dataset summaries (counts, label balance, region coverage,
pixel resolution / window stats) and describes how many overlap entries were
resolved into co-located embedding pairs. Use `--json` to emit a machine-friendly report.

If you provide `integration_metadata_path` in the config, the workspace will
pull dataset-level fallbacks (e.g. min resolution, window spacing) from the
STAC-integrated `combined_metadata.json` whenever per-tile metadata is missing
inside the embedding bundles. When a `log_dir` (or `output_dir`) is supplied,
the CLI also emits `overlap_pairs.json` that enumerates every resolved overlap
pair together with the label type (`positive_common`, `positive_<dataset>`, or
`unlabelled`) and the source indices for each dataset.

## Stage-1 Overlap Alignment Training (positive-only)

The stage-1 trainer freezes encoder features and optimises only the projection
heads using a positive-only, negative-free alignment loss.

```bash
python -m Methods.1_UniversalGFM4MPM_v1.overlap_alignment.train --config /home/qubuntu25/Desktop/Research/Data/2_UFM_v1/config_ufm_v1_debug.json --debug --use-positive-only --use-positive-augmentation
```

By default the trainer:

- Groups fine-scale tiles into coarse windows via weighted pooling (settable
  with `--aggregator`).
- Optimises a Deep CCA objective (overrideable via `--objective`; `barlow`
  raises a “not implemented” placeholder) and reports per-epoch loss / mean
  correlation with a progress bar when `tqdm` is available.
- Learns small projection heads with configurable batch size, learning rate,
  and projection dimension (overridable via CLI flags or config).
- Auto-selects `--max-coord-error` when not provided, using the coarser tile
  spacing/resolution, and exposes the value in the saved summary.
- `--debug` saves a centroid scatter (`bridge_visualizations/overlap/debug_positive_overlap.png`,
  requires matplotlib) alongside the standard logs.

Outputs:
- `overlap_alignment_stage1.pt` – projection-head weights plus an experiment
  summary.
- `overlap_alignment_stage1_metrics.json` – JSON dump of the summary and the
  full epoch history (loss, mean canonical correlation) for downstream
  analysis.

## Next steps

- Implement overlap alignment losses on top of `OverlapAlignmentWorkspace.iter_pairs`.
- Extend overlap diagnostics with STAC-derived masks (e.g., study-area rasters).
- Plug the workspace into a modular training stack (prototype heads, InfoNCE, etc.).
