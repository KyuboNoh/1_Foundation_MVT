# Universal GFM4MPM v1 (DCCA overlap alignment + CLS training)

## Layout

- `scripts/train.py` — DCCA and CLS training.
- `scripts/cli.py` — quick diagnostics over the loaded workspace.
- `scripts/config.py` — JSON schema loader (updated format from `v0`).

## Stage-1 Overlap Alignment Training 

The stage-1 trainer freezes encoder features and optimises only the projection
heads using a positive-only, negative-free alignment loss.

```bash
python -m Methods.1_UniversalGFM4MPM_v1.scripts.train --config ./ufm_v1_config_wsl_debug.json --debug
```

## Stage-2 Train CLS after Overlap Alignment Training

```bash
python -m Methods.1_UniversalGFM4MPM_v1.overlap_alignment.train --config ./ufm_v1_config_wsl_debug.json --train-cls --read-dcca --no-train-dcca --dcca-weights-path /home/wslqubuntu24/Research/Data/1_Foundation_MVT_Result/2_UFM_v1/Ex1_DCCA_dim128/overlap_alignment_stage1_dcca.pt --debug
```

## Stage-1+2 Overlap Alignment Training (positive-only)

```bash
python -m Methods.1_UniversalGFM4MPM_v1.overlap_alignment.train --config ./ufm_v1_config_wsl_debug.json --train-cls --debug 
```

By default the trainer:

- Groups fine-scale tiles into coarse windows via weighted pooling (settable with `--aggregator`).
- Optimises a Deep CCA objective (overrideable via `--objective`; `barlow`
  raises a “not implemented” placeholder) and reports per-epoch loss / mean
  correlation with a progress bar when `tqdm` is available.
- Learns small projection heads with configurable batch size, learning rate,
  and projection dimension (overridable via CLI flags or config).
- Auto-selects `--max-coord-error` when not provided, using the coarser tile
  spacing/resolution, and exposes the value in the saved summary.
- `--debug` saves a centroid scatter (`overlap_visualizations/overlap/debug_positive_overlap.png`,
  requires matplotlib) alongside the standard logs.

Outputs:
- `overlap_alignment_stage1.pt` – projection-head weights plus an experiment summary.
- `overlap_alignment_stage1_metrics.json` – JSON dump of the summary and the
  full epoch history (loss, mean canonical correlation) for downstream analysis.

## Overlap Utility Helpers

Several helper functions in `overlap_alignment/train.py` are being prepared for
promotion into `Common/Unifying/Labels_TwoDatasets` so v2 can reuse them:

- `align_overlap_embeddings_for_pn_one_to_one(z_a, z_b)` (now published under
  `Common/Unifying/Labels_TwoDatasets/fusion_utils.py`) trims the re-embedding payloads
  down to the spatial overlap and aligns their PN labels, emitting paired
  `z_a ∈ ℝ^{N×d_a}` / `z_b ∈ ℝ^{N×d_b}` tensors plus consistent `pair_ids`.
- `prepare_fusion_overlap_dataset_one_to_one(...)` consumes the aligned payloads and
  constructs the fused feature table `φ(u,v)` used to train the PN head. It
  supports both the simple `[u; v]` template and the strong
  `[u; v; |u−v|; u⊙v; cos(u,v); m]` template, bundling metadata/labels alongside
  the fused matrix.
- `prepare_fusion_overlap_dataset_for_inference(...)` mirrors the previous
  function but emits NumPy tensors and lookup tables suited for inference over
  raster grids. It works with method_id `v1_simple` or `v1_strong` for this package. 

These descriptions should be kept in sync once the helpers move into the shared
module so downstream tooling (v2 transformer aggregators, distillation logic,
etc.) can discover their expected inputs and outputs quickly.

## Next steps

- Implement overlap alignment losses on top of `OverlapAlignmentWorkspace.iter_pairs`.
- Extend overlap diagnostics with STAC-derived masks (e.g., study-area rasters).
- Plug the workspace into a modular training stack (prototype heads, InfoNCE, etc.).
