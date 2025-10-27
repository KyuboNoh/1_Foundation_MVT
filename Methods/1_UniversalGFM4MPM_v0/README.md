# Universal GFM4MPM Toolkit

Integrating foundation models across multiple mineral prospectivity datasets.

## Contents

- `integrate_stac.py`: tools to merge per-dataset STAC exports into a unified metadata bundle and compute overlap products (GeoJSON/TIFF).
- `scripts/1_integrate_FMs/`: experimental pipeline for integrating dataset-specific foundation models using pre-computed embeddings.


Running the command writes the blended metadata tree under `/tmp/output/Integrated_Project`, including:

- `study_area_overlap.geojson` and `study_area_overlap.tif` (1 inside the overlap window, 0 outside, in EPSG:8857).
- `data/<DATASET_A>_<DATASET_B>_overlap_pairs.json` with tile-level overlap records suitable for cross-view alignment.
- Per-bridge visualisations under `bridge_visualizations/`.

## Foundation Model Integration

The FM integration pipeline consumes embeddings derived from pretrained MAE-ViT models and trains lightweight heads to align prospectivity predictions across datasets.

1. Prepare a JSON configuration (example `config_example.json`):

2. Launch training:

```bash
python Methods/1_UniversalGFM4MPM_v0/scripts/1_integrate_FMs/integrate_fms.py \
--config /home/qubuntu25/Desktop/Research/Data/2_UFM_v0/config_fm_integration_debug.json \
--use-previous-negatives --debug  --inference
```

The training loop performs:

- **Phase 1**: positive-manifold supervision (SupCon + prototype losses) and nnPU classification per dataset.
- **Phase 2**: cross-view alignment via InfoNCE/ KL losses on overlapping tiles (skipped when no overlap pairs are available).
- **Phase 3**: domain adaptation with gradient reversal, optional MMD, and shared mask-aware clipping using `study_area_overlap.tif`.

Outputs are stored in `output_dir/<dataset_name>/` as projection heads, classification heads, and prototypes, alongside a `state.json` summary.

> **Input expectations**: `.npz` embedding files should contain `embeddings`, `labels`, and optionally `tile_ids`, `coords`, and `metadata` arrays.

### Overlap Pairs JSON Structure

`integrate_stac.py` writes per-dataset overlap lists (stored in the project `data/` directory). Each record contains the intersecting tiles and the size of their shared footprint:

Point `overlap_pairs_path` in the FM integration configuration to one of these files when cross-view alignment should be enabled.
