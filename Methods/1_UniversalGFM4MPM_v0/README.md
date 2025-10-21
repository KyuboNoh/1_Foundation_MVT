# Universal GFM4MPM Toolkit

Utilities for merging STAC outputs and integrating foundation models across multiple mineral prospectivity datasets.

## Contents

- `integrate_stac.py`: tools to merge per-dataset STAC exports into a unified metadata bundle and compute overlap products (GeoJSON/TIFF).
- `scripts/1_integrate_FMs/`: experimental pipeline for integrating dataset-specific foundation models using pre-computed embeddings.

## STAC Integration

```bash
python -m Methods.1_UniversalGFM4MPM_v0.integrate_stac \
--collections /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/                /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/ \
--projectname 2_Integrate_MVT_gcs_bcgs   --output /home/qubuntu25/Desktop/Research/Data/ \
--dataset-ids NA_AU BC   --region-select "{NA; GLOBAL}"   --bridge-guess-number 1 \
--bridge "{Gravity_Bouguer, Gravity_Bouguer_HGM; NEBC_Canada_2_km___GRAV___Bouguer, NEBC_Canada_2_km___GRAV___Horizontal_Gradient}"   --visualize
```

Running the command writes the blended metadata tree under `/tmp/output/Integrated_Project`, including:

- `study_area_overlap.geojson` and `study_area_overlap.tif` (1 inside the overlap window, 0 outside, in EPSG:8857).
- `data/<DATASET_A>_<DATASET_B>_overlap_pairs.json` with tile-level overlap records suitable for cross-view alignment.
- Per-bridge visualisations under `bridge_visualizations/`.

## Foundation Model Integration

The FM integration pipeline consumes embeddings derived from pretrained MAE-ViT models and trains lightweight heads to align prospectivity predictions across datasets.

1. Prepare a JSON configuration (example `config_example.json`):

```json
{
  "output_dir": "/home/qubuntu25/Desktop/Research/Data/2_Integrate_MVT_gcs_bcgs//experiments/fm_integration",
  "log_dir": "/home/qubuntu25/Desktop/Research/Data/2_Integrate_MVT_gcs_bcgs//experiments/fm_integration/logs",
  "datasets": [
    {
      "name": "NA_AU",
      "embedding_path": "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/2_Labeling_01_10/embeddings.npy",
      "metadata_path":  "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/training_metadata.json",
      "region_filter": ["NA"],
      "class_prior": 0.12
    },
    {
      "name": "BC",
      "embedding_path": "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/2_Labeling_01_10/embeddings.npy",
      "metadata_path":  "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/training_metadata.json",
      "region_filter": ["GLOBAL"],
      "class_prior": 0.08
    }
  ],
  "overlap_pairs_path": "/home/qubuntu25/Desktop/Research/Data/2_Integrate_MVT_gcs_bcgs//data/BC_NA_AU_overlap_pairs.json",
  "overlap_mask_path": "/home/qubuntu25/Desktop/Research/Data/2_Integrate_MVT_gcs_bcgs//data/study_area_overlap.tif",
  "device": "cuda",
  "optimization": {"batch_size": 256, "epochs": 50, "lr": 1e-4}
}
```

2. Launch training:

```bash
python Methods/1_UniversalGFM4MPM_v0/scripts/1_integrate_FMs/integrate_fms.py --config /home/qubuntu25/Desktop/Research/Data/2_Integrate_MVT_gcs_bcgs/config.json
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
