# GFM4MPM Benchmark Kit — methodology‑first (PyTorch)

**Goal:** Benchmark GFM4MPM methodology* (SSL masked image modeling + positive–unlabeled undersampling + small classifier with MC‑Dropout + IG attributions) on own raster/vector stacks.

## Repository layout
```
gfm4mpm_bench/
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/gfm4mpm/
│  ├─ data/geo_stack.py
│  ├─ models/mae_vit.py
│  ├─ models/mlp_dropout.py
│  ├─ sampling/likely_negatives.py
│  ├─ training/train_ssl.py
│  ├─ training/train_cls.py
│  ├─ infer/infer_maps.py
│  ├─ explain/ig_maps.py
│  └─ eval/metrics.py
└─ scripts/
   ├─ pretrain_ssl.py
   ├─ build_splits.py
   ├─ train_classifier.py
   ├─ eval_ablation_sparse.py
   ├─ make_maps.py
   └─ make_ig.py
```

## Quickstart

<!-- 0) **Inspect label** :
```bash
python -m scripts.inspect_labels \
  --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 \
  --out-json ./work/label_summary.json \
  --out-plot ./work/label_summary.png
``` -->

1) **Pretrain SSL encoder** (masked autoencoder):
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 --lat-column Latitude_EPSG4326 --lon-column Longitude_EPSG4326 --features Dict_Sedimentary Dict_Igneous Dict_Metamorphic Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Seismic_LAB_Hoggard Seismic_Moho Gravity_Bouguer Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM_Worms_Proximity --mask-ratio 0.75 --preview-samples 2 --lr 2.5e-4 --epochs 30 --out ./f21_Geol_9_GOCE_5_Seis_2_Grav_3_Mag_2 --check-feature Gravity_Bouguer_HGM_Worms_Proximity --patch 16 --window 224
```

2) **Load pretrained model and do only Inference** (masked autoencoder):
```bash
python -m scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 --lat-column Latitude_EPSG4326 --lon-column Longitude_EPSG4326 --mask-ratio 0.75 --preview-samples 2 --out ./f21_Geol_9_GOCE_5_Seis_2_Grav_3_Mag_2 --patch 16 --window 224 --button-inference
```

3) **Build PU splits**:
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.build_splits   --bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/assets/rasters/2021_Table04_Datacube_selected_Norm_*.tif"   --pos_geojson /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/assets/Training_MVT_Deposit_NA.geojson   --encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_Geol_9_GOCE_5_Seis_2_Grav_3_Mag_2/mae_encoder.pth   --out ./work/splits_raster_na   --filter_top_pct 0.10   --negs_per_pos 5   --validation
```

##################################################################################################################################################

3) **Train classifier**:
```bash
python -m scripts.train_classifier \
    --bands "/data/rasters/*.tif" \
    --splits ./work/splits/splits.json \
    --encoder ./work/ssl/mae_encoder.pth \
    --epochs 60
```

4) **Maps & Uncertainty (MC‑Dropout)**:
```bash
python -m scripts.make_maps \
    --bands "/data/rasters/*.tif" \
    --encoder ./work/ssl/mae_encoder.pth \
    --mlp mlp_classifier.pth \
    --out ./work/prospectivity \
    --patch 32 \
    --stride 16 \
    --passes 30
```

5) **Explainability (IG)**:
```bash
python -m scripts.make_ig \
    --bands "/data/rasters/*.tif" \
    --encoder ./work/ssl/mae_encoder.pth \
    --mlp mlp_classifier.pth \
    --row 12345 \
    --col 67890 \
    --patch 32
```

### Pretraining knobs & telemetry
- `--mask-ratio` (default 0.75) determines the fraction of MAE tokens hidden each step.
- `--encoder-depth` / `--decoder-depth` configure the number of transformer blocks (defaults 6 / 2).
- `--patch` sets the spatial window sampled from rasters (defaults 16). Random window sampling is used—no sliding window tiling occurs during SSL.
- `--optimizer` (`adamw` or `adam`), `--lr`, and `--batch` expose the optimiser family, learning rate, and batch size.
- Each run writes `ssl_history.json` alongside the encoder checkpoint, capturing epoch-wise reconstruction loss, mean absolute error, PSNR, and SSIM (SSIM reported only when patches are larger than 1×1, so STAC table runs will show `null`).
- `--preview-samples` (default 0) saves side-by-side original/masked/reconstructed panels for the requested number of samples in `out/previews/` (one PNG per sample with rows for each feature/channel).

## Working with GSC STAC tables
The helper scripts can also ingest the STAC directory produced by
`stacify_GCS_data.py` (Parquet table). In this mode each table row becomes a
1×1 "patch" whose channels are the numeric feature columns, and outputs are
written as CSV instead of GeoTIFFs.

Example using `/home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/` and focusing on the
`Magnetotelluric` + `HeatFlow` predictors with the `Training_MVT_Occurrence`
label:

```bash
# 1) Pretrain (patch size forced to 1 for tables)
python -m scripts.pretrain_ssl \
    --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 \
    --features Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --mask-ratio 0.75 \
    --encoder-depth 6 \
    --decoder-depth 2 \
    --preview-samples 3 \
    --window 16 \
    --optimizer adamw \
    --lr 2.5e-4 \
    --batch 128 \
    --epochs 2 \
    --out ./work/test


python -m scripts.pretrain_ssl \
    --stac-root /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021 \
    --features Terrane_Majority Terrane_Minority Terrane_Contact Terrane_Proximity Geology_Eon_Maximum_Majority Geology_Eon_Maximum_Minority Geology_Eon_Minimum_Majority Geology_Eon_Minimum_Minority Geology_Era_Maximum_Majority Geology_Era_Maximum_Minority Geology_Era_Minimum_Majority Geology_Era_Minimum_Minority Geology_Period_Maximum_Majority Geology_Period_Maximum_Minority Geology_Period_Minimum_Majority Geology_Period_Minimum_Minority Geology_Period_Contact Geology_Lithology_Majority Geology_Lithology_Minority Geology_Lithology_Contact Geology_Dictionary_Alkalic Geology_Dictionary_Anatectic Geology_Dictionary_Calcareous Geology_Dictionary_Carbonaceous Geology_Dictionary_Cherty Geology_Dictionary_CoarseClastic Geology_Dictionary_Evaporitic Geology_Dictionary_Felsic Geology_Dictionary_FineClastic Geology_Dictionary_Gneissose Geology_Dictionary_Igneous Geology_Dictionary_Intermediate Geology_Dictionary_Pegmatitic Geology_Dictionary_RedBed Geology_Dictionary_Schistose Geology_Dictionary_Sedimentary Geology_Dictionary_UltramaficMafic Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_CoverThickness Geology_Paleolongitude_Period_Maximum Geology_Paleolongitude_Period_Minimum Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_LAB_Priestley Seismic_Moho Seismic_Moho_GEMMA Seismic_Moho_Szwillus Seismic_Velocity_050km Seismic_Velocity_100km Seismic_Velocity_150km Seismic_Velocity_200km Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_BGI Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_RTP Magnetic_EMAG2v3 Magnetic_EMAG2v3_CuriePoint Magnetic_1VD Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity HeatFlow Magnetotelluric Litmod_Density_Asthenosphere Litmod_Density_Crust Litmod_Density_Lithosphere Crust1_Type Crust1_CrustalThickness Crust1_SedimentThickness \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --mask-ratio 0.75 \
    --encoder-depth 6 \
    --decoder-depth 2 \
    --window 16 \
    --preview-samples 16 \
    --optimizer adamw \
    --lr 2.5e-4 \
    --batch 128 \
    --epochs 2 \
    --out ./work/test

# 2) Build PU splits from the Training_MVT_Occurrence label
python -m scripts.build_splits \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube.parquet \
    --features Magnetotelluric HeatFlow \
    --label-column Training_MVT_Occurrence \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --encoder ./work/ssl_table/mae_encoder.pth \
    --out ./work/splits_table

# 3) Train classifier using the generated splits
python -m scripts.train_classifier \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --splits ./work/splits_table/splits.json \
    --encoder ./work/ssl_table/mae_encoder.pth

# 4) Prospectivity scores saved to CSV (adding per-row metadata)
python -m scripts.make_maps \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --encoder ./work/ssl_table/mae_encoder.pth \
    --mlp mlp_classifier.pth \
    --out ./work/prospectivity_table

# 5) Integrated gradients per feature
python -m scripts.make_ig \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --encoder ./work/ssl_table/mae_encoder.pth \
    --mlp mlp_classifier.pth \
    --index 0

# (Optional) Inspect label columns and create a prevalence plot
python -m scripts.inspect_labels \
    --stac-root /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021 \
    --out-json ./work/label_summary.json \
    --out-plot ./work/label_summary.png

# (Optional) Plot SSL training metrics
python -m scripts.plot_ssl_history \
    ./work/ssl_table/ssl_history.json \
    --out ./work/ssl_table/ssl_metrics.png
```

Notes:
- String features are automatically expanded to one-hot columns when present, so mixed numeric/categorical selections (e.g., `Terrane_Majority`) now work out-of-the-box.
- Label column defaults to `Training_MVT_Deposit`; override with `--label-column` (e.g., `Training_MVT_Occurrence`) and ensure it actually contains `Present` entries.
- Outputs from `make_maps`/`make_ig` become CSV files for easier inspection.
- STAC table runs operate on 1×1 pseudo-patches, so SSIM will be reported as `null` in `ssl_history.json`.

## Notes
- Bring aligned GeoTIFF stacks and positive points (GeoJSON). Reprojection is out‑of‑scope here.
- For huge rasters, consider windowed/cached IO and precomputed global band stats.
