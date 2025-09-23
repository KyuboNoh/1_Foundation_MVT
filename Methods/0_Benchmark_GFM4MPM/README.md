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

0) **Inspect label** :
```bash
python -m scripts.inspect_labels \
  --stac-root /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021 \
  --out-json ./work/label_summary.json \
  --out-plot ./work/label_summary.png
```


1) **Pretrain SSL encoder** (masked autoencoder):
```bash
python -m scripts.pretrain_ssl \
  --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
  --features Magnetotelluric HeatFlow \
  --lat-column Latitude_EPSG4326 \
  --lon-column Longitude_EPSG4326 \
  --out ./work/ssl_table \
  --epochs 30
```

2) **Build PU splits**:
```bash
python -m scripts.build_splits \
    --bands "/data/rasters/*.tif" \
    --pos_geojson deposits.geojson \
    --encoder ./work/ssl/mae_encoder.pth \
    --out ./work/splits \
    --filter_top_pct 0.10 \
    --negs_per_pos 5
```

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
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --out ./work/ssl_table \
    --epochs 30

# 2) Build PU splits from the Training_MVT_Occurrence label
python -m scripts.build_splits \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
    --features Magnetotelluric HeatFlow \
    --label-column Training_MVT_Occurrence \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --encoder ./work/ssl_table/mae_encoder.pth \
    --out ./work/splits_table

# 3) Train classifier using the generated splits
python -m scripts.train_classifier \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --splits ./work/splits_table/splits.json \
    --encoder ./work/ssl_table/mae_encoder.pth

# 4) Prospectivity scores saved to CSV (adding per-row metadata)
python -m scripts.make_maps \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
    --features Magnetotelluric HeatFlow \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --encoder ./work/ssl_table/mae_encoder.pth \
    --mlp mlp_classifier.pth \
    --out ./work/prospectivity_table

# 5) Integrated gradients per feature
python -m scripts.make_ig \
    --stac-table /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/gsc-2021/assets/tables/2021_Table04_Datacube_temp.parquet \
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
```

Notes:
- Only numeric columns are used as features; provide `--features` to pick an explicit subset.
- Label column defaults to `Training_MVT_Deposit`; override with `--label-column` (e.g., `Training_MVT_Occurrence`) and ensure it actually contains `Present` entries.
- Outputs from `make_maps`/`make_ig` become CSV files for easier inspection.

## Notes
- Bring aligned GeoTIFF stacks and positive points (GeoJSON). Reprojection is out‑of‑scope here.
- For huge rasters, consider windowed/cached IO and precomputed global band stats.
