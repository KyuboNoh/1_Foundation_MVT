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
python -m Methods.0_Benchmark_GFM4MPM.scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 --lat-column Latitude_EPSG4326 --lon-column Longitude_EPSG4326 --features Dict_Sedimentary Dict_Igneous Dict_Metamorphic Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Seismic_LAB_Hoggard Seismic_Moho Gravity_Bouguer Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM_Worms_Proximity --mask-ratio 0.75 --preview-samples 2 --lr 5.0e-4 --epochs 50 --out ./f21_2_10 --check-feature Gravity_Bouguer_HGM_Worms_Proximity --patch 2 --window 10 --preview-window-centers  --ssim
```

```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.pretrain_ssl   --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Down30/   --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Down30/1_SSL_re   --mask-ratio 0.75 --preview-samples 2 --lr 5.0e-4 --epochs 30 --check-feature Mag_RTF_Binary --patch 6 --window 36 --preview-window-centers --ssim
```
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.pretrain_ssl   --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/   --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/1_SSL_re   --mask-ratio 0.75 --preview-samples 2 --lr 5.0e-4 --epochs 30 --patch 6 --window 36 --preview-window-centers --ssim
```

```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.pretrain_ssl   --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/   --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/1_SSL_re   --mask-ratio 0.75 --preview-samples 2 --lr 5.0e-4 --epochs 30 --patch 2 --window 10 --preview-window-centers --ssim
```

2) **Load pretrained model and do only Inference** (masked autoencoder):
```bash
python -m scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021 --lat-column Latitude_EPSG4326 --lon-column Longitude_EPSG4326 --mask-ratio 0.75 --preview-samples 2 --out ./f21_2_10 --patch 16 --window 224 --button-inference
```

```bash
python -m scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Down30 --mask-ratio 0.75 --preview-samples 2 --out ./f21_Geol_9_GOCE_5_Seis_2_Grav_3_Mag_2 --patch 6 --window 36 --button-inference
```
```bash
python -m scripts.pretrain_ssl --stac-root /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Down30 --mask-ratio 0.75 --preview-samples 2 --out ./f21_Geol_9_GOCE_5_Seis_2_Grav_3_Mag_2 --patch 6 --window 36 --button-inference
```



3) **Build PU splits**:
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.build_splits \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/assets/rasters/2021_Table04_Datacube_selected_Norm_*.tif" \
--encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/1_SSL/mae_encoder.pth \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/2_Labeling_01_10 \
--pos_geojson /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/assets/labels/geojson/Training_MVT_Occurrence.geojson \
--filter_top_pct 0.10   --negs_per_pos 10 --positive-augmentation --debug 
```


```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.build_splits \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/assets/rasters/*.tif" \
--encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/1_SSL_re/mae_encoder.pth \
--pos_geojson /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/assets/labels/geojson/label_value.geojson \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/2_Labeling_01_10 \
--filter_top_pct 0.10   --negs_per_pos 10 --positive-augmentation --debug
```
##################################################################################################################################################

3) **Train classifier**:
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.train_classifier \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/assets/rasters/2021_Table04_Datacube_selected_Norm_*.tif" \
--step1 /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/1_SSL/ \
--step2 /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/2_Labeling_01_10/ \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/3_cls_01_10 \
--save-prediction --epochs 40 --test-ratio 0.3 --stride 10 --positive-augmentation 
```
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.train_classifier \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/assets/rasters/2021_Table04_Datacube_selected_Norm_*.tif" \
--encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/1_SSL/mae_encoder.pth \
--splits /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/2_Labeling_01_10/splits.json \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/3_cls_01_10 \
--save-prediction --epochs 20 --test-ratio 0.3
```


```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.train_classifier \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/assets/rasters/*.tif" \
--step1 /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/1_SSL_re/ \
--step2 /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/2_Labeling_01_10/ \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/3_cls_01_10 \
--save-prediction --epochs 40 --test-ratio 0.3 --stride 10 --positive-augmentation 
```




4) **Plot again**:
```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.plot_prospectivity \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/assets/rasters/2021_Table04_Datacube_selected_Norm_*.tif" \
--encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/1_SSL/mae_encoder.pth \
--splits /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/2_Labeling_01_10/splits.json \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/3_cls_01_10 \
--prediction-glob "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/work/f21_2_10/3_cls_01_10/*_predictions.npy" 
```

```bash
python -m Methods.0_Benchmark_GFM4MPM.scripts.plot_prospectivity \
--bands "/home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/assets/rasters/*.tif" \
--encoder /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/1_SSL_re/mae_encoder.pth \
--splits /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/2_Labeling_01_10/splits.json \
--out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/3_cls_01_10 \
--prediction-glob /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/3_cls_01_10/prediction_predictions.npy 
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
`Gravity_Bouguer` predictors with the `Training_MVT_Occurrence`
label:

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
