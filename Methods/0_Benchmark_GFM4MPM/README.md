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
1) **Pretrain SSL encoder** (masked autoencoder):
```bash
python -m scripts.pretrain_ssl       --bands "/data/rasters/*.tif"       --out ./work/ssl       --patch 32 --epochs 30
```

2) **Build PU splits**:
```bash
python -m scripts.build_splits       --bands "/data/rasters/*.tif"       --pos_geojson deposits.geojson       --encoder ./work/ssl/mae_encoder.pth       --out ./work/splits       --filter_top_pct 0.10 --negs_per_pos 5
```

3) **Train classifier**:
```bash
python -m scripts.train_classifier       --bands "/data/rasters/*.tif"       --splits ./work/splits/splits.json       --encoder ./work/ssl/mae_encoder.pth       --epochs 60
```

4) **Maps & Uncertainty (MC‑Dropout)**:
```bash
python -m scripts.make_maps       --bands "/data/rasters/*.tif"       --encoder ./work/ssl/mae_encoder.pth       --mlp mlp_classifier.pth       --out ./work/prospectivity       --patch 32 --stride 16 --passes 30
```

5) **Explainability (IG)**:
```bash
python -m scripts.make_ig --bands "/data/rasters/*.tif"       --encoder ./work/ssl/mae_encoder.pth --mlp mlp_classifier.pth       --row 12345 --col 67890 --patch 32
```

## Notes
- Bring aligned GeoTIFF stacks and positive points (GeoJSON). Reprojection is out‑of‑scope here.
- For huge rasters, consider windowed/cached IO and precomputed global band stats.
