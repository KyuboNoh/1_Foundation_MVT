# Universal GFM4MPM Toolkit

Utilities for merging STAC outputs and integrating foundation models across multiple mineral prospectivity datasets.

## Contents

- `integrate_stac.py`: tools to merge per-dataset STAC exports into a unified metadata bundle and compute overlap products (GeoJSON/TIFF).

## STAC Integration
```bash
python -m Methods.1_Integrating_TwoDS.integrate_stac \
--collections /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/                /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/ \
--embedding-path /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/2_Labeling_01_10/embeddings.npz /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/2_Labeling_01_10/embeddings.npz \
--projectname 2_Integrate_MVT_gcs_bcgs_occ   --output /home/qubuntu25/Desktop/Research/Data \
--dataset-ids NA_AU BC   --region-select "{NA; GLOBAL}" \
--bridge-guess-number 1 --bridge "{Gravity_Bouguer, Gravity_Bouguer_HGM; NEBC_Canada_2_km___GRAV___Bouguer, NEBC_Canada_2_km___GRAV___Horizontal_Gradient}"   --visualize
```

```bash
python -m Methods.1_Integrating_TwoDS.integrate_stac \
--collections /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/                /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/ \
--embedding-path /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021-minocc/work/f21_2_10/2_Labeling_01_10/embeddings.npz /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down5/2_Labeling_01_10/embeddings.npz \
--projectname 2_Integrate_MVT_gcs_bcgs_occ_posaug   --output /home/qubuntu25/Desktop/Research/Data \
--dataset-ids NA_AU BC   --region-select "{NA; GLOBAL}" \
--bridge-guess-number 1 --bridge "{Gravity_Bouguer, Gravity_Bouguer_HGM; NEBC_Canada_2_km___GRAV___Bouguer, NEBC_Canada_2_km___GRAV___Horizontal_Gradient}"   --visualize  --use-positive-augmentation --debug
```


Running the command writes the blended metadata tree under `/tmp/output/Integrated_Project`, including:

- `study_area_overlap.geojson` and `study_area_overlap.tif` (1 inside the overlap window, 0 outside, in EPSG:8857).
- `data/<DATASET_A>_<DATASET_B>_overlap_pairs.json` with tile-level overlap records suitable for cross-view alignment.
- Per-bridge visualisations under `bridge_visualizations/`.
