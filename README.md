## Quickstart

0) **STACify data** :
```bash
python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Data/GSC/2021_Table04_Datacube.csv   --collection-id gsc-2021  \
    --out /home/qubuntu25/Desktop/GitHub/1_Foundation_MVT_Result/   --title "GSC 2021 Table"   \
    --keywords GSC Datacube 2021 --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   \
    --features Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM Magnetic_LongWavelength_HGM_Worms_Proximity Geology_Period_Maximum_Majority Geology_Period_Minimum_Majority Geology_Lithology_Majority Geology_Lithology_Minority \
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --validate   --check-raster --check-raster-features Gravity_Bouguer_HGM_Worms_Proximity
```

0) **Inspect label** :
```bash

```