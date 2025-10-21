## Quickstart

0) **STACify data** :
```bash
python stacify_GCS_data.py   --csv /home/qubuntu25/Desktop/Research/Data/GSC/2021_Table04_Datacube.csv \
    --collection-id gsc-2021_geol_10_seis_2_goce_5_grav5_mag_4  \
    --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/   --title "GSC 2021 Table"   \
    --keywords GSC Datacube 2021 --description "GSC 2021 Datacube Table"   --license "CC-BY-4.0"   \
    --features Dict_Sedimentary Dict_Igneous Dict_Metamorphic Terrane_Proximity Geology_PassiveMargin_Proximity Geology_BlackShale_Proximity Geology_Fault_Proximity Geology_Paleolatitude_Period_Maximum Geology_Paleolatitude_Period_Minimum Seismic_LAB_Hoggard Seismic_Moho  Gravity_GOCE_Differential Gravity_GOCE_MaximumCurve Gravity_GOCE_MinimumCurve Gravity_GOCE_MeanCurve Gravity_GOCE_ShapeIndex Gravity_Bouguer Gravity_Bouguer_HGM Gravity_Bouguer_HGM_Worms_Proximity Gravity_Bouguer_UpCont30km_HGM Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity Magnetic_HGM Magnetic_HGM_Worms_Proximity Magnetic_LongWavelength_HGM_Worms_Proximity \ 
    --lat-column Latitude_EPSG4326 \
    --lon-column Longitude_EPSG4326 \
    --validate   --check-raster --check-raster-features Gravity_Bouguer_HGM_Worms_Proximity
```


```bash
python stacify_bc_data.py  --raw-dir /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary/ --label /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary/NEBC_MVT_TP.shp  --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/ \
--license "CC-BY-4.0"   --keywords MVT British_Columbia \
--project-boundary /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_except2/NEBC_StudyArea_1.tif \
--cogify   --validate --collection-id Out_Data_Binary_Down30 --downsample-factor 30 

python stacify_bc_data.py  --raw-dir /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/ --label /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/NEBC_MVT_TP.shp  --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/ --license "CC-BY-4.0"   --keywords MVT British_Columbia --project-boundary /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/NEBC_StudyArea_1.tif --cogify   --validate --collection-id Out_Data_Binary_Geophy_Float_Down10 --downsample-factor 10 

python stacify_bc_data.py  --raw-dir /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/ --label /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/NEBC_MVT_TP.shp  --out /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/ --license "CC-BY-4.0"   --keywords MVT British_Columbia --project-boundary /home/qubuntu25/Desktop/Research/Data/BCGS_OF2024-11/Data_Binary_Geophy_Float/NEBC_StudyArea_1.tif --cogify   --validate --collection-id Out_Data_Binary_Geophy_Float_Down5 --downsample-factor 5

```