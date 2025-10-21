<!-- python -m Methods.1_UniversalGFM4MPM_v0.integrate_stac   --collections /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/                /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/   --projectname 2_Integrate_MVT_gcs_bcgs   --output /home/qubuntu25/Desktop/Research/Data/   --dataset-ids NA_AU BC   --region-select "{NA; GLOBAL}"   --bridge-guess-number 1   --bridge "{Gravity_Bouguer, Gravity_Bouguer_HGM; NEBC_Canada_2_km___GRAV___Bouguer, NEBC_Canada_2_km___GRAV___Horizontal_Gradient}"   --visualize   --crs-input EPSG6933 EPSG3005 -->
 

python -m Methods.1_UniversalGFM4MPM_v0.integrate_stac \
  --collections /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/gsc-2021/ /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/Out_Data_Binary_Geophy_Float_Down10/ \
  --projectname 2_Integrate_MVT_gcs_bcgs \
  --output /home/qubuntu25/Desktop/Research/Data/ \
  --dataset-ids NA_AU BC \
  --region-select "{NA; GLOBAL}" \
  --bridge-guess-number 1 \
  --bridge "{Gravity_Bouguer, Gravity_Bouguer_HGM; NEBC_Canada_2_km___GRAV___Bouguer, NEBC_Canada_2_km___GRAV___Horizontal_Gradient}" \
  --visualize

