activate venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

Check
where python
python -V
python -m pip -V

# --- Run your code below ---
DATA_DIR="/home/kyubonoh/scratch/Data/1_Foundation_MVT_Result/gsc-2021"

cd Methods/0_Benchmark_GFM4MPM
srun python -m scripts.pretrain_ssl   --stac-root "${DATA_DIR}" \
--lat-column Latitude_EPSG4326   --lon-column Longitude_EPSG4326       \
--mask-ratio 0.75  --patch 16   --window 224 --lr 2.5e-4  \
--preview-samples 5   --out ./Default   --epochs 40