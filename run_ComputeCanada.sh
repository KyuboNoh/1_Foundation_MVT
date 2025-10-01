#!/bin/bash
#SBATCH --account=rrg-aswidins
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=30G                   # memory; default unit is megabytes
#SBATCH --time=3-00:00              # time (DD-HH:MM)

#salloc --time=1:0:0 --mem-per-cpu=30G --nodes=1 --gpus-per-node=2 --account=rrg-aswidins

module purge

module load StdEnv/2023
module load cuda/12.9 

module load gcc arrow/21.0.0 
module load python/3.11

# --- Virtualenv (create once; afterwards you can reuse it) ---
if [[ ! -d .venv ]]; then
  # create venv AFTER loading Arrow/GDAL so their env leaks into venv
  virtualenv --no-download .venv
fi
source .venv/bin/activate

#python -m pip install -U pip wheel 
pip install -r requirements.txt

# Quick sanity check (prints to job log)
python - <<'PY'
import torch, pyarrow, rasterio, numpy
print("CUDA avail:", torch.cuda.is_available(), "Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("pyarrow:", pyarrow.__version__, "numpy:", numpy.__version__, "rasterio:", rasterio.__version__)
PY

# --- Run your code below ---
DATA_DIR="/home/kyubonoh/scratch/Data/1_Foundation_MVT_Result/gsc-2021"

cd Methods/0_Benchmark_GFM4MPM
srun python -m scripts.pretrain_ssl   --stac-root "${DATA_DIR}" \
--lat-column Latitude_EPSG4326   --lon-column Longitude_EPSG4326       \
--mask-ratio 0.75  --patch 16   --window 224 --lr 2.5e-4  \
--preview-samples 5   --out ./Default   --epochs 40