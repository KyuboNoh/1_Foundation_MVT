# 1) Use Python 3.11
#pyenv local 3.11.9      # creates .python-version here
pyenv local 3.11
 python3.11 -V

# 2) Create & activate venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# 3) Install dependencies
pip install -r requirements.txt

# 4) run in UI mode
#python app.py --ui


# --- Run your code below ---
DATA_DIR="/home/kyubonoh/scratch/Data/1_Foundation_MVT_Result/gsc-2021"

cd Methods/0_Benchmark_GFM4MPM
srun python -m scripts.pretrain_ssl   --stac-root "${DATA_DIR}" \
--lat-column Latitude_EPSG4326   --lon-column Longitude_EPSG4326       \
--mask-ratio 0.75  --patch 16   --window 224 --lr 2.5e-4  \
--preview-samples 5   --out ./Default   --epochs 40

