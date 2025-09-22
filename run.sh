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