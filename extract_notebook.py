
import json
from pathlib import Path

notebook_path = Path(r"Methods/1_UniversalGFM4MPM_v2_module/scripts/summarise_meta_eval.ipynb")
output_path = Path(r"Methods/1_UniversalGFM4MPM_v2_module/scripts/summarise_meta_eval.py")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        code_cells.append(source)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n\n# %% [markdown]\n# Extracted from summarise_meta_eval.ipynb\n\n")
    for i, code in enumerate(code_cells):
        f.write(f"# %% [code] Cell {i}\n")
        f.write(code)
        f.write("\n\n")

print(f"Extracted {len(code_cells)} code cells to {output_path}")
