from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import one_d, vectors


def normalize_label_vectors(
    vector_path: Path,
    out_parquet: Path,
    target_epsg: str,
    class_property: str,
    additional_properties: Optional[List[str]] = None,
) -> Path:
    out_parquet = Path(out_parquet)
    label_gpq = vectors.to_geoparquet(vector_path, out_parquet, target_epsg)
    return label_gpq


def normalize_chip_labels(
    table_path: Path,
    out_parquet: Path,
    schema_hints: Optional[Dict[str, str]] = None,
) -> Path:
    return one_d.csv_to_parquet(table_path, out_parquet, schema_hints=schema_hints)


def build_label_metadata(
    class_property: str,
    classes: Iterable[str],
    tasks: Iterable[str],
    label_properties: Optional[Iterable[str]] = None,
) -> Dict:
    classes_list = list(classes)
    tasks_list = list(tasks)
    props_list = list(label_properties or [class_property])
    return {
        "label:properties": props_list,
        "label:classes": [
            {
                "name": class_property,
                "classes": classes_list,
            }
        ],
        "label:tasks": tasks_list,
    }

