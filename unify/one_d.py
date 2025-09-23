from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)


def _apply_schema_hints(df: pd.DataFrame, schema_hints: Optional[Dict[str, str]]) -> pd.DataFrame:
    if not schema_hints:
        return df
    for column, hint in schema_hints.items():
        if column not in df.columns:
            _LOG.warning("Schema hint for missing column %s", column)
            continue
        hint_lower = hint.lower()
        if hint_lower in {"datetime", "timestamp"}:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
        elif hint_lower in {"int", "integer"}:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif hint_lower in {"float", "number"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif hint_lower in {"bool", "boolean"}:
            df[column] = df[column].astype("boolean")
        else:
            df[column] = df[column].astype("string")
    return df


def csv_to_parquet(csv_path: Path, out_parquet: Path, schema_hints: Optional[Dict[str, str]] = None) -> Path:
    csv_path = Path(csv_path)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _apply_schema_hints(df, schema_hints)

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_parquet)
    return out_parquet

