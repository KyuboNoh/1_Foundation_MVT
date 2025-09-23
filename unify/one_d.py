from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)


_DEFAULT_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "latin1",
    "ISO-8859-1",
    "cp1252",
)


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

# TODO: later implement this for other csv readers
def _read_csv_with_fallback(csv_path: Path, read_csv_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load a CSV, trying a cascade of encodings when none is specified."""
    kwargs = dict(read_csv_kwargs or {})
    if "encoding" in kwargs:
        return pd.read_csv(csv_path, **kwargs)

    errors: list[tuple[str, Exception]] = []
    for enc in _DEFAULT_ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=enc, **kwargs)
            if enc != _DEFAULT_ENCODINGS[0]:
                _LOG.info("Read %s with fallback encoding %s", csv_path, enc)
            return df
        except Exception as exc:  # pragma: no cover - relies on pandas behaviour
            errors.append((enc, exc))
            _LOG.debug("Failed reading %s with encoding %s: %s", csv_path, enc, exc)
    if errors:
        last_enc, last_exc = errors[-1]
        _LOG.error(
            "Unable to read %s with candidate encodings %s", csv_path, ", ".join(e for e, _ in errors)
        )
        raise last_exc
    raise RuntimeError(f"No encodings attempted for CSV: {csv_path}")


def csv_to_parquet(
    csv_path: Path,
    out_parquet: Path,
    schema_hints: Optional[Dict[str, str]] = None,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    csv_path = Path(csv_path)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = _read_csv_with_fallback(csv_path, read_csv_kwargs)
    df = _apply_schema_hints(df, schema_hints)

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_parquet)
    return out_parquet
