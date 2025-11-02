"""Utilities to treat a STAC tabular collection as a pseudo image stack.

This bridges the GFM4MPM raster-based pipeline to STAC table exports (e.g. the
`stacify_GCS_data.py` output) by projecting numeric columns into pseudo "band"
channels. Each table row is mapped to a synthetic 1×1 patch whose channel values
are the normalized feature columns. Coordinates follow the `GeoStack` interface
so that the training/inference scripts can operate without large rewrites.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "pyarrow is required to read STAC Parquet tables. Install with `pip install pyarrow`."
    ) from exc


_LOG = logging.getLogger(__name__)


def _candidate_roots(root: Path) -> List[Path]:
    """Return possible directories that may contain table assets."""
    if not root.exists():
        return []

    candidates: List[Path] = []
    if root.is_dir():
        assets_dir = root / "assets"
        if assets_dir.exists():
            for sub in ("tables", "table", "tabular"):
                maybe = assets_dir / sub
                if maybe.exists():
                    candidates.append(maybe)
            candidates.append(assets_dir)
        for sub in ("tables", "table", "tabular"):
            maybe = root / sub
            if maybe.exists():
                candidates.append(maybe)
        candidates.append(root)
    return list(dict.fromkeys(path.resolve() for path in candidates))


def _find_parquet(root: Path) -> Path:
    if root.is_file() and root.suffix.lower() == ".parquet":
        return root

    search_roots = _candidate_roots(root) or [root]
    for base in search_roots:
        candidates = sorted(base.rglob("*.parquet"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"No Parquet assets found under {root}")


def _to_numpy_float(col: pa.ChunkedArray) -> np.ndarray:
    arr = col.to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=np.float32)


def _to_numpy_any(col: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(col.to_numpy(zero_copy_only=False))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _ordered_unique(values: Sequence[object]) -> List[object]:
    seen = set()
    ordered: List[object] = []
    for val in values:
        if _is_missing(val):
            continue
        key = val if isinstance(val, (int, float, bool)) else str(val)
        if key not in seen:
            seen.add(key)
            ordered.append(val)
    return ordered


def _encode_categorical(name: str, column: pa.ChunkedArray) -> Tuple[List[str], List[np.ndarray]]:
    # Consolidate to avoid per-chunk conversions when creating indicator bands.
    combined = column.combine_chunks()
    data = combined.to_numpy(zero_copy_only=False)

    normalized = np.empty(data.shape, dtype=object)
    for idx, val in enumerate(data):
        if isinstance(val, bytes):
            normalized[idx] = val.decode("utf-8", errors="replace")
        else:
            normalized[idx] = val

    categories = _ordered_unique(normalized)
    arrays: List[np.ndarray] = []
    names: List[str] = []

    for cat in categories:
        label = str(cat).replace("\n", " ")
        if label == "":
            label = "__empty__"
        mask = normalized == cat
        arrays.append(mask.astype(np.float32))
        names.append(f"{name}={label}")

    missing_mask = np.fromiter((_is_missing(v) for v in normalized), dtype=bool, count=len(normalized))
    if missing_mask.any():
        arrays.append(missing_mask.astype(np.float32))
        names.append(f"{name}=__missing__")

    return names, arrays


def _binary_from_values(values: Sequence) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.int64)
    for idx, val in enumerate(values):
        if val is None:
            continue
        if isinstance(val, (int, np.integer, float, np.floating)):
            result[idx] = int(val > 0)
        elif isinstance(val, (bool, np.bool_)):
            result[idx] = int(val)
        else:
            text = str(val).strip().lower()
            if text in {"1", "true", "t", "yes", "present", "y"}:
                result[idx] = 1
    return result


@dataclass
class StacTableStack:
    root: Path
    label_columns: Optional[Iterable[str]] = None
    drop_columns: Optional[Iterable[str]] = None
    feature_columns: Optional[Iterable[str]] = None
    latitude_column: Optional[str] = None
    longitude_column: Optional[str] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.is_table = True
        self.kind = "table"
        self._requested_feature_columns = list(self.feature_columns) if self.feature_columns is not None else None
        if self.root.is_file():
            self.table_path = self.root
        else:
            self.table_path = _find_parquet(self.root)
        self.feature_columns = []
        self.feature_metadata: List[Dict[str, Optional[str]]] = []
        self.categorical_expansions: Dict[str, List[str]] = {}
        self._load_table()
        self._prepare_feature_matrix()
        self._prepare_labels()
        self.height = self.features.shape[0]
        self.width = 1  # synthetic dimension
        self.count = self.features.shape[1]
        self.transform = None
        self.crs = None
        self.srcs: List = []  # placeholder for API parity with GeoStack

    # ---------------------------------------------------------------------
    def _load_table(self) -> None:
        self._table = pq.read_table(self.table_path)
        self._schema = self._table.schema

    def _prepare_feature_matrix(self) -> None:
        drop = set(self.drop_columns or [])
        label_cols = set(self.label_columns or [])

        requested = [c for c in (self._requested_feature_columns or []) if c not in drop]
        feature_candidates: List[str] = []
        if requested:
            for name in requested:
                if name not in self._schema.names:
                    _LOG.warning("Requested feature '%s' not found in table", name)
                    continue
                feature_candidates.append(name)
        else:
            for field in self._schema:
                if field.name in drop or field.name in label_cols:
                    continue
                feature_candidates.append(field.name)

        feature_arrays: List[np.ndarray] = []
        feature_names: List[str] = []
        feature_meta: List[Dict[str, Optional[str]]] = []
        categorical_expansions: Dict[str, List[str]] = {}

        for name in feature_candidates:
            field = self._schema.field(name)
            column = self._table.column(name)
            _LOG.info("Processing feature '%s' (%s)", name, field.type)
            print(f"Processing feature '{name}' ({field.type})")

            if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
                arr = _to_numpy_float(column)
                feature_arrays.append(arr)
                feature_names.append(name)
                feature_meta.append({"source": name, "category": None})
            elif pa.types.is_boolean(field.type):
                arr = _to_numpy_any(column).astype(np.float32)
                feature_arrays.append(arr)
                feature_names.append(name)
                feature_meta.append({"source": name, "category": None})
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type) or pa.types.is_dictionary(field.type):
                encoded_names, encoded_arrays = _encode_categorical(name, column)
                if not encoded_arrays:
                    _LOG.warning("Categorical feature '%s' has no valid categories and will be skipped", name)
                    continue
                feature_arrays.extend(encoded_arrays)
                feature_names.extend(encoded_names)
                feature_meta.extend(
                    {"source": name, "category": encoded.partition("=")[2]} for encoded in encoded_names
                )
                categorical_expansions[name] = encoded_names
                _LOG.info("  expanded to %d indicator columns", len(encoded_names))
                print(f"  expanded to {len(encoded_names)} indicator columns")
            else:
                _LOG.warning("Feature '%s' with type '%s' is not supported and will be skipped", name, field.type)
                print(f"[warn] Feature '{name}' with type '{field.type}' skipped")

        if not feature_arrays:
            raise ValueError("No usable columns found to use as features in STAC table")

        cols = [np.asarray(col, dtype=np.float32) for col in feature_arrays]
        matrix = np.column_stack(cols).astype(np.float32)
        raw_gb = matrix.nbytes / (1024 ** 3)
        total_gb = raw_gb * 2  # raw + normalized copy
        _LOG.info(
            "Stacked %d feature arrays into shape %s (raw ~%.2f GiB, normalized copy pushes total ~%.2f GiB)",
            len(feature_names),
            matrix.shape,
            raw_gb,
            total_gb,
        )
        print(
            f"Stacked {len(feature_names)} feature arrays into shape {matrix.shape} "
            f"(raw ~{raw_gb:.2f} GiB, normalized copy pushes total ~{total_gb:.2f} GiB)"
        )
        if total_gb > 6:
            _LOG.warning(
                "Large STAC feature matrix detected (~%.2f GiB in memory). Consider reducing features or running on a high-memory host.",
                total_gb,
            )
            print(
                f"[warn] Large STAC feature matrix detected (~{total_gb:.2f} GiB). "
                "Consider reducing features or using a high-memory host."
            )
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        std[std < 1e-6] = 1.0
        self.features = matrix
        self.feature_columns = feature_names
        self.feature_metadata = feature_meta
        self.categorical_expansions = categorical_expansions
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)
        normalized = (self.features - self.feature_mean) / self.feature_std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        self._normalized = normalized.astype(np.float32)

        # Optional metadata columns (if present)
        self.h3_addresses = None
        if "H3_Address" in self._schema.names:
            self.h3_addresses = _to_numpy_any(self._table.column("H3_Address"))

        self.latitudes = None
        self.longitudes = None
        lat_name = self.latitude_column or ("Latitude_EPSG4326" if "Latitude_EPSG4326" in self._schema.names else None)
        lon_name = self.longitude_column or ("Longitude_EPSG4326" if "Longitude_EPSG4326" in self._schema.names else None)
        if lat_name and lat_name in self._schema.names:
            self.latitudes = _to_numpy_float(self._table.column(lat_name))
            self.latitude_column = lat_name
        elif self.latitude_column:
            _LOG.warning("Latitude column '%s' not found", self.latitude_column)
            self.latitude_column = None
        if lon_name and lon_name in self._schema.names:
            self.longitudes = _to_numpy_float(self._table.column(lon_name))
            self.longitude_column = lon_name
        elif self.longitude_column:
            _LOG.warning("Longitude column '%s' not found", self.longitude_column)
            self.longitude_column = None

    def _prepare_labels(self) -> None:
        if self.label_columns is None:
            inferred = [name for name in self._schema.names if name.lower().startswith("training_")]
            self.label_columns = inferred
        else:
            self.label_columns = list(self.label_columns)
        self.labels: Dict[str, np.ndarray] = {}
        for name in self.label_columns:
            if name not in self._schema.names:
                continue
            vals = _to_numpy_any(self._table.column(name))
            self.labels[name] = _binary_from_values(vals)

    # ------------------------------------------------------------------
    def read_patch(self, row: int, col: int, size: int, nodata_val: Optional[float] = None) -> np.ndarray:
        if row < 0 or row >= self.height:
            raise IndexError(f"Row {row} out of bounds for STAC table with {self.height} rows")
        vec = self._normalized[row]
        patch = np.repeat(vec[:, None, None], size, axis=1)
        patch = np.repeat(patch, size, axis=2)
        return patch.astype(np.float32)

    def grid_centers(self, stride: int) -> List[Tuple[int, int]]:
        # Stride is ignored; every row is a candidate sample.
        return [(idx, 0) for idx in range(self.height)]

    # Convenience helpers ------------------------------------------------
    def random_coord(self, patch: int, rng: np.random.Generator) -> Tuple[int, int]:
        idx = int(rng.integers(0, self.height))
        return idx, 0

    def coord_to_index(self, coord: Tuple[int, int]) -> int:
        return coord[0]

    def index_to_coord(self, index: int) -> Tuple[int, int]:
        return int(index), 0

    def label_array(self, name: str) -> np.ndarray:
        arr = self.labels.get(name)
        if arr is None:
            raise KeyError(f"Label column '{name}' not available in STAC table")
        return arr

    def positive_coords(self, name: str) -> List[Tuple[int, int]]:
        lab = self.label_array(name)
        return [(idx, 0) for idx, val in enumerate(lab) if val == 1]

    def metadata_row(self, index: int) -> Dict[str, Optional[float]]:
        meta = {"index": int(index)}
        if self.h3_addresses is not None:
            meta["h3_address"] = self.h3_addresses[index]
        if self.latitudes is not None and self.latitude_column:
            meta["latitude"] = float(self.latitudes[index])
        if self.longitudes is not None and self.longitude_column:
            meta["longitude"] = float(self.longitudes[index])
        return meta

    def iter_metadata(self) -> List[Dict[str, Optional[float]]]:
        return [self.metadata_row(i) for i in range(self.height)]

    def raw_column(self, name: str) -> np.ndarray:
        if name not in self._schema.names:
            raise KeyError(f"Column '{name}' not found in STAC table")
        return _to_numpy_any(self._table.column(name))
