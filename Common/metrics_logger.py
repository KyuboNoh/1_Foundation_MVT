from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

DEFAULT_METRIC_ORDER: Sequence[str] = (
    "loss",
    "f1",
    "mcc",
    "auprc",
    "auroc",
    "accuracy",
    "balanced_accuracy",
)

DEFAULT_PRECISION: Dict[str, int] = {
    "loss": 4,
    "auroc": 4,
    "auprc": 4,
}


def _default_json_encoder(obj: Any):
    if isinstance(obj, (Path,)):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def normalize_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Convert metric values to plain floats when possible."""
    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _format_value(value: float, precision: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def format_metrics_line(
    metrics: Mapping[str, float],
    order: Optional[Sequence[str]] = None,
    precision: Optional[Mapping[str, int]] = None,
    fallback_precision: int = 3,
) -> str:
    ordered_keys: Iterable[str]
    if order is not None:
        ordered_keys = order
    else:
        ordered_keys = metrics.keys()

    pieces = []
    seen = set()
    precision_map = precision or {}

    for key in ordered_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        prec = precision_map.get(key, fallback_precision)
        pieces.append(f"{key}={_format_value(value, prec)}")
        seen.add(key)

    for key, value in metrics.items():
        if key in seen:
            continue
        prec = precision_map.get(key, fallback_precision)
        pieces.append(f"{key}={_format_value(value, prec)}")
    return " ".join(pieces)


def log_metrics(
    label: str,
    metrics: Mapping[str, Any],
    order: Optional[Sequence[str]] = None,
    precision: Optional[Mapping[str, int]] = None,
    printer=print,
) -> None:
    normalized = normalize_metrics(metrics)
    if not normalized:
        printer(f"[{label.upper()}] no metrics")
        return
    line = format_metrics_line(
        normalized,
        order=order or DEFAULT_METRIC_ORDER,
        precision=precision or DEFAULT_PRECISION,
    )
    printer(f"[{label.upper()}] {line}")


def save_metrics_json(metrics: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=indent, default=_default_json_encoder)
