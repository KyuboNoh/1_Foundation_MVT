from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from Common.Unifying.DCCA import _prepare_output_dir


def _prepare_classifier_dir(
    cfg: Any,
    method_id: int,
    *name_tokens: str,
) -> Optional[Path]:
    primary = cfg.output_dir if getattr(cfg, "output_dir", None) is not None else getattr(cfg, "log_dir", None)
    suffix = f"CLS_Method_{method_id}"
    sanitized: List[str] = []
    for token in name_tokens:
        text = str(token).strip()
        if not text:
            continue
        text = text.replace("\\", "_").replace("/", "_").replace(" ", "_")
        sanitized.append(text)
    if sanitized:
        suffix += "_Data_" + "_".join(sanitized)
    return _prepare_output_dir(primary, suffix)
