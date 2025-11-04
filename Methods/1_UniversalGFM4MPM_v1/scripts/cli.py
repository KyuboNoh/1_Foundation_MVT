from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import load_config
from Common.Unifying.Labels_TwoDatasets.datasets import (
    auto_coord_error,
    load_embedding_records,
    summarise_records,
)
from Common.Unifying.Labels_TwoDatasets.fusion_utils.workspace import OverlapAlignmentWorkspace
from Common.Unifying.Labels_TwoDatasets.overlaps import load_overlap_pairs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect overlap alignment pairs for the v1 pipeline.")
    parser.add_argument("--config", required=True, type=str, help="Path to alignment configuration JSON.")
    return parser.parse_args()


def build_report(
    workspace: OverlapAlignmentWorkspace,
    pairs: Sequence,
    max_coord_error: Optional[float],
) -> Dict[str, Any]:
    total_pairs = len(workspace.overlap.pairs) if workspace.overlap is not None else 0
    unresolved = max(total_pairs - len(pairs), 0)
    report: Dict[str, Any] = {
        "datasets": workspace.dataset_summaries(),
        "overlap": workspace.overlap_summary(),
        "pair_diagnostics": workspace.pair_diagnostics(max_coord_error=max_coord_error, pairs=pairs),
        "unresolved_pairs": unresolved,
    }
    return report


def _print_text_report(report: Dict[str, Any]) -> None:
    datasets = report.get("datasets", {})
    for name, summary in datasets.items():
        print(f"[dataset] {name}")
        if isinstance(summary, dict):
            print(f"  count: {summary.get('count')}")
            labels = summary.get("labels") or {}
            print(f"  labels: {labels}")
            regions = summary.get("regions") or {}
            print(f"  regions: {regions}")
            if summary.get("pixel_resolution"):
                print(f"  pixel_resolution: {summary['pixel_resolution']}")
            if summary.get("window_size"):
                print(f"  window_size: {summary['window_size']}")
        print()

    overlap_summary = report.get("overlap") or {}
    if overlap_summary:
        datasets = overlap_summary.get("datasets")
        print(f"[overlap] datasets={datasets} pairs={overlap_summary.get('pair_count')}")
        print(f"  generated_from: {overlap_summary.get('generated_from')}")
        print(f"  approx_spacing: {overlap_summary.get('approx_spacing')}")
        print(f"  dataset_resolutions: {overlap_summary.get('dataset_resolutions')}")
        if overlap_summary.get("window_stats"):
            print(f"  window_stats: {overlap_summary.get('window_stats')}")
        print()

    pair_diag = report.get("pair_diagnostics") or {}
    print("[pairs]")
    print(f"  total_overlap_pairs: {pair_diag.get('total_overlap_pairs')}")
    print(f"  resolved_pairs: {pair_diag.get('resolved_pairs')}")
    print(f"  resolution_ratio_stats: {pair_diag.get('resolution_ratio_stats')}")
    if pair_diag.get("window_ratio_stats"):
        print(f"  window_ratio_stats: {pair_diag.get('window_ratio_stats')}")
    print(f"  unresolved_pairs: {report.get('unresolved_pairs')}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    workspace = OverlapAlignmentWorkspace(cfg)
    max_coord_error = auto_coord_error(workspace, cfg.datasets[0].name, cfg.datasets[1].name)
    pairs = list(workspace.iter_pairs(max_coord_error=max_coord_error))
    report = build_report(workspace, pairs, max_coord_error=max_coord_error)

    json.dump(report, fp=sys.stdout, indent=2)
    _print_text_report(report)

    _maybe_write_pairs(cfg, pairs)


def _maybe_write_pairs(cfg, pairs: Sequence) -> None:
    target_dir = cfg.log_dir or cfg.output_dir
    if target_dir is None:
        return
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[warn] Unable to create log directory {target_dir}: {exc}", file=sys.stderr)
        return
    dataset_order = [dataset.name for dataset in cfg.datasets]
    payload = _serialise_pairs(pairs, dataset_order)
    output_path = target_dir / "overlap_pairs.json"
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        print(f"[warn] Unable to write pair log {output_path}: {exc}", file=sys.stderr)


def _serialise_pairs(pairs: Sequence, dataset_order: Sequence[str]) -> List[Dict[str, Any]]:
    serialised_with_labels: List[Tuple[str, Dict[str, Any]]] = []
    for pair in pairs:
        centroid = None
        if pair.centroid is not None:
            try:
                centroid = [float(pair.centroid[0]), float(pair.centroid[1])]
            except Exception:
                centroid = None
        label_type = pair.label_type()
        serialised_with_labels.append(
            (
                label_type,
                {
                    "datasets": [pair.anchor_dataset, pair.target_dataset],
                    "label_type": label_type,
                    "anchor": {
                        "dataset": pair.anchor_dataset,
                        "index": int(pair.anchor_record.index),
                        "label": int(pair.anchor_record.label),
                        "tile_id": pair.anchor_record.tile_id,
                    },
                    "target": {
                        "dataset": pair.target_dataset,
                        "index": int(pair.target_record.index),
                        "label": int(pair.target_record.label),
                        "tile_id": pair.target_record.tile_id,
                    },
                    "centroid": centroid,
                    "notes": pair.notes,
                },
            )
        )

    def _priority(label: str) -> Tuple[int, int]:
        if label == "positive_common":
            return (0, 0)
        if label.startswith("positive_"):
            dataset_name = label[len("positive_"):]
            try:
                dataset_index = dataset_order.index(dataset_name)
            except ValueError:
                dataset_index = len(dataset_order)
            return (1, dataset_index)
        if label == "unlabelled":
            return (2, 0)
        return (3, 0)

    serialised_with_labels.sort(key=lambda item: _priority(item[0]))
    return [entry for _, entry in serialised_with_labels]


if __name__ == "__main__":
    main()
