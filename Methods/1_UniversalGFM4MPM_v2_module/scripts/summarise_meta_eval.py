from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


def parse_meta_file_args(items: List[str]) -> List[Tuple[Optional[str], Path]]:
    """Parse --meta-file TAG:PATH arguments (supports directories)."""
    parsed: List[Tuple[Optional[str], Path]] = []
    for item in items:
        if ":" in item:
            tag, path_str = item.split(":", 1)
        else:
            tag, path_str = None, item
        path = Path(path_str).expanduser().resolve()
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found under directory '{path}'")
            for file in json_files:
                parsed.append((tag, file))
        else:
            parsed.append((tag, path))
    return parsed


def infer_method_cluster(path: Path, methods: List[str], clusters: List[int]) -> Tuple[str, int]:
    """Infer method+n_clusters token from filename."""
    name = path.stem
    matches: List[Tuple[str, int]] = []
    for method in methods:
        idx = name.find(method)
        while idx != -1:
            start = idx + len(method)
            digits = []
            while start < len(name) and name[start].isdigit():
                digits.append(name[start])
                start += 1
            if digits:
                cluster_val = int("".join(digits))
                matches.append((method, cluster_val))
            idx = name.find(method, idx + 1)
    if not matches:
        raise ValueError(
            f"Unable to infer method/n_clusters from filename '{path.name}'. "
            "Filenames should contain tokens like 'random10', 'kmeans6', etc."
        )
    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous method/n_clusters inference for '{path.name}': {matches}")
    method, cluster = matches[0]
    if clusters and cluster not in clusters:
        raise ValueError(
            f"Cluster {cluster} inferred from '{path.name}' is not listed in --meta-evaluation-n-clusters {clusters}."
        )
    return method, cluster


def filter_metrics(raw_metrics: Dict[str, object], selected: List[str]) -> Dict[str, object]:
    if not selected:
        return raw_metrics
    return {metric: raw_metrics[metric] for metric in selected if metric in raw_metrics}


def derive_tag_main(
    path: Path,
    method: str,
    cluster: int,
    override: Optional[str] = None,
    allowed_tags: Optional[Set[str]] = None,
) -> str:
    if override:
        return override
    stem = path.stem
    if stem.endswith("_meta_eval"):
        stem = stem[:-len("_meta_eval")]
    token = f"_{method}{cluster}"
    if stem.endswith(token):
        candidate = stem[:-len(token)]
    else:
        candidate = stem
    if allowed_tags:
        for tag in allowed_tags:
            if candidate == tag or candidate.startswith(tag):
                return tag
        raise ValueError(
            f"Unable to match derived tag '{candidate}' to any of the provided --tags {sorted(allowed_tags)}"
        )
    return candidate


def summarise_meta_eval(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    entries = parse_meta_file_args(args.meta_file)
    methods = args.clustering_methods
    clusters = args.meta_evaluation_n_clusters
    metrics_to_keep = list(args.meta_evaluation) if args.meta_evaluation else []
    allowed_tags = set(args.tags) if args.tags else None

    summary: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for tag_override, path in entries:
        if not path.exists():
            raise FileNotFoundError(f"Meta evaluation file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        method, cluster = infer_method_cluster(path, methods, clusters)
        cleaned = filter_metrics(data, metrics_to_keep)

        tag_main = derive_tag_main(
            path,
            method,
            cluster,
            override=tag_override,
            allowed_tags=allowed_tags,
        )
        summary.setdefault(tag_main, {}).setdefault(method, {})[str(cluster)] = cleaned

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise multiple *_meta_eval.json files.")
    parser.add_argument(
        "--meta-file",
        metavar="TAG:PATH",
        action="append",
        required=True,
        help="Meta-eval JSON path with optional tag (TAG:PATH). "
             "PATH can be a file or directory (all *.json inside will be processed). "
             "If TAG is omitted, tag_main is inferred from the filename.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of allowed tag_main prefixes. When provided, inferred tags must match one of these.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("meta_eval_summary.json"),
        help="Optional path to write structured summary JSON (default: ./meta_eval_summary.json).",
    )
    # Reuse key train.py arguments so filtering remains familiar
    parser.add_argument("--meta-evaluation", type=str, nargs="+",
                        default=["PosDrop_Acc", "Focus", "topk", "pauc"],
                        help="Metrics to retain in the summary (subset of those saved).")
    parser.add_argument("--clustering-methods", type=str, nargs="+",
                        default=["random"], choices=["random", "kmeans", "hierarchical"],
                        help="Clustering methods considered when inferring filenames.")
    parser.add_argument("--meta-evaluation-n-clusters", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10],
                        help="Cluster counts considered when inferring filenames.")
    parser.add_argument("--topk-k-ratio", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0])
    parser.add_argument("--topk-k-values", type=int, nargs="+", default=[])
    parser.add_argument("--topk-area-percentages", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0])
    parser.add_argument("--pauc-prior-variants", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--pu-prior-range", type=float, nargs=2, default=[0.01, 0.20])
    parser.add_argument("--pu-metric-thresholds", type=float, nargs="+", default=[])
    parser.add_argument("--pu-metric-threshold-count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarise_meta_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary for {len(summary)} tags to {args.output}")


if __name__ == "__main__":
    main()
