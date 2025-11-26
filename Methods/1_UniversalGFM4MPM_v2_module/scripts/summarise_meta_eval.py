

# %% [markdown]
# Extracted from summarise_meta_eval.ipynb

# %% [code] Cell 0
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


def parse_meta_file_args(items: List[str]) -> List[Tuple[Optional[str], Path]]:
    """Parse inputs in optional TAG:PATH format (directories supported)."""
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
                raise FileNotFoundError(f"No JSON files under directory '{path}'")
            for file in json_files:
                parsed.append((tag, file))
        else:
            parsed.append((tag, path))
    return parsed


def infer_method_cluster(path: Path, methods: List[str], clusters: List[int]) -> Tuple[str, int]:
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
                matches.append((method, int("".join(digits))))
            idx = name.find(method, idx + 1)
    if not matches:
        raise ValueError(f"Cannot infer method/n_clusters from '{path.name}'.")
    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous method tokens in '{path.name}': {matches}")
    method, cluster = matches[0]
    if clusters and cluster not in clusters:
        raise ValueError(f"Cluster {cluster} for '{path.name}' not in allowed {clusters}")
    return method, cluster


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
    candidate = stem[:-len(token)] if stem.endswith(token) else stem
    if allowed_tags:
        for tag in allowed_tags:
            if candidate == tag or candidate.startswith(tag):
                return tag
        raise ValueError(f"Inferred tag '{candidate}' not in --tags {sorted(allowed_tags)}")
    return candidate


def filter_metrics(raw_metrics: Dict[str, object], selected: List[str]) -> Dict[str, object]:
    if not selected:
        return raw_metrics
    return {metric: raw_metrics[metric] for metric in selected if metric in raw_metrics}


def summarise_meta_eval_interactive(
    meta_files: List[str],
    *,
    clustering_methods: List[str],
    cluster_counts: List[int],
    metrics: Optional[List[str]] = None,
    allowed_tags: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    entries_raw = parse_meta_file_args(meta_files)
    entries = [
        (tag_override, path)
        for tag_override, path in entries_raw
        if path.name != "meta_eval_summary.json"
    ]
    metrics_to_keep = metrics or []
    allowed_set = set(allowed_tags) if allowed_tags else None

    summary: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for tag_override, path in entries:
        if not path.exists():
            raise FileNotFoundError(f"Meta eval file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        method, cluster = infer_method_cluster(path, clustering_methods, cluster_counts)
        
        # Extract rigorous evaluation data before filtering
        rigorous_data = data.get('rigorous_evaluation')
        
        cleaned = filter_metrics(data, metrics_to_keep)
        
        # Re-attach rigorous data if present
        if rigorous_data:
            cleaned['rigorous_evaluation'] = rigorous_data
            
        tag_main = derive_tag_main(path, method, cluster, override=tag_override, allowed_tags=allowed_set)
        summary.setdefault(tag_main, {}).setdefault(method, {})[str(cluster)] = cleaned
    return summary



# %% [code] Cell 1
# Example placeholder arguments.
# Replace strings below with real directories or TAG:PATH pairs.
meta_inputs = [
    "/home/wslqubuntu24/Research/Data/1_Foundation_MVT_Result/2_UFM_v2/TransAggDCCA_Ex1_dim256/cls_1_training_results/meta_evaluation_results",  # directory (all *.json will be processed)
]

summary = summarise_meta_eval_interactive(
    meta_files=meta_inputs,
    clustering_methods=["random", "kmeans", "hierarchical"],
    cluster_counts=[2, 3, 4, 5, 6, 7, 8, 10],
    metrics=["PosDrop_Acc", "Focus", "topk", "pauc", "pu_tpr", "pu_fpr", "pu_npv", "background_rejection"],
    # allowed_tags=["base_MAEViT_BC", "Method2_concat_MAEViT_plus_TransDCCA_BC"],
    allowed_tags=["M3_Uni_KL_Overlap_Shared_MAEViT_plus_TransDCCA_NA_AU_BC"],
)

summary

# %% [code] Cell 2
import matplotlib.pyplot as plt
import numpy as np

def plot_scalar_metric(summary_dict, metric_name='PosDrop_Acc',
                       include_tags=None, include_methods=None, include_clusters=None):
    include_tags = set(include_tags) if include_tags else None
    include_methods = set(include_methods) if include_methods else None
    include_clusters = set(int(c) for c in include_clusters) if include_clusters else None

    rows = []
    for tag_main, methods_dict in summary_dict.items():
        if include_tags and tag_main not in include_tags:
            continue
        for method, clusters_dict in methods_dict.items():
            if include_methods and method not in include_methods:
                continue
            for cluster, metrics in clusters_dict.items():
                cluster_id = int(cluster)
                if include_clusters and cluster_id not in include_clusters:
                    continue
                metric = metrics.get(metric_name)
                if isinstance(metric, dict) and isinstance(metric.get('mean'), (int, float)):
                    rows.append((tag_main, method, cluster_id, float(metric['mean'])))
    if not rows:
        raise RuntimeError(f"No scalar '{metric_name}' mean values found after filtering.")

    palettes = {
        'base_MAEViT_BC': plt.cm.Blues,
        'Method2_concat_MAEViT_plus_TransDCCA_BC': plt.cm.Reds,
    }
    default_cmap = plt.cm.Greys

    plt.figure(figsize=(8, 5))
    for tag in sorted({r[0] for r in rows}):
        tag_rows = [r for r in rows if r[0] == tag]
        tag_rows.sort(key=lambda x: x[2])
        clusters = [r[2] for r in tag_rows]
        values = [r[3] for r in tag_rows]
        cmap = palettes.get(tag, default_cmap)(np.linspace(0.3, 0.9, len(tag_rows)))
        plt.plot(clusters, values, marker='o', color=cmap[-1], label=tag)

    plt.xlabel('Cluster Count')
    plt.ylabel(f'{metric_name} mean')
    plt.title(f'{metric_name} across cluster counts by tag')
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [code] Cell 3
# Example:
plot_scalar_metric(summary, metric_name='PosDrop_Acc', include_methods=['kmeans'])

# %% [code] Cell 4
# Example:
plot_scalar_metric(summary, metric_name='Focus', include_methods=['kmeans'])

# %% [code] Cell 5
import matplotlib.pyplot as plt

def plot_pauc(summary_dict, prior_key='prior_0.010'):
    by_tag = {}
    for tag_main, methods_dict in summary_dict.items():
        entries = []
        for method, clusters_dict in methods_dict.items():
            for cluster, metrics in clusters_dict.items():
                entry = metrics.get('pauc', {}).get(prior_key)
                if isinstance(entry, dict) and isinstance(entry.get('mean'), (int, float)):
                    entries.append((f'{method}{cluster}', float(entry['mean'])))
        if entries:
            by_tag[tag_main] = entries
    if not by_tag:
        raise RuntimeError(f'No PAUC data found for {prior_key}.')
    plt.figure(figsize=(8, 5))
    for tag, entries in by_tag.items():
        entries.sort(key=lambda x: x[0])
        labels = [label for label, _ in entries]
        values = [val for _, val in entries]
        plt.plot(labels, values, marker='o', label=tag)
    plt.ylabel(f'pauc mean ({prior_key})')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'PAUC means for {prior_key}')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_pauc(summary, prior_key='prior_0.010')


# %% [code] Cell 6
import matplotlib.pyplot as plt

def plot_topk_metric(summary_dict, ratio_keys=None, metric='mean_capture_rate', include_methods=None):
    include_methods = set(include_methods) if include_methods else None

    if ratio_keys is None:
        first_tag = next(iter(summary_dict.values()))
        first_method = next(iter(first_tag.values()))
        first_clusters = next(iter(first_method.values()))
        ratio_keys = sorted(first_clusters.get('topk', {}).get('ratios', {}).keys())
    if not ratio_keys:
        raise RuntimeError('No ratio keys available in summary.')

    plt.figure(figsize=(8, 5))
    for ratio_key in ratio_keys:
        entries = []
        for tag_main, methods in summary_dict.items():
            for method, clusters_dict in methods.items():
                if include_methods and method not in include_methods:
                    continue
                for cluster, metrics in clusters_dict.items():
                    ratio_entry = metrics.get('topk', {}).get('ratios', {}).get(ratio_key)
                    if isinstance(ratio_entry, dict) and isinstance(ratio_entry.get(metric), (int, float)):
                        entries.append((tag_main, int(cluster), float(ratio_entry[metric])))
        if entries:
            entries.sort(key=lambda x: (x[0], x[1]))
            tags = sorted(set(tag for tag, _, _ in entries))
            for tag in tags:
                tag_entries = [entry for entry in entries if entry[0] == tag]
                clusters = [entry[1] for entry in tag_entries]
                values = [entry[2] for entry in tag_entries]
                plt.plot(clusters, values, marker='o', label=f"{tag} {ratio_key}")

    plt.xlabel('Cluster Count')
    plt.ylabel(metric)
    plt.title('TopK ratios vs cluster count')
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% [code] Cell 7
plot_topk_metric(summary,
                 ratio_keys=['ratio_0.50','ratio_1.00'],
                 metric='mean_capture_rate',
                 include_methods=['kmeans'])

# %% [code] Cell 8
import matplotlib.pyplot as plt

def plot_topk_absolute(summary_dict, k_keys=None, metric='mean_capture_rate'):
    if k_keys is None:
        first_tag = next(iter(summary_dict.values()))
        first_method = next(iter(first_tag.values()))
        first_clusters = next(iter(first_method.values()))
        k_keys = sorted(first_clusters.get('topk', {}).get('absolute', {}).keys())
    if not k_keys:
        raise RuntimeError('No absolute K keys available in summary.')
    plt.figure(figsize=(8, 5))
    for k_key in k_keys:
        entries = []
        for tag_main, methods in summary_dict.items():
            for method, clusters_dict in methods.items():
                for cluster, metrics in clusters_dict.items():
                    abs_entry = metrics.get('topk', {}).get('absolute', {}).get(k_key)
                    if isinstance(abs_entry, dict) and isinstance(abs_entry.get(metric), (int, float)):
                        entries.append((tag_main, int(cluster), float(abs_entry[metric])))
        if entries:
            entries.sort(key=lambda x: (x[0], x[1]))
            tags = sorted(set(tag for tag, _, _ in entries))
            for tag in tags:
                tag_entries = [entry for entry in entries if entry[0] == tag]
                clusters = [entry[1] for entry in tag_entries]
                values = [entry[2] for entry in tag_entries]
                plt.plot(clusters, values, marker='o', label=f"{tag} {k_key}")
    plt.xlabel('Cluster Count')
    plt.ylabel(metric)
    plt.title('TopK absolute K capture rate vs cluster count')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_topk_absolute(summary, k_keys=['k_50', 'k_100'], metric='mean_capture_rate')



# %% [code] Cell 9
import matplotlib.pyplot as plt

def plot_topk_area(summary_dict, area_keys=None, metric='mean_capture_rate', include_methods=None):
    if area_keys is None:
        first_tag = next(iter(summary_dict.values()))
        first_method = next(iter(first_tag.values()))
        first_clusters = next(iter(first_method.values()))
        area_keys = sorted(first_clusters.get('topk', {}).get('area_percentages', {}).keys())
    if not area_keys:
        raise RuntimeError('No area percentage keys available in summary.')
    plt.figure(figsize=(8, 5))
    for area_key in area_keys:
        entries = []
        for tag_main, methods in summary_dict.items():
            for method, clusters_dict in methods.items():
                if include_methods and method not in include_methods:
                    continue
                for cluster, metrics in clusters_dict.items():
                    area_entry = metrics.get('topk', {}).get('area_percentages', {}).get(area_key)
                    if isinstance(area_entry, dict) and isinstance(area_entry.get(metric), (int, float)):
                        entries.append((tag_main, method, int(cluster), float(area_entry[metric])))
        if entries:
            entries.sort(key=lambda x: (x[0], x[2]))
            tags = sorted(set(entry[0] for entry in entries))
            for tag in tags:
                tag_entries = [entry for entry in entries if entry[0] == tag]
                clusters = [entry[2] for entry in tag_entries]
                values = [entry[3] for entry in tag_entries]
                methods_present = sorted(set(entry[1] for entry in tag_entries))
                label = f"{tag} {area_key}"
                if methods_present:
                    label = f"{tag} ({','.join(methods_present)}) {area_key}"
                plt.plot(clusters, values, marker='o', label=label)
    plt.xlabel('Cluster Count')
    plt.ylabel(metric)
    plt.title('TopK area percentages vs cluster count')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [code] Cell 10
plot_topk_area(summary, area_keys=['area_0.1pct', 'area_0.5pct', 'area_1.0pct', 'area_2.0pct'], metric='mean_capture_rate', include_methods=['kmeans'])

# %% [code] Cell 11
plot_topk_area(summary, area_keys=['area_0.1pct', 'area_0.5pct', 'area_1.0pct', 'area_10.0pct'], metric='mean_precision', include_methods=['kmeans'])

# %% [code] Cell 12
import matplotlib.pyplot as plt
import numpy as np

def plot_pu_curve(
    summary_dict,
    metric_name="pu_fpr",
    include_tags=None,
    include_methods=None,
    include_clusters=None,
):
    include_tags = set(include_tags) if include_tags else None
    include_methods = set(include_methods) if include_methods else None
    include_clusters = set(int(c) for c in include_clusters) if include_clusters else None

    tag_palettes = {
        "base_MAEViT_BC": plt.cm.Blues,
        "Method2_concat_MAEViT_plus_TransDCCA_BC": plt.cm.Reds,
    }
    default_cmap = plt.cm.Greys

    aggregated_curves = []
    for tag, methods in summary_dict.items():
        if include_tags and tag not in include_tags:
            continue
        # count clusters for this tag to size the colormap
        total_clusters = sum(len(clusters) for clusters in methods.values())
        cmap = tag_palettes.get(tag, default_cmap)(np.linspace(0.3, 0.9, max(1, total_clusters)))
        cmap_iter = iter(cmap)

        for method, clusters in methods.items():
            if include_methods and method not in include_methods:
                continue
            for cluster, metrics in clusters.items():
                cluster_id = int(cluster)
                if include_clusters and cluster_id not in include_clusters:
                    continue
                block = metrics.get(metric_name)
                if not block:
                    continue
                color = next(cmap_iter, cmap[-1])
                aggregated_curves.append((tag, method, cluster_id, block, color))

    if not aggregated_curves:
        raise RuntimeError("No matching PU curves found for the specified filters.")

    plt.figure(figsize=(9, 5))
    for tag, method, cluster_id, block, color in aggregated_curves:
        thresholds = block["thresholds"]
        base_label = f"{tag} {method}{cluster_id}"
        if metric_name == "background_rejection":
            mean, std = block["mean"], block["std"]
            plt.plot(thresholds, mean, color=color, label=base_label)
            plt.fill_between(
                thresholds,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=color,
                alpha=0.15,
            )
        else:
            for entry in block.get("pi_stats", []):
                pi = entry["pi"]
                mean, std = entry["mean"], entry["std"]
                plt.plot(thresholds, mean, color=color, label=f"{base_label} π={pi}")
                plt.fill_between(
                    thresholds,
                    [m - s for m, s in zip(mean, std)],
                    [m + s for m, s in zip(mean, std)],
                    color=color,
                    alpha=0.12,
                )

    plt.xscale("log")
    plt.xlabel("Threshold")
    plt.ylabel(f"{metric_name} mean")
    plt.title(f"{metric_name}: combined view")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% [code] Cell 13
# plot_pu_curve(summary, metric_name="pu_fpr", include_tags=["base_MAEViT_BC", "Method2_concat_MAEViT_plus_TransDCCA_BC"],
#               include_methods=["random", "kmeans", "hierarchical"],)

plot_pu_curve(summary, metric_name="pu_fpr", include_tags=["base_MAEViT_BC", "Method2_concat_MAEViT_plus_TransDCCA_BC"],
              include_methods=["kmeans"], include_clusters= [10])

# %% [code] Cell 14
plot_pu_curve(summary, metric_name="background_rejection", include_tags=["base_MAEViT_BC", "Method2_concat_MAEViT_plus_TransDCCA_BC"],
              include_methods=["kmeans"], include_clusters= [2, 10])

# %% [code] Cell 15
import matplotlib.pyplot as plt
import numpy as np

def plot_pu_npv(
    summary_dict,
    include_tags=None,
    include_methods=None,
    include_clusters=None,
    pi_keys=None,
):
    include_tags = set(include_tags) if include_tags else None
    include_methods = set(include_methods) if include_methods else None
    include_clusters = set(int(c) for c in include_clusters) if include_clusters else None

    tag_palettes = {
        "base_MAEViT_BC": plt.cm.Blues,
        "Method2_concat_MAEViT_plus_TransDCCA_BC": plt.cm.Reds,
    }
    default_cmap = plt.cm.Greys

    entries = []
    for tag, methods in summary_dict.items():
        if include_tags and tag not in include_tags:
            continue
        total_clusters = sum(len(clusters) for clusters in methods.values())
        cmap = tag_palettes.get(tag, default_cmap)(np.linspace(0.3, 0.9, max(1, total_clusters)))
        cmap_iter = iter(cmap)

        for method, clusters in methods.items():
            if include_methods and method not in include_methods:
                continue
            for cluster, metrics in clusters.items():
                cluster_id = int(cluster)
                if include_clusters and cluster_id not in include_clusters:
                    continue
                block = metrics.get("pu_npv")
                if not block:
                    continue
                color = next(cmap_iter, cmap[-1])
                entries.append((tag, method, cluster_id, block, color))

    if not entries:
        raise RuntimeError("No pu_npv data found for given filters.")

    plt.figure(figsize=(10, 5))
    for tag, method, cluster_id, block, color in entries:
        npv_block = block.get("npv", {})
        thresholds = npv_block.get("thresholds")
        pi_stats = npv_block.get("pi_stats", [])
        if not thresholds or not pi_stats:
            continue
        active_pis = pi_keys or [entry["pi"] for entry in pi_stats]
        for entry in pi_stats:
            if entry["pi"] not in active_pis:
                continue
            mean = entry["mean"]
            std = entry["std"]
            label = f"{tag} {method}{cluster_id} π={entry['pi']}"
            plt.plot(thresholds, mean, color=color, label=label)
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            plt.fill_between(thresholds, lower, upper, color=color, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("Threshold")
    plt.ylabel("PU-NPV mean")
    plt.title("PU-NPV across tags/methods")
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% [code] Cell 16
plot_pu_npv(
    summary,
    include_tags=["base_MAEViT_BC", "Method2_concat_MAEViT_plus_TransDCCA_BC"],
    include_methods=["kmeans"],
    include_clusters=[10],
    pi_keys=[0.01, 0.2],
)

# %% [code] Cell 17



# %% [code] Cell 17
def print_rigorous_metrics(summary_data):
    """Helper to print rigorous evaluation metrics from the summary."""
    for tag, methods in summary_data.items():
        print(f"\n=== Rigorous Evaluation for {tag} ===")
        for method, clusters in methods.items():
            for cluster, metrics in clusters.items():
                if 'rigorous_evaluation' in metrics:
                    print(f"  Method: {method}, Cluster: {cluster}")
                    rig = metrics['rigorous_evaluation']
                    
                    if 'target_shuffling' in rig:
                        ts = rig['target_shuffling']
                        print(f"    Target Shuffling:")
                        # Handle nested structure where keys might be method names
                        for k, v in ts.items():
                            if isinstance(v, dict):
                                print(f"      [{k}] Z-Score: {v.get('z_score')}")
                        
                    if 'stability_selection' in rig:
                        ss = rig['stability_selection']
                        print(f"    Stability Selection:")
                        for k, v in ss.items():
                             if isinstance(v, dict):
                                print(f"      [{k}] Spatial Entropy: {v.get('spatial_entropy')}")
                                print(f"      [{k}] Spatial Jaccard: {v.get('spatial_jaccard')}")

# %% [code] Cell 18
import matplotlib.pyplot as plt
import numpy as np

def plot_shuffle_zscore(summary_dict, include_tags=None):
    """
    Plot Z-Scores from Target Shuffling.
    """
    include_tags = set(include_tags) if include_tags else None
    
    data_points = []
    
    for tag_main, methods_dict in summary_dict.items():
        if include_tags and tag_main not in include_tags:
            continue
            
        # We usually just need one representative entry per tag if they are all the same run
        # But structure is hierarchical. Let's iterate.
        for method, clusters_dict in methods_dict.items():
            for cluster, metrics in clusters_dict.items():
                rig = metrics.get('rigorous_evaluation', {})
                shuffle = rig.get('target_shuffling', {})
                
                # shuffle keys are the baseline methods (e.g. 'standard', 'posdrop_...')
                for baseline_key, res in shuffle.items():
                    if res and 'z_score' in res and res['z_score'] is not None:
                        label = f"{tag_main} [{baseline_key}]"
                        data_points.append((label, res['z_score']))
                        
    if not data_points:
        print("No Z-Score data found.")
        return

    # Sort by Z-Score
    data_points.sort(key=lambda x: x[1], reverse=True)
    
    labels = [x[0] for x in data_points]
    values = [x[1] for x in data_points]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color='skyblue')
    plt.axvline(x=1.96, color='r', linestyle='--', label='p=0.05 (Z=1.96)')
    plt.xlabel('Z-Score (Higher is better)')
    plt.title('Target Shuffling Z-Scores')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example
# plot_shuffle_zscore(summary)

# %% [code] Cell 19
def plot_stability_jaccard(summary_dict, include_tags=None):
    """
    Plot Spatial Jaccard Index from Stability Selection.
    """
    include_tags = set(include_tags) if include_tags else None
    
    data_points = []
    
    for tag_main, methods_dict in summary_dict.items():
        if include_tags and tag_main not in include_tags:
            continue
            
        for method, clusters_dict in methods_dict.items():
            for cluster, metrics in clusters_dict.items():
                rig = metrics.get('rigorous_evaluation', {})
                stability = rig.get('stability_selection', {})
                
                for baseline_key, res in stability.items():
                    if res and 'spatial_jaccard' in res and res['spatial_jaccard'] is not None:
                        label = f"{tag_main} [{baseline_key}]"
                        data_points.append((label, res['spatial_jaccard']))
                        
    if not data_points:
        print("No Spatial Jaccard data found.")
        return

    data_points.sort(key=lambda x: x[1], reverse=True)
    
    labels = [x[0] for x in data_points]
    values = [x[1] for x in data_points]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='lightgreen')
    plt.xlabel('Spatial Jaccard Index (Higher is more stable)')
    plt.title('Stability Selection: Spatial Jaccard')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example
# plot_stability_jaccard(summary)

# %% [code] Cell 20
def plot_stability_entropy(summary_dict, include_tags=None):
    """
    Plot Spatial Entropy from Stability Selection.
    """
    include_tags = set(include_tags) if include_tags else None
    
    data_points = []
    
    for tag_main, methods_dict in summary_dict.items():
        if include_tags and tag_main not in include_tags:
            continue
            
        for method, clusters_dict in methods_dict.items():
            for cluster, metrics in clusters_dict.items():
                rig = metrics.get('rigorous_evaluation', {})
                stability = rig.get('stability_selection', {})
                
                for baseline_key, res in stability.items():
                    if res and 'spatial_entropy' in res and res['spatial_entropy'] is not None:
                        label = f"{tag_main} [{baseline_key}]"
                        data_points.append((label, res['spatial_entropy']))
                        
    if not data_points:
        print("No Spatial Entropy data found.")
        return

    data_points.sort(key=lambda x: x[1]) # Lower entropy might be better (more certain)? Or higher (more informative)? 
    # Usually lower entropy = more confident predictions.
    
    labels = [x[0] for x in data_points]
    values = [x[1] for x in data_points]
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='salmon')
    plt.xlabel('Spatial Entropy (Lower is more confident)')
    plt.title('Stability Selection: Spatial Entropy')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example
# plot_stability_entropy(summary)

