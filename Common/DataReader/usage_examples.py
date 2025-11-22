"""
Usage examples and helper functions for the DataReader module.
"""

from typing import Any, Dict
from .original_data import get_original_labels_from_cfg, get_original_embeddings_from_cfg, extract_positive_samples_from_original


def load_anchor_labels_for_substitution(cfg: Any, anchor_name: str, run_logger: Any) -> Dict[str, Any]:
    """
    Convenience function to load anchor dataset labels for label substitution in PU/PN learning.
    
    Args:
        cfg: Configuration object
        anchor_name: Name of the anchor dataset
        run_logger: Logger instance
        
    Returns:
        Dictionary suitable for use with substitute_label function
    """
    run_logger.log(f"[load_anchor_labels] Loading anchor labels from {anchor_name} for substitution")
    
    anchor_labels = get_original_labels_from_cfg(
        cfg=cfg,
        data_name=anchor_name,
        run_logger=run_logger
    )
    
    if anchor_labels['n_positive'] == 0:
        run_logger.log(f"[load_anchor_labels] WARNING: No positive labels found in anchor dataset {anchor_name}")
    else:
        run_logger.log(f"[load_anchor_labels] Successfully loaded {anchor_labels['n_positive']} positive labels from {anchor_name}")
    
    return anchor_labels


def compare_label_sources(cfg: Any, anchor_name: str, dcca_sets: Dict[str, Any], run_logger: Any) -> None:
    """
    Compare original labels from cfg vs DCCA processed labels for debugging.
    
    Args:
        cfg: Configuration object
        anchor_name: Name of the anchor dataset
        dcca_sets: DCCA processed datasets
        run_logger: Logger instance
    """
    run_logger.log("[compare_labels] Comparing original vs DCCA processed labels")
    
    # Load original labels
    original_labels = get_original_labels_from_cfg(cfg, anchor_name, run_logger)
    
    # Get DCCA labels
    dcca_data = dcca_sets.get(anchor_name, {})
    dcca_labels = dcca_data.get('labels')
    
    if dcca_labels is not None:
        if hasattr(dcca_labels, 'cpu'):
            dcca_labels = dcca_labels.cpu().numpy()
        
        dcca_positive = int((dcca_labels > 0).sum())
        dcca_negative = int((dcca_labels <= 0).sum())
        
        run_logger.log(f"[compare_labels] Original: {original_labels['n_positive']} pos, {original_labels['n_negative']} neg")
        run_logger.log(f"[compare_labels] DCCA: {dcca_positive} pos, {dcca_negative} neg")
        
        # Check if counts match
        if original_labels['n_positive'] == dcca_positive:
            run_logger.log("[compare_labels] ✅ Positive label counts match")
        else:
            run_logger.log(f"[compare_labels] ⚠️ Positive label count mismatch: original={original_labels['n_positive']}, dcca={dcca_positive}")
    else:
        run_logger.log("[compare_labels] ❌ No DCCA labels found for comparison")


def load_dataset_for_pu_learning(cfg: Any, dataset_name: str, run_logger: Any) -> Dict[str, Any]:
    """
    Load complete dataset optimized for PU learning scenarios.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of dataset to load
        run_logger: Logger instance
        
    Returns:
        Dictionary with embeddings, labels, and positive sample info
    """
    run_logger.log(f"[load_pu_dataset] Loading {dataset_name} for PU learning")
    
    # Load complete dataset
    dataset = get_original_embeddings_from_cfg(cfg, dataset_name, run_logger)
    
    # Extract positive samples separately for convenience
    positives = extract_positive_samples_from_original(dataset, run_logger)
    
    # Add PU learning specific info
    dataset['positive_samples'] = positives
    dataset['pu_ready'] = True
    
    run_logger.log(f"[load_pu_dataset] Loaded {dataset['n_samples']} samples with {dataset['n_positive']} positives")
    
    return dataset


def get_original_dataset_for_training(cfg: Any, dataset_name: str, run_logger: Any) -> Dict[str, Any]:
    """
    Load original dataset in the format expected by train_cls_1_PN_PosDrop function.
    Returns data with 'features', 'labels', and 'coords' keys.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of dataset to load
        run_logger: Logger instance
        
    Returns:
        Dictionary in format expected by training functions:
        {
            'features': np.array,       # Original embeddings [N, dim]
            'labels': np.array,         # Original labels
            'coords': List[Tuple],      # Spatial coordinates (renamed from 'coordinates')
            'n_samples': int,           # Total samples
            'n_positive': int,          # Positive samples
            'dataset_name': str,        # Dataset name
        }
    """
    run_logger.log(f"[get_training_dataset] Loading {dataset_name} for training")
    
    # Load complete dataset
    dataset = get_original_embeddings_from_cfg(cfg, dataset_name, run_logger)
    
    # Convert to training format
    training_data = {
        'features': dataset['embeddings'],      # Rename 'embeddings' to 'features'
        'labels': dataset['labels'],
        'coords': dataset['coordinates'],       # Rename 'coordinates' to 'coords'
        'indices': dataset.get('indices'),
        'n_samples': dataset['n_samples'],
        'n_positive': dataset['n_positive'],
        'n_negative': dataset['n_negative'],
        'dataset_name': dataset['dataset_name'],
        'embedding_path': dataset['embedding_path'],
        'region_filter': dataset['region_filter'],
    }
    
    run_logger.log(f"[get_training_dataset] Prepared {training_data['n_samples']} samples with {training_data['n_positive']} positives for training")
    
    return training_data
