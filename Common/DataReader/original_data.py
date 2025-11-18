"""
Original data loading utilities for extracting embeddings and labels from configuration files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import the existing data loading utilities
from Common.Unifying.Labels_TwoDatasets.datasets import (
    load_embedding_records,
)


def get_original_labels_from_cfg(
    cfg: Any, 
    data_name: str,
    run_logger: Any
) -> Dict[str, Any]:
    """
    Extract only original labels from specified dataset for label substitution.
    Maintains compatibility with current data_use structure while providing access to original labels.
    
    Args:
        cfg: Configuration object containing dataset information
        data_name: Name of the dataset to extract labels from (e.g., anchor_name)
        run_logger: Logger for debugging information
        
    Returns:
        Dictionary containing:
        {
            'labels': np.array,         # Original labels from dataset
            'coordinates': List[Tuple], # Spatial coordinates for matching
            'indices': List,            # Original sample indices
            'n_samples': int,           # Total number of samples
            'n_positive': int,          # Number of positive samples
            'n_negative': int,          # Number of negative/unknown samples
            'embedding_path': str,      # Source file path
            'region_filter': List[str], # Applied region filter
            'dataset_name': str,        # Dataset name
        }
    """
    run_logger.log(f"[get_original_labels] Loading original labels for dataset: {data_name}")
    
    # Find the dataset configuration
    target_dataset_cfg = None
    for dataset_cfg in cfg.datasets:
        if dataset_cfg.name == data_name:
            target_dataset_cfg = dataset_cfg
            break
    
    if target_dataset_cfg is None:
        raise ValueError(f"Dataset '{data_name}' not found in configuration. Available datasets: {[d.name for d in cfg.datasets]}")
    
    embedding_path = target_dataset_cfg.embedding_path
    region_filter = target_dataset_cfg.region_filter
    
    if region_filter:
        run_logger.log(f"[get_original_labels] Applying region filter: {region_filter}")
    
    try:
        # Load embedding records using existing infrastructure
        # Note: load_embedding_records expects the dataset_cfg object, not individual parameters
        records = load_embedding_records(target_dataset_cfg)
        
        if not records:
            run_logger.log(f"[get_original_labels] WARNING: No records loaded from {embedding_path}")
            return _create_empty_label_dict(data_name, embedding_path, region_filter)
        
        
        # Extract labels, coordinates, and indices
        labels = []
        coordinates = []
        indices = []
        
        # Track coordinate extraction methods for debugging
        coord_methods = {'coord': 0, 'row_col': 0, 'metadata': 0, 'tile_id': 0, 'grid_fallback': 0}
        
        for i, record in enumerate(records):
            # Extract label - handle different possible label values
            if hasattr(record, 'label') and record.label is not None:
                # Convert label to int, handling different formats
                if isinstance(record.label, (int, np.integer)):
                    label_val = int(record.label)
                elif isinstance(record.label, (float, np.floating)):
                    label_val = int(record.label)
                elif isinstance(record.label, str):
                    try:
                        label_val = int(float(record.label))
                    except (ValueError, TypeError):
                        label_val = 0  # Default to unknown
                else:
                    label_val = 0  # Default to unknown
            else:
                label_val = 0  # Default to unknown/unlabeled
            
            labels.append(label_val)
            
            # Extract coordinate - try multiple sources
            coord_found = False
            
            # Try record.coord first
            if hasattr(record, 'coord') and record.coord is not None:
                if isinstance(record.coord, (list, tuple)) and len(record.coord) >= 2:
                    coordinates.append((float(record.coord[0]), float(record.coord[1])))
                    coord_found = True
                    coord_methods['coord'] += 1
            
            # Try row_col conversion if coord not found
            if not coord_found and hasattr(record, 'row_col') and record.row_col is not None:
                if isinstance(record.row_col, (list, tuple)) and len(record.row_col) >= 2:
                    # Convert row,col to approximate coordinates (simple grid assumption)
                    row, col = record.row_col
                    # Use row,col as pseudo-coordinates for spatial distribution
                    coordinates.append((float(col), float(row)))
                    coord_found = True
                    coord_methods['row_col'] += 1
            
            # Try metadata coord if available
            if not coord_found and hasattr(record, 'metadata') and record.metadata is not None:
                if isinstance(record.metadata, dict):
                    meta_coord = record.metadata.get('coord') or record.metadata.get('coordinate')
                    if meta_coord and isinstance(meta_coord, (list, tuple)) and len(meta_coord) >= 2:
                        coordinates.append((float(meta_coord[0]), float(meta_coord[1])))
                        coord_found = True
                        coord_methods['metadata'] += 1
            
            # Try extracting from tile_id pattern (e.g., "REGION_row_col")
            if not coord_found and hasattr(record, 'tile_id') and record.tile_id is not None:
                try:
                    parts = str(record.tile_id).split('_')
                    if len(parts) >= 3:  # Assumes format like "REGION_row_col"
                        row_part = parts[-2]
                        col_part = parts[-1]
                        row_val = float(row_part)
                        col_val = float(col_part)
                        coordinates.append((col_val, row_val))
                        coord_found = True
                        coord_methods['tile_id'] += 1
                except (ValueError, IndexError):
                    pass
            
            # Generate spatial distribution if no coordinates found
            if not coord_found:
                # Create a simple grid distribution instead of all (0,0)
                grid_size = int(np.sqrt(len(records))) + 1
                row = i // grid_size
                col = i % grid_size
                coordinates.append((float(col), float(row)))
                coord_found = True
                coord_methods['grid_fallback'] += 1
            
            # Extract index
            if hasattr(record, 'index') and record.index is not None:
                indices.append(record.index)
            else:
                indices.append(i)  # Use position as index
        
        # Convert to numpy array
        labels_array = np.array(labels, dtype=np.int32)
        
        # Calculate statistics
        n_samples = len(labels_array)
        n_positive = int(np.sum(labels_array > 0))
        n_negative = int(np.sum(labels_array <= 0))
        
        
        # Log coordinate extraction methods used
        total_valid_coords = sum(coord_methods.values())
        if coord_methods['grid_fallback'] == total_valid_coords:
            run_logger.log(f"[get_original_labels] WARNING: All coordinates using grid fallback - original data may lack spatial info")
        
        if n_positive == 0:
            run_logger.log(f"[get_original_labels] WARNING: No positive labels found in {data_name}")
        
        return {
            'labels': labels_array,
            'coordinates': coordinates,
            'indices': indices,
            'n_samples': n_samples,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'embedding_path': str(embedding_path),
            'region_filter': region_filter,
            'dataset_name': data_name,
        }
        
    except Exception as e:
        run_logger.log(f"[get_original_labels] ERROR loading {data_name} from {embedding_path}: {e}")
        raise RuntimeError(f"Failed to load original labels for dataset '{data_name}': {e}")


def get_original_embeddings_from_cfg(
    cfg: Any, 
    data_name: str,
    run_logger: Any
) -> Dict[str, Any]:
    """
    Load complete original embeddings and metadata for a specific dataset from cfg file.
    
    Args:
        cfg: Configuration object containing dataset information
        data_name: Name of the dataset to load (e.g., anchor_name, target_name)
        run_logger: Logger for debugging information
        
    Returns:
        Dictionary containing full dataset:
        {
            'embeddings': np.array,      # Original embeddings [N, dim]
            'labels': np.array,         # Original labels
            'coordinates': List[Tuple],  # Spatial coordinates
            'indices': List,            # Sample indices
            'tile_ids': List,           # Tile identifiers
            'metadata': List[Dict],     # Full metadata records
            'n_samples': int,           # Total samples
            'embedding_dim': int,       # Embedding dimension
            'n_positive': int,          # Positive samples
            'n_negative': int,          # Negative/unknown samples
            'embedding_path': str,      # Source file path
            'region_filter': List[str], # Applied region filter
            'dataset_name': str,        # Dataset name
        }
    """
    
    # Get labels first (reuses the label loading logic)
    label_data = get_original_labels_from_cfg(cfg, data_name, run_logger)
    
    # Find the dataset configuration
    target_dataset_cfg = None
    for dataset_cfg in cfg.datasets:
        if dataset_cfg.name == data_name:
            target_dataset_cfg = dataset_cfg
            break
    
    if target_dataset_cfg is None:
        raise ValueError(f"Dataset '{data_name}' not found in configuration")
    
    embedding_path = target_dataset_cfg.embedding_path
    
    try:
        # Load embedding records
        # Note: load_embedding_records expects the dataset_cfg object, not individual parameters
        records = load_embedding_records(target_dataset_cfg)
        
        if not records:
            run_logger.log(f"[get_original_embeddings] WARNING: No records loaded from {embedding_path}")
            return _create_empty_embedding_dict(data_name, embedding_path, label_data.get('region_filter'))
        
        # Extract embeddings and additional metadata
        embeddings = []
        tile_ids = []
        metadata = []
        
        for i, record in enumerate(records):
            # Extract embedding vector
            if hasattr(record, 'embedding') and record.embedding is not None:
                if hasattr(record.embedding, 'numpy'):
                    embeddings.append(record.embedding.numpy())
                elif isinstance(record.embedding, np.ndarray):
                    embeddings.append(record.embedding)
                else:
                    embeddings.append(np.array(record.embedding))
            else:
                run_logger.log(f"[get_original_embeddings] WARNING: Record {i} missing embedding")
                continue
            
            # Extract tile ID
            if hasattr(record, 'tile_id') and record.tile_id is not None:
                tile_ids.append(record.tile_id)
            else:
                tile_ids.append(f"tile_{i}")  # Generate default tile ID
            
            # Store full record as metadata
            metadata.append({
                'embedding_shape': embeddings[-1].shape if embeddings else None,
                'coordinate': record.coordinate if hasattr(record, 'coordinate') else None,
                'label': record.label if hasattr(record, 'label') else None,
                'tile_id': record.tile_id if hasattr(record, 'tile_id') else None,
                'index': record.index if hasattr(record, 'index') else i,
                'dataset_name': data_name,
            })
        
        # Convert embeddings to numpy array
        if embeddings:
            embeddings_array = np.array(embeddings)
            embedding_dim = embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 0
        else:
            embeddings_array = np.empty((0, 0))
            embedding_dim = 0
        
        run_logger.log(f"[get_original_embeddings] Loaded {len(embeddings)} embeddings with dimension {embedding_dim}")
        
        # Combine with label data
        result = {
            'embeddings': embeddings_array,
            'tile_ids': tile_ids,
            'metadata': metadata,
            'embedding_dim': embedding_dim,
            **label_data  # Include all label data
        }
        
        return result
        
    except Exception as e:
        run_logger.log(f"[get_original_embeddings] ERROR loading {data_name}: {e}")
        raise RuntimeError(f"Failed to load original embeddings for dataset '{data_name}': {e}")


def extract_positive_samples_from_original(
    original_data: Dict[str, Any], 
    run_logger: Any
) -> Dict[str, Any]:
    """
    Extract only positive samples from loaded original dataset.
    
    Args:
        original_data: Output from get_original_embeddings_from_cfg or get_original_labels_from_cfg
        run_logger: Logger for debugging
        
    Returns:
        Dictionary containing only positive samples:
        {
            'labels': np.array,         # All ones (positive labels)
            'coordinates': List,        # Positive sample coordinates
            'indices': List,            # Original indices of positive samples
            'n_positives': int,         # Number of positive samples
            'source_dataset': str,      # Source dataset name
            'embeddings': np.array,     # Positive embeddings (if available)
            'tile_ids': List,           # Positive tile IDs (if available)
            'metadata': List,           # Positive metadata (if available)
        }
    """
    dataset_name = original_data.get('dataset_name', 'unknown')
    labels = original_data['labels']
    
    # Find positive samples
    positive_mask = labels > 0
    n_positives = int(positive_mask.sum())
    
    run_logger.log(f"[extract_positives] Found {n_positives} positive samples in {dataset_name}")
    
    if n_positives == 0:
        run_logger.log(f"[extract_positives] WARNING: No positive samples found in {dataset_name}")
        return {
            'labels': np.empty((0,), dtype=np.int32),
            'coordinates': [],
            'indices': [],
            'n_positives': 0,
            'source_dataset': dataset_name,
            'embeddings': np.empty((0, original_data.get('embedding_dim', 0))),
            'tile_ids': [],
            'metadata': [],
        }
    
    # Extract positive samples
    positive_labels = labels[positive_mask]
    positive_coordinates = [coord for i, coord in enumerate(original_data['coordinates']) if positive_mask[i]]
    positive_indices = [idx for i, idx in enumerate(original_data['indices']) if positive_mask[i]]
    
    result = {
        'labels': positive_labels,
        'coordinates': positive_coordinates,
        'indices': positive_indices,
        'n_positives': n_positives,
        'source_dataset': dataset_name,
    }
    
    # Add embeddings if available
    if 'embeddings' in original_data and original_data['embeddings'].size > 0:
        positive_embeddings = original_data['embeddings'][positive_mask]
        result['embeddings'] = positive_embeddings
        run_logger.log(f"[extract_positives] Extracted {len(positive_embeddings)} positive embeddings")
    
    # Add tile IDs if available
    if 'tile_ids' in original_data:
        positive_tile_ids = [tid for i, tid in enumerate(original_data['tile_ids']) if positive_mask[i]]
        result['tile_ids'] = positive_tile_ids
    
    # Add metadata if available
    if 'metadata' in original_data:
        positive_metadata = [meta for i, meta in enumerate(original_data['metadata']) if positive_mask[i]]
        result['metadata'] = positive_metadata
    
    return result


def _create_empty_label_dict(data_name: str, embedding_path: str, region_filter: Optional[List[str]]) -> Dict[str, Any]:
    """Create an empty label dictionary for error cases."""
    return {
        'labels': np.empty((0,), dtype=np.int32),
        'coordinates': [],
        'indices': [],
        'n_samples': 0,
        'n_positive': 0,
        'n_negative': 0,
        'embedding_path': str(embedding_path),
        'region_filter': region_filter,
        'dataset_name': data_name,
    }


def _create_empty_embedding_dict(data_name: str, embedding_path: str, region_filter: Optional[List[str]]) -> Dict[str, Any]:
    """Create an empty embedding dictionary for error cases."""
    return {
        'embeddings': np.empty((0, 0)),
        'tile_ids': [],
        'metadata': [],
        'embedding_dim': 0,
        **_create_empty_label_dict(data_name, embedding_path, region_filter)
    }