"""
Utilities for extracting overlap-region data from datasets using spatial masks.

This module provides functions to filter geospatial datasets based on overlap masks,
keeping only samples that fall within designated overlap regions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from affine import Affine
except ImportError:
    Affine = None


def extract_overlap_only_using_masks(
    data: Dict[str, Any],
    overlap_mask: Dict[str, Any],
    run_logger: Any
) -> Dict[str, Any]:
    """
    Extract samples from dataset that fall within the overlap region defined by a spatial mask.
    
    This function filters a geospatial dataset to retain only samples whose coordinates
    fall within regions marked as overlap in the provided mask. The mask is typically
    a rasterized representation of overlap regions between two datasets.
    
    Args:
        data: Dictionary containing dataset with keys:
            - 'features': np.ndarray of shape [N, dim] - feature vectors/embeddings
            - 'labels': np.ndarray of shape [N] - sample labels
            - 'coords': List[Tuple[float, float]] - spatial coordinates (x, y) or (lon, lat)
            - 'n_samples': int - total number of samples
            - 'n_positive': int - number of positive samples
            - Additional metadata fields are preserved
            
        overlap_mask: Dictionary containing mask information from _load_overlap_mask_data():
            - 'array': np.ndarray - 2D raster mask (1=overlap, 0=non-overlap)
            - 'transform': affine.Affine - affine transform for coordinate-to-pixel conversion
            - 'shape': Tuple[int, int] - (height, width) of mask array
            - 'nodata': Optional value representing nodata/masked pixels
            
        run_logger: Logger instance for progress reporting
        
    Returns:
        Dictionary with same structure as input data but filtered to overlap samples only:
        {
            'features': np.ndarray [M, dim] where M <= N,
            'labels': np.ndarray [M],
            'coords': List[Tuple] of length M,
            'n_samples': M,
            'n_positive': count of positive samples in M,
            ... (other metadata preserved)
        }
        
    Example:
        >>> data = get_original_dataset_for_training(cfg, "dataset_A", logger)
        >>> mask = _load_overlap_mask_data(Path("overlap_mask.tif"))
        >>> overlap_data = extract_overlap_only_using_masks(data, mask, logger)
        >>> print(f"Retained {overlap_data['n_samples']} / {data['n_samples']} samples")
    
    Notes:
        - Coordinates are assumed to be in the same coordinate reference system as the mask
        - Samples with None/invalid coordinates are excluded
        - The mask value is checked: pixels with value 1 (or non-zero) are considered overlap
        - Out-of-bounds coordinates (outside mask extent) are excluded
    """
    
    if overlap_mask is None:
        run_logger.log("[extract_overlap] No overlap mask provided, returning original data")
        return data
    
    # Extract mask components
    mask_array = overlap_mask.get("array")
    mask_transform = overlap_mask.get("transform")
    mask_shape = overlap_mask.get("shape")
    nodata_value = overlap_mask.get("nodata")
    
    if mask_array is None or mask_transform is None:
        run_logger.log("[extract_overlap] Invalid mask data, returning original data")
        return data
    
    if not isinstance(mask_transform, Affine):
        run_logger.log("[extract_overlap] Mask transform is not an Affine object, attempting conversion")
        try:
            # Try to convert from tuple/list to Affine
            if isinstance(mask_transform, (list, tuple)) and len(mask_transform) >= 6:
                mask_transform = Affine(*mask_transform[:6])
            else:
                run_logger.log("[extract_overlap] Could not convert mask transform, returning original data")
                return data
        except Exception as e:
            run_logger.log(f"[extract_overlap] Error converting mask transform: {e}, returning original data")
            return data
    
    # Get dataset components
    features = data.get("features")
    labels = data.get("labels")
    # Prioritize 'coords' since that's what get_original_dataset_for_training provides
    coords = data.get("coords", data.get("coordinates", []))
    
    if features is None or labels is None or not coords:
        run_logger.log("[extract_overlap] Missing required data fields (features/labels/coords), returning original data")
        return data
    
    n_original = len(coords)
    run_logger.log(f"[extract_overlap] Processing {n_original} samples with overlap mask")
    
    # Determine which samples fall within overlap region
    keep_indices = []
    for idx, coord in enumerate(coords):
        if coord is None or not isinstance(coord, (tuple, list)) or len(coord) < 2:
            continue
            
        x, y = coord[0], coord[1]
        
        # Convert geographic coordinate to pixel indices using inverse transform
        try:
            # Use the inverse transform to convert from geographic to pixel coordinates
            col, row = ~mask_transform * (x, y)
            col_int, row_int = int(round(col)), int(round(row))
            
            # Check if pixel is within mask bounds
            if 0 <= row_int < mask_shape[0] and 0 <= col_int < mask_shape[1]:
                pixel_value = mask_array[row_int, col_int]
                
                # Check if pixel is in overlap region (non-zero and not nodata)
                if nodata_value is not None and pixel_value == nodata_value:
                    continue
                    
                # Consider non-zero values as overlap
                if pixel_value != 0:
                    keep_indices.append(idx)
        except Exception:
            # If coordinate transformation fails, skip this sample
            continue
    
    n_kept = len(keep_indices)
    run_logger.log(f"[extract_overlap] Retained {n_kept} / {n_original} samples ({100.0*n_kept/n_original:.1f}%) in overlap region")
    
    if n_kept == 0:
        run_logger.log("[extract_overlap] WARNING: No samples found in overlap region!")
        # Return empty dataset with same structure
        return {
            'features': features[:0],  # Empty array with same dtype
            'labels': labels[:0],
            'coords': [],
            'n_samples': 0,
            'n_positive': 0,
            'n_negative': 0,
            'dataset_name': data.get('dataset_name', ''),
            'embedding_path': data.get('embedding_path', ''),
            'region_filter': data.get('region_filter', []),
        }
    
    # Filter all arrays/lists by keep_indices
    keep_indices_array = np.array(keep_indices)
    
    filtered_features = features[keep_indices_array] if isinstance(features, np.ndarray) else np.array([features[i] for i in keep_indices])
    filtered_labels = labels[keep_indices_array] if isinstance(labels, np.ndarray) else np.array([labels[i] for i in keep_indices])
    filtered_coords = [coords[i] for i in keep_indices]
    
    # Count positive/negative samples in filtered data
    n_positive = int(np.sum(filtered_labels > 0))
    n_negative = int(np.sum(filtered_labels == 0))
    
    run_logger.log(f"[extract_overlap] Filtered data contains {n_positive} positive and {n_negative} negative samples")
    
    # Build filtered dataset dictionary
    filtered_data = {
        'features': filtered_features,
        'labels': filtered_labels,
        'coords': filtered_coords,  # Primary key used by training functions
        'n_samples': n_kept,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'dataset_name': data.get('dataset_name', ''),
        'embedding_path': data.get('embedding_path', ''),
        'region_filter': data.get('region_filter', []),
    }
    
    # Preserve any additional fields from original data
    for key, value in data.items():
        if key not in filtered_data:
            # For list/array fields that should be filtered
            if isinstance(value, (list, tuple)) and len(value) == n_original:
                try:
                    filtered_data[key] = [value[i] for i in keep_indices]
                except (IndexError, TypeError):
                    filtered_data[key] = value  # Keep original if filtering fails
            elif isinstance(value, np.ndarray) and value.shape[0] == n_original:
                try:
                    filtered_data[key] = value[keep_indices_array]
                except (IndexError, ValueError):
                    filtered_data[key] = value  # Keep original if filtering fails
            else:
                # Scalar/metadata fields - preserve as-is
                filtered_data[key] = value
    
    return filtered_data


__all__ = [
    'extract_overlap_only_using_masks',
]
