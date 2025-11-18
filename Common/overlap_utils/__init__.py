"""
Overlap utilities for processing and filtering geospatial datasets.

This package provides utilities for working with overlap regions,
including overlap region extraction and spatial filtering operations.
"""

from .overlap_extraction import extract_overlap_only_using_masks

__all__ = [
    'extract_overlap_only_using_masks',
]
