"""
Feature extraction modules for RNA motif classification.

This package contains extractors for size-invariant features from PDB structures.
"""

from .sequence_features import SequenceFeatureExtractor

__all__ = ['SequenceFeatureExtractor']
