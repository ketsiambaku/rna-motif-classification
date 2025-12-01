#!/usr/bin/env python3
"""
RNA Sequence Feature Extractor

Extracts size-invariant sequence features from PDB files for RNA motif classification.
These features capture nucleotide composition and patterns without encoding motif size.

Features extracted (24 total):
1. Nucleotide composition (4): A%, U%, G%, C%
2. Chemical composition (3): GC content, purine%, pyrimidine%
3. Dinucleotide frequencies (16): AA, AU, AG, AC, UA, UU, UG, UC, GA, GU, GG, GC, CA, CU, CG, CC
4. Sequence complexity (1): Shannon entropy (normalized)

All features are normalized by sequence length to ensure size-invariance.

Author: RNA Motif Classification Project
Date: December 1, 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceFeatureExtractor:
    """
    Extract size-invariant sequence features from PDB files.
    
    This extractor parses SEQRES records from PDB files to obtain the RNA sequence,
    then computes composition-based features that are independent of motif size.
    """
    
    def __init__(self):
        """Initialize the sequence feature extractor."""
        self.nucleotides = ['A', 'U', 'G', 'C']
        self.purines = ['A', 'G']
        self.pyrimidines = ['U', 'C']
        
        # Generate all dinucleotide pairs
        self.dinucleotides = [n1 + n2 for n1 in self.nucleotides for n2 in self.nucleotides]
        
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for all 24 features."""
        names = []
        
        # Nucleotide composition (4)
        names.extend([f'{nuc}_percent' for nuc in self.nucleotides])
        
        # Chemical composition (3)
        names.extend(['GC_content', 'purine_percent', 'pyrimidine_percent'])
        
        # Dinucleotide frequencies (16)
        names.extend([f'dinuc_{d}' for d in self.dinucleotides])
        
        # Sequence complexity (1)
        names.append('shannon_entropy')
        
        return names
    
    def extract_sequence_from_pdb(self, pdb_path: str) -> Optional[str]:
        """
        Extract RNA sequence from PDB file by parsing SEQRES records.
        
        SEQRES records contain the full sequence of the molecule. Format:
        SEQRES   1 A   30  G   C   G   A   U   ...
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            RNA sequence as string, or None if extraction fails
        """
        try:
            pdb_path = Path(pdb_path)
            if not pdb_path.exists():
                logger.warning(f"PDB file not found: {pdb_path}")
                return None
            
            sequence_parts = []
            
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('SEQRES'):
                        # SEQRES format: SEQRES   1 A   30  G   C   G   A   U   ...
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        
                        # Residues start at index 4
                        residues = parts[4:]
                        
                        # Convert 3-letter codes to 1-letter if needed
                        for residue in residues:
                            # Common RNA residues
                            residue_map = {
                                'A': 'A', 'ADE': 'A',
                                'U': 'U', 'URA': 'U', 'URI': 'U',
                                'G': 'G', 'GUA': 'G',
                                'C': 'C', 'CYT': 'C',
                                # Modified nucleotides (map to standard)
                                'PSU': 'U',  # pseudouridine
                                'H2U': 'U',  # dihydrouridine
                                'M2G': 'G',  # dimethylguanosine
                                '1MA': 'A',  # 1-methyladenosine
                            }
                            
                            residue_upper = residue.upper()
                            if residue_upper in residue_map:
                                sequence_parts.append(residue_map[residue_upper])
            
            if not sequence_parts:
                logger.warning(f"No SEQRES records found in {pdb_path}")
                return None
            
            sequence = ''.join(sequence_parts)
            
            # Validate sequence contains only valid nucleotides
            valid_nucs = set(self.nucleotides)
            if not all(nuc in valid_nucs for nuc in sequence):
                logger.warning(f"Sequence contains invalid nucleotides: {pdb_path}")
                # Filter to valid nucleotides only
                sequence = ''.join([nuc for nuc in sequence if nuc in valid_nucs])
            
            return sequence if sequence else None
            
        except Exception as e:
            logger.error(f"Error extracting sequence from {pdb_path}: {e}")
            return None
    
    def compute_nucleotide_composition(self, sequence: str) -> np.ndarray:
        """
        Compute nucleotide composition (A%, U%, G%, C%).
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            Array of 4 percentages (sum = 100%)
        """
        if not sequence:
            return np.zeros(4)
        
        seq_length = len(sequence)
        counts = Counter(sequence)
        
        composition = np.array([
            (counts.get(nuc, 0) / seq_length) * 100
            for nuc in self.nucleotides
        ])
        
        return composition
    
    def compute_chemical_composition(self, sequence: str) -> np.ndarray:
        """
        Compute chemical composition: GC content, purine%, pyrimidine%.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            Array of 3 percentages
        """
        if not sequence:
            return np.zeros(3)
        
        seq_length = len(sequence)
        counts = Counter(sequence)
        
        gc_count = counts.get('G', 0) + counts.get('C', 0)
        purine_count = sum(counts.get(nuc, 0) for nuc in self.purines)
        pyrimidine_count = sum(counts.get(nuc, 0) for nuc in self.pyrimidines)
        
        gc_content = (gc_count / seq_length) * 100
        purine_percent = (purine_count / seq_length) * 100
        pyrimidine_percent = (pyrimidine_count / seq_length) * 100
        
        return np.array([gc_content, purine_percent, pyrimidine_percent])
    
    def compute_dinucleotide_frequencies(self, sequence: str) -> np.ndarray:
        """
        Compute dinucleotide frequencies (16 features: AA, AU, AG, ..., CC).
        
        Normalized by total number of dinucleotides (length - 1).
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            Array of 16 frequencies (percentages)
        """
        if not sequence or len(sequence) < 2:
            return np.zeros(16)
        
        # Count dinucleotides
        dinuc_counts = Counter()
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            if all(nuc in self.nucleotides for nuc in dinuc):
                dinuc_counts[dinuc] += 1
        
        total_dinucs = len(sequence) - 1
        
        # Compute frequencies in order
        frequencies = np.array([
            (dinuc_counts.get(dinuc, 0) / total_dinucs) * 100
            for dinuc in self.dinucleotides
        ])
        
        return frequencies
    
    def compute_shannon_entropy(self, sequence: str) -> float:
        """
        Compute Shannon entropy of nucleotide distribution (normalized).
        
        H = -sum(p_i * log2(p_i)) / log2(4)
        
        Normalized by log2(4) so max entropy = 1.0 (uniform distribution).
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            Normalized Shannon entropy (0-1)
        """
        if not sequence:
            return 0.0
        
        seq_length = len(sequence)
        counts = Counter(sequence)
        
        # Compute probabilities
        probs = np.array([counts.get(nuc, 0) / seq_length for nuc in self.nucleotides])
        
        # Filter out zero probabilities
        probs = probs[probs > 0]
        
        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by maximum possible entropy (log2(4) = 2)
        normalized_entropy = entropy / 2.0
        
        return normalized_entropy
    
    def extract_features(self, pdb_path: str) -> Optional[np.ndarray]:
        """
        Extract all 24 size-invariant sequence features from PDB file.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Array of 24 features, or None if extraction fails
        """
        # Extract sequence
        sequence = self.extract_sequence_from_pdb(pdb_path)
        
        if sequence is None or len(sequence) == 0:
            logger.warning(f"Failed to extract valid sequence from {pdb_path}")
            return None
        
        # Compute all feature groups
        nuc_comp = self.compute_nucleotide_composition(sequence)  # 4
        chem_comp = self.compute_chemical_composition(sequence)   # 3
        dinuc_freq = self.compute_dinucleotide_frequencies(sequence)  # 16
        entropy = self.compute_shannon_entropy(sequence)  # 1
        
        # Concatenate all features
        features = np.concatenate([
            nuc_comp,
            chem_comp,
            dinuc_freq,
            [entropy]
        ])
        
        assert len(features) == 24, f"Expected 24 features, got {len(features)}"
        
        return features
    
    def validate_size_invariance(self, 
                                  pdb_paths: List[str], 
                                  motif_sizes: List[int],
                                  threshold: float = 0.1) -> Dict[str, float]:
        """
        Validate that extracted features are size-invariant.
        
        Computes correlation between each feature and motif size.
        Features should have |correlation| < threshold to be size-invariant.
        
        Args:
            pdb_paths: List of PDB file paths
            motif_sizes: List of motif sizes (same length as pdb_paths)
            threshold: Maximum acceptable |correlation| (default: 0.1)
            
        Returns:
            Dictionary with feature correlations and validation results
        """
        assert len(pdb_paths) == len(motif_sizes), "Length mismatch"
        
        # Extract features for all samples
        all_features = []
        valid_sizes = []
        
        for pdb_path, size in zip(pdb_paths, motif_sizes):
            features = self.extract_features(pdb_path)
            if features is not None:
                all_features.append(features)
                valid_sizes.append(size)
        
        if len(all_features) == 0:
            logger.error("No features extracted for validation")
            return {}
        
        features_array = np.array(all_features)  # (n_samples, 24)
        sizes_array = np.array(valid_sizes)
        
        # Compute correlation for each feature
        correlations = {}
        violations = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features_array[:, i]
            corr = np.corrcoef(feature_values, sizes_array)[0, 1]
            correlations[feature_name] = corr
            
            if abs(corr) >= threshold:
                violations.append((feature_name, corr))
        
        # Report results
        logger.info(f"Size-invariance validation on {len(all_features)} samples:")
        logger.info(f"  Threshold: |correlation| < {threshold}")
        logger.info(f"  Violations: {len(violations)}/{len(self.feature_names)}")
        
        if violations:
            logger.warning("Features violating size-invariance:")
            for name, corr in violations:
                logger.warning(f"  {name}: {corr:.4f}")
        else:
            logger.info("✓ All features are size-invariant!")
        
        return {
            'correlations': correlations,
            'violations': violations,
            'threshold': threshold,
            'pass': len(violations) == 0
        }


def main():
    """Test the sequence feature extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract sequence features from PDB file')
    parser.add_argument('pdb_file', help='Path to PDB file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = SequenceFeatureExtractor()
    
    print(f"Extracting features from: {args.pdb_file}")
    
    # Extract sequence
    sequence = extractor.extract_sequence_from_pdb(args.pdb_file)
    if sequence:
        print(f"Sequence length: {len(sequence)}")
        print(f"Sequence: {sequence[:50]}..." if len(sequence) > 50 else f"Sequence: {sequence}")
    
    # Extract features
    features = extractor.extract_features(args.pdb_file)
    
    if features is not None:
        print(f"\nExtracted {len(features)} features:")
        for name, value in zip(extractor.feature_names, features):
            print(f"  {name:20s}: {value:8.4f}")
        print(f"\nFeature vector: {features}")
    else:
        print("Failed to extract features")


if __name__ == '__main__':
    main()
