#!/usr/bin/env python3
"""
RNA Base Pairing Feature Extractor

Extracts size-invariant base pairing features from PDB files for RNA motif classification.
Features capture Watson-Crick and non-canonical base pairing topology without encoding motif size.

Features extracted (~10 total):
1. Pairing ratio (% of residues in pairs)
2. Pairing density (pairs per nucleotide, normalized)
3. Average pairing distance (Å, normalized by motif size)
4. Pairing distance std dev (normalized)
5. Stem length statistics (mean, max, normalized)
6. Loop closure indicator (binary)
7. GC pairing percentage (among all pairs)
8. AU pairing percentage (among all pairs)
9. Non-canonical pair percentage
10. Paired vs unpaired ratio

Detection method:
- Watson-Crick pairs: N1(purine)...N3(pyrimidine) distance < 3.5 Å
- A-U: N1(A) ... N3(U)
- G-C: N1(G) ... N3(C)
- Non-canonical: other N1/N3 distances < 4.0 Å

Author: RNA Motif Classification Project
Date: December 1, 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging

try:
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Residue import Residue
except ImportError:
    raise ImportError("BioPython is required. Install with: pip install biopython")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePairingExtractor:
    """
    Extract size-invariant base pairing features from PDB files.
    
    This extractor parses PDB files to obtain atomic coordinates, detects
    Watson-Crick and non-canonical base pairs using N1/N3 atom distances,
    then computes topology-based features that are independent of motif size.
    """
    
    def __init__(self, wc_threshold: float = 3.5, noncanonical_threshold: float = 4.0):
        """
        Initialize the base pairing feature extractor.
        
        Args:
            wc_threshold: Distance threshold (Å) for Watson-Crick pairs (default: 3.5)
            noncanonical_threshold: Distance threshold (Å) for non-canonical pairs (default: 4.0)
        """
        self.wc_threshold = wc_threshold
        self.noncanonical_threshold = noncanonical_threshold
        
        # RNA nucleotides
        self.purines = {'A', 'G', 'DA', 'DG'}  # Include DNA variants
        self.pyrimidines = {'U', 'C', 'T', 'DU', 'DC', 'DT'}  # T for DNA, U for RNA
        
        # Watson-Crick pairs
        self.wc_pairs = {
            ('A', 'U'), ('U', 'A'),
            ('G', 'C'), ('C', 'G'),
            ('A', 'T'), ('T', 'A'),  # DNA compatibility
        }
        
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for all ~10 features."""
        return [
            'pairing_ratio',           # % of residues in pairs
            'pairing_density',         # pairs per nucleotide
            'avg_pairing_distance',    # mean distance (normalized)
            'pairing_distance_std',    # std dev (normalized)
            'mean_stem_length',        # average consecutive pairs (normalized)
            'max_stem_length',         # longest stem (normalized)
            'loop_closure',            # binary: has closing pair
            'gc_pair_percentage',      # % GC among pairs
            'au_pair_percentage',      # % AU among pairs
            'noncanonical_percentage', # % non-WC pairs
        ]
    
    def parse_pdb_structure(self, pdb_path: str) -> Optional[Structure]:
        """
        Parse PDB file and return BioPython Structure object.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Structure object or None if parsing fails
        """
        try:
            pdb_path = Path(pdb_path)
            if not pdb_path.exists():
                logger.warning(f"PDB file not found: {pdb_path}")
                return None
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('rna', str(pdb_path))
            
            return structure
            
        except Exception as e:
            logger.error(f"Error parsing PDB {pdb_path}: {e}")
            return None
    
    def get_nucleotide_atoms(self, structure: Structure) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Extract nucleotide residues and their N1/N3 atom coordinates.
        
        Args:
            structure: BioPython Structure object
            
        Returns:
            Dict mapping residue_id to {'resname': str, 'n1': coords, 'n3': coords}
        """
        nucleotides = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    
                    # Check if RNA/DNA nucleotide
                    if resname not in (self.purines | self.pyrimidines):
                        continue
                    
                    res_id = residue.get_id()[1]  # Get residue number
                    
                    # Get N1 (purine) or N3 (pyrimidine) atoms
                    n1_coord = None
                    n3_coord = None
                    
                    try:
                        if resname in self.purines and 'N1' in residue:
                            n1_coord = residue['N1'].get_coord()
                        if resname in self.pyrimidines and 'N3' in residue:
                            n3_coord = residue['N3'].get_coord()
                        
                        # Store residue info
                        nucleotides[res_id] = {
                            'resname': resname,
                            'n1': n1_coord,
                            'n3': n3_coord,
                            'residue': residue
                        }
                    except KeyError:
                        # Atom not found, skip this residue
                        logger.debug(f"N1/N3 atoms not found for residue {res_id} ({resname})")
                        continue
        
        return nucleotides
    
    def detect_base_pairs(self, nucleotides: Dict[int, Dict]) -> List[Tuple[int, int, float, str]]:
        """
        Detect base pairs using N1-N3 distance criterion.
        
        Args:
            nucleotides: Dict from get_nucleotide_atoms()
            
        Returns:
            List of tuples: (res_id1, res_id2, distance, pair_type)
            pair_type: 'WC' (Watson-Crick) or 'NC' (non-canonical)
        """
        pairs = []
        res_ids = sorted(nucleotides.keys())
        
        for i, res_id1 in enumerate(res_ids):
            nuc1 = nucleotides[res_id1]
            resname1 = nuc1['resname']
            
            for res_id2 in res_ids[i+1:]:  # Only check pairs once (i < j)
                nuc2 = nucleotides[res_id2]
                resname2 = nuc2['resname']
                
                # Skip if same residue or adjacent residues (no self-pairing)
                if abs(res_id2 - res_id1) <= 1:
                    continue
                
                # Calculate N1-N3 distance
                distance = None
                
                # Purine-Pyrimidine pairing (standard)
                if resname1 in self.purines and resname2 in self.pyrimidines:
                    if nuc1['n1'] is not None and nuc2['n3'] is not None:
                        distance = np.linalg.norm(nuc1['n1'] - nuc2['n3'])
                
                elif resname1 in self.pyrimidines and resname2 in self.purines:
                    if nuc1['n3'] is not None and nuc2['n1'] is not None:
                        distance = np.linalg.norm(nuc1['n3'] - nuc2['n1'])
                
                # Check if distance indicates pairing
                if distance is not None:
                    # Determine pair type
                    is_wc = (resname1, resname2) in self.wc_pairs
                    
                    if distance < self.wc_threshold and is_wc:
                        pairs.append((res_id1, res_id2, distance, 'WC'))
                    elif distance < self.noncanonical_threshold:
                        pairs.append((res_id1, res_id2, distance, 'NC'))
        
        return pairs
    
    def compute_stem_lengths(self, pairs: List[Tuple[int, int, float, str]]) -> List[int]:
        """
        Compute lengths of consecutive base pair stems.
        
        A stem is a set of consecutive base pairs (i, j), (i+1, j-1), (i+2, j-2), ...
        
        Args:
            pairs: List of base pairs from detect_base_pairs()
            
        Returns:
            List of stem lengths
        """
        if not pairs:
            return []
        
        # Sort pairs by first residue
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        
        stems = []
        current_stem = 1
        
        for i in range(1, len(sorted_pairs)):
            prev_pair = sorted_pairs[i-1]
            curr_pair = sorted_pairs[i]
            
            # Check if consecutive stem: (i, j) → (i+1, j-1)
            if (curr_pair[0] == prev_pair[0] + 1 and 
                curr_pair[1] == prev_pair[1] - 1):
                current_stem += 1
            else:
                stems.append(current_stem)
                current_stem = 1
        
        stems.append(current_stem)  # Add last stem
        
        return stems
    
    def compute_pairing_features(self, 
                                  nucleotides: Dict[int, Dict], 
                                  pairs: List[Tuple[int, int, float, str]]) -> np.ndarray:
        """
        Compute all ~10 size-invariant base pairing features.
        
        Args:
            nucleotides: Dict from get_nucleotide_atoms()
            pairs: List of base pairs from detect_base_pairs()
            
        Returns:
            Array of 10 features
        """
        n_residues = len(nucleotides)
        n_pairs = len(pairs)
        
        if n_residues == 0:
            return np.zeros(10)
        
        # Feature 1: Pairing ratio (% residues in pairs)
        paired_residues = set()
        for res1, res2, _, _ in pairs:
            paired_residues.add(res1)
            paired_residues.add(res2)
        
        pairing_ratio = (len(paired_residues) / n_residues) * 100
        
        # Feature 2: Pairing density (pairs per nucleotide)
        pairing_density = n_pairs / n_residues
        
        # Feature 3-4: Average and std dev of pairing distances (normalized by sqrt(n_residues))
        if n_pairs > 0:
            distances = [dist for _, _, dist, _ in pairs]
            avg_distance = np.mean(distances) / np.sqrt(n_residues)
            std_distance = np.std(distances) / np.sqrt(n_residues) if len(distances) > 1 else 0.0
        else:
            avg_distance = 0.0
            std_distance = 0.0
        
        # Feature 5-6: Stem length statistics (normalized by n_residues)
        stem_lengths = self.compute_stem_lengths(pairs)
        if stem_lengths:
            mean_stem = np.mean(stem_lengths) / n_residues
            max_stem = np.max(stem_lengths) / n_residues
        else:
            mean_stem = 0.0
            max_stem = 0.0
        
        # Feature 7: Loop closure (binary: 1 if first and last residues are paired)
        res_ids = sorted(nucleotides.keys())
        if len(res_ids) >= 2:
            first_res = res_ids[0]
            last_res = res_ids[-1]
            loop_closure = 1.0 if (first_res, last_res) in [(p[0], p[1]) for p in pairs] else 0.0
        else:
            loop_closure = 0.0
        
        # Feature 8-9: GC and AU pair percentages
        if n_pairs > 0:
            gc_pairs = sum(1 for p in pairs if self._is_gc_pair(nucleotides[p[0]]['resname'], 
                                                                 nucleotides[p[1]]['resname']))
            au_pairs = sum(1 for p in pairs if self._is_au_pair(nucleotides[p[0]]['resname'], 
                                                                 nucleotides[p[1]]['resname']))
            gc_percentage = (gc_pairs / n_pairs) * 100
            au_percentage = (au_pairs / n_pairs) * 100
        else:
            gc_percentage = 0.0
            au_percentage = 0.0
        
        # Feature 10: Non-canonical pair percentage
        if n_pairs > 0:
            nc_pairs = sum(1 for p in pairs if p[3] == 'NC')
            nc_percentage = (nc_pairs / n_pairs) * 100
        else:
            nc_percentage = 0.0
        
        # Combine all features
        features = np.array([
            pairing_ratio,
            pairing_density,
            avg_distance,
            std_distance,
            mean_stem,
            max_stem,
            loop_closure,
            gc_percentage,
            au_percentage,
            nc_percentage
        ])
        
        return features
    
    def _is_gc_pair(self, res1: str, res2: str) -> bool:
        """Check if pair is G-C."""
        return (res1, res2) in {('G', 'C'), ('C', 'G')}
    
    def _is_au_pair(self, res1: str, res2: str) -> bool:
        """Check if pair is A-U or A-T."""
        return (res1, res2) in {('A', 'U'), ('U', 'A'), ('A', 'T'), ('T', 'A')}
    
    def extract_features(self, pdb_path: str) -> Optional[np.ndarray]:
        """
        Extract all ~10 size-invariant base pairing features from PDB file.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Array of 10 features, or None if extraction fails
        """
        # Parse structure
        structure = self.parse_pdb_structure(pdb_path)
        if structure is None:
            logger.warning(f"Failed to parse structure from {pdb_path}")
            return None
        
        # Get nucleotide atoms
        nucleotides = self.get_nucleotide_atoms(structure)
        if not nucleotides:
            logger.warning(f"No nucleotides found in {pdb_path}")
            return None
        
        # Detect base pairs
        pairs = self.detect_base_pairs(nucleotides)
        
        # Compute features
        features = self.compute_pairing_features(nucleotides, pairs)
        
        assert len(features) == 10, f"Expected 10 features, got {len(features)}"
        
        return features
    
    def validate_size_invariance(self, 
                                  pdb_paths: List[str], 
                                  motif_sizes: List[int],
                                  threshold: float = 0.3) -> Dict[str, float]:
        """
        Validate that extracted features are size-invariant.
        
        Computes correlation between each feature and motif size.
        Features should have |correlation| < threshold to be size-invariant.
        
        Args:
            pdb_paths: List of PDB file paths
            motif_sizes: List of motif sizes (same length as pdb_paths)
            threshold: Maximum acceptable |correlation| (default: 0.3)
            
        Returns:
            Dict mapping feature names to correlations
        """
        if len(pdb_paths) != len(motif_sizes):
            raise ValueError("pdb_paths and motif_sizes must have same length")
        
        # Extract features for all samples
        feature_matrix = []
        for pdb_path in pdb_paths:
            features = self.extract_features(pdb_path)
            if features is not None:
                feature_matrix.append(features)
            else:
                # Use zeros for failed extractions
                feature_matrix.append(np.zeros(10))
        
        feature_matrix = np.array(feature_matrix)
        
        # Compute correlations
        correlations = {}
        for i, feature_name in enumerate(self.feature_names):
            corr = np.corrcoef(feature_matrix[:, i], motif_sizes)[0, 1]
            correlations[feature_name] = corr
            
            if abs(corr) > threshold:
                logger.warning(f"Feature '{feature_name}' has high correlation with size: {corr:.3f}")
        
        return correlations


def test_base_pairing_extractor():
    """Test the BasePairingExtractor on sample data."""
    import time
    
    print("="*80)
    print("Testing BasePairingExtractor")
    print("="*80)
    
    # Initialize extractor
    extractor = BasePairingExtractor()
    
    # Test on sample PDB file
    dataset_path = Path("/Users/ketsiambaku/Repositories/rna-motif-classification/dataset2")
    
    # Find a sample PDB file
    sample_classes = ['1x1', 'bulge1', 'hairpin3']
    
    for class_name in sample_classes:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        pdb_files = list(class_dir.glob('*.pdb'))
        if not pdb_files:
            continue
        
        pdb_path = pdb_files[0]
        print(f"\nTesting on: {pdb_path.name} (class: {class_name})")
        print("-" * 60)
        
        start_time = time.time()
        features = extractor.extract_features(str(pdb_path))
        elapsed = time.time() - start_time
        
        if features is not None:
            print(f"Extraction time: {elapsed*1000:.2f} ms")
            print(f"\nFeatures extracted ({len(features)} total):")
            for name, value in zip(extractor.feature_names, features):
                print(f"  {name:30s}: {value:8.4f}")
        else:
            print("❌ Feature extraction failed")
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == '__main__':
    test_base_pairing_extractor()
