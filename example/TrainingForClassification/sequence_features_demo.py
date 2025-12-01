"""
Demonstration of sequence feature extraction from PDB files.

This script shows different ways to encode RNA sequence information:
1. One-hot encoding (4D per nucleotide)
2. K-mer composition (di-nucleotide, tri-nucleotide frequencies)
3. Physicochemical properties (purine/pyrimidine, ring structure)
4. Position-specific encoding
"""

import numpy as np
from Bio.PDB import PDBParser
from collections import Counter
import pandas as pd

# Nucleotide mappings
NUCLEOTIDES = ['A', 'C', 'G', 'U']  # RNA bases
NUCLEOTIDE_TO_IDX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# Physicochemical properties
PURINE = {'A', 'G'}  # Two rings
PYRIMIDINE = {'C', 'U'}  # One ring
AMINO = {'A', 'C'}  # Amino group at position 6/4
KETO = {'G', 'U'}  # Keto group at position 6/4


def extract_sequence_from_pdb(pdb_file):
    """Extract RNA sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_file)
    
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                # Handle modified nucleotides
                if resname in ['A', 'G', 'C', 'U']:
                    sequence.append(resname)
                elif resname in ['ADE', 'GUA', 'CYT', 'URA']:
                    sequence.append(resname[0])
                # Handle common modifications
                elif resname in ['PSU', 'H2U']:  # Pseudouridine
                    sequence.append('U')
                elif resname in ['5MC', 'OMC']:  # Modified C
                    sequence.append('C')
                elif resname in ['1MA', '6MA']:  # Modified A
                    sequence.append('A')
                elif resname in ['7MG', 'OMG']:  # Modified G
                    sequence.append('G')
    
    return ''.join(sequence)


def one_hot_encode(sequence, max_length=30):
    """
    One-hot encode RNA sequence.
    
    Args:
        sequence: String of nucleotides (A, C, G, U)
        max_length: Maximum sequence length (pad or truncate)
    
    Returns:
        numpy array of shape (max_length, 4)
    """
    # Truncate or pad sequence
    seq = sequence[:max_length]
    seq = seq + 'N' * (max_length - len(seq))  # Pad with N
    
    encoding = np.zeros((max_length, 4))
    for i, nt in enumerate(seq):
        if nt in NUCLEOTIDE_TO_IDX:
            encoding[i, NUCLEOTIDE_TO_IDX[nt]] = 1
    
    return encoding


def kmer_composition(sequence, k=2):
    """
    Compute k-mer composition features.
    
    For k=2 (dinucleotides): 4^2 = 16 features
    For k=3 (trinucleotides): 4^3 = 64 features
    
    Args:
        sequence: String of nucleotides
        k: k-mer length
    
    Returns:
        Dictionary of k-mer frequencies
    """
    if len(sequence) < k:
        return {}
    
    # Generate all possible k-mers
    all_kmers = [''.join(combo) for combo in 
                 np.array(np.meshgrid(*[NUCLEOTIDES]*k)).T.reshape(-1, k)]
    
    # Count k-mers in sequence
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    counts = Counter(kmers)
    
    # Normalize to frequencies
    total = len(kmers)
    frequencies = {kmer: counts.get(kmer, 0) / total for kmer in all_kmers}
    
    return frequencies


def physicochemical_features(sequence):
    """
    Compute physicochemical property features.
    
    Features:
    - Purine/Pyrimidine ratio
    - GC content
    - AT content
    - Amino/Keto ratio
    - Position-weighted features
    
    Returns:
        numpy array of features
    """
    if not sequence:
        return np.zeros(10)
    
    seq_set = set(sequence)
    n = len(sequence)
    
    # Base composition
    purine_count = sum(1 for nt in sequence if nt in PURINE)
    pyrimidine_count = sum(1 for nt in sequence if nt in PYRIMIDINE)
    gc_count = sequence.count('G') + sequence.count('C')
    at_count = sequence.count('A') + sequence.count('U')
    
    features = [
        purine_count / n,                # Purine content
        pyrimidine_count / n,            # Pyrimidine content
        gc_count / n,                    # GC content
        at_count / n,                    # AU content
        sequence.count('A') / n,         # A content
        sequence.count('C') / n,         # C content
        sequence.count('G') / n,         # G content
        sequence.count('U') / n,         # U content
        len(seq_set) / 4,                # Nucleotide diversity
        n,                               # Sequence length
    ]
    
    return np.array(features)


def position_weighted_encoding(sequence, max_length=30):
    """
    Encode sequence with position weighting.
    
    Important for capturing position-dependent patterns in motifs.
    
    Returns:
        Flattened array of position-weighted one-hot encoding
    """
    one_hot = one_hot_encode(sequence, max_length)
    
    # Apply position weights (center positions weighted higher)
    positions = np.arange(max_length)
    center = max_length / 2
    weights = np.exp(-((positions - center) ** 2) / (2 * (max_length / 4) ** 2))
    weights = weights.reshape(-1, 1)
    
    weighted = one_hot * weights
    return weighted.flatten()


def extract_all_sequence_features(pdb_file, max_length=30):
    """
    Extract comprehensive sequence features from PDB file.
    
    Returns:
        Dictionary with different feature representations
    """
    sequence = extract_sequence_from_pdb(pdb_file)
    
    features = {
        'sequence': sequence,
        'length': len(sequence),
        'one_hot': one_hot_encode(sequence, max_length),
        'dinucleotide': kmer_composition(sequence, k=2),
        'trinucleotide': kmer_composition(sequence, k=3),
        'physicochemical': physicochemical_features(sequence),
        'position_weighted': position_weighted_encoding(sequence, max_length)
    }
    
    return features


def compare_sequence_features_by_class():
    """
    Compare sequence features across motif classes.
    """
    import os
    
    base_path = '../../dataset'
    classes = ['bulge', 'hairpin', 'internal']
    
    results = []
    
    for motif_class in classes:
        class_path = os.path.join(base_path, motif_class)
        pdb_files = [f for f in os.listdir(class_path) if f.endswith('.pdb')][:5]
        
        for pdb_file in pdb_files:
            filepath = os.path.join(class_path, pdb_file)
            try:
                features = extract_all_sequence_features(filepath)
                
                # Extract key statistics
                dinuc_features = np.array(list(features['dinucleotide'].values()))
                
                results.append({
                    'class': motif_class,
                    'file': pdb_file,
                    'length': features['length'],
                    'gc_content': features['physicochemical'][2],
                    'purine_content': features['physicochemical'][0],
                    'dinuc_entropy': -np.sum(dinuc_features * np.log(dinuc_features + 1e-10)),
                    'sequence': features['sequence'][:20] + '...' if len(features['sequence']) > 20 else features['sequence']
                })
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SEQUENCE FEATURE COMPARISON BY MOTIF CLASS")
    print("="*80)
    
    print("\nSample sequences:")
    for motif_class in classes:
        class_data = df[df['class'] == motif_class]
        print(f"\n{motif_class.upper()}:")
        for _, row in class_data.head(2).iterrows():
            print(f"  {row['file']}: {row['sequence']}")
    
    print("\n\nStatistical summary:")
    print(df.groupby('class')[['length', 'gc_content', 'purine_content', 'dinuc_entropy']].mean())
    
    return df


if __name__ == "__main__":
    # Demo 1: Extract features from a single file
    print("="*80)
    print("DEMO 1: Single file feature extraction")
    print("="*80)
    
    pdb_file = '../../dataset/bulge/3J5L_2919.pdb'
    features = extract_all_sequence_features(pdb_file)
    
    print(f"\nFile: {pdb_file}")
    print(f"Sequence: {features['sequence']}")
    print(f"Length: {features['length']}")
    print(f"\nPhysicochemical features:")
    print(f"  Purine content: {features['physicochemical'][0]:.3f}")
    print(f"  GC content: {features['physicochemical'][2]:.3f}")
    print(f"  AU content: {features['physicochemical'][3]:.3f}")
    
    print(f"\nOne-hot encoding shape: {features['one_hot'].shape}")
    print(f"Dinucleotide features: {len(features['dinucleotide'])} dimensions")
    print(f"Trinucleotide features: {len(features['trinucleotide'])} dimensions")
    
    # Show top 5 dinucleotides
    dinuc_sorted = sorted(features['dinucleotide'].items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 dinucleotides:")
    for kmer, freq in dinuc_sorted[:5]:
        print(f"  {kmer}: {freq:.3f}")
    
    # Demo 2: Compare across classes
    print("\n" + "="*80)
    print("DEMO 2: Cross-class comparison")
    print("="*80)
    
    df = compare_sequence_features_by_class()
    
    print("\n" + "="*80)
    print("RECOMMENDED FEATURE COMBINATIONS FOR MODELING")
    print("="*80)
    print("""
Option 1: Lightweight (26 features)
  - Physicochemical properties: 10 features
  - Dinucleotide composition: 16 features
  - Total: 26 features

Option 2: Medium (90 features)  
  - Physicochemical properties: 10 features
  - Dinucleotide composition: 16 features
  - Trinucleotide composition: 64 features
  - Total: 90 features

Option 3: Rich representation (120 features)
  - One-hot encoding (flattened): 30×4 = 120 features
  - Preserves position information

Option 4: Hybrid (1020 features) - RECOMMENDED
  - Phosphate distances: 900 features (existing)
  - One-hot sequence: 30×4 = 120 features
  - Total: 1020 features
  - Combines structure + sequence information

Option 5: Normalized hybrid (920 features)
  - Phosphate distances (size-normalized): 900 features
  - Physicochemical + dinucleotides: 10 + 16 = 26 features
  - Total: 926 features
  - Addresses size leakage while adding sequence info
""")
