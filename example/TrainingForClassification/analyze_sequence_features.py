#!/usr/bin/env python3
"""
Analyze RNA sequence features from PDB files for dataset2.

Examines:
1. Nucleotide composition (A, U, G, C percentages)
2. Sequence patterns and motifs
3. Di-nucleotide frequencies
4. Purine/Pyrimidine ratios
5. GC content
6. Correlation with motif class

Goal: Determine if sequence features are:
- Size-invariant ✓
- Informative for classification
- Available in all PDB files
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import matplotlib.pyplot as plt
import seaborn as sns

# RNA nucleotide mapping
NUCLEOTIDES = ['A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC']  # Include DNA variants
RNA_MAP = {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
           'DA': 'A', 'DT': 'U', 'DG': 'G', 'DC': 'C'}  # DNA→RNA

def extract_sequence_from_pdb(pdb_path):
    """Extract RNA sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('RNA', pdb_path)
        sequence = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in RNA_MAP:
                        sequence.append(RNA_MAP[resname])
        
        return ''.join(sequence) if sequence else None
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None

def compute_sequence_features(sequence):
    """Compute size-invariant sequence features."""
    if not sequence or len(sequence) == 0:
        return None
    
    length = len(sequence)
    counter = Counter(sequence)
    
    # Basic composition (percentages - size invariant)
    composition = {
        'A_pct': counter.get('A', 0) / length,
        'U_pct': counter.get('U', 0) / length,
        'G_pct': counter.get('G', 0) / length,
        'C_pct': counter.get('C', 0) / length,
    }
    
    # Purine/Pyrimidine ratio
    purines = counter.get('A', 0) + counter.get('G', 0)
    pyrimidines = counter.get('U', 0) + counter.get('C', 0)
    composition['purine_pct'] = purines / length if length > 0 else 0
    composition['pyrimidine_pct'] = pyrimidines / length if length > 0 else 0
    
    # GC content (important for stability)
    gc_count = counter.get('G', 0) + counter.get('C', 0)
    composition['gc_content'] = gc_count / length if length > 0 else 0
    
    # Di-nucleotide frequencies (size invariant)
    dinuc_counts = defaultdict(int)
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        dinuc_counts[dinuc] += 1
    
    total_dinuc = sum(dinuc_counts.values())
    if total_dinuc > 0:
        for dinuc in ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC',
                      'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']:
            composition[f'dinuc_{dinuc}'] = dinuc_counts[dinuc] / total_dinuc
    else:
        for dinuc in ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC',
                      'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']:
            composition[f'dinuc_{dinuc}'] = 0.0
    
    # Sequence complexity (Shannon entropy - size invariant)
    if length > 0:
        entropy = -sum((count/length) * np.log2(count/length) 
                      for count in counter.values() if count > 0)
        composition['entropy'] = entropy / 2.0  # Normalize by max entropy (log2(4))
    else:
        composition['entropy'] = 0.0
    
    composition['length'] = length  # For validation only
    
    return composition

def analyze_dataset_sequences(csv_path, dataset_root):
    """Analyze sequences across all classes."""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f"Analyzing sequences from: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"{'='*80}\n")
    
    results = []
    failed = 0
    
    for idx, row in df.iterrows():
        mrc_filename = row['mrc_file']
        pdb_filename = mrc_filename.replace('.mrc', '.pdb')
        fine_label = row['fine_label']
        coarse_label = row['coarse_label']
        
        # Find PDB file
        pdb_path = os.path.join(dataset_root, fine_label, pdb_filename)
        
        if not os.path.exists(pdb_path):
            failed += 1
            continue
        
        # Extract sequence
        sequence = extract_sequence_from_pdb(pdb_path)
        if sequence is None:
            failed += 1
            continue
        
        # Compute features
        features = compute_sequence_features(sequence)
        if features is None:
            failed += 1
            continue
        
        features['fine_label'] = fine_label
        features['coarse_label'] = coarse_label
        features['sequence'] = sequence
        features['filename'] = pdb_filename
        
        results.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")
    
    print(f"\nSuccessfully processed: {len(results)}/{len(df)} samples")
    print(f"Failed: {failed}")
    
    return pd.DataFrame(results)

def visualize_sequence_features(df_seq):
    """Generate visualizations for sequence features."""
    
    # 1. Nucleotide composition by coarse class
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    nucleotides = ['A_pct', 'U_pct', 'G_pct', 'C_pct']
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    
    for idx, (nuc, color) in enumerate(zip(nucleotides, colors)):
        ax = axes[idx // 2, idx % 2]
        
        data_by_class = [df_seq[df_seq['coarse_label'] == cls][nuc].values 
                        for cls in ['bulge', 'hairpin', 'internal']]
        
        bp = ax.boxplot(data_by_class, labels=['Bulge', 'Hairpin', 'Internal'],
                       patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
        
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title(f'{nuc.replace("_pct", "").upper()} Content', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sequence_nucleotide_composition.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sequence_nucleotide_composition.png")
    plt.close()
    
    # 2. GC content and Purine/Pyrimidine ratio
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GC content
    data_gc = [df_seq[df_seq['coarse_label'] == cls]['gc_content'].values 
               for cls in ['bulge', 'hairpin', 'internal']]
    bp = axes[0].boxplot(data_gc, labels=['Bulge', 'Hairpin', 'Internal'],
                         patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#FF7F00')
    axes[0].set_ylabel('GC Content', fontsize=12)
    axes[0].set_title('GC Content by Motif Type', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Purine ratio
    data_pur = [df_seq[df_seq['coarse_label'] == cls]['purine_pct'].values 
                for cls in ['bulge', 'hairpin', 'internal']]
    bp = axes[1].boxplot(data_pur, labels=['Bulge', 'Hairpin', 'Internal'],
                         patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#A65628')
    axes[1].set_ylabel('Purine Percentage', fontsize=12)
    axes[1].set_title('Purine Content by Motif Type', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sequence_gc_purine_content.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sequence_gc_purine_content.png")
    plt.close()
    
    # 3. Sequence entropy (complexity)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_entropy = [df_seq[df_seq['coarse_label'] == cls]['entropy'].values 
                   for cls in ['bulge', 'hairpin', 'internal']]
    bp = ax.boxplot(data_entropy, labels=['Bulge', 'Hairpin', 'Internal'],
                   patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#F781BF')
    ax.set_ylabel('Normalized Entropy', fontsize=12)
    ax.set_title('Sequence Complexity by Motif Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sequence_entropy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sequence_entropy.png")
    plt.close()
    
    # 4. Correlation heatmap of top di-nucleotides
    top_dinucs = ['dinuc_GG', 'dinuc_GC', 'dinuc_CG', 'dinuc_AA', 'dinuc_UU', 'dinuc_AU']
    dinuc_cols = [col for col in top_dinucs if col in df_seq.columns]
    
    if dinuc_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df_seq[dinuc_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Di-nucleotide Frequency Correlations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sequence_dinucleotide_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: sequence_dinucleotide_correlation.png")
        plt.close()

def statistical_analysis(df_seq):
    """Perform statistical tests to check if features differ by class."""
    from scipy import stats
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: Do sequence features differ by motif class?")
    print("="*80 + "\n")
    
    features_to_test = ['A_pct', 'U_pct', 'G_pct', 'C_pct', 'gc_content', 
                       'purine_pct', 'entropy']
    
    results = []
    
    for feature in features_to_test:
        bulge = df_seq[df_seq['coarse_label'] == 'bulge'][feature].values
        hairpin = df_seq[df_seq['coarse_label'] == 'hairpin'][feature].values
        internal = df_seq[df_seq['coarse_label'] == 'internal'][feature].values
        
        # ANOVA test
        f_stat, p_value = stats.f_oneway(bulge, hairpin, internal)
        
        # Mean and std for each class
        results.append({
            'Feature': feature,
            'Bulge_mean': np.mean(bulge),
            'Bulge_std': np.std(bulge),
            'Hairpin_mean': np.mean(hairpin),
            'Hairpin_std': np.std(hairpin),
            'Internal_mean': np.mean(internal),
            'Internal_std': np.std(internal),
            'F_statistic': f_stat,
            'p_value': p_value,
            'Significant': 'YES' if p_value < 0.05 else 'NO'
        })
    
    results_df = pd.DataFrame(results)
    
    print(results_df.to_string(index=False))
    print("\n")
    
    # Check correlation with sequence length (to verify size-invariance)
    print("="*80)
    print("SIZE-INVARIANCE CHECK: Correlation with sequence length")
    print("="*80 + "\n")
    
    length_corrs = []
    for feature in features_to_test:
        corr = df_seq[feature].corr(df_seq['length'])
        length_corrs.append({
            'Feature': feature,
            'Correlation_with_length': corr,
            'Size_Invariant': 'YES ✓' if abs(corr) < 0.3 else 'NO ✗'
        })
    
    length_df = pd.DataFrame(length_corrs)
    print(length_df.to_string(index=False))
    print("\n")
    
    return results_df, length_df

def main():
    # Paths
    dataset_root = "../../dataset2"
    csv_path = "rna_train_dataset2_small.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    if not os.path.exists(dataset_root):
        print(f"Error: {dataset_root} not found!")
        return
    
    # Analyze sequences
    df_seq = analyze_dataset_sequences(csv_path, dataset_root)
    
    if len(df_seq) == 0:
        print("No sequences extracted!")
        return
    
    # Save results
    df_seq.to_csv('sequence_features_analysis.csv', index=False)
    print(f"\n✓ Saved sequence features to: sequence_features_analysis.csv")
    
    # Statistical analysis
    stats_df, size_inv_df = statistical_analysis(df_seq)
    
    # Summary statistics by class
    print("="*80)
    print("SUMMARY: Sequence Feature Statistics by Motif Class")
    print("="*80 + "\n")
    
    summary = df_seq.groupby('coarse_label').agg({
        'length': ['mean', 'std', 'min', 'max'],
        'A_pct': ['mean', 'std'],
        'G_pct': ['mean', 'std'],
        'C_pct': ['mean', 'std'],
        'U_pct': ['mean', 'std'],
        'gc_content': ['mean', 'std'],
        'purine_pct': ['mean', 'std'],
        'entropy': ['mean', 'std']
    })
    print(summary)
    print("\n")
    
    # Generate visualizations
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    visualize_sequence_features(df_seq)
    
    # Show some example sequences
    print("="*80)
    print("EXAMPLE SEQUENCES (first 3 per class)")
    print("="*80 + "\n")
    
    for cls in ['bulge', 'hairpin', 'internal']:
        print(f"\n{cls.upper()}:")
        samples = df_seq[df_seq['coarse_label'] == cls].head(3)
        for idx, row in samples.iterrows():
            print(f"  {row['filename']}: {row['sequence'][:50]}... (len={row['length']})")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Check sequence_features_analysis.csv for full feature data")
    print("2. Check visualizations for feature distributions")
    print("3. Review statistical tests for feature significance")
    print("4. Review size-invariance check (correlation with length)")

if __name__ == "__main__":
    main()
