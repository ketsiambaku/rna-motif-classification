#!/usr/bin/env python3
"""
Dataset Statistics Generator

Analyzes dataset2 structure and generates comprehensive statistics including:
- Class distribution
- File pair validation
- MRC dimension statistics
- PDB residue count statistics
- Sequence composition analysis

Author: RNA Motif Classification Project
Date: November 28, 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Try importing mrcfile and BioPython
try:
    import mrcfile
except ImportError:
    print("Warning: mrcfile not installed. MRC analysis will be limited.")
    mrcfile = None

try:
    from Bio import PDB
except ImportError:
    print("Warning: BioPython not installed. PDB analysis will be limited.")
    PDB = None


class DatasetStatistics:
    """Generate comprehensive dataset statistics."""
    
    def __init__(self, dataset_root: str):
        """
        Initialize dataset statistics generator.
        
        Args:
            dataset_root: Path to dataset2 directory
        """
        self.dataset_root = Path(dataset_root)
        self.stats = {}
        
        # Define class categories (15 classes total)
        self.size_classes = ['1x1', '2x2', '3x3', '4x4', '5x5']
        self.bulge_classes = ['bulge1', 'bulge2', 'bulge3', 'bulge4', 'bulge5']
        self.hairpin_classes = ['hairpin3', 'hairpin4', 'hairpin5', 'hairpin6', 'hairpin7']
        self.all_classes = self.size_classes + self.bulge_classes + self.hairpin_classes
        
    def scan_dataset(self) -> Dict:
        """
        Scan entire dataset and collect file information.
        
        Returns:
            Dictionary with file counts and paths
        """
        print("Scanning dataset structure...")
        
        results = {
            'class_counts': {},
            'file_pairs': {},
            'missing_pairs': [],
            'total_samples': 0
        }
        
        for class_name in self.all_classes:
            class_dir = self.dataset_root / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_name}")
                continue
            
            # Find all MRC files
            mrc_files = sorted(class_dir.glob('*.mrc'))
            pdb_files = sorted(class_dir.glob('*.pdb'))
            
            # Create sets for fast lookup
            mrc_basenames = {f.stem for f in mrc_files}
            pdb_basenames = {f.stem for f in pdb_files}
            
            # Find matching pairs
            paired = mrc_basenames & pdb_basenames
            mrc_only = mrc_basenames - pdb_basenames
            pdb_only = pdb_basenames - mrc_basenames
            
            results['class_counts'][class_name] = len(paired)
            results['file_pairs'][class_name] = {
                'paired': len(paired),
                'mrc_only': len(mrc_only),
                'pdb_only': len(pdb_only),
                'paired_files': sorted(list(paired))
            }
            
            # Track missing pairs
            for basename in mrc_only:
                results['missing_pairs'].append({
                    'class': class_name,
                    'basename': basename,
                    'missing': 'pdb'
                })
            
            for basename in pdb_only:
                results['missing_pairs'].append({
                    'class': class_name,
                    'basename': basename,
                    'missing': 'mrc'
                })
            
            results['total_samples'] += len(paired)
            
            print(f"  {class_name}: {len(paired)} pairs "
                  f"({len(mrc_only)} MRC-only, {len(pdb_only)} PDB-only)")
        
        self.stats['scan'] = results
        return results
    
    def analyze_mrc_dimensions(self, sample_size: Optional[int] = None) -> Dict:
        """
        Analyze MRC file dimensions and statistics.
        
        Args:
            sample_size: Number of files to sample per class (None = all)
            
        Returns:
            Dictionary with MRC dimension statistics
        """
        if mrcfile is None:
            print("Skipping MRC analysis (mrcfile not installed)")
            return {}
        
        print("\nAnalyzing MRC dimensions...")
        
        results = {
            'dimensions': [],
            'voxel_spacing': [],
            'density_stats': [],
            'class': []
        }
        
        for class_name in tqdm(self.all_classes, desc="Classes"):
            class_dir = self.dataset_root / class_name
            
            if not class_dir.exists():
                continue
            
            mrc_files = sorted(class_dir.glob('*.mrc'))
            
            # Sample if requested
            if sample_size and len(mrc_files) > sample_size:
                mrc_files = np.random.choice(mrc_files, sample_size, replace=False)
            
            for mrc_path in mrc_files:
                try:
                    with mrcfile.open(mrc_path, permissive=True) as mrc:
                        data = mrc.data
                        
                        results['dimensions'].append(data.shape)
                        results['voxel_spacing'].append(mrc.voxel_size)
                        results['density_stats'].append({
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'min': float(np.min(data)),
                            'max': float(np.max(data))
                        })
                        results['class'].append(class_name)
                        
                except Exception as e:
                    print(f"Error reading {mrc_path}: {e}")
                    continue
        
        self.stats['mrc'] = results
        return results
    
    def analyze_pdb_structures(self, sample_size: Optional[int] = None) -> Dict:
        """
        Analyze PDB structures for residue counts and sequences.
        
        Args:
            sample_size: Number of files to sample per class (None = all)
            
        Returns:
            Dictionary with PDB structure statistics
        """
        print("\nAnalyzing PDB structures...")
        
        results = {
            'residue_counts': [],
            'sequences': [],
            'nucleotide_counts': [],
            'class': []
        }
        
        for class_name in tqdm(self.all_classes, desc="Classes"):
            class_dir = self.dataset_root / class_name
            
            if not class_dir.exists():
                continue
            
            pdb_files = sorted(class_dir.glob('*.pdb'))
            
            # Sample if requested
            if sample_size and len(pdb_files) > sample_size:
                pdb_files = np.random.choice(pdb_files, sample_size, replace=False)
            
            for pdb_path in pdb_files:
                try:
                    # Parse SEQRES records for sequence
                    sequence = self._extract_sequence_from_pdb(pdb_path)
                    
                    if sequence:
                        results['sequences'].append(sequence)
                        results['residue_counts'].append(len(sequence))
                        
                        # Count nucleotides
                        counts = Counter(sequence)
                        results['nucleotide_counts'].append({
                            'A': counts.get('A', 0),
                            'U': counts.get('U', 0),
                            'G': counts.get('G', 0),
                            'C': counts.get('C', 0)
                        })
                        results['class'].append(class_name)
                    
                except Exception as e:
                    print(f"Error reading {pdb_path}: {e}")
                    continue
        
        self.stats['pdb'] = results
        return results
    
    def _extract_sequence_from_pdb(self, pdb_path: Path) -> str:
        """
        Extract RNA sequence from PDB SEQRES records.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            RNA sequence string
        """
        sequence = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('SEQRES'):
                    # SEQRES records contain sequence information
                    # Format: SEQRES   1 A   30  G G U U A A G C ...
                    parts = line.split()
                    if len(parts) > 4:
                        # Skip first 4 fields (SEQRES, serial, chain, count)
                        nucleotides = parts[4:]
                        # Convert 3-letter to 1-letter code
                        for nuc in nucleotides:
                            if nuc in ['A', 'U', 'G', 'C']:
                                sequence.append(nuc)
        
        return ''.join(sequence)
    
    def analyze_class_balance(self) -> Dict:
        """
        Analyze class distribution and imbalance.
        
        Returns:
            Dictionary with class balance statistics
        """
        print("\nAnalyzing class balance...")
        
        if 'scan' not in self.stats:
            self.scan_dataset()
        
        counts = self.stats['scan']['class_counts']
        total = sum(counts.values())
        
        results = {
            'counts': counts,
            'percentages': {k: v/total*100 for k, v in counts.items()},
            'total': total,
            'min_class': min(counts.items(), key=lambda x: x[1]),
            'max_class': max(counts.items(), key=lambda x: x[1]),
            'imbalance_ratio': max(counts.values()) / min(counts.values()) if min(counts.values()) > 0 else float('inf')
        }
        
        # Group by category
        results['by_category'] = {
            'size': sum(counts.get(c, 0) for c in self.size_classes),
            'bulge': sum(counts.get(c, 0) for c in self.bulge_classes),
            'hairpin': sum(counts.get(c, 0) for c in self.hairpin_classes)
        }
        
        self.stats['balance'] = results
        return results
    
    def generate_visualizations(self, output_dir: str = 'docs/figures'):
        """
        Generate visualization plots for dataset statistics.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        # 1. Class distribution
        if 'balance' in self.stats:
            self._plot_class_distribution(output_path)
        
        # 2. MRC dimensions
        if 'mrc' in self.stats and self.stats['mrc']:
            self._plot_mrc_statistics(output_path)
        
        # 3. PDB residue counts
        if 'pdb' in self.stats and self.stats['pdb']:
            self._plot_pdb_statistics(output_path)
        
        print("Visualizations saved!")
    
    def _plot_class_distribution(self, output_path: Path):
        """Plot class distribution bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # All classes
        counts = self.stats['balance']['counts']
        classes = list(counts.keys())
        values = list(counts.values())
        
        axes[0].bar(range(len(classes)), values, color='steelblue')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Dataset2: Class Distribution (14 Classes)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Category summary
        cat_counts = self.stats['balance']['by_category']
        axes[1].bar(cat_counts.keys(), cat_counts.values(), color=['coral', 'lightblue', 'lightgreen'])
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Dataset2: Category Summary')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mrc_statistics(self, output_path: Path):
        """Plot MRC dimension and density statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        mrc_data = self.stats['mrc']
        
        # Dimension distribution
        dims = [d[0] for d in mrc_data['dimensions']]  # Assuming cubic
        axes[0, 0].hist(dims, bins=20, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Dimension (voxels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MRC Dimension Distribution')
        
        # Density mean by class
        density_means = [s['mean'] for s in mrc_data['density_stats']]
        classes = mrc_data['class']
        df = pd.DataFrame({'mean': density_means, 'class': classes})
        df.boxplot(column='mean', by='class', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Mean Density')
        axes[0, 1].set_title('Density Distribution by Class')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45, ha='right')
        
        # Density range
        density_mins = [s['min'] for s in mrc_data['density_stats']]
        density_maxs = [s['max'] for s in mrc_data['density_stats']]
        axes[1, 0].scatter(density_mins, density_maxs, alpha=0.3, c='steelblue')
        axes[1, 0].set_xlabel('Min Density')
        axes[1, 0].set_ylabel('Max Density')
        axes[1, 0].set_title('Density Range')
        
        # Density std
        density_stds = [s['std'] for s in mrc_data['density_stats']]
        axes[1, 1].hist(density_stds, bins=30, color='coral', alpha=0.7)
        axes[1, 1].set_xlabel('Density Std Dev')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Density Variability Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'mrc_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pdb_statistics(self, output_path: Path):
        """Plot PDB residue count and sequence statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        pdb_data = self.stats['pdb']
        
        # Residue count distribution
        axes[0, 0].hist(pdb_data['residue_counts'], bins=30, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Residue Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Residue Count Distribution')
        
        # Residue count by class
        df = pd.DataFrame({
            'count': pdb_data['residue_counts'],
            'class': pdb_data['class']
        })
        df.boxplot(column='count', by='class', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Residue Count')
        axes[0, 1].set_title('Residue Count by Class')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45, ha='right')
        
        # Nucleotide composition
        total_nucs = {nuc: sum(c[nuc] for c in pdb_data['nucleotide_counts']) 
                     for nuc in ['A', 'U', 'G', 'C']}
        axes[1, 0].bar(total_nucs.keys(), total_nucs.values(), 
                       color=['red', 'blue', 'green', 'orange'])
        axes[1, 0].set_ylabel('Total Count')
        axes[1, 0].set_title('Overall Nucleotide Composition')
        
        # GC content distribution
        gc_contents = []
        for counts in pdb_data['nucleotide_counts']:
            total = sum(counts.values())
            if total > 0:
                gc = (counts['G'] + counts['C']) / total * 100
                gc_contents.append(gc)
        
        axes[1, 1].hist(gc_contents, bins=30, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('GC Content (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('GC Content Distribution')
        axes[1, 1].axvline(np.mean(gc_contents), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(gc_contents):.1f}%')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'pdb_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_path: str = 'docs/dataset-statistics.md'):
        """
        Generate comprehensive markdown report.
        
        Args:
            output_path: Path to save markdown report
        """
        print(f"\nGenerating report: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# Dataset2 Statistics Report\n\n")
            f.write(f"**Generated**: November 28, 2025\n\n")
            f.write("---\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            if 'balance' in self.stats:
                total = self.stats['balance']['total']
                f.write(f"**Total samples**: {total:,}\n")
                f.write(f"**Number of classes**: {len(self.all_classes)}\n")
                f.write(f"**Class imbalance ratio**: {self.stats['balance']['imbalance_ratio']:.2f}:1\n\n")
            
            # Class distribution
            f.write("## Class Distribution\n\n")
            if 'balance' in self.stats:
                f.write("| Class | Count | Percentage |\n")
                f.write("|-------|------:|-----------:|\n")
                
                counts = self.stats['balance']['counts']
                percentages = self.stats['balance']['percentages']
                
                for class_name in self.all_classes:
                    if class_name in counts:
                        f.write(f"| {class_name} | {counts[class_name]:,} | {percentages[class_name]:.2f}% |\n")
                
                f.write("\n### By Category\n\n")
                by_cat = self.stats['balance']['by_category']
                f.write(f"- **Size classes (2x2-5x5)**: {by_cat['size']:,} samples\n")
                f.write(f"- **Bulge loops**: {by_cat['bulge']:,} samples\n")
                f.write(f"- **Hairpin loops**: {by_cat['hairpin']:,} samples\n\n")
            
            # File validation
            f.write("## File Pair Validation\n\n")
            if 'scan' in self.stats:
                missing = len(self.stats['scan']['missing_pairs'])
                if missing > 0:
                    f.write(f"⚠️ **{missing} files missing pairs**\n\n")
                    f.write("| Class | Missing Type | Count |\n")
                    f.write("|-------|-------------|------:|\n")
                    
                    missing_summary = defaultdict(lambda: {'mrc': 0, 'pdb': 0})
                    for item in self.stats['scan']['missing_pairs']:
                        missing_summary[item['class']][item['missing']] += 1
                    
                    for class_name, counts in missing_summary.items():
                        if counts['mrc'] > 0:
                            f.write(f"| {class_name} | MRC | {counts['mrc']} |\n")
                        if counts['pdb'] > 0:
                            f.write(f"| {class_name} | PDB | {counts['pdb']} |\n")
                else:
                    f.write("✅ All MRC files have matching PDB files\n")
                f.write("\n")
            
            # MRC statistics
            if 'mrc' in self.stats and self.stats['mrc']:
                f.write("## MRC File Statistics\n\n")
                
                dims = self.stats['mrc']['dimensions']
                if dims:
                    unique_dims = set(tuple(d) for d in dims)
                    f.write(f"**Samples analyzed**: {len(dims)}\n")
                    f.write(f"**Unique dimensions**: {len(unique_dims)}\n\n")
                    
                    f.write("### Common Dimensions\n\n")
                    dim_counts = Counter(tuple(d) for d in dims)
                    for dim, count in dim_counts.most_common(5):
                        f.write(f"- `{dim}`: {count} files\n")
                    
                    f.write("\n### Density Statistics\n\n")
                    density_stats = self.stats['mrc']['density_stats']
                    all_means = [s['mean'] for s in density_stats]
                    all_stds = [s['std'] for s in density_stats]
                    
                    f.write(f"- **Mean density**: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}\n")
                    f.write(f"- **Density std dev**: {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}\n\n")
            
            # PDB statistics
            if 'pdb' in self.stats and self.stats['pdb']:
                f.write("## PDB Structure Statistics\n\n")
                
                residue_counts = self.stats['pdb']['residue_counts']
                if residue_counts:
                    f.write(f"**Samples analyzed**: {len(residue_counts)}\n")
                    f.write(f"**Mean residue count**: {np.mean(residue_counts):.1f} ± {np.std(residue_counts):.1f}\n")
                    f.write(f"**Residue count range**: [{min(residue_counts)}, {max(residue_counts)}]\n\n")
                    
                    # Nucleotide composition
                    nuc_counts = self.stats['pdb']['nucleotide_counts']
                    total_nucs = {nuc: sum(c[nuc] for c in nuc_counts) for nuc in ['A', 'U', 'G', 'C']}
                    total = sum(total_nucs.values())
                    
                    f.write("### Nucleotide Composition\n\n")
                    f.write("| Nucleotide | Count | Percentage |\n")
                    f.write("|-----------|------:|-----------:|\n")
                    for nuc in ['A', 'U', 'G', 'C']:
                        pct = total_nucs[nuc] / total * 100
                        f.write(f"| {nuc} | {total_nucs[nuc]:,} | {pct:.2f}% |\n")
                    
                    # GC content
                    gc_contents = []
                    for counts in nuc_counts:
                        t = sum(counts.values())
                        if t > 0:
                            gc_contents.append((counts['G'] + counts['C']) / t * 100)
                    
                    f.write(f"\n**Mean GC content**: {np.mean(gc_contents):.2f}% ± {np.std(gc_contents):.2f}%\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("See generated figures in `docs/figures/`:\n\n")
            f.write("- `class_distribution.png`: Class distribution charts\n")
            if 'mrc' in self.stats and self.stats['mrc']:
                f.write("- `mrc_statistics.png`: MRC dimension and density statistics\n")
            if 'pdb' in self.stats and self.stats['pdb']:
                f.write("- `pdb_statistics.png`: PDB residue count and sequence statistics\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("*Generated by `src/utils/dataset_stats.py`*\n")
        
        print(f"Report saved to {output_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dataset statistics')
    parser.add_argument('--dataset', type=str, default='dataset2',
                       help='Path to dataset directory (default: dataset2)')
    parser.add_argument('--sample-mrc', type=int, default=None,
                       help='Number of MRC files to sample per class (None = all)')
    parser.add_argument('--sample-pdb', type=int, default=None,
                       help='Number of PDB files to sample per class (None = all)')
    parser.add_argument('--output', type=str, default='docs/dataset-statistics.md',
                       help='Output path for markdown report')
    parser.add_argument('--figures', type=str, default='docs/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset directory not found: {args.dataset}")
        sys.exit(1)
    
    # Create statistics generator
    stats = DatasetStatistics(args.dataset)
    
    # Run analyses
    print("=" * 60)
    print("Dataset Statistics Generator")
    print("=" * 60)
    
    # 1. Scan dataset structure
    stats.scan_dataset()
    
    # 2. Analyze class balance
    stats.analyze_class_balance()
    
    # 3. Analyze MRC files (with sampling if specified)
    if mrcfile:
        stats.analyze_mrc_dimensions(sample_size=args.sample_mrc)
    
    # 4. Analyze PDB files (with sampling if specified)
    stats.analyze_pdb_structures(sample_size=args.sample_pdb)
    
    # 5. Generate visualizations
    stats.generate_visualizations(output_dir=args.figures)
    
    # 6. Generate report
    stats.generate_report(output_path=args.output)
    
    print("\n" + "=" * 60)
    print("Dataset statistics generation complete!")
    print("=" * 60)
    print(f"\nReport: {args.output}")
    print(f"Figures: {args.figures}/")


if __name__ == '__main__':
    main()
