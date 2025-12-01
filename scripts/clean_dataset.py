"""
Dataset cleaning script: Identify and optionally remove samples with missing features.

This script checks all .mrc/.pdb pairs in dataset2/ and identifies samples where:
1. PDB file is missing SEQRES records (no sequence features)
2. PDB file is corrupted or unreadable
3. MRC file is corrupted or has invalid dimensions

Usage:
    python scripts/clean_dataset.py --check         # Just report issues
    python scripts/clean_dataset.py --remove        # Remove problematic pairs
    python scripts/clean_dataset.py --move backup/  # Move to backup folder
"""

import os
import sys
from pathlib import Path
import argparse
import shutil
from collections import defaultdict
import mrcfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from features.sequence_features import SequenceFeatureExtractor


def check_pdb_features(pdb_path, extractor):
    """
    Check if PDB file has valid SEQRES features.
    
    Returns:
        (bool, str): (is_valid, reason)
    """
    try:
        features = extractor.extract_features(pdb_path)
        if features is None:
            return False, "No SEQRES records"
        if len(features) != 24:
            return False, f"Invalid feature count: {len(features)}"
        # Check if features are all zeros (fallback case)
        if (features == 0).all():
            return False, "All zero features"
        return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)[:50]}"


def check_mrc_file(mrc_path):
    """
    Check if MRC file is valid.
    
    Returns:
        (bool, str): (is_valid, reason)
    """
    try:
        with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
            data = mrc.data
            if data is None:
                return False, "No data"
            if data.ndim != 3:
                return False, f"Invalid dimensions: {data.ndim}D"
            if data.size == 0:
                return False, "Empty data"
            return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)[:50]}"


def scan_dataset(root_dir, verbose=False):
    """
    Scan entire dataset and identify problematic samples.
    
    Returns:
        dict: Statistics and list of problematic files
    """
    root = Path(root_dir)
    extractor = SequenceFeatureExtractor()
    
    class_names = [
        '1x1', '2x2', '3x3', '4x4', '5x5',
        'bulge1', 'bulge2', 'bulge3', 'bulge4', 'bulge5',
        'hairpin3', 'hairpin4', 'hairpin5', 'hairpin6', 'hairpin7'
    ]
    
    stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'missing_pdb_features': 0,
        'corrupted_mrc': 0,
        'corrupted_pdb': 0,
        'by_class': defaultdict(lambda: {'total': 0, 'valid': 0, 'invalid': 0}),
        'problematic_files': []
    }
    
    print("Scanning dataset...")
    print("=" * 80)
    
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        
        mrc_files = sorted(class_dir.glob('*.mrc'))
        
        if verbose:
            print(f"\nChecking {class_name}/  ({len(mrc_files)} files)...")
        
        for mrc_path in mrc_files:
            pdb_path = mrc_path.with_suffix('.pdb')
            
            stats['total_samples'] += 1
            stats['by_class'][class_name]['total'] += 1
            
            # Check if PDB exists
            if not pdb_path.exists():
                stats['problematic_files'].append({
                    'mrc': str(mrc_path),
                    'pdb': str(pdb_path),
                    'class': class_name,
                    'reason': 'Missing PDB file'
                })
                stats['by_class'][class_name]['invalid'] += 1
                if verbose:
                    print(f"  ✗ {mrc_path.name}: Missing PDB")
                continue
            
            # Check MRC file
            mrc_valid, mrc_reason = check_mrc_file(mrc_path)
            if not mrc_valid:
                stats['corrupted_mrc'] += 1
                stats['problematic_files'].append({
                    'mrc': str(mrc_path),
                    'pdb': str(pdb_path),
                    'class': class_name,
                    'reason': f'Corrupted MRC: {mrc_reason}'
                })
                stats['by_class'][class_name]['invalid'] += 1
                if verbose:
                    print(f"  ✗ {mrc_path.name}: Bad MRC - {mrc_reason}")
                continue
            
            # Check PDB features
            pdb_valid, pdb_reason = check_pdb_features(pdb_path, extractor)
            if not pdb_valid:
                if "No SEQRES" in pdb_reason or "All zero" in pdb_reason:
                    stats['missing_pdb_features'] += 1
                else:
                    stats['corrupted_pdb'] += 1
                
                stats['problematic_files'].append({
                    'mrc': str(mrc_path),
                    'pdb': str(pdb_path),
                    'class': class_name,
                    'reason': f'Bad PDB: {pdb_reason}'
                })
                stats['by_class'][class_name]['invalid'] += 1
                if verbose:
                    print(f"  ✗ {mrc_path.name}: Bad PDB - {pdb_reason}")
                continue
            
            # Sample is valid
            stats['valid_samples'] += 1
            stats['by_class'][class_name]['valid'] += 1
            if verbose:
                print(f"  ✓ {mrc_path.name}")
    
    return stats


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("DATASET CLEANING REPORT")
    print("=" * 80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total samples:              {stats['total_samples']:>6}")
    print(f"  Valid samples:              {stats['valid_samples']:>6} ({100*stats['valid_samples']/stats['total_samples']:.1f}%)")
    print(f"  Problematic samples:        {len(stats['problematic_files']):>6} ({100*len(stats['problematic_files'])/stats['total_samples']:.1f}%)")
    
    print(f"\nBreakdown by Issue:")
    print(f"  Missing PDB features:       {stats['missing_pdb_features']:>6}")
    print(f"  Corrupted MRC files:        {stats['corrupted_mrc']:>6}")
    print(f"  Corrupted PDB files:        {stats['corrupted_pdb']:>6}")
    
    print(f"\nBy Class:")
    print(f"{'Class':<12} {'Total':>8} {'Valid':>8} {'Invalid':>8} {'Valid %':>10}")
    print("-" * 60)
    
    for class_name in sorted(stats['by_class'].keys()):
        class_stats = stats['by_class'][class_name]
        total = class_stats['total']
        valid = class_stats['valid']
        invalid = class_stats['invalid']
        pct = 100 * valid / total if total > 0 else 0
        print(f"{class_name:<12} {total:>8} {valid:>8} {invalid:>8} {pct:>9.1f}%")
    
    print("\n" + "=" * 80)


def remove_problematic_files(stats, dry_run=True, backup_dir=None):
    """
    Remove or backup problematic files.
    
    Args:
        stats: Statistics dict from scan_dataset
        dry_run: If True, only print what would be removed
        backup_dir: If provided, move files here instead of deleting
    """
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY RUN - No files will be removed")
    elif backup_dir:
        print(f"BACKUP MODE - Files will be moved to {backup_dir}")
    else:
        print("REMOVAL MODE - Files will be permanently deleted")
    print("=" * 80)
    
    backup_path = None
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        # Create class subdirectories
        for class_name in stats['by_class'].keys():
            (backup_path / class_name).mkdir(parents=True, exist_ok=True)
    
    removed_count = 0
    
    for item in stats['problematic_files']:
        mrc_path = Path(item['mrc'])
        pdb_path = Path(item['pdb'])
        
        print(f"\n{mrc_path.name}")
        print(f"  Class: {item['class']}")
        print(f"  Reason: {item['reason']}")
        
        if dry_run:
            print(f"  Would remove: {mrc_path.name}, {pdb_path.name}")
        elif backup_dir and backup_path:
            # Move to backup
            backup_class_dir = backup_path / item['class']
            if mrc_path.exists():
                shutil.move(str(mrc_path), str(backup_class_dir / mrc_path.name))
                print(f"  Moved MRC to backup")
            if pdb_path.exists():
                shutil.move(str(pdb_path), str(backup_class_dir / pdb_path.name))
                print(f"  Moved PDB to backup")
            removed_count += 1
        else:
            # Delete
            if mrc_path.exists():
                mrc_path.unlink()
                print(f"  Deleted MRC")
            if pdb_path.exists():
                pdb_path.unlink()
                print(f"  Deleted PDB")
            removed_count += 1
    
    print("\n" + "=" * 80)
    if dry_run:
        print(f"Would remove {len(stats['problematic_files'])} sample pairs")
    else:
        print(f"Processed {removed_count} sample pairs")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Clean RNA motif dataset by identifying/removing problematic samples'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset2',
        help='Path to dataset directory (default: dataset2)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check and report issues (default mode)'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove problematic files permanently'
    )
    parser.add_argument(
        '--move',
        type=str,
        metavar='BACKUP_DIR',
        help='Move problematic files to backup directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress for each file'
    )
    
    args = parser.parse_args()
    
    # Scan dataset
    stats = scan_dataset(args.dataset, verbose=args.verbose)
    
    # Print summary
    print_summary(stats)
    
    # Handle files based on mode
    if args.remove:
        confirm = input("\n⚠️  Are you sure you want to PERMANENTLY DELETE problematic files? (yes/no): ")
        if confirm.lower() == 'yes':
            remove_problematic_files(stats, dry_run=False, backup_dir=None)
        else:
            print("Aborted.")
    elif args.move:
        remove_problematic_files(stats, dry_run=False, backup_dir=args.move)
    else:
        # Default: dry run
        print("\n" + "=" * 80)
        print("Use --remove to delete or --move BACKUP_DIR to backup problematic files")
        print("=" * 80)


if __name__ == '__main__':
    main()
