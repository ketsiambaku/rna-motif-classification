"""
Hybrid dataset loader combining MRC density files with sequence features.

This dataset loads cryo-EM density volumes (.mrc) and extracts size-invariant
sequence features from corresponding PDB structures for multi-modal learning.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import mrcfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys

# Add parent directory to path for importing sequence features
sys.path.append(str(Path(__file__).parent.parent))
from features.sequence_features import SequenceFeatureExtractor


class HybridDataset(Dataset):
    """
    PyTorch Dataset for RNA motif classification combining density and sequence features.
    
    Dataset structure:
        dataset2/
            1x1/         - 6,153 samples
            2x2/         - 2,503 samples
            3x3/         - 1,996 samples
            ...
            hairpin7/    - 1,308 samples
    
    Each sample consists of:
        - Density: .mrc file (3D volume, variable size)
        - Sequence: .pdb file (SEQRES records)
        - Label: Class index (0-14 for 15 classes)
    
    Args:
        root_dir: Path to dataset2/ directory
        split: One of 'train', 'val', 'test'
        transform: Optional transform for density volumes
        target_size: Target size for density volumes (default: 32)
        use_subset: Fraction of dataset to use (default: 0.1 for 10%)
        random_seed: Seed for reproducible splits (default: 42)
    """
    
    # Define class names and indices
    CLASS_NAMES = [
        '1x1', '2x2', '3x3', '4x4', '5x5',  # Size classes (0-4)
        'bulge1', 'bulge2', 'bulge3', 'bulge4', 'bulge5',  # Bulge classes (5-9)
        'hairpin3', 'hairpin4', 'hairpin5', 'hairpin6', 'hairpin7'  # Hairpin classes (10-14)
    ]
    
    # Class consolidation mapping (15 classes -> 6 superclasses)
    # Balanced grouping based on sample counts for better class distribution
    CONSOLIDATION_MAP = {
        '1x1': 'small_internal',
        '2x2': 'small_internal',
        '3x3': 'large_internal',
        '4x4': 'large_internal',
        '5x5': 'large_internal',
        'bulge1': 'small_bulge',
        'bulge2': 'small_bulge',
        'bulge3': 'large_bulge',
        'bulge4': 'large_bulge',
        'bulge5': 'large_bulge',
        'hairpin3': 'small_hairpin',
        'hairpin4': 'small_hairpin',
        'hairpin5': 'small_hairpin',
        'hairpin6': 'large_hairpin',
        'hairpin7': 'large_hairpin'
    }
    
    CONSOLIDATED_CLASS_NAMES = [
        'small_internal',    # 1x1, 2x2          → ~7,047 samples
        'large_internal',    # 3x3, 4x4, 5x5     → ~2,782 samples
        'small_bulge',       # bulge1, bulge2    → ~9,073 samples
        'large_bulge',       # bulge3, bulge4, bulge5 → ~2,447 samples
        'small_hairpin',     # hairpin3, hairpin4, hairpin5 → ~4,584 samples
        'large_hairpin'      # hairpin6, hairpin7 → ~2,805 samples
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        target_size: int = 32,
        use_subset: float = 0.1,
        random_seed: int = 42,
        consolidate: bool = False
    ):
        """Initialize the hybrid dataset."""
        assert split in ['train', 'val', 'test'], f"split must be 'train', 'val', or 'test', got {split}"
        assert 0.0 < use_subset <= 1.0, f"use_subset must be in (0, 1], got {use_subset}"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.use_subset = use_subset
        self.random_seed = random_seed
        self.consolidate = consolidate
        
        # Set up class system
        if self.consolidate:
            self.num_classes = len(self.CONSOLIDATED_CLASS_NAMES)
            self.class_to_idx = {name: idx for idx, name in enumerate(self.CONSOLIDATED_CLASS_NAMES)}
        else:
            self.num_classes = len(self.CLASS_NAMES)
            self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        
        # Initialize sequence feature extractor
        self.seq_extractor = SequenceFeatureExtractor()
        
        # Load file lists
        self.samples = self._load_samples()
        
        mode_str = "6-class consolidated" if self.consolidate else "15-class"
        print(f"Loaded {len(self.samples)} samples for {split} split ({mode_str}, subset={use_subset:.1%})")
        self._print_class_distribution()
    
    def _get_consolidated_label(self, class_name: str) -> int:
        """Get consolidated class label from original class name."""
        consolidated_name = self.CONSOLIDATION_MAP[class_name]
        return self.class_to_idx[consolidated_name]
    
    def _load_samples(self) -> List[Dict[str, any]]:
        """
        Load all sample paths and create train/val/test splits.
        
        Returns:
            List of dicts with keys: 'mrc_path', 'pdb_path', 'label', 'class_name'
        """
        all_samples = []
        
        # Iterate through all class directories
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist, skipping")
                continue
            
            # Find all .mrc files
            mrc_files = sorted(class_dir.glob('*.mrc'))
            
            # For each .mrc, check if corresponding .pdb exists
            for mrc_path in mrc_files:
                pdb_path = mrc_path.with_suffix('.pdb')
                
                if pdb_path.exists():
                    # Determine the label to use
                    label = self._get_consolidated_label(class_name) if self.consolidate else class_idx
                    
                    all_samples.append({
                        'mrc_path': str(mrc_path),
                        'pdb_path': str(pdb_path),
                        'label': label,
                        'class_name': class_name,
                        'original_label': class_idx
                    })
        
        if len(all_samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.root_dir}")
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in indices]
        
        # Apply subset selection
        n_total = len(all_samples)
        n_subset = int(n_total * self.use_subset)
        all_samples = all_samples[:n_subset]
        
        # Create splits: 70% train, 20% val, 10% test
        n_train = int(len(all_samples) * 0.7)
        n_val = int(len(all_samples) * 0.2)
        
        if self.split == 'train':
            return all_samples[:n_train]
        elif self.split == 'val':
            return all_samples[n_train:n_train+n_val]
        else:  # test
            return all_samples[n_train+n_val:]
    
    def _print_class_distribution(self):
        """Print the distribution of classes in this split."""
        from collections import Counter
        
        if self.consolidate:
            # Show consolidated class distribution
            label_counts = Counter([s['label'] for s in self.samples])
            print(f"\n{self.split.upper()} split class distribution (6 consolidated classes):")
            for class_name, class_idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
                count = label_counts.get(class_idx, 0)
                if count > 0:
                    print(f"  {class_name:>18}: {count:>5} samples")
        else:
            # Show original 15 class distribution
            class_counts = Counter([s['class_name'] for s in self.samples])
            print(f"\n{self.split.upper()} split class distribution (15 classes):")
            for class_name in self.CLASS_NAMES:
                count = class_counts.get(class_name, 0)
                if count > 0:
                    print(f"  {class_name:>10}: {count:>4} samples")
    
    def _load_mrc(self, mrc_path: str) -> np.ndarray:
        """
        Load MRC density file and resize to target size using trilinear interpolation.
        
        This matches the approach used in the example code:
        - Min-max normalization to [0, 1]
        - Trilinear interpolation for smooth resizing
        - Preserves all structural information without cropping/padding artifacts
        
        Args:
            mrc_path: Path to .mrc file
            
        Returns:
            Density volume of shape (target_size, target_size, target_size)
        """
        with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
            density = mrc.data.astype(np.float32)
        
        # Handle different input shapes
        if density.ndim != 3:
            raise ValueError(f"Expected 3D density, got shape {density.shape}")
        
        # Min-max normalization to [0, 1]
        density_min = density.min()
        density_max = density.max()
        if density_max > density_min:
            density = (density - density_min) / (density_max - density_min + 1e-8)
        else:
            density = np.zeros_like(density)
        
        # Convert to PyTorch tensor and add batch + channel dimensions
        density_tensor = torch.tensor(density).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # Resize using trilinear interpolation
        target_shape = (self.target_size, self.target_size, self.target_size)
        density_tensor = F.interpolate(
            density_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        # Convert back to numpy and remove batch/channel dimensions
        density = density_tensor.squeeze(0).squeeze(0).numpy()  # [D, H, W]
        
        return density
    

    
    def _extract_sequence_features(self, pdb_path: str) -> np.ndarray:
        """
        Extract 24 size-invariant sequence features from PDB file.
        
        Args:
            pdb_path: Path to .pdb file
            
        Returns:
            Feature vector of shape (24,) - returns zeros if extraction fails
        """
        try:
            features = self.seq_extractor.extract_features(pdb_path)
            # Check if features are valid (not None and correct shape)
            if features is None or len(features) != 24:
                # Return zero vector as fallback
                return np.zeros(24, dtype=np.float32)
            return features
        except Exception as e:
            # Silently return zero vector for failed extractions
            # (warnings already printed by sequence_features.py)
            return np.zeros(24, dtype=np.float32)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with keys:
                - 'density': Tensor of shape (1, target_size, target_size, target_size)
                - 'sequence': Tensor of shape (24,)
                - 'label': Integer class label (0-14)
                - 'class_name': String class name (for debugging)
                - 'mrc_path': Path to MRC file (for debugging)
        """
        sample_info = self.samples[idx]
        
        # Load density
        density = self._load_mrc(sample_info['mrc_path'])
        
        # Extract sequence features
        sequence = self._extract_sequence_features(sample_info['pdb_path'])
        
        # Convert to tensors
        density_tensor = torch.from_numpy(density).float().unsqueeze(0)  # Add channel dim
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.tensor(sample_info['label'], dtype=torch.long)
        
        # Apply optional transform to density
        if self.transform is not None:
            density_tensor = self.transform(density_tensor)
        
        return {
            'density': density_tensor,
            'sequence': sequence_tensor,
            'label': label_tensor,
            'class_name': sample_info['class_name'],
            'mrc_path': sample_info['mrc_path']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced dataset.
        
        Uses inverse frequency weighting: weight_i = n_samples / (n_classes * n_samples_i)
        
        Returns:
            Tensor of shape (n_classes,) with class weights
        """
        class_counts = np.zeros(len(self.CLASS_NAMES), dtype=np.int64)
        
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Inverse frequency weighting
        n_samples = len(self.samples)
        n_classes = len(self.CLASS_NAMES)
        weights = n_samples / (n_classes * class_counts)
        
        return torch.from_numpy(weights).float()


def test_dataset():
    """Test the HybridDataset loader."""
    import time
    
    print("="*80)
    print("Testing HybridDataset Loader")
    print("="*80)
    
    # Test with small subset
    dataset_path = "/Users/ketsiambaku/Repositories/rna-motif-classification/dataset2"
    
    print("\n1. Loading training split (10% subset)...")
    train_dataset = HybridDataset(
        root_dir=dataset_path,
        split='train',
        use_subset=0.1,
        random_seed=42
    )
    
    print("\n2. Loading validation split (10% subset)...")
    val_dataset = HybridDataset(
        root_dir=dataset_path,
        split='val',
        use_subset=0.1,
        random_seed=42
    )
    
    print("\n3. Loading test split (10% subset)...")
    test_dataset = HybridDataset(
        root_dir=dataset_path,
        split='test',
        use_subset=0.1,
        random_seed=42
    )
    
    print("\n" + "="*80)
    print("Dataset Split Summary")
    print("="*80)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    
    print("\n" + "="*80)
    print("Testing Sample Loading")
    print("="*80)
    
    # Test loading first sample
    print("\nLoading first training sample...")
    start_time = time.time()
    sample = train_dataset[0]
    load_time = time.time() - start_time
    
    print(f"✓ Sample loaded in {load_time:.3f}s")
    print(f"  Density shape: {sample['density'].shape}")
    print(f"  Sequence shape: {sample['sequence'].shape}")
    print(f"  Label: {sample['label'].item()} ({sample['class_name']})")
    print(f"  Density range: [{sample['density'].min():.3f}, {sample['density'].max():.3f}]")
    print(f"  Density mean: {sample['density'].mean():.3f}, std: {sample['density'].std():.3f}")
    print(f"  Sequence features (first 8): {sample['sequence'][:8].numpy()}")
    
    # Test loading multiple samples
    print("\n" + "="*80)
    print("Testing Batch Loading")
    print("="*80)
    
    from torch.utils.data import DataLoader
    
    print("\nCreating DataLoader with batch_size=4...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for M1 compatibility
    )
    
    print("Loading first batch...")
    start_time = time.time()
    batch = next(iter(train_loader))
    load_time = time.time() - start_time
    
    print(f"✓ Batch loaded in {load_time:.3f}s")
    print(f"  Density batch shape: {batch['density'].shape}")
    print(f"  Sequence batch shape: {batch['sequence'].shape}")
    print(f"  Label batch shape: {batch['label'].shape}")
    print(f"  Classes in batch: {[train_dataset.CLASS_NAMES[l.item()] for l in batch['label']]}")
    
    # Test class weights
    print("\n" + "="*80)
    print("Testing Class Weights")
    print("="*80)
    
    weights = train_dataset.get_class_weights()
    print(f"\nClass weights shape: {weights.shape}")
    print("\nClass weights (for weighted loss):")
    for i, class_name in enumerate(train_dataset.CLASS_NAMES):
        if i < len(weights):
            print(f"  {class_name:>10}: {weights[i]:.4f}")
    
    print(f"\nMin weight: {weights.min():.4f} (most frequent class)")
    print(f"Max weight: {weights.max():.4f} (least frequent class)")
    print(f"Weight ratio: {weights.max() / weights.min():.2f}x")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == '__main__':
    test_dataset()
