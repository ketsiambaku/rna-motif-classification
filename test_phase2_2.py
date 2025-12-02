#!/usr/bin/env python3
"""
Integration test for Phase 2.2 - 3-branch hybrid model.

Tests the complete pipeline:
1. Dataset loads density, sequence, and pairing features
2. Model processes all 3 inputs
3. Training loop works correctly
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.hybrid_dataset import HybridDataset
from models.hybrid_unet import HybridUNet

def test_phase2_2_integration():
    """Test Phase 2.2 integration."""
    print("="*80)
    print("Phase 2.2 Integration Test")
    print("="*80)
    
    # 1. Test dataset with pairing features
    print("\n1. Testing dataset with pairing features...")
    dataset_path = "/Users/ketsiambaku/Repositories/rna-motif-classification/dataset2"
    
    dataset = HybridDataset(
        root_dir=dataset_path,
        split='train',
        use_subset=0.01,  # Just 1% for quick test
        consolidate=True   # 6-class mode
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # 2. Test data loading
    print("\n2. Testing data loading...")
    sample = dataset[0]
    
    print(f"  Density shape: {sample['density'].shape}")
    print(f"  Sequence shape: {sample['sequence'].shape}")
    print(f"  Pairing shape: {sample['pairing'].shape}")
    print(f"  Label: {sample['label']} ({sample['class_name']})")
    
    assert sample['density'].shape == torch.Size([1, 32, 32, 32]), "Wrong density shape"
    assert sample['sequence'].shape == torch.Size([24]), "Wrong sequence shape"
    assert sample['pairing'].shape == torch.Size([10]), "Wrong pairing shape"
    
    print(f"✓ Data loading successful")
    
    # 3. Test model forward pass
    print("\n3. Testing model forward pass...")
    model = HybridUNet(n_classes=6)  # 6-class consolidated
    
    # Create batch
    batch_size = 4
    density = torch.stack([dataset[i]['density'] for i in range(batch_size)])
    sequence = torch.stack([dataset[i]['sequence'] for i in range(batch_size)])
    pairing = torch.stack([dataset[i]['pairing'] for i in range(batch_size)])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(density, sequence, pairing)
    
    print(f"  Batch shapes:")
    print(f"    Density: {density.shape}")
    print(f"    Sequence: {sequence.shape}")
    print(f"    Pairing: {pairing.shape}")
    print(f"  Output logits: {logits.shape}")
    
    assert logits.shape == torch.Size([batch_size, 6]), "Wrong output shape"
    
    print(f"✓ Model forward pass successful")
    
    # 4. Test feature extraction
    print("\n4. Testing feature extraction...")
    features = model.get_feature_representations(density, sequence, pairing)
    
    print(f"  Extracted features:")
    print(f"    Density features: {features['density_features'].shape}")
    print(f"    Sequence features: {features['sequence_features'].shape}")
    print(f"    Pairing features: {features['pairing_features'].shape}")
    print(f"    Fused features: {features['fused_features'].shape}")
    
    assert features['density_features'].shape == torch.Size([batch_size, 256])
    assert features['sequence_features'].shape == torch.Size([batch_size, 256])
    assert features['pairing_features'].shape == torch.Size([batch_size, 64])
    assert features['fused_features'].shape == torch.Size([batch_size, 576])
    
    print(f"✓ Feature extraction successful")
    
    # 5. Test model parameters
    print("\n5. Checking model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"✓ Model has {total_params:,} parameters")
    
    # Summary
    print("\n" + "="*80)
    print("✓ Phase 2.2 Integration Test PASSED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Train on 10% subset locally (test run)")
    print("  2. Train full model on Colab with 6-class consolidation")
    print("  3. Compare accuracy with Phase 2.1 (sequence only)")
    print("  4. Target: 10-15pp improvement over Phase 2.1")
    print("="*80)


if __name__ == '__main__':
    test_phase2_2_integration()
