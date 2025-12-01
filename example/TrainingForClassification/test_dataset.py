import pandas as pd
from rna_dataset import RNADensityDataset

# Load a small sample
df = pd.read_csv('rna_train.csv').head(2)
print(f"Testing with {len(df)} samples:")
print(df)

try:
    dataset = RNADensityDataset(df, base_path='../../dataset/')
    print(f"\nDataset created with {len(dataset)} samples")
    
    # Try loading first sample
    vol, pdb_feat, label = dataset[0]
    print(f"\nFirst sample loaded successfully:")
    print(f"  Volume shape: {vol.shape}")
    print(f"  PDB features shape: {pdb_feat.shape}")
    print(f"  Label: {label}")
    print("\n✓ Dataset works correctly!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
