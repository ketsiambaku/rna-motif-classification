import torch
from unet_classifier_fixed import UNetClassifier

# Test model instantiation
model = UNetClassifier(pdb_feat_dim=900, num_classes=3)
print("Model created successfully")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass with dummy data
batch_size = 2
vol = torch.randn(batch_size, 1, 64, 64, 64)
pdb_feat = torch.randn(batch_size, 900)

output = model(vol, pdb_feat)
print(f"\nInput volume shape: {vol.shape}")
print(f"Input PDB features shape: {pdb_feat.shape}")
print(f"Output shape: {output.shape}")
print(f"Output logits: {output}")

print("\n✓ Model works correctly!")
