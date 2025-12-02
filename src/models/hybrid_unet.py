"""
Hybrid U-Net architecture combining 3D density with sequence features.

This model uses a dual-branch architecture:
1. Density Branch: 3D U-Net encoder for cryo-EM density volumes
2. Sequence Branch: MLP for size-invariant sequence features
3. Fusion: Concatenate both branches and classify into 15 motif classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityEncoder(nn.Module):
    """
    3D U-Net encoder for processing cryo-EM density volumes.
    
    Architecture:
        Input: (B, 1, 32, 32, 32)
        → Conv blocks with downsampling
        → Output: (B, 256) feature vector
    """
    
    def __init__(self, in_channels=1):
        super(DensityEncoder, self).__init__()
        
        # Encoder pathway with increasing channels
        # Block 1: 32x32x32 -> 16x16x16
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        # Block 2: 16x16x16 -> 8x8x8
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        # Block 3: 8x8x8 -> 4x4x4
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck: 4x4x4
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling to get fixed-size feature vector
        self.gap = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        """
        Forward pass through density encoder.
        
        Args:
            x: Input tensor of shape (B, 1, 32, 32, 32)
            
        Returns:
            Feature vector of shape (B, 256)
        """
        # Encoder pathway
        e1 = self.enc1(x)      # (B, 32, 32, 32, 32)
        p1 = self.pool1(e1)    # (B, 32, 16, 16, 16)
        
        e2 = self.enc2(p1)     # (B, 64, 16, 16, 16)
        p2 = self.pool2(e2)    # (B, 64, 8, 8, 8)
        
        e3 = self.enc3(p2)     # (B, 128, 8, 8, 8)
        p3 = self.pool3(e3)    # (B, 128, 4, 4, 4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p3)  # (B, 256, 4, 4, 4)
        
        # Global average pooling
        features = self.gap(bottleneck)   # (B, 256, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 256)
        
        return features


class SequenceEncoder(nn.Module):
    """
    MLP for processing size-invariant sequence features.
    
    Architecture:
        Input: (B, 24)
        → MLP with hidden layers
        → Output: (B, 256) feature vector
    """
    
    def __init__(self, in_features=24, hidden_dim=128, out_features=256):
        super(SequenceEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through sequence encoder.
        
        Args:
            x: Input tensor of shape (B, 24)
            
        Returns:
            Feature vector of shape (B, 256)
        """
        return self.mlp(x)


class PairingEncoder(nn.Module):
    """
    MLP for processing size-invariant base pairing features.
    
    Architecture:
        Input: (B, 10)
        → MLP with hidden layer
        → Output: (B, 64) feature vector
    """
    
    def __init__(self, in_features=10, hidden_dim=32, out_features=64):
        super(PairingEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through pairing encoder.
        
        Args:
            x: Input tensor of shape (B, 10)
            
        Returns:
            Feature vector of shape (B, 64)
        """
        return self.mlp(x)


class HybridUNet(nn.Module):
    """
    Hybrid U-Net combining 3D density with sequence and pairing features for RNA motif classification.
    
    Architecture:
        Density (B, 1, 32, 32, 32) → DensityEncoder → (B, 256)
        Sequence (B, 24) → SequenceEncoder → (B, 256)
        Pairing (B, 10) → PairingEncoder → (B, 64)
        Concatenate → (B, 576)
        Classifier → (B, n_classes)
    
    Args:
        n_classes: Number of output classes (default: 6 for consolidated)
        in_channels: Number of input channels for density (default: 1)
        seq_features: Number of sequence features (default: 24)
        pairing_features: Number of pairing features (default: 10)
        dropout: Dropout rate for classifier (default: 0.5)
    """
    
    def __init__(
        self,
        n_classes=6,
        in_channels=1,
        seq_features=24,
        pairing_features=10,
        dropout=0.5
    ):
        super(HybridUNet, self).__init__()
        
        self.n_classes = n_classes
        
        # Three branches
        self.density_encoder = DensityEncoder(in_channels=in_channels)
        self.sequence_encoder = SequenceEncoder(in_features=seq_features, out_features=256)
        self.pairing_encoder = PairingEncoder(in_features=pairing_features, out_features=64)
        
        # Fusion and classification head (256 + 256 + 64 = 576)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, density, sequence, pairing):
        """
        Forward pass through hybrid model.
        
        Args:
            density: Density tensor of shape (B, 1, 32, 32, 32)
            sequence: Sequence features of shape (B, 24)
            pairing: Base pairing features of shape (B, 10)
            
        Returns:
            Logits of shape (B, n_classes)
        """
        # Extract features from all three branches
        density_features = self.density_encoder(density)      # (B, 256)
        sequence_features = self.sequence_encoder(sequence)   # (B, 256)
        pairing_features = self.pairing_encoder(pairing)      # (B, 64)
        
        # Concatenate features
        fused = torch.cat([density_features, sequence_features, pairing_features], dim=1)  # (B, 576)
        
        # Classify
        logits = self.classifier(fused)  # (B, n_classes)
        
        return logits
    
    def get_feature_representations(self, density, sequence, pairing):
        """
        Extract intermediate feature representations (for analysis/visualization).
        
        Args:
            density: Density tensor of shape (B, 1, 32, 32, 32)
            sequence: Sequence features of shape (B, 24)
            pairing: Base pairing features of shape (B, 10)
            
        Returns:
            Dictionary with keys:
                - 'density_features': (B, 256)
                - 'sequence_features': (B, 256)
                - 'pairing_features': (B, 64)
                - 'fused_features': (B, 576)
                - 'logits': (B, n_classes)
        """
        density_features = self.density_encoder(density)
        sequence_features = self.sequence_encoder(sequence)
        pairing_features = self.pairing_encoder(pairing)
        fused = torch.cat([density_features, sequence_features, pairing_features], dim=1)
        logits = self.classifier(fused)
        
        return {
            'density_features': density_features,
            'sequence_features': sequence_features,
            'pairing_features': pairing_features,
            'fused_features': fused,
            'logits': logits
        }


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model():
    """Test the HybridUNet model architecture."""
    print("="*80)
    print("Testing HybridUNet Model")
    print("="*80)
    
    # Create model
    print("\n1. Creating model...")
    model = HybridUNet(n_classes=15)
    print(f"✓ Model created")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    density = torch.randn(batch_size, 1, 32, 32, 32)
    sequence = torch.randn(batch_size, 24)
    pairing = torch.randn(batch_size, 10)
    
    print(f"  Input density shape: {density.shape}")
    print(f"  Input sequence shape: {sequence.shape}")
    print(f"  Input pairing shape: {pairing.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(density, sequence, pairing)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"✓ Forward pass successful")
    
    # Test feature extraction
    print("\n3. Testing feature extraction...")
    with torch.no_grad():
        features = model.get_feature_representations(density, sequence, pairing)
    
    print(f"  Density features shape: {features['density_features'].shape}")
    print(f"  Sequence features shape: {features['sequence_features'].shape}")
    print(f"  Pairing features shape: {features['pairing_features'].shape}")
    print(f"  Fused features shape: {features['fused_features'].shape}")
    print(f"  Logits shape: {features['logits'].shape}")
    print(f"✓ Feature extraction successful")
    
    # Test prediction
    print("\n4. Testing prediction...")
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions.tolist()}")
    print(f"  Sample probabilities (first sample):")
    print(f"    Max prob: {probs[0].max().item():.4f} (class {predictions[0].item()})")
    print(f"✓ Prediction successful")
    
    # Test each component separately
    print("\n5. Testing individual components...")
    
    # Density encoder
    density_enc = model.density_encoder
    with torch.no_grad():
        dens_feat = density_enc(density)
    print(f"  DensityEncoder output: {dens_feat.shape}")
    
    # Sequence encoder
    seq_enc = model.sequence_encoder
    with torch.no_grad():
        seq_feat = seq_enc(sequence)
    print(f"  SequenceEncoder output: {seq_feat.shape}")
    
    # Pairing encoder
    pairing_enc = model.pairing_encoder
    with torch.no_grad():
        pair_feat = pairing_enc(pairing)
    print(f"  PairingEncoder output: {pair_feat.shape}")
    
    print(f"✓ All components working")
    
    # Model summary
    print("\n" + "="*80)
    print("Model Architecture Summary")
    print("="*80)
    print(f"\nDensity Branch (DensityEncoder):")
    print(f"  Input: (B, 1, 32, 32, 32)")
    print(f"  Conv blocks: 32 → 64 → 128 → 256 channels")
    print(f"  Output: (B, 256) via global average pooling")
    
    print(f"\nSequence Branch (SequenceEncoder):")
    print(f"  Input: (B, 24)")
    print(f"  MLP: 24 → 128 → 128 → 256")
    print(f"  Dropout: 0.3 after each hidden layer")
    print(f"  Output: (B, 256)")
    
    print(f"\nPairing Branch (PairingEncoder):")
    print(f"  Input: (B, 10)")
    print(f"  MLP: 10 → 32 → 64")
    print(f"  Dropout: 0.2 after hidden layer")
    print(f"  Output: (B, 64)")
    
    print(f"\nFusion & Classifier:")
    print(f"  Concatenate: (B, 256) + (B, 256) + (B, 64) → (B, 576)")
    print(f"  MLP: 576 → 256 → 128 → 15")
    print(f"  Dropout: 0.5 in classifier")
    print(f"  Output: (B, 15) class logits")
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == '__main__':
    test_model()
