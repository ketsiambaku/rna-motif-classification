import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityOnlyUNet(nn.Module):
    """
    3D U-Net that uses ONLY density maps (no PDB features).
    This forces the model to learn structural patterns from 3D geometry.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Global average pooling for classification
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, density):
        # Encoder path
        e1 = self.enc1(density)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Global pooling and classification
        features = self.gap(b)
        output = self.classifier(features)
        
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Test the model
    model = DensityOnlyUNet(num_classes=14)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 64, 64)  # Batch of 2, 64^3 volumes
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
