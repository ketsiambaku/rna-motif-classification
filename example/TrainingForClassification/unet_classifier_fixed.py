import torch.nn as nn
import torch

class UNetClassifier(nn.Module):
    """
    Fixed U-Net Classifier that combines 3D volumetric data with PDB features.
    
    Architecture:
    - Encoder: Two conv blocks with pooling (1->32->64 channels)
    - Decoder: Two transposed conv blocks (64->32->16 channels)
    - Classifier: Global pooling + FC layers combining volume and PDB features
    """
    def __init__(self, pdb_feat_dim, num_classes):
        super().__init__()
        # Encoder path
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool3d(2)
        )
        
        # Decoder path
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 2, stride=2), 
            nn.ReLU()
        )
        
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(16 + pdb_feat_dim, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_vol, x_pdb):
        # Encode volumetric data
        x = self.encoder(x_vol)
        
        # Decode back to spatial resolution
        x = self.decoder(x)
        
        # Global pooling to get fixed-size features
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Concatenate volume features with PDB features
        x_all = torch.cat((x, x_pdb), dim=1)
        
        # Classification
        return self.classifier(x_all)
