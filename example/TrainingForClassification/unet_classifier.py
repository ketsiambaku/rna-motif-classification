import torch.nn as nn
import torch

class UNetClassifier(nn.Module):
    def __init__(self, pdb_feat_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 2, stride=2), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(16 + pdb_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_vol, x_pdb):
        x = self.encoder(x_vol)
        x = self.decoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x_all = torch.cat((x, x_pdb), dim=1)  
        return self.classifier(x_all)


