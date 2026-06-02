"""
M1 — 3D CNN Baseline

Three-stage 3D convolutional encoder with a flat 25-class classification head.
Each stage applies two Conv3d → BN → ReLU blocks followed by MaxPool3d(2).
Channel progression: in_channels → 32 → 64 → 256.
A global average pool collapses the spatial dims to a 256-dim embedding before
the linear classifier.

This model returns a plain Tensor (not a dict), so train.py auto-selects
FlatLoss and evaluate.py maps predictions back to L1/L2 via the label hierarchy.

Architecture (default in_channels=1, input 64³):

    Stage 1:  Conv(in,  32) → Conv(32,  32) → MaxPool  →  [32, 32, 32]
    Stage 2:  Conv(32,  64) → Conv(64,  64) → MaxPool  →  [16, 16, 16]
    Stage 3:  Conv(64, 128) → Conv(128,256) → MaxPool  →  [ 8,  8,  8]
    Head:     AdaptiveAvgPool3d(1) → Flatten → Dropout → Linear(256, num_classes)

Future upgrade: set in_channels=4 to accept [density, backbone, ribose, base]
when PDB label maps become available (no other changes needed).
"""
import torch
import torch.nn as nn


class CNN3DBaseline(nn.Module):
    """3D CNN baseline classifier (M1).

    Args:
        in_channels:  Number of input channels. Default 1 (density only).
                      Set to 4 when PDB label maps are added.
        num_classes:  Output classes. Default 25 (full CoSSMos taxonomy).
        dropout:      Dropout rate applied before the final linear layer.
    """

    name = "cnn_baseline"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 25,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 32),
            nn.MaxPool3d(2),             # → [B, 32, 32, 32]
        )
        self.stage2 = nn.Sequential(
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool3d(2),             # → [B, 64, 16, 16]
        )
        self.stage3 = nn.Sequential(
            _conv_block(64, 128),
            _conv_block(128, 256),
            nn.MaxPool3d(2),             # → [B, 256, 8, 8]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),     # → [B, 256, 1, 1, 1]
            nn.Flatten(),                # → [B, 256]
            nn.Dropout(dropout),
            nn.Linear(256, num_classes), # → [B, num_classes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [B, in_channels, 64, 64, 64]

        Returns:
            logits: FloatTensor [B, num_classes]
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv3d(3×3×3) → BatchNorm → ReLU. bias=False since BN follows."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )
