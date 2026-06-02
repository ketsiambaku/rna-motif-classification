from __future__ import annotations
"""
M2 — 3D ResNet-18 with Hierarchical Multi-Head Loss

Shared 3D ResNet-18 backbone with three independent classification heads
trained jointly under the weighted hierarchical loss:

    L = α · CE(L1) + β · CE(L2) + γ · CE(L3, masked)

L3 loss is masked so asymmetric loop samples (l3_idx == -1) do not
contribute to the L3 head gradient — those classes stop at L2.

This model returns a dict {"l1", "l2", "l3"} so train.py auto-selects
HierarchicalLoss.

Architecture:

    Stem:    Conv(in, 32, 3) → BN → ReLU → MaxPool     → [32, 32, 32]
    Layer 1: ResBlock(32,  32)  × 2                     → [32, 32, 32]
    Layer 2: ResBlock(32,  64,  s=2) × 2                → [16, 16, 16]
    Layer 3: ResBlock(64,  128, s=2) × 2                → [ 8,  8,  8]
    Layer 4: ResBlock(128, 256, s=2) × 2                → [ 4,  4,  4]
    Pool:    AdaptiveAvgPool3d(1) → Flatten             → [256]

    Head L1: Dropout → Linear(256,  3)   topology
    Head L2: Dropout → Linear(256,  4)   symmetry
    Head L3: Dropout → Linear(256, 15)   fine-grained (15 L3-eligible classes)

Future upgrade: set in_channels=4 for label-map input.
"""
import torch
import torch.nn as nn


class _ResBlock3D(nn.Module):
    """Standard (post-activation) 3D residual block.

    Uses a 1×1×1 projection shortcut when spatial resolution or channel
    width changes, otherwise a plain identity skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet3DBaseline(nn.Module):
    """M2 — 3D ResNet-18 with three hierarchical classification heads.

    Args:
        in_channels:  Input channels. Default 1 (density only).
                      Set to 4 when PDB label maps are added.
        dropout:      Dropout rate before each classification head.
    """

    name = "resnet3d"

    def __init__(
        self,
        in_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                 # → [B, 32, 32, 32]
        )

        self.layer1 = nn.Sequential(         # → [B, 32, 32, 32]
            _ResBlock3D(32,  32),
            _ResBlock3D(32,  32),
        )
        self.layer2 = nn.Sequential(         # → [B, 64, 16, 16]
            _ResBlock3D(32,  64,  stride=2),
            _ResBlock3D(64,  64),
        )
        self.layer3 = nn.Sequential(         # → [B, 128, 8, 8]
            _ResBlock3D(64,  128, stride=2),
            _ResBlock3D(128, 128),
        )
        self.layer4 = nn.Sequential(         # → [B, 256, 4, 4]
            _ResBlock3D(128, 256, stride=2),
            _ResBlock3D(256, 256),
        )

        self.pool    = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

        self.head_l1 = nn.Linear(256,  3)   # hairpin / internal_loop / bulge
        self.head_l2 = nn.Linear(256,  4)   # hairpin / symmetric / asymmetric / bulge
        self.head_l3 = nn.Linear(256, 15)   # 15 L3-eligible classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: FloatTensor [B, in_channels, 64, 64, 64]

        Returns:
            dict with keys "l1" [B,3], "l2" [B,4], "l3" [B,15]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)   # [B, 256]
        x = self.dropout(x)

        return {
            "l1": self.head_l1(x),
            "l2": self.head_l2(x),
            "l3": self.head_l3(x),
        }
