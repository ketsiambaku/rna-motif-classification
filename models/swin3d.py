from __future__ import annotations
"""
M3 — 3D Swin Transformer with Conditional Hierarchical Inference

Backbone: 3D Swin Transformer operating on 64³ density volumes.
  - 4×4×4 patch embedding → 16³ token grid
  - 4 stages with window-based shifted multi-head self-attention
  - Patch merging between stages doubles channels, halves spatial dims

Conditional heads (soft gating, fully differentiable):
  L1 logits = Linear(384, 3)
  p_L1       = softmax(L1 logits)

  L2 input   = concat(features, p_L1)     # 387-dim
  L2 logits  = Linear(387, 4)
  p_L2       = softmax(L2 logits)

  L3 input   = concat(features, p_L2)     # 388-dim
  L3 logits  = Linear(388, 15)            # 15 L3-eligible classes

The L1 soft probabilities "gate" L2, and L2 gates L3 — each level's
uncertainty is propagated forward before the next decision is made.
During inference, optional hard-constraint decoding can zero out
logits that are impossible given the L2 prediction (see decode()).

Returns dict {"l1", "l2", "l3"} → train.py auto-selects HierarchicalLoss.

Architecture summary (default in_channels=1):

    PatchEmbed: Conv3d(1, 48, 4, stride=4) → [B, 48, 16, 16, 16]

    Stage 1: SwinLayer(dim=48,  depth=2, heads=3,  window=4)  → [B, 48,  16,16,16]
    Merge  : PatchMerging(48  → 96)                           → [B, 96,   8, 8, 8]

    Stage 2: SwinLayer(dim=96,  depth=2, heads=6,  window=4)  → [B, 96,   8, 8, 8]
    Merge  : PatchMerging(96  → 192)                          → [B, 192,  4, 4, 4]

    Stage 3: SwinLayer(dim=192, depth=6, heads=12, window=4)  → [B, 192,  4, 4, 4]
    Merge  : PatchMerging(192 → 384)                          → [B, 384,  2, 2, 2]

    Stage 4: SwinLayer(dim=384, depth=2, heads=24, window=2)  → [B, 384,  2, 2, 2]

    GlobalAvgPool → [B, 384]

Memory note: gradient checkpointing is applied to stages 3–4 if
torch.utils.checkpoint is available and the model is in training mode.
"""
import math
from typing import Optional
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _window_partition(x: torch.Tensor, w: int) -> torch.Tensor:
    """Partition a [B, D, H, W, C] tensor into non-overlapping windows.

    Returns: [num_windows*B, w, w, w, C]
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // w, w, H // w, w, W // w, w, C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, w, w, w, C)


def _window_reverse(windows: torch.Tensor, w: int, D: int, H: int, W: int) -> torch.Tensor:
    """Reverse _window_partition. Returns [B, D, H, W, C]."""
    B = int(windows.shape[0] / (D * H * W / w ** 3))
    x = windows.view(B, D // w, H // w, W // w, w, w, w, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


@lru_cache(maxsize=None)
def _relative_position_index(w: int) -> torch.Tensor:
    """Pre-compute relative position index table for a window of size w³."""
    coords = torch.stack(torch.meshgrid(
        torch.arange(w), torch.arange(w), torch.arange(w), indexing="ij"
    ))                                                          # [3, w, w, w]
    flat   = coords.flatten(1).T                               # [w³, 3]
    rel    = flat.unsqueeze(0) - flat.unsqueeze(1)             # [w³, w³, 3]
    rel   += w - 1
    rel[:, :, 0] *= (2 * w - 1) ** 2
    rel[:, :, 1] *= (2 * w - 1)
    return rel.sum(-1)                                          # [w³, w³]


# ---------------------------------------------------------------------------
# Window-based Multi-Head Self-Attention (W-MSA / SW-MSA)
# ---------------------------------------------------------------------------

class _WindowAttention3D(nn.Module):
    """3D window attention with optional shifted windows and relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Learnable relative position bias table
        n_rel = (2 * window_size - 1) ** 3
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_rel, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [num_windows*B, w³, C]
            mask: [num_windows, w³, w³] attention mask for shifted windows, or None
        """
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)    # each [Bw, heads, N, C/heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        idx  = _relative_position_index(self.window_size).to(x.device)
        bias = self.rel_pos_bias[idx].permute(2, 0, 1).unsqueeze(0)  # [1, heads, N, N]
        attn = attn + bias

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------

class _SwinBlock3D(nn.Module):
    """One Swin Transformer block (W-MSA or SW-MSA + FFN)."""

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int,
        shift:       bool,
        mlp_ratio:   float = 4.0,
        drop:        float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def _attn_mask(self, D: int, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        s  = self.shift_size
        w  = self.window_size
        img_mask = torch.zeros(1, D, H, W, 1, device=device)
        for d_slice in (slice(0, -w), slice(-w, -s), slice(-s, None)):
            for h_slice in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                for w_slice in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                    img_mask[:, d_slice, h_slice, w_slice, :] += 1
        # each region gets a unique count → use as label
        windows = _window_partition(img_mask, w).squeeze(-1).view(-1, w ** 3)
        mask    = windows.unsqueeze(1) - windows.unsqueeze(2)   # [nW, w³, w³]
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            s = self.shift_size
            x = torch.roll(x, shifts=(-s, -s, -s), dims=(1, 2, 3))

        # Partition → attention → reverse
        windows = _window_partition(x, self.window_size).view(-1, self.window_size ** 3, C)
        mask    = self._attn_mask(D, H, W, x.device)
        windows = self.attn(windows, mask=mask)
        x       = _window_reverse(windows.view(-1, self.window_size, self.window_size, self.window_size, C),
                                   self.window_size, D, H, W)

        # Reverse shift
        if self.shift_size > 0:
            s = self.shift_size
            x = torch.roll(x, shifts=(s, s, s), dims=(1, 2, 3))

        x = x.view(B, N, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Merging (downsampling between stages)
# ---------------------------------------------------------------------------

class _PatchMerging3D(nn.Module):
    """Concatenate 2×2×2 neighbouring patches and project to 2×input_dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.proj = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> tuple[torch.Tensor, int, int, int]:
        B, _, C = x.shape
        x = x.view(B, D, H, W, C)
        x = torch.cat([
            x[:, 0::2, 0::2, 0::2, :],
            x[:, 1::2, 0::2, 0::2, :],
            x[:, 0::2, 1::2, 0::2, :],
            x[:, 0::2, 0::2, 1::2, :],
            x[:, 1::2, 1::2, 0::2, :],
            x[:, 1::2, 0::2, 1::2, :],
            x[:, 0::2, 1::2, 1::2, :],
            x[:, 1::2, 1::2, 1::2, :],
        ], dim=-1)
        D2, H2, W2 = D // 2, H // 2, W // 2
        x = x.view(B, D2 * H2 * W2, 8 * C)
        return self.proj(self.norm(x)), D2, H2, W2


# ---------------------------------------------------------------------------
# Stage (sequence of Swin blocks)
# ---------------------------------------------------------------------------

class _SwinStage3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock3D(dim, num_heads, window_size, shift=(i % 2 == 1))
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        for blk in self.blocks:
            if self.training:
                x = checkpoint.checkpoint(blk, x, D, H, W, use_reentrant=False)
            else:
                x = blk(x, D, H, W)
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class _PatchEmbed3D(nn.Module):
    """Split volume into non-overlapping 4×4×4 patches and embed."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        x = self.proj(x)                        # [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # [B, D*H*W, C]
        return self.norm(x), D, H, W


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SwinTransformer3D(nn.Module):
    """M3 — 3D Swin Transformer with conditional hierarchical inference.

    Args:
        in_channels:  Input channels (default 1; set to 4 with label maps).
        embed_dim:    Base embedding dimension (default 48).
        depths:       Number of Swin blocks per stage (default [2,2,6,2]).
        num_heads:    Attention heads per stage (default [3,6,12,24]).
        window_size:  Attention window size (default 4).
        dropout:      Dropout rate in attention and FFN (default 0.1).
    """

    name = "swin3d"

    def __init__(
        self,
        in_channels:  int        = 1,
        embed_dim:    int        = 48,
        depths:       list[int]  = None,
        num_heads:    list[int]  = None,
        window_size:  int        = 4,
        dropout:      float      = 0.1,
    ) -> None:
        super().__init__()
        depths    = depths    or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.patch_embed = _PatchEmbed3D(in_channels, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)

        # 4 stages with patch merging between them
        dims = [embed_dim * (2 ** i) for i in range(4)]   # [48, 96, 192, 384]

        self.stage1 = _SwinStage3D(dims[0], depths[0], num_heads[0], window_size)
        self.merge1 = _PatchMerging3D(dims[0])

        self.stage2 = _SwinStage3D(dims[1], depths[1], num_heads[1], window_size)
        self.merge2 = _PatchMerging3D(dims[1])

        self.stage3 = _SwinStage3D(dims[2], depths[2], num_heads[2], window_size)
        self.merge3 = _PatchMerging3D(dims[2])

        self.stage4 = _SwinStage3D(dims[3], depths[3], num_heads[3], window_size=2)

        self.norm = nn.LayerNorm(dims[3])

        feat_dim = dims[3]   # 384

        # Conditional heads
        self.head_l1 = nn.Linear(feat_dim,          3)   # topology
        self.head_l2 = nn.Linear(feat_dim + 3,      4)   # + p_L1
        self.head_l3 = nn.Linear(feat_dim + 4,     15)   # + p_L2

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: FloatTensor [B, in_channels, 64, 64, 64]

        Returns:
            dict with keys "l1" [B,3], "l2" [B,4], "l3" [B,15]
        """
        x, D, H, W = self.patch_embed(x)        # [B, 16³, 48]
        x = self.pos_drop(x)

        x = self.stage1(x, D, H, W)
        x, D, H, W = self.merge1(x, D, H, W)    # → [B, 8³, 96]

        x = self.stage2(x, D, H, W)
        x, D, H, W = self.merge2(x, D, H, W)    # → [B, 4³, 192]

        x = self.stage3(x, D, H, W)
        x, D, H, W = self.merge3(x, D, H, W)    # → [B, 2³, 384]

        x = self.stage4(x, D, H, W)
        x = self.norm(x)

        f = x.mean(dim=1)                        # global avg pool → [B, 384]

        # Conditional heads
        l1_logits = self.head_l1(f)
        p_l1      = F.softmax(l1_logits, dim=-1).detach()   # stop-grad for gating

        l2_logits = self.head_l2(torch.cat([f, p_l1], dim=-1))
        p_l2      = F.softmax(l2_logits, dim=-1).detach()

        l3_logits = self.head_l3(torch.cat([f, p_l2], dim=-1))

        return {"l1": l1_logits, "l2": l2_logits, "l3": l3_logits}

    def decode(
        self,
        output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Hard-constraint decoding (inference only).

        After predicting L2, zero out L3 logits for classes that are
        impossible given the predicted L2 group:
          L2=hairpin    → L3 must be hairpin3–7     (indices 0–4)
          L2=symmetric  → L3 must be 1×1–5×5        (indices 5–9)
          L2=asymmetric → no L3 prediction
          L2=bulge      → L3 must be bulge1–5        (indices 10–14)
        """
        # L2 index → valid L3 index range
        L2_TO_L3 = {
            0: (0,  5),   # hairpin
            1: (5, 10),   # symmetric
            2: None,      # asymmetric — no L3
            3: (10, 15),  # bulge
        }
        l2_pred   = output["l2"].argmax(1)       # [B]
        l3_logits = output["l3"].clone()

        for b in range(l2_pred.size(0)):
            valid = L2_TO_L3[l2_pred[b].item()]
            if valid is None:
                l3_logits[b] = float("-inf")     # mark as undefined
            else:
                mask = torch.ones(15, dtype=torch.bool, device=l3_logits.device)
                mask[valid[0]:valid[1]] = False
                l3_logits[b, mask] = float("-inf")

        return {**output, "l3": l3_logits}
