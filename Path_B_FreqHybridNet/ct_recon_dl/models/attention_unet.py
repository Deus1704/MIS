"""Attention U-Net: U-Net augmented with channel + spatial attention gates.

Channel attention: Squeeze-Excitation (SE) blocks emphasize diagnostically
  important feature channels.
Spatial attention gates: Focus decoder on relevant spatial regions from skip
  connections, suppressing irrelevant background activations.

References:
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas." 2018.
- Hu et al., "Squeeze-and-Excitation Networks." CVPR, 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    """Channel attention via global average pooling → FC → sigmoid gating."""

    def __init__(self, n_channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, n_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttentionGate(nn.Module):
    """Additive attention gate for skip connections.

    Aligns skip-connection features (g from decoder, x from encoder skip)
    and produces a spatial attention map to weight the skip.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """g: gating signal (decoder), x: skip connection (encoder)."""
        # Upsample g to match x if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # attended skip features


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConvSE(nn.Module):
    """DoubleConv with Squeeze-Excitation channel attention."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class DownSE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConvSE(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpAttn(nn.Module):
    """Upsample + spatial attention gate + concat + DoubleConvSE."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = SpatialAttentionGate(F_g=in_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv = DoubleConvSE(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip spatial size
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

class AttentionUNet(nn.Module):
    """U-Net with channel (SE) + spatial attention gates at every skip connection.

    Encoder:   1 → 64 → 128 → 256 → 512
    Bottleneck: 512 → 1024 with SE
    Decoder:   1024 → 512 → 256 → 128 → 64 → 1
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_ch: int = 64):
        super().__init__()
        b = base_ch
        self.inc = DoubleConvSE(n_channels, b)
        self.down1 = DownSE(b, b * 2)
        self.down2 = DownSE(b * 2, b * 4)
        self.down3 = DownSE(b * 4, b * 8)
        self.down4 = DownSE(b * 8, b * 16)

        self.up1 = UpAttn(in_ch=b * 16, skip_ch=b * 8, out_ch=b * 8)
        self.up2 = UpAttn(in_ch=b * 8, skip_ch=b * 4, out_ch=b * 4)
        self.up3 = UpAttn(in_ch=b * 4, skip_ch=b * 2, out_ch=b * 2)
        self.up4 = UpAttn(in_ch=b * 2, skip_ch=b, out_ch=b)

        self.outc = nn.Conv2d(b, n_classes, kernel_size=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
