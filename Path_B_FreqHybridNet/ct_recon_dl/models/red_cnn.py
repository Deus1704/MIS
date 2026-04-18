"""RED-CNN: Residual Encoder-Decoder CNN for CT image denoising/reconstruction.

Reference: Chen, H., et al. "Low-dose CT with a residual encoder-decoder convolutional
neural network." IEEE TMI, 2017.

Architecture: Symmetric encoder-decoder with residual skip connection.
Input: noisy FBP reconstruction → Output: clean CT image.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class REDCNN(nn.Module):
    """RED-CNN: 10-layer symmetric encoder-decoder with global residual connection.

    The network predicts the NOISE/ARTIFACT residual; the clean image is
    recovered as:  clean = noisy_fbp + network(noisy_fbp)
    (network learns negative of noise, i.e., net output + input = clean).
    """

    def __init__(self, n_channels: int = 1, n_filters: int = 64, n_layers: int = 10, kernel: int = 3):
        super().__init__()
        assert n_layers % 2 == 0, "n_layers must be even for symmetric encoder-decoder"

        layers: list[nn.Module] = []
        # Encoder
        layers.append(ConvBlock(n_channels, n_filters, kernel=kernel))
        for _ in range(n_layers // 2 - 1):
            layers.append(ConvBlock(n_filters, n_filters, kernel=kernel))
        # Decoder
        for _ in range(n_layers // 2 - 1):
            layers.append(ConvBlock(n_filters, n_filters, kernel=kernel))
        # Final layer — no BN, no ReLU (output can be negative residual)
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=kernel, padding=kernel//2, bias=True))

        self.encoder_decoder = nn.Sequential(*layers)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] noisy FBP image → clean image."""
        residual = self.encoder_decoder(x)
        return x + residual  # global residual learning
