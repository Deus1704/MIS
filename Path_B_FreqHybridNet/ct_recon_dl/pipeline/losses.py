"""Loss functions for CT reconstruction training.

All losses operate on [B, 1, H, W] tensors in [0, 1] range.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------

def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Differentiable SSIM loss using separable Gaussian filter."""
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    g1 = gaussian_kernel(kernel_size, sigma).to(pred.device)
    g2 = g1.unsqueeze(0)  # [1, ks]
    g1 = g1.unsqueeze(1)  # [ks, 1]

    # Build 2D kernel as outer product
    kernel_2d = g1 @ g2  # [ks, ks]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, ks, ks]

    padding = kernel_size // 2

    def gauss_filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])

    mu1 = gauss_filter(pred)
    mu2 = gauss_filter(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = gauss_filter(pred ** 2) - mu1_sq
    sigma2_sq = gauss_filter(target ** 2) - mu2_sq
    sigma12 = gauss_filter(pred * target) - mu12

    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)

    return 1.0 - ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _ssim_loss(pred, target, self.kernel_size, self.sigma)


# ---------------------------------------------------------------------------
# Gradient / edge-consistency loss
# ---------------------------------------------------------------------------

class GradientLoss(nn.Module):
    """Penalizes differences in image gradients (edge sharpness)."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_pred = F.conv2d(pred, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred, self.sobel_y, padding=1)
        gx_tgt = F.conv2d(target, self.sobel_x, padding=1)
        gy_tgt = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)


# ---------------------------------------------------------------------------
# Sinogram consistency loss (for FreqHybridNet)
# ---------------------------------------------------------------------------

class SinogramConsistencyLoss(nn.Module):
    """Penalize frequency-filtered sinogram deviation from the input sinogram.

    The filtered sinogram should retain overall structure (data consistency).
    We use L1 loss weighted by inverse frequency to preserve low-freq structure.
    """

    def forward(self, filtered_sino: torch.Tensor, orig_sino: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(filtered_sino, orig_sino)


# ---------------------------------------------------------------------------
# Combined loss functions per model
# ---------------------------------------------------------------------------

class REDCNNLoss(nn.Module):
    """MSE loss for RED-CNN."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mse = F.mse_loss(pred, target)
        return {"total": mse, "mse": mse}


class UNetLoss(nn.Module):
    """0.8 × MSE + 0.2 × SSIM loss for U-Net."""

    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mse = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        total = 0.8 * mse + 0.2 * ssim
        return {"total": total, "mse": mse, "ssim_loss": ssim}


class AttentionUNetLoss(nn.Module):
    """0.7 × MSE + 0.2 × SSIM + 0.1 × Gradient loss for Attention U-Net."""

    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mse = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        grad = self.grad(pred, target)
        total = 0.7 * mse + 0.2 * ssim + 0.1 * grad
        return {"total": total, "mse": mse, "ssim_loss": ssim, "grad_loss": grad}


class FreqHybridLoss(nn.Module):
    """0.6 × MSE + 0.2 × SSIM + 0.2 × SinogramConsistency."""

    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()
        self.sino_cons = SinogramConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        filtered_sino: torch.Tensor,
        orig_sino: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mse = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        sc = self.sino_cons(filtered_sino, orig_sino)
        total = 0.6 * mse + 0.2 * ssim + 0.2 * sc
        return {"total": total, "mse": mse, "ssim_loss": ssim, "sino_cons": sc}
