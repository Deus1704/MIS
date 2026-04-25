from __future__ import annotations

import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DifferentiableParallelBeamRadon(nn.Module):
    """Approximate parallel-beam Radon transform using differentiable image rotations.

    The forward projection rotates the image by each angle, integrates along the
    vertical axis, and optionally resamples the detector axis to the requested
    number of bins. The adjoint performs the corresponding backprojection by
    expanding each 1D projection into a stripe, rotating it back, and summing.

    This is a CPU-friendly approximation that is fully differentiable in PyTorch.
    """

    def __init__(
        self,
        image_size: int,
        num_angles: int,
        num_detectors: int | None = None,
        *,
        align_corners: bool = False,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_angles = int(num_angles)
        self.num_detectors = int(num_detectors or image_size)
        self.align_corners = align_corners

        angles = torch.linspace(0.0, 180.0, self.num_angles + 1, dtype=torch.float32)[:-1]
        self.register_buffer("angles_deg", angles, persistent=False)

        detector_coords = torch.linspace(-1.0, 1.0, self.num_detectors, dtype=torch.float32)
        self.register_buffer("detector_coords", detector_coords, persistent=False)

    @staticmethod
    def _rotation_matrix(angle_deg: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        angle_rad = torch.deg2rad(angle_deg)
        if inverse:
            angle_rad = -angle_rad

        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)
        zeros = torch.zeros_like(cos_theta)

        row1 = torch.stack([cos_theta, -sin_theta, zeros], dim=-1)
        row2 = torch.stack([sin_theta, cos_theta, zeros], dim=-1)
        return torch.stack([row1, row2], dim=0)

    def _rotate(self, image: torch.Tensor, angle_deg: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        batch_size = image.shape[0]
        matrix = self._rotation_matrix(angle_deg, inverse=inverse).to(device=image.device, dtype=image.dtype)
        grid = F.affine_grid(matrix.expand(batch_size, 2, 3), image.shape, align_corners=self.align_corners)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )

    def forward_project(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected image tensor with shape [B, 1, H, W]")
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=self.align_corners,
            )

        projections = []
        pixel_scale = 2.0 / max(self.image_size - 1, 1)

        for angle_deg in self.angles_deg:
            rotated = self._rotate(image, angle_deg, inverse=True)
            projection = rotated.sum(dim=2) * pixel_scale
            if projection.shape[-1] != self.num_detectors:
                projection = F.interpolate(
                    projection,
                    size=self.num_detectors,
                    mode="linear",
                    align_corners=self.align_corners,
                )
            projections.append(projection)

        sinogram = torch.stack(projections, dim=2)
        return sinogram

    def back_project(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.ndim != 4 or sinogram.shape[1] != 1:
            raise ValueError("Expected sinogram tensor with shape [B, 1, A, D]")
        if sinogram.shape[2] != self.num_angles:
            raise ValueError(f"Expected {self.num_angles} angles, got {sinogram.shape[2]}")

        batch_size = sinogram.shape[0]
        detector_grid = self.image_size
        reconstruction = sinogram.new_zeros((batch_size, 1, detector_grid, detector_grid))
        pixel_scale = 2.0 / max(self.image_size - 1, 1)

        for angle_index, angle_deg in enumerate(self.angles_deg):
            projection = sinogram[:, :, angle_index, :]
            if projection.shape[-1] != detector_grid:
                projection = F.interpolate(
                    projection,
                    size=detector_grid,
                    mode="linear",
                    align_corners=self.align_corners,
                )

            stripe = projection.unsqueeze(2).expand(-1, -1, detector_grid, -1)
            rotated = self._rotate(stripe, angle_deg, inverse=False)
            reconstruction = reconstruction + rotated

        return reconstruction * pixel_scale / float(self.num_angles)


class ResidualCNNBlock(nn.Module):
    """A small residual network of the form Id + Conv-PReLU-Conv-PReLU-Conv."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.net(x)


class LearnedPrimalDual(nn.Module):
    """Minimal learned primal-dual unrolling with two primal and two dual channels."""

    def __init__(
        self,
        projector: DifferentiableParallelBeamRadon,
        num_iterations: int = 5,
        num_primal_channels: int = 2,
        num_dual_channels: int = 2,
        hidden_channels: int = 16,
    ):
        super().__init__()
        self.projector = projector
        self.num_iterations = int(num_iterations)
        self.num_primal_channels = int(num_primal_channels)
        self.num_dual_channels = int(num_dual_channels)
        self.primal_projection_index = 1 if self.num_primal_channels > 1 else 0

        dual_in_channels = self.num_dual_channels + 2
        primal_in_channels = self.num_primal_channels + 1

        self.dual_blocks = nn.ModuleList(
            [ResidualCNNBlock(dual_in_channels, self.num_dual_channels, hidden_channels) for _ in range(self.num_iterations)]
        )
        self.primal_blocks = nn.ModuleList(
            [ResidualCNNBlock(primal_in_channels, self.num_primal_channels, hidden_channels) for _ in range(self.num_iterations)]
        )

        self.output_head = nn.Conv2d(self.num_primal_channels, 1, kernel_size=1)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.ndim != 4 or sinogram.shape[1] != 1:
            raise ValueError("LPD expects sinogram shape [B, 1, A, D]")

        batch_size = sinogram.shape[0]
        image_size = self.projector.image_size
        device = sinogram.device
        dtype = sinogram.dtype

        primal = sinogram.new_zeros((batch_size, self.num_primal_channels, image_size, image_size))
        dual = sinogram.new_zeros((batch_size, self.num_dual_channels, self.projector.num_angles, self.projector.num_detectors))

        for iteration in range(self.num_iterations):
            f_proj = self.projector.forward_project(primal[:, self.primal_projection_index : self.primal_projection_index + 1])
            dual_input = torch.cat([dual, f_proj, sinogram], dim=1)
            dual = self.dual_blocks[iteration](dual_input)

            backproj = self.projector.back_project(dual[:, 0:1])
            primal_input = torch.cat([primal, backproj], dim=1)
            primal = self.primal_blocks[iteration](primal_input)

        return self.output_head(primal)


@dataclass(frozen=True)
class PhantomSample:
    image: torch.Tensor
    sinogram: torch.Tensor


class CircularPhantomDataset(Dataset[PhantomSample]):
    """On-the-fly synthetic dataset of random circular phantoms."""

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        projector: DifferentiableParallelBeamRadon,
        *,
        min_circles: int = 1,
        max_circles: int = 4,
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.projector = projector
        self.min_circles = int(min_circles)
        self.max_circles = int(max_circles)
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _draw_single_phantom(self, generator: torch.Generator) -> torch.Tensor:
        coords = torch.linspace(-1.0, 1.0, self.image_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        phantom = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)

        background = 0.02 + 0.03 * torch.rand((), generator=generator)
        phantom += background

        num_circles = int(torch.randint(self.min_circles, self.max_circles + 1, (), generator=generator).item())
        for _ in range(num_circles):
            center_x = torch.rand((), generator=generator).mul(0.9).sub(0.45).item()
            center_y = torch.rand((), generator=generator).mul(0.9).sub(0.45).item()
            radius = torch.rand((), generator=generator).mul(0.25).add(0.10).item()
            intensity = torch.rand((), generator=generator).mul(0.80).add(0.20).item()
            mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= (radius ** 2)
            phantom = torch.where(mask, torch.full_like(phantom, intensity), phantom)

        phantom = phantom.clamp(0.0, 1.0)
        return phantom.unsqueeze(0)

    def __getitem__(self, index: int) -> PhantomSample:
        generator = torch.Generator().manual_seed(self.base_seed + index)
        image = self._draw_single_phantom(generator)
        with torch.no_grad():
            sinogram = self.projector.forward_project(image.unsqueeze(0)).squeeze(0)
        return PhantomSample(image=image, sinogram=sinogram)


def _collate_phantoms(batch: list[PhantomSample]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([sample.image for sample in batch], dim=0)
    sinograms = torch.stack([sample.sinogram for sample in batch], dim=0)
    return images, sinograms


def train_minimal_lpd(
    image_size: int = 64,
    num_angles: int = 45,
    num_detectors: int = 64,
    num_iterations: int = 5,
    hidden_channels: int = 16,
    train_samples: int = 512,
    eval_samples: int = 64,
    batch_size: int = 16,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Minimal CPU Learned Primal-Dual Demo ===")
    print(f"Device: {device}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Angles: {num_angles}")
    print(f"Detectors: {num_detectors}")
    print(f"Iterations: {num_iterations}")

    projector = DifferentiableParallelBeamRadon(
        image_size=image_size,
        num_angles=num_angles,
        num_detectors=num_detectors,
    ).to(device)

    model = LearnedPrimalDual(
        projector=projector,
        num_iterations=num_iterations,
        num_primal_channels=2,
        num_dual_channels=2,
        hidden_channels=hidden_channels,
    ).to(device)

    train_dataset = CircularPhantomDataset(
        num_samples=train_samples,
        image_size=image_size,
        projector=projector,
        seed=seed,
    )
    eval_dataset = CircularPhantomDataset(
        num_samples=eval_samples,
        image_size=image_size,
        projector=projector,
        seed=seed + 10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_phantoms,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_phantoms,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_eval_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, sinograms in train_loader:
            images = images.to(device)
            sinograms = sinograms.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(sinograms)
            loss = criterion(predictions, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)

        model.eval()
        eval_running = 0.0
        eval_batches = 0
        with torch.no_grad():
            for images, sinograms in eval_loader:
                images = images.to(device)
                sinograms = sinograms.to(device)
                predictions = model(sinograms)
                loss = criterion(predictions, images)
                eval_running += loss.item()
                eval_batches += 1

        eval_loss = eval_running / max(eval_batches, 1)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | train MSE {train_loss:.6f} | eval MSE {eval_loss:.6f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), "checkpoints/minimal_lpd_cpu_best.pth")

    torch.save(model.state_dict(), "checkpoints/minimal_lpd_cpu_final.pth")
    print(f"Saved best checkpoint to checkpoints/minimal_lpd_cpu_best.pth")
    print(f"Saved final checkpoint to checkpoints/minimal_lpd_cpu_final.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal differentiable LPD training demo on synthetic circular phantoms.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-angles", type=int, default=45)
    parser.add_argument("--num-detectors", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_minimal_lpd(
        image_size=args.image_size,
        num_angles=args.num_angles,
        num_detectors=args.num_detectors,
        num_iterations=args.num_iterations,
        hidden_channels=args.hidden_channels,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
