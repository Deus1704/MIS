import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_rotation_matrices(angles_deg: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Create 2x3 affine matrices for batched image rotation."""
    angles_rad = torch.deg2rad(angles_deg)
    if inverse:
        angles_rad = -angles_rad

    cos_t = torch.cos(angles_rad)
    sin_t = torch.sin(angles_rad)
    zeros = torch.zeros_like(cos_t)

    row1 = torch.stack([cos_t, -sin_t, zeros], dim=-1)
    row2 = torch.stack([sin_t, cos_t, zeros], dim=-1)
    return torch.stack([row1, row2], dim=1)


class ParallelBeamOperatorMixin:
    """Utility shared by forward and adjoint projector implementations."""

    chunk_size: int
    image_size: int
    num_angles: int
    num_detectors: int
    align_corners: bool
    pad_value: float

    def _resize_or_pad_to_detector_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Map image-domain tensors to a square detector grid."""
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=self.align_corners,
            )

        if self.num_detectors == self.image_size:
            return x

        if self.num_detectors < self.image_size:
            return F.interpolate(
                x,
                size=(self.num_detectors, self.num_detectors),
                mode="bilinear",
                align_corners=self.align_corners,
            )

        pad_total = self.num_detectors - self.image_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return F.pad(
            x,
            (pad_before, pad_after, pad_before, pad_after),
            mode="constant",
            value=self.pad_value,
        )

    def _crop_to_image_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Crop detector-grid tensors back to the requested image size."""
        if x.shape[-2:] == (self.image_size, self.image_size):
            return x

        start = max((x.shape[-1] - self.image_size) // 2, 0)
        end = start + self.image_size
        return x[..., start:end, start:end]


class ForwardOperator(nn.Module, ParallelBeamOperatorMixin):
    """Differentiable parallel-beam Radon transform approximation in PyTorch.

    This operator rotates the image onto a detector-aligned grid and integrates
    along one axis to obtain line integrals. It is not a perfect substitute for
    ODL/ASTRA geometry, but it is a physically meaningful and differentiable
    forward projector, unlike the previous placeholder.
    """

    def __init__(
        self,
        image_size: int,
        num_angles: int,
        num_detectors: int,
        *,
        chunk_size: int = 8,
        align_corners: bool = False,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors
        self.chunk_size = max(1, chunk_size)
        self.align_corners = align_corners
        self.pad_value = pad_value

        angles = torch.linspace(0.0, 180.0, num_angles + 1, dtype=torch.float32)[:-1]
        self.register_buffer("angles_deg", angles, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("ForwardOperator expects input of shape [B, 1, H, W]")

        x = self._resize_or_pad_to_detector_grid(x)
        batch_size = x.shape[0]
        detector_grid = x.shape[-1]
        projections: list[torch.Tensor] = []

        for start in range(0, self.num_angles, self.chunk_size):
            stop = min(start + self.chunk_size, self.num_angles)
            chunk_angles = self.angles_deg[start:stop].to(device=x.device, dtype=x.dtype)
            matrices = _build_rotation_matrices(chunk_angles).to(device=x.device, dtype=x.dtype)
            chunk = stop - start

            x_rep = (
                x[:, None, :, :, :]
                .expand(batch_size, chunk, 1, detector_grid, detector_grid)
                .reshape(batch_size * chunk, 1, detector_grid, detector_grid)
            )
            theta = (
                matrices[None, :, :, :]
                .expand(batch_size, chunk, 2, 3)
                .reshape(batch_size * chunk, 2, 3)
            )

            grid = F.affine_grid(
                theta,
                size=x_rep.shape,
                align_corners=self.align_corners,
            )
            rotated = F.grid_sample(
                x_rep,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )

            projection = rotated.mean(dim=2)
            projection = projection.reshape(batch_size, chunk, detector_grid)
            projections.append(projection)

        sinogram = torch.cat(projections, dim=1)
        return sinogram.unsqueeze(1)


class AdjointOperator(nn.Module, ParallelBeamOperatorMixin):
    """Differentiable adjoint of the parallel-beam projection approximation."""

    def __init__(
        self,
        image_size: int,
        num_angles: int,
        num_detectors: int,
        *,
        chunk_size: int = 8,
        align_corners: bool = False,
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors
        self.chunk_size = max(1, chunk_size)
        self.align_corners = align_corners
        self.pad_value = pad_value

        angles = torch.linspace(0.0, 180.0, num_angles + 1, dtype=torch.float32)[:-1]
        self.register_buffer("angles_deg", angles, persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError("AdjointOperator expects input of shape [B, 1, angles, detectors]")

        batch_size, _, num_angles, num_detectors = y.shape
        if num_angles != self.num_angles:
            raise ValueError(f"Expected {self.num_angles} angles, got {num_angles}")
        if num_detectors != self.num_detectors:
            raise ValueError(f"Expected {self.num_detectors} detectors, got {num_detectors}")

        detector_grid = self.num_detectors
        recon = y.new_zeros((batch_size, 1, detector_grid, detector_grid))

        for start in range(0, self.num_angles, self.chunk_size):
            stop = min(start + self.chunk_size, self.num_angles)
            chunk_angles = self.angles_deg[start:stop].to(device=y.device, dtype=y.dtype)
            matrices = _build_rotation_matrices(chunk_angles, inverse=True).to(device=y.device, dtype=y.dtype)
            chunk = stop - start

            slice_proj = y[:, :, start:stop, :]
            stripes = (
                slice_proj.permute(0, 2, 1, 3)[:, :, :, None, :]
                .expand(batch_size, chunk, 1, detector_grid, detector_grid)
                .reshape(batch_size * chunk, 1, detector_grid, detector_grid)
            )
            theta = (
                matrices[None, :, :, :]
                .expand(batch_size, chunk, 2, 3)
                .reshape(batch_size * chunk, 2, 3)
            )

            grid = F.affine_grid(
                theta,
                size=stripes.shape,
                align_corners=self.align_corners,
            )
            rotated = F.grid_sample(
                stripes,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            recon = recon + rotated.reshape(batch_size, chunk, 1, detector_grid, detector_grid).sum(dim=1)

        recon = recon / float(self.num_angles)
        recon = self._crop_to_image_grid(recon)
        return recon

class DualBlock(nn.Module):
    """
    Dual Block (Data domain network).
    Operates in the measurement (sinogram) space.
    Inputs:
    - h_prev: hidden state of the dual variable from the previous iteration.
    - f_proj: forward projection of the current primal variable (Kf_i).
    - g: the measured data (sinogram).
    """
    def __init__(self, num_channels=3, hidden_channels=32):
        super().__init__()
        # Input channels: h_prev (c_h) + f_proj (1) + g (1)
        # We assume h_prev has num_channels, f_proj has 1, g has 1. Total = num_channels + 2
        in_channels = num_channels + 2
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, h_prev, f_proj, g):
        # Concatenate inputs along the channel dimension
        x = torch.cat([h_prev, f_proj, g], dim=1)
        # Residual connection updates the hidden state
        h_next = h_prev + self.net(x)
        return h_next

class PrimalBlock(nn.Module):
    """
    Primal Block (Image domain network).
    Operates in the reconstruction (image) space.
    Inputs:
    - f_prev: hidden state of the primal variable from the previous iteration.
    - h_backproj: adjoint projection of the updated dual variable (K^* h_{i+1}).
    """
    def __init__(self, num_channels=3, hidden_channels=32):
        super().__init__()
        # Input channels: f_prev (c_f) + h_backproj (1)
        # We assume h_backproj collapses its channels or we use the first channel.
        # To strictly follow LPD, f_prev has num_channels. h_next might have multiple channels,
        # but the adjoint operator usually acts on a 1-channel pseudo-sinogram or we sum them.
        # Here we assume the adjoint operator provides a 1-channel image space representation
        # for each channel of h, or we just project the "first" channel of h for simplicity.
        # Let's project channel 0 of h, so h_backproj has 1 channel.
        in_channels = num_channels + 1
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, f_prev, h_backproj):
        # Concatenate inputs
        x = torch.cat([f_prev, h_backproj], dim=1)
        # Residual connection
        f_next = f_prev + self.net(x)
        return f_next

class LearnedPrimalDual(nn.Module):
    """
    Learned Primal-Dual (LPD) architecture for CT Reconstruction.
    Paper: Adler & Öktem (2018), "Learned Primal-Dual Reconstruction".
    
    Unrolls a block-iterative scheme taking physics (forward/adjoint operators)
    into account at every step.
    """
    def __init__(self, 
                 num_iterations=10, 
                 num_primal_channels=5, 
                 num_dual_channels=5,
                 image_size=128,
                 num_angles=180,
                 num_detectors=180):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_primal_channels = num_primal_channels
        self.num_dual_channels = num_dual_channels
        
        # Physics operators (mocked)
        self.primal_projection_idx = 1 if num_primal_channels > 1 else 0
        self.op_forward = ForwardOperator(
            image_size=image_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
        )
        self.op_adjoint = AdjointOperator(
            image_size=image_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
        )
        
        # We create separate Primal and Dual blocks for each iteration
        # to allow them to learn iteration-specific features.
        self.dual_blocks = nn.ModuleList([
            DualBlock(num_channels=num_dual_channels) for _ in range(num_iterations)
        ])
        self.primal_blocks = nn.ModuleList([
            PrimalBlock(num_channels=num_primal_channels) for _ in range(num_iterations)
        ])
        
        # Final convolution to output a 1-channel image
        self.final_conv = nn.Conv2d(num_primal_channels, 1, kernel_size=1)

    def forward(self, g):
        """
        g: measured data (sinogram), shape (B, 1, num_angles, num_detectors)
        """
        B = g.shape[0]
        device = g.device
        
        # Initialize primal (f) and dual (h) states with zeros
        # f: image space hidden variables
        # h: measurement space hidden variables
        f = torch.zeros(B, self.num_primal_channels, self.op_adjoint.image_size, self.op_adjoint.image_size, device=device)
        h = torch.zeros(B, self.num_dual_channels, self.op_forward.num_angles, self.op_forward.num_detectors, device=device)
        
        for i in range(self.num_iterations):
            # --- DUAL STEP ---
            # Forward-project a dedicated primal channel, following the LPD idea
            # that not every latent channel is the final reconstructed image.
            f_proj = self.op_forward(f[:, self.primal_projection_idx:self.primal_projection_idx + 1, :, :])
            # 2. Update dual variable
            h = self.dual_blocks[i](h, f_proj, g)
            
            # --- PRIMAL STEP ---
            # 1. Backproject the FIRST channel of the updated dual variable
            h_backproj = self.op_adjoint(h[:, 0:1, :, :])
            # 2. Update primal variable
            f = self.primal_blocks[i](f, h_backproj)
            
        # The first channel of f is conventionally the reconstructed image
        out = self.final_conv(f)
        return out
