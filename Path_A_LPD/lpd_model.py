import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardOperator(nn.Module):
    """
    Mock Forward Operator (Radon Transform).
    In a real implementation, this would use ODL or TorchRadon to project
    the 2D image domain into the sinogram (measurement) domain.
    """
    def __init__(self, num_angles, num_detectors):
        super().__init__()
        self.num_angles = num_angles
        self.num_detectors = num_detectors
        # A simple linear projection for mocking the physics
        # Real physics would be implemented via grid_sample or specialized CUDA kernels.
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        # Mock projection: flatten spatial dims and project to sinogram dims
        # Output shape should be (B, 1, num_angles, num_detectors)
        # This is purely structural to allow the network to run.
        device = x.device
        mock_sinogram = torch.mean(x, dim=(-1, -2), keepdim=True).expand(-1, -1, self.num_angles, self.num_detectors)
        return mock_sinogram

class AdjointOperator(nn.Module):
    """
    Mock Adjoint Operator (Filtered Back Projection or Back Projection).
    Projects from sinogram domain back to image domain.
    """
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        
    def forward(self, y):
        # y shape: (B, C, num_angles, num_detectors)
        B, C, N_A, N_D = y.shape
        # Mock backprojection: 
        # Output shape should be (B, 1, image_size, image_size)
        mock_image = torch.mean(y, dim=(-1, -2), keepdim=True).expand(-1, -1, self.image_size, self.image_size)
        return mock_image

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
        self.op_forward = ForwardOperator(num_angles, num_detectors)
        self.op_adjoint = AdjointOperator(image_size)
        
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
            # 1. Forward project the SECOND channel of primal variable (f[:, 1:2])
            # (In original LPD, f2 is used for projection, f1 is the actual image).
            # Here, we project f[:, 0:1] for simplicity.
            f_proj = self.op_forward(f[:, 0:1, :, :])
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
