from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.transform import iradon, radon


def default_theta(width: int) -> np.ndarray:
    """Default projection angles for parallel-beam simulation."""
    return np.linspace(0.0, 180.0, max(width, 180), endpoint=False)


def simulate_sinogram(image_2d: np.ndarray, theta: Sequence[float] | None = None) -> np.ndarray:
    """Simulate parallel-beam sinogram from a 2D slice."""
    if image_2d.ndim != 2:
        raise ValueError("simulate_sinogram expects a 2D array")
    thetas = np.asarray(theta) if theta is not None else default_theta(image_2d.shape[1])
    return radon(image_2d.astype(np.float32), theta=thetas, circle=False)


def fbp_reconstruct(
    sinogram: np.ndarray,
    theta: Sequence[float] | None = None,
    filter_name: str = "ramp",
) -> np.ndarray:
    """Reconstruct image from sinogram using Filtered Back Projection."""
    if sinogram.ndim != 2:
        raise ValueError("fbp_reconstruct expects a 2D sinogram")
    thetas = np.asarray(theta) if theta is not None else default_theta(sinogram.shape[1])
    return iradon(sinogram.astype(np.float32), theta=thetas, filter_name=filter_name, circle=False)
