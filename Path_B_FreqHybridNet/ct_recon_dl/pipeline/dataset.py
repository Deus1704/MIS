"""CT Reconstruction Dataset.

Loads real CT slices (OrganAMNIST), simulates physics-accurate sinograms via
Radon transform, adds realistic noise, and provides paired (noisy_fbp, target)
and (noisy_sinogram, target) samples for training all reconstruction methods.

All data is REAL (OrganAMNIST = real CT slices from LiTS), all sinograms
are PHYSICS-ACCURATE via Radon transform. NO dummy data.

Performance: All sinograms + FBP images are pre-computed once at dataset
construction time and cached in RAM. This avoids expensive Radon transforms
inside the training loop.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import iradon, radon
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data statistics
# ---------------------------------------------------------------------------

class DatasetStats(NamedTuple):
    n_train: int
    n_val: int
    n_test: int
    image_shape: tuple[int, int]
    hu_min: float
    hu_max: float
    hu_mean: float
    hu_std: float
    load_time_ms: float


# ---------------------------------------------------------------------------
# Physics simulation helpers
# ---------------------------------------------------------------------------

def make_theta(n_angles: int) -> np.ndarray:
    """Projection angles for parallel-beam CT."""
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] using robust percentiles."""
    img = img.astype(np.float32)
    lo = float(np.percentile(img, 1))
    hi = float(np.percentile(img, 99))
    if np.isclose(lo, hi):
        lo, hi = float(np.min(img)), float(np.max(img))
    denom = max(hi - lo, 1e-8)
    return np.clip((img - lo) / denom, 0.0, 1.0)


def simulate_sinogram(
    image: np.ndarray,
    theta: np.ndarray,
    dose_fraction: float = 0.25,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Physics-accurate sinogram simulation with noise.

    Args:
        image: 2D CT slice [H, W] normalized to [0, 1]
        theta: projection angles
        dose_fraction: fraction of full dose (0.25 = quarter dose)
        noise_std: base Gaussian noise std
        rng: random generator

    Returns:
        noisy_sinogram: [n_angles, n_detectors]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Radon transform: skimage returns [detectors × angles], we want [angles × detectors]
    sino = radon(image.astype(np.float32), theta=theta, circle=False).T  # [angles, detectors]

    # Scale noise by dose: lower dose = more noise
    effective_noise = noise_std / max(dose_fraction ** 0.5, 1e-4)
    noise = rng.normal(0.0, effective_noise, size=sino.shape).astype(np.float32)
    return sino + noise


def fbp_from_sinogram(
    sinogram: np.ndarray,
    theta: np.ndarray,
    filter_name: str = "ramp",
    target_size: int | None = None,
) -> np.ndarray:
    """Filtered back projection from a [n_angles, n_detectors] sinogram.

    Returns normalized reconstruction resized to target_size × target_size.
    """
    recon = iradon(sinogram.T, theta=theta, filter_name=filter_name, circle=False)
    recon = normalize_image(recon)
    if target_size is not None and recon.shape[0] != target_size:
        t = torch.from_numpy(recon).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target_size, target_size),
                          mode="bilinear", align_corners=True)
        recon = t.squeeze().numpy()
    return recon


# ---------------------------------------------------------------------------
# Dataset — pre-caches all sinograms + FBP in RAM for fast training
# ---------------------------------------------------------------------------

class CTReconDataset(Dataset):
    """CT Reconstruction dataset from OrganAMNIST real CT slices.

    All sinograms and FBP reconstructions are pre-computed at construction
    and stored in RAM. This is critical for training speed: Radon is expensive.

    At 64×64, 60 angles, 2000 samples:
      RAM usage: ~130 MB (sinos ~43 MB + fbps ~31 MB + targets ~31 MB)
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: Literal["train", "val", "test"],
        target_size: int = 64,
        n_angles: int = 60,
        dose_fraction: float = 0.25,
        noise_std: float = 0.02,
        fbp_filter: str = "ramp",
        max_samples: int | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.target_size = target_size
        self.n_angles = n_angles
        self.dose_fraction = dose_fraction
        self.noise_std = noise_std
        self.fbp_filter = fbp_filter
        self.theta = make_theta(n_angles)

        # Load raw images from .npz
        t0 = time.perf_counter()
        data = np.load(npz_path)
        split_key = {"train": "train_images", "val": "val_images", "test": "test_images"}[split]
        images_raw = data[split_key]  # [N, H, W] uint8 or float

        if max_samples is not None:
            images_raw = images_raw[:max_samples]

        n = len(images_raw)
        self.load_time_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"  Pre-computing sinograms + FBP for {n} {split} slices "
                  f"({target_size}×{target_size}, {n_angles} angles)...")

        rng = np.random.default_rng(seed)

        # Determine sinogram shape from first sample
        img0 = normalize_image(images_raw[0].astype(np.float32))
        if img0.shape[0] != target_size:
            t_t = torch.from_numpy(img0).float().unsqueeze(0).unsqueeze(0)
            t_t = F.interpolate(t_t, size=(target_size, target_size),
                                mode="bilinear", align_corners=True)
            img0 = t_t.squeeze().numpy()
        sino0 = simulate_sinogram(img0, self.theta, dose_fraction, noise_std, rng)
        sino_h, sino_w = sino0.shape

        # Pre-allocate all arrays
        self._targets = np.zeros((n, target_size, target_size), dtype=np.float32)
        self._fbps = np.zeros((n, target_size, target_size), dtype=np.float32)
        self._sinos = np.zeros((n, sino_h, sino_w), dtype=np.float32)
        self._sino_mins = np.zeros(n, dtype=np.float32)
        self._sino_maxs = np.ones(n, dtype=np.float32)

        # Reset RNG for deterministic processing
        rng = np.random.default_rng(seed)
        t_proc = time.perf_counter()

        for i, img in enumerate(images_raw):
            target = normalize_image(img.astype(np.float32))
            if target.shape[0] != target_size:
                t_t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
                t_t = F.interpolate(t_t, size=(target_size, target_size),
                                    mode="bilinear", align_corners=True)
                target = t_t.squeeze().numpy()

            noisy_sino = simulate_sinogram(target, self.theta, dose_fraction, noise_std, rng)
            fbp = fbp_from_sinogram(noisy_sino, self.theta, fbp_filter, target_size=target_size)

            smin = float(noisy_sino.min())
            smax = float(noisy_sino.max())
            denom = max(smax - smin, 1e-8)

            self._targets[i] = target
            self._fbps[i] = fbp
            self._sinos[i] = (noisy_sino - smin) / denom
            self._sino_mins[i] = smin
            self._sino_maxs[i] = smax

            if verbose and n > 0 and (i + 1) % max(1, n // 5) == 0:
                elapsed = time.perf_counter() - t_proc
                rate = (i + 1) / max(elapsed, 1e-6)
                remaining = (n - i - 1) / max(rate, 1e-6)
                print(f"    {i+1}/{n} ({100*(i+1)/n:.0f}%) — "
                      f"{rate:.1f} imgs/s, ~{remaining:.0f}s remaining")

        proc_time = time.perf_counter() - t_proc
        if verbose:
            total_mb = (self._sinos.nbytes + self._fbps.nbytes + self._targets.nbytes) / 1e6
            print(f"  ✓ Pre-computation: {proc_time:.1f}s | RAM: {total_mb:.0f} MB")

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "target": torch.from_numpy(self._targets[idx]).unsqueeze(0),    # [1, H, W]
            "fbp": torch.from_numpy(self._fbps[idx]).unsqueeze(0),          # [1, H, W]
            "sinogram": torch.from_numpy(self._sinos[idx]).unsqueeze(0),    # [1, angles, det]
            "sino_min": torch.tensor(self._sino_mins[idx], dtype=torch.float32),
            "sino_max": torch.tensor(self._sino_maxs[idx], dtype=torch.float32),
        }


def compute_dataset_stats(dataset: CTReconDataset) -> DatasetStats:
    """Compute summary statistics over all dataset images."""
    all_pixels = dataset._targets.ravel()
    return DatasetStats(
        n_train=len(dataset),
        n_val=0,
        n_test=0,
        image_shape=(dataset.target_size, dataset.target_size),
        hu_min=float(all_pixels.min()),
        hu_max=float(all_pixels.max()),
        hu_mean=float(all_pixels.mean()),
        hu_std=float(all_pixels.std()),
        load_time_ms=dataset.load_time_ms,
    )
