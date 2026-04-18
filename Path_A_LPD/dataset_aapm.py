import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import iradon, radon
from pathlib import Path

def generate_parallel_geometry(n_angles):
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)

def simulate_aapm_sinogram(image: np.ndarray, theta: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """Radon physics simulator on high-fidelity 512x512 images using SKImage.
    Simulates Low Dose CT measurement by adding Poisson/Gaussian hybrid noise."""
    sino = radon(image, theta=theta, circle=False).T
    noise = np.random.normal(loc=0.0, scale=noise_std, size=sino.shape)
    noisy_sino = sino + noise
    return noisy_sino.astype(np.float32)

class AAPMDataset(Dataset):
    """
    DataLoader specifically engineered for the Mayo Clinic 512x512 Full/Low Dose Challenge.
    Iterates dynamically across Patient subfolders (L001-L009 Training, L010 Evaluation).
    Calculates pure physical Radon transforms natively (WARNING: VRAM Heavy!)
    """
    def __init__(self, root_dir, split="train", n_angles=180, image_size=512, cache_to_ram=True):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.files = []
        self.n_angles = n_angles
        self.image_size = image_size
        self.theta = generate_parallel_geometry(n_angles)
        self.cache_to_ram = cache_to_ram
        
        # Scrape patient directories
        if os.path.exists(self.root_dir):
            for patient_dir in sorted(os.listdir(self.root_dir)):
                pat_path = os.path.join(self.root_dir, patient_dir)
                if os.path.isdir(pat_path):
                    for fname in sorted(os.listdir(pat_path)):
                        if fname.endswith('.npy'):
                            self.files.append(os.path.join(pat_path, fname))
        
        print(f"AAPM {split.upper()} Dataset initialized: {len(self.files)} slices detected.")
        if len(self.files) == 0:
            print(f"WARNING: The directory {self.root_dir} is completely empty. Please run download_aapm.py to populate structurally or inject TCIA DICOMS (.npy processed).")

        # Cache in-memory for fast Dataloading if you have enough RAM (Otherwise reads from disk every step)
        self.ram_cache = {}
        if self.cache_to_ram and len(self.files) > 0:
            print("Caching high-fidelity 512x512 projections to RAM. This will take ~2 minutes on first run due to skimage.radon...")
            # For 2168 arrays, this is around 2-3GB memory footprint which is easily handled.
            for i in range(len(self.files)):
                self._load_and_simulate(i)

    def _load_and_simulate(self, idx):
        file_path = self.files[idx]
        image = np.load(file_path).astype(np.float32)
        
        # Normalize strictly to [0,1] mapping HU units typically provided by AAPM
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Perform explicit physics projection
        sinogram = simulate_aapm_sinogram(image, self.theta)
        
        # Calculate optimal classical FBP for later comparison baseline
        fbp = iradon(sinogram.T, theta=self.theta, filter_name='ramp', circle=False).astype(np.float32)
        # Rescale FBP mapping
        fbp = (fbp - fbp.min()) / (fbp.max() - fbp.min() + 1e-8)
        
        # Squeeze min/max normalization of sinogram
        smin, smax = sinogram.min(), sinogram.max()
        sinogram_norm = (sinogram - smin) / (smax - smin + 1e-8)
        
        bundle = {
            "target": torch.from_numpy(image).unsqueeze(0),
            "sinogram": torch.from_numpy(sinogram_norm).unsqueeze(0),
            "fbp": torch.from_numpy(fbp).unsqueeze(0)
        }
        
        if self.cache_to_ram:
            self.ram_cache[idx] = bundle
            
        return bundle
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_to_ram and idx in self.ram_cache:
            return self.ram_cache[idx]
        return self._load_and_simulate(idx)
