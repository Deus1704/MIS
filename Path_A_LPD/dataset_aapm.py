import os
from pathlib import Path

import numpy as np
import torch
from skimage.transform import iradon
from torch.utils.data import Dataset


def generate_parallel_geometry(n_angles):
    """Generate parallel beam geometry angles."""
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


class AAPMSinogramDataset(Dataset):
    """
    Dataset using ACTUAL measured sinograms from AAPM Mayo Clinic data.

    The sinogram path and the target image path are intentionally separate.
    If paired target images are unavailable, the loader fails fast unless an
    explicit FBP fallback is requested for debugging.

    Data structure:
    - real_data/aapm_ldct/train/Phantom_Train/*.npy: Raw sinograms (736x64)
    - target_root_dir/*.npy: Paired ground-truth phantom images (optional)
    """

    def __init__(
        self,
        root_dir,
        target_root_dir=None,
        split="train",
        image_size=512,
        max_samples=None,
        cache_to_ram=False,
        eval_split=0.1,
        allow_fbp_target=False,
    ):
        """
        Args:
            root_dir: Base directory (e.g., '../../real_data/aapm_ldct')
            split: 'train' or 'eval'
            image_size: Target reconstruction size (default 512)
            max_samples: Maximum samples to use
            cache_to_ram: Cache data in RAM
            eval_split: Fraction of data to use for eval (default 10%)
        """
        super().__init__()

        self.root_dir = root_dir
        self.target_root_dir = target_root_dir
        self.split = split
        self.image_size = image_size
        self.eval_split = eval_split
        self.allow_fbp_target = allow_fbp_target

        # Actual AAPM sinogram geometry
        self.num_detectors = 736
        self.num_angles = 64

        # Load ALL sinogram files from Phantom_Train
        sinogram_dir = os.path.join(root_dir, "train", "Phantom_Train")

        self.all_sinogram_files = []
        if os.path.exists(sinogram_dir):
            self.all_sinogram_files = sorted(
                [
                    os.path.join(sinogram_dir, f)
                    for f in os.listdir(sinogram_dir)
                    if f.endswith(".npy")
                ]
            )

        print(
            f"AAPM Dataset: Found {len(self.all_sinogram_files)} total sinogram files"
        )

        # Split into train/eval
        n_total = len(self.all_sinogram_files)
        n_eval = int(n_total * eval_split)
        n_train = n_total - n_eval

        if split == "train":
            self.sinogram_files = self.all_sinogram_files[:n_train]
            print(f"  - Train split: {len(self.sinogram_files)} samples")
        else:  # eval
            self.sinogram_files = self.all_sinogram_files[n_train:]
            print(f"  - Eval split: {len(self.sinogram_files)} samples")

        if max_samples:
            self.sinogram_files = self.sinogram_files[:max_samples]
            print(f"  - Limited to {max_samples} samples")

        self.target_files = {}
        self._target_dir_resolved = None
        if self.target_root_dir:
            self._index_target_files(self.target_root_dir)
            print(
                f"  - Paired targets: {len(self.target_files)} files indexed from {self._target_dir_resolved}"
            )

        self.cache_to_ram = cache_to_ram
        self.ram_cache = {}

        if self.cache_to_ram and len(self.sinogram_files) > 0:
            print(f"  - Caching to RAM...")
            for i in range(min(len(self.sinogram_files), 50)):
                self.ram_cache[i] = self._load_sample(i)

    def _index_target_files(self, target_root_dir):
        target_dir = Path(target_root_dir)
        if not target_dir.exists():
            raise FileNotFoundError(
                f"Paired target directory not found: {target_dir}"
            )

        indexed = {}
        for file_path in sorted(target_dir.rglob("*.npy")):
            key = file_path.name
            if key in indexed:
                raise RuntimeError(
                    f"Duplicate target filename {key} found under {target_dir}. "
                    "Use unique names or a mirrored directory structure."
                )
            indexed[key] = str(file_path)

        if not indexed:
            raise RuntimeError(f"No .npy target files found under {target_dir}")

        self.target_files = indexed
        self._target_dir_resolved = str(target_dir)

    def _resolve_target_path(self, sinogram_path):
        if not self.target_files:
            return None

        target_path = self.target_files.get(os.path.basename(sinogram_path))
        if target_path is None:
            raise FileNotFoundError(
                f"No paired target found for {os.path.basename(sinogram_path)} in {self._target_dir_resolved}"
            )
        return target_path

    def _load_sample(self, idx):
        """Load sinogram, paired target image, and FBP reconstruction."""
        # Load actual sinogram
        sinogram_path = self.sinogram_files[idx]
        sinogram = np.load(sinogram_path).astype(np.float32)

        # Sinogram shape: (736, 64) = (detectors, angles)
        sinogram_for_fbp = sinogram.T  # (angles, detectors)

        # Normalize sinogram to [0, 1]
        smin, smax = sinogram.min(), sinogram.max()
        sinogram_norm = (sinogram - smin) / (smax - smin + 1e-8)

        # Compute FBP reconstruction as target
        theta = generate_parallel_geometry(sinogram_for_fbp.shape[0])
        fbp = iradon(
            sinogram_for_fbp.T, theta=theta, filter_name="ramp", circle=False
        ).astype(np.float32)

        # Normalize FBP to [0, 1]
        fbp = (fbp - fbp.min()) / (fbp.max() - fbp.min() + 1e-8)

        # Resize to target image_size if needed
        if fbp.shape != (self.image_size, self.image_size):
            from skimage.transform import resize

            fbp = resize(
                fbp, (self.image_size, self.image_size), anti_aliasing=True
            ).astype(np.float32)
            sinogram_norm = resize(
                sinogram_norm, (self.num_angles, self.num_detectors), anti_aliasing=True
            ).astype(np.float32)

        target = None
        target_path = self._resolve_target_path(sinogram_path)
        if target_path is not None:
            target = np.load(target_path).astype(np.float32)
            if target.shape != (self.image_size, self.image_size):
                from skimage.transform import resize

                target = resize(
                    target, (self.image_size, self.image_size), anti_aliasing=True
                ).astype(np.float32)

            tmin, tmax = target.min(), target.max()
            target = (target - tmin) / (tmax - tmin + 1e-8)
        elif self.allow_fbp_target:
            target = fbp.copy()
        else:
            raise RuntimeError(
                "No paired ground-truth target image was found for this sample. "
                "Provide target_root_dir with matched .npy images or set allow_fbp_target=True "
                "only for synthetic/debug runs."
            )

        return {
            "target": torch.from_numpy(target).unsqueeze(0),
            "sinogram": torch.from_numpy(sinogram_norm).unsqueeze(0),
            "fbp": torch.from_numpy(fbp).unsqueeze(0),
        }

    def __len__(self):
        return len(self.sinogram_files)

    def __getitem__(self, idx):
        if self.cache_to_ram and idx in self.ram_cache:
            return self.ram_cache[idx]
        return self._load_sample(idx)
