#!/usr/bin/env python3
"""
Visualize actual AAPM phantom sinogram data and reconstructions.
This script shows what the real data looks like without any simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon


def generate_parallel_geometry(n_angles):
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def visualize_phantom_data():
    root_dir = "/teamspace/studios/this_studio/real_data/aapm_ldct"

    # Load actual sinogram
    sinogram_path = os.path.join(root_dir, "train/Phantom_Train/00000001.npy")
    sinogram = np.load(sinogram_path)

    # Load sample image (if available)
    image_path = os.path.join(root_dir, "train/L001/slice_000.npy")
    image = np.load(image_path) if os.path.exists(image_path) else None

    print(f"Sinogram shape: {sinogram.shape}")
    print(f"  - Detectors: {sinogram.shape[0]}")
    print(f"  - Angles: {sinogram.shape[1]}")

    # Transpose sinogram to (angles, detectors)
    sinogram_t = sinogram.T

    # Compute FBP reconstruction
    theta = generate_parallel_geometry(sinogram_t.shape[0])
    fbp = iradon(sinogram_t.T, theta=theta, filter_name="ramp", circle=False)

    # Normalize for display
    sinogram_norm = (sinogram_t - sinogram_t.min()) / (
        sinogram_t.max() - sinogram_t.min()
    )
    fbp_norm = (fbp - fbp.min()) / (fbp.max() - fbp.min())

    if image is not None:
        image_norm = (image - image.min()) / (image.max() - image.min())

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Sinogram visualization
    im1 = axes[0, 0].imshow(sinogram, aspect="auto", cmap="gray")
    axes[0, 0].set_title(f"Raw Sinogram\nShape: {sinogram.shape}")
    axes[0, 0].set_xlabel("Angle index")
    axes[0, 0].set_ylabel("Detector index")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(sinogram_t, aspect="auto", cmap="hot")
    axes[0, 1].set_title(f"Sinogram (transposed)\nShape: {sinogram_t.shape}")
    axes[0, 1].set_xlabel("Detector index")
    axes[0, 1].set_ylabel("Angle index")
    plt.colorbar(im2, ax=axes[0, 1])

    # Show sinogram profile
    axes[0, 2].plot(sinogram_t[0, :])
    axes[0, 2].set_title(f"Sinogram profile (angle 0)")
    axes[0, 2].set_xlabel("Detector index")
    axes[0, 2].set_ylabel("Intensity")
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Reconstructions
    im3 = axes[1, 0].imshow(fbp_norm, cmap="gray")
    axes[1, 0].set_title(
        f"FBP Reconstruction\nShape: {fbp.shape}\nRange: [{fbp.min():.3f}, {fbp.max():.3f}]"
    )
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0])

    # Zoom into FBP center
    h, w = fbp_norm.shape
    crop_h, crop_w = h // 4, w // 4
    fbp_crop = fbp_norm[
        h // 2 - crop_h : h // 2 + crop_h, w // 2 - crop_w : w // 2 + crop_w
    ]
    im4 = axes[1, 1].imshow(fbp_crop, cmap="gray")
    axes[1, 1].set_title(
        f"FBP Center Crop\n[{h // 2 - crop_h}:{h // 2 + crop_h}, {w // 2 - crop_w}:{w // 2 + crop_w}]"
    )
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1])

    # Show image if available
    if image is not None:
        im5 = axes[1, 2].imshow(image_norm, cmap="gray")
        axes[1, 2].set_title(
            f"Reconstructed Image (L001)\nShape: {image.shape}\nRange: [{image.min():.3f}, {image.max():.3f}]"
        )
        axes[1, 2].axis("off")
        plt.colorbar(im5, ax=axes[1, 2])
    else:
        axes[1, 2].text(0.5, 0.5, "No image available", ha="center", va="center")
        axes[1, 2].axis("off")

    plt.tight_layout()

    # Save
    save_path = "/teamspace/studios/this_studio/MIS_Project/Path_A_LPD/phantom_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {save_path}")

    # Show statistics
    print(f"\n=== Sinogram Statistics ===")
    print(f"Min: {sinogram_t.min():.4f}")
    print(f"Max: {sinogram_t.max():.4f}")
    print(f"Mean: {sinogram_t.mean():.4f}")
    print(f"Std: {sinogram_t.std():.4f}")

    print(f"\n=== FBP Reconstruction Statistics ===")
    print(f"Min: {fbp.min():.4f}")
    print(f"Max: {fbp.max():.4f}")
    print(f"Mean: {fbp.mean():.4f}")
    print(f"Std: {fbp.std():.4f}")

    if image is not None:
        print(f"\n=== Image (L001) Statistics ===")
        print(f"Min: {image.min():.4f}")
        print(f"Max: {image.max():.4f}")
        print(f"Mean: {image.mean():.4f}")
        print(f"Std: {image.std():.4f}")


if __name__ == "__main__":
    visualize_phantom_data()
