"""Smoke tests — verify all model architectures + pipeline components."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_red_cnn_shape():
    from models.red_cnn import REDCNN
    model = REDCNN(n_channels=1, n_filters=64, n_layers=10)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  ✓ RED-CNN: {model.num_parameters:,} params, output {out.shape}")


def test_unet_shape():
    from models.unet import UNet
    model = UNet(n_channels=1, n_classes=1, base_ch=32)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  ✓ U-Net: {model.num_parameters:,} params, output {out.shape}")


def test_attention_unet_shape():
    from models.attention_unet import AttentionUNet
    model = AttentionUNet(n_channels=1, n_classes=1, base_ch=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  ✓ AttentionUNet: {model.num_parameters:,} params, output {out.shape}")


def test_freq_hybrid_shape():
    from models.freq_hybrid_net import FreqHybridNet
    n_angles, n_det, img_h, img_w = 64, 64, 64, 64
    model = FreqHybridNet(n_angles=n_angles, n_detectors=n_det, img_h=img_h, img_w=img_w)
    sino = torch.randn(2, 1, n_angles, n_det)
    recon, filtered = model(sino)
    assert recon.shape == (2, 1, img_h, img_w), f"Recon shape mismatch: {recon.shape}"
    assert filtered.shape == sino.shape, f"Filtered sino shape mismatch: {filtered.shape}"
    print(f"  ✓ FreqHybridNet: {model.num_parameters:,} params, recon {recon.shape}")


def test_losses():
    from pipeline.losses import REDCNNLoss, UNetLoss, AttentionUNetLoss, FreqHybridLoss
    pred = torch.rand(2, 1, 32, 32)
    tgt = torch.rand(2, 1, 32, 32)
    sino = torch.rand(2, 1, 32, 32)

    for name, fn, kwargs in [
        ("RED-CNN", REDCNNLoss(), {}),
        ("U-Net", UNetLoss(), {}),
        ("AttUNet", AttentionUNetLoss(), {}),
        ("FreqHybrid", FreqHybridLoss(), {"filtered_sino": sino, "orig_sino": sino}),
    ]:
        loss_dict = fn(pred, tgt, **kwargs)
        assert "total" in loss_dict
        assert loss_dict["total"].item() >= 0
        print(f"  ✓ {name} loss: {loss_dict['total'].item():.6f}")


def test_dataset_smoke():
    """Quick sanity check on dataset loading (uses real organamnist.npz)."""
    npz = Path(__file__).resolve().parent.parent.parent / "real_data/organamnist/raw/organamnist.npz"
    if not npz.exists():
        print(f"  SKIP dataset test (file not found: {npz})")
        return

    from pipeline.dataset import CTReconDataset
    ds = CTReconDataset(npz, "test", target_size=64, n_angles=45, max_samples=4, seed=42)
    sample = ds[0]
    assert "target" in sample
    assert "fbp" in sample
    assert "sinogram" in sample
    assert sample["target"].shape == (1, 64, 64)
    assert sample["fbp"].shape == (1, 64, 64)
    assert sample["sinogram"].shape[0] == 1
    print(f"  ✓ Dataset: {len(ds)} samples, "
          f"sino {tuple(sample['sinogram'].shape)}, "
          f"fbp [{sample['fbp'].min():.3f}, {sample['fbp'].max():.3f}]")


if __name__ == "__main__":
    print("\nRunning smoke tests...\n")
    test_red_cnn_shape()
    test_unet_shape()
    test_attention_unet_shape()
    test_freq_hybrid_shape()
    test_losses()
    test_dataset_smoke()
    print("\n✓ All smoke tests passed!\n")
