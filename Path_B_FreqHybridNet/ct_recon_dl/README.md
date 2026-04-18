# CT Reconstruction: End-to-End Deep Learning Pipeline

A rigorous, fully-measured end-to-end pipeline for CT image reconstruction using deep learning.
**No dummy data. No fraud. Everything measured.**

## What This Does

Reconstructs CT images from **real, physics-accurate sinogram data** using multiple DL methods
and compares them rigorously against the classical Filtered Back Projection (FBP) baseline.

### Data: Real CT Scans
- **OrganAMNIST**: Real abdominal CT slices from the LiTS (Liver Tumor Segmentation) challenge
- **Physics-accurate sinograms**: Radon transform of each real slice
- **Realistic noise**: Poisson + Gaussian model simulating quarter-dose (25%) CT acquisition

### Methods Compared
| Method | Type | Key Feature |
|--------|------|-------------|
| FBP | Classical baseline | Ramp-filtered back projection |
| RED-CNN | Image-domain DL | Residual encoder-decoder, global skip |
| U-Net | Image-domain DL | 4-level skip connections, batch norm |
| AttentionUNet | Image-domain DL | Channel (SE) + spatial attention gates |
| **FreqHybridNet** | **Sinogram-domain DL (Novel)** | Learns frequency weighting in sinogram space + spatial fusion |

### FreqHybridNet (Proposed Novel Method)
Classical FBP applies a **fixed ramp filter** in Fourier space before back-projection.
FreqHybridNet learns this filter from data:
- Operates on **raw sinograms** (not FBP output) 
- **Frequency branch**: 2D FFT → learned filter → iFFT (generalizes ramp filter)
- **Spatial branch**: Learnable back-projection approximation
- **Fusion decoder**: Combines both representations → reconstructed image
- Trained with sinogram consistency loss to ensure physics-faithful reconstruction

### Everything Measured
- Per-slice: SSIM, PSNR, RMSE, inference time (ms)
- Statistical: Wilcoxon signed-rank test, paired permutation test, bootstrap 95% CI, rank-biserial effect size
- Frequency domain: Radial power spectrum, FBP distortion analysis
- Training: Loss curves, LR schedule, per-epoch SSIM/PSNR/RMSE
- Model: Parameter counts, model size (MB), throughput (slices/sec)

## Structure

```
ct_recon_dl/
├── ct_recon/            # Evaluation library (metrics, stats, ROI)
├── models/
│   ├── red_cnn.py       # RED-CNN
│   ├── unet.py          # U-Net
│   ├── attention_unet.py # Attention U-Net (SE + spatial gates)
│   └── freq_hybrid_net.py # FreqHybridNet (NOVEL)
├── pipeline/
│   ├── dataset.py       # CTReconDataset with physics sinogram simulation
│   ├── losses.py        # SSIM, gradient, sinogram-consistency losses
│   ├── train.py         # Training loop with full metric tracking
│   └── infer.py         # Inference + statistical comparison
├── scripts/
│   └── run_dl_pipeline.py # Main runner
├── tests/
│   └── test_smoke.py    # Architecture + pipeline smoke tests
└── results/             # All outputs (CSVs, PNGs, checkpoints)
```

## Quick Start

```bash
cd ct_recon_dl

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke tests
python tests/test_smoke.py

# 3. Quick-test mode (2 epochs, small subset)
python scripts/run_dl_pipeline.py \
    --data ../real_data/organamnist/raw/organamnist.npz \
    --image-size 64 \
    --epochs 2 \
    --max-train 200 \
    --max-test 50

# 4. Full training (all methods, 30 epochs)
python scripts/run_dl_pipeline.py \
    --data ../real_data/organamnist/raw/organamnist.npz \
    --image-size 128 \
    --n-angles 90 \
    --dose-fraction 0.25 \
    --epochs 30 \
    --batch-size 8 \
    --methods red_cnn unet attention_unet freq_hybrid
```

## Outputs (in `results/`)
- `training_curves.csv` + `training_curves.png` — per-epoch metrics
- `inference_metrics.csv` — per-slice SSIM/PSNR/RMSE/time for all methods
- `method_summary.csv` — aggregate metrics + statistical tests
- `ct_panel.png` — side-by-side CT slice comparison
- `metrics_comparison.png` — bar charts SSIM/PSNR/RMSE
- `speed_quality_plot.png` — inference time vs SSIM tradeoff
- `radar_chart.png` — multi-metric radar
- `freq_spectrum.png` — frequency domain analysis
- `checkpoints/` — best model weights per method

## References
1. Chen et al., "Low-dose CT with RED-CNN." IEEE TMI, 2017.
2. Ronneberger et al., "U-Net." MICCAI, 2015.
3. Oktay et al., "Attention U-Net." 2018.
4. Hu et al., "Squeeze-and-Excitation Networks." CVPR, 2018.
5. MedMNIST v2: Yang et al., Scientific Data, 2023.
