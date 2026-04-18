# Compute Requirements & Running Info

## Branch A: Learned Primal-Dual (LPD) on AAPM 512x512 Data

### GPU Requirements
- **T4 (16GB VRAM):** INSUFFICIENT. Will hit `CUDA OutOfMemoryError` due to the unrolled network architecture keeping 10 iterations of intermediate feature maps in memory at 512x512. Dropping batch size to 1 will still likely OOM.
- **L40S (48GB) / A100 (40GB/80GB):** MINIMUM RECOMMENDED. Provides enough VRAM for backpropagation through the spatial and frequency CNN blocks with a batch size of 2-4.
- **H100 (80GB):** IDEAL. Can handle batch sizes of up to 8.

### Time Expectations (For ~2,168 slices / 50 Epochs)
- **Dataset Caching:** 2-5 minutes initially to compute `skimage.radon` transforms before Epoch 1. Ensure your instance has >64GB system RAM to hold the dataset.
- **L40S / A100:** 2.5 to 3.5 hours for full training.
- **H100:** <1.5 hours for full training.

### Fallback Instructions
If you are forced to use a GPU with less than 24GB VRAM, you must lower `num_iterations` in `train_aapm_lpd.py` from 10 down to ~3, though this will significantly erode the theoretical reconstruction performance.
