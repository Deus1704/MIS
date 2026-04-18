# Compute Requirements & Running Info

## Branch A: Learned Primal-Dual (LPD) on AAPM 512x512 Data

### GPU Requirements
- **T4 (16GB VRAM):** INSUFFICIENT. Will hit `CUDA OutOfMemoryError` due to the unrolled network architecture keeping 10 iterations of intermediate feature maps in memory at 512x512. Dropping batch size to 1 will still likely OOM.
- **L40S (48GB) / A100 (40GB/80GB):** MINIMUM RECOMMENDED. Provides enough VRAM for backpropagation through the spatial and frequency CNN blocks with a batch size of 2-4.
- **H100 (80GB):** IDEAL. Can handle batch sizes of up to 8.

### Time Expectations (For 1.8GB Phantom Object / ~15,000 Train Slices)
- **Dataset Caching:** It will take a few minutes initially to compute `skimage.radon` transforms on 15,000 slices before Epoch 1. Ensure your instance has robust system RAM.
- **Epoch Iteration:** Because the dataset increased from 2,168 slices to 15,000 slices, you only need to run **~10-15 Epochs** to reach structural convergence, down from 50. 
- **L40S / A100:** Expect ~4 to 6 hours for full training with batch sizes of 2-4.
- **H100:** Expect ~2.5 hours for full training.

### Fallback Instructions
If you are forced to use a GPU with less than 24GB VRAM, you must lower `num_iterations` in `train_aapm_lpd.py` from 10 down to ~3, though this will significantly erode the theoretical reconstruction performance.
