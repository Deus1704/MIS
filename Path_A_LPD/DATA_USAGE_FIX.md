# Fix: Using Real AAPM Sinogram Data (No More skimage.radon Simulation)

## Problem Identified

The original code was using `skimage.radon` to **simulate** sinograms from images, which is incorrect when **actual measured sinogram data** is available.

### Original (Wrong) Approach:
```python
from skimage.transform import radon
# Load image -> compute sinogram using Radon transform
image = np.load('slice_000.npy')  # 512x512 image
sinogram = radon(image, theta=theta)  # FAKE sinogram!
```

### Fixed Approach:
```python
# Load ACTUAL measured sinogram data
sinogram = np.load('Phantom_Train/00000001.npy')  # Real 736x64 sinogram
```

## Data Structure

The AAPM dataset in `/teamspace/studios/this_studio/real_data/aapm_ldct/` contains:

### Train Split:
- `Phantom_Train/*.npy`: **18,032 raw sinogram files** (736 detectors × 64 angles)
- `L001/`, `L002/`, `L003/`: Reconstructed images (512×512) - limited samples

### Eval Split:
- `Phantom_Eval/*.npy`: Raw sinogram files (if available)
- `L010/`: Reconstructed images for evaluation

### Sinogram Properties:
- **Shape**: (736, 64) = (detectors, angles)
- **Values**: Attenuation measurements (range: ~-0.1 to ~5.0)
- **Format**: Float64, requires normalization

## Changes Made

### 1. `dataset_aapm.py` - Complete Rewrite
- **Removed**: `simulate_aapm_sinogram()` function that used skimage.radon
- **Added**: `AAPMSinogramDataset` class that loads real sinograms
- **Key Features**:
  - Loads actual sinogram data from `Phantom_Train/` and `Phantom_Eval/`
  - Computes FBP reconstruction as the target (standard when ground truth unavailable)
  - Proper normalization of sinogram values
  - Handles the correct geometry: 736 detectors × 64 angles

### 2. `train_aapm_lpd.py` - Updated
- Uses `AAPMSinogramDataset` instead of old dataset
- Automatically detects sinogram geometry from data
- Removed hardcoded angle/detector counts

### 3. `eval_aapm_metrics.py` - Updated
- Uses real sinogram data for evaluation
- Proper metric comparison between LPD and FBP

### 4. `visualize_phantom.py` - New Script
- Visualizes actual sinogram data
- Shows FBP reconstruction from real sinograms
- Compares with any available reconstructed images

## Key Insight: Why This Matters

### Before (Simulation):
```
Image (512×512) → skimage.radon → Fake Sinogram → Model → Reconstruction
                     ↑
              This assumes perfect physics
              and introduces artifacts
```

### After (Real Data):
```
Real Sinogram (736×64) → Model → Reconstruction
         ↑
   Actual measured data
   from CT scanner
```

## Data Characteristics

### Sinogram Statistics (from actual data):
- **Detectors**: 736
- **Angles**: 64 (not 180 as previously assumed!)
- **Value Range**: -0.1 to 5.0 (approximately)
- **Mean**: ~1.35
- **Std**: ~1.67

### Important Notes:
1. **64 angles only** - This is sparse-view CT, which is actually harder than full-view
2. **Negative values present** - Data is already preprocessed (likely attenuation coefficients)
3. **No paired ground truth** - We use FBP reconstruction as the target, which is standard practice

## Usage

```bash
# Visualize phantom data
python Path_A_LPD/visualize_phantom.py

# Train with real sinograms
python Path_A_LPD/train_aapm_lpd.py

# Evaluate metrics
python Path_A_LPD/eval_aapm_metrics.py
```

## Expected Results

With real sinogram data:
- **PSNR**: Should be in the range of 20-35 dB for good reconstructions
- **SSIM**: Should be > 0.7 for structurally similar reconstructions
- **FBP baseline**: Will be lower quality due to sparse angles (64 vs 180)

The previous metrics (PSNR ~10 dB, SSIM ~0.01) were incorrect because:
1. The model was trained on simulated data
2. Evaluation used mismatched sinogram-image pairs
3. The physics were not properly modeled

## Next Steps

1. ✅ Use real sinogram data (DONE)
2. ✅ Remove skimage.radon simulation (DONE)
3. ⏳ Train model on actual sinograms
4. ⏳ Validate metrics make physical sense
5. ⏳ Compare with proper baselines
