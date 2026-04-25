# Path C Objective Report

## Objective
Apply deep learning post-processing on top of Filtered Back Projection (FBP) to improve reconstruction quality under low-dose CT conditions.

## Evidence Stored In This Run
- Training/validation metrics per epoch (`training_curves.csv`).
- Per-slice test metrics (`inference_metrics.csv`).
- Method-level aggregate metrics and statistical tests (`method_summary.csv`).
- Visual comparisons and diagnostic plots (`visualizations/`).

## Test Summary (Mean Metrics)

```text
 method  mean_ssim  mean_psnr  mean_rmse  mean_inference_ms
    FBP   0.863145  23.991774   0.066847           0.002232
RED-CNN   0.851140  23.325912   0.071963          23.917207
```

## Training Coverage
- Models trained: RED-CNN
- Total epoch records: 1
