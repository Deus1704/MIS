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
        U-Net   0.938528  24.428965   0.061682           6.100309
AttentionUNet   0.938215  24.515464   0.061339           7.649195
      RED-CNN   0.932827  24.017917   0.064772           6.113128
          FBP   0.851970  20.372506   0.098550           0.001528
   FreqHybrid   0.219755  12.326905   0.244075          30.090171
```

## Training Coverage
- Models trained: FreqHybrid
- Total epoch records: 3
