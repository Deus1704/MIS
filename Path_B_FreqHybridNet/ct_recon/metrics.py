from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _safe_data_range(target: np.ndarray, data_range: float | None) -> float:
    if data_range is not None:
        return float(data_range)
    dynamic = float(np.max(target) - np.min(target))
    return dynamic if dynamic > 0 else 1.0


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float | None = None,
) -> Dict[str, float]:
    """Compute SSIM, PSNR, RMSE for two same-shape arrays."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    pred_f = pred.astype(np.float32, copy=False)
    target_f = target.astype(np.float32, copy=False)
    dr = _safe_data_range(target_f, data_range)

    mse = float(np.mean((pred_f - target_f) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(peak_signal_noise_ratio(target_f, pred_f, data_range=dr))
    ssim = float(structural_similarity(target_f, pred_f, data_range=dr))

    return {"ssim": ssim, "psnr": psnr, "rmse": rmse}


def compute_masked_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    data_range: float | None = None,
) -> Dict[str, float]:
    """Compute metrics on masked pixels only."""
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("Pred, target, and mask must have same shape")

    valid = mask.astype(bool)
    if np.count_nonzero(valid) == 0:
        raise ValueError("Mask contains zero valid pixels")

    pred_vals = pred[valid]
    target_vals = target[valid]
    return compute_metrics(pred_vals, target_vals, data_range=data_range)
