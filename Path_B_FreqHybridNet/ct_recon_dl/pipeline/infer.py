"""Inference + comprehensive measurement module.

Measures:
- Per-slice inference time (ms)
- Throughput (slices/second)
- SSIM, PSNR, RMSE per slice and aggregate
- FBP baseline metrics
- Statistical comparison (Wilcoxon, permutation, bootstrap CI)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ct_recon.stats import bootstrap_ci, paired_permutation_test, paired_wilcoxon, rank_biserial_effect_size


@dataclass
class SliceMetrics:
    ssim: float
    psnr: float
    rmse: float
    inference_time_ms: float


@dataclass
class MethodResult:
    method_name: str
    slice_metrics: list[SliceMetrics]
    mean_ssim: float
    mean_psnr: float
    mean_rmse: float
    mean_inference_ms: float
    throughput_slices_per_sec: float
    n_parameters: int
    model_size_mb: float
    # Statistical comparison vs FBP (filled in after all methods run)
    ssim_vs_fbp_wilcoxon_p: float = float("nan")
    ssim_vs_fbp_perm_p: float = float("nan")
    ssim_vs_fbp_effect_r: float = float("nan")
    ssim_vs_fbp_ci95_low: float = float("nan")
    ssim_vs_fbp_ci95_high: float = float("nan")


def _compute_slice_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Compute SSIM, PSNR, RMSE for numpy 2D arrays."""
    pred = np.clip(pred.astype(np.float32), 0, 1)
    target = np.clip(target.astype(np.float32), 0, 1)
    dr = float(target.max() - target.min())
    if dr < 1e-8:
        dr = 1.0
    ssim = float(structural_similarity(target, pred, data_range=dr))
    psnr = float(peak_signal_noise_ratio(target, pred, data_range=dr))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return ssim, psnr, rmse


def run_fbp_inference(
    test_loader: DataLoader,
    device: torch.device,
) -> MethodResult:
    """Measure FBP reconstruction metrics (no model — just noisy_fbp vs target)."""
    slice_metrics: list[SliceMetrics] = []

    for batch in test_loader:
        fbp_batch = batch["fbp"].numpy()
        target_batch = batch["target"].numpy()

        for i in range(fbp_batch.shape[0]):
            t0 = time.perf_counter()
            fbp = fbp_batch[i, 0]  # [H, W]
            inf_ms = (time.perf_counter() - t0) * 1000

            target = target_batch[i, 0]
            ssim, psnr, rmse = _compute_slice_metrics(fbp, target)
            slice_metrics.append(SliceMetrics(ssim, psnr, rmse, inf_ms))

    ssims = [m.ssim for m in slice_metrics]
    psnrs = [m.psnr for m in slice_metrics]
    rmses = [m.rmse for m in slice_metrics]
    times = [m.inference_time_ms for m in slice_metrics]
    n = len(slice_metrics)
    throughput = n / (sum(times) / 1000.0) if sum(times) > 0 else float("inf")

    return MethodResult(
        method_name="FBP",
        slice_metrics=slice_metrics,
        mean_ssim=float(np.mean(ssims)),
        mean_psnr=float(np.mean(psnrs)),
        mean_rmse=float(np.mean(rmses)),
        mean_inference_ms=float(np.mean(times)),
        throughput_slices_per_sec=throughput,
        n_parameters=0,
        model_size_mb=0.0,
    )


def run_model_inference(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    device: torch.device,
    is_freq_hybrid: bool = False,
    n_warmup: int = 3,
) -> MethodResult:
    """Run model inference on test set and measure all metrics.

    Args:
        model: trained PyTorch model
        model_name: string identifier
        test_loader: test DataLoader
        device: cpu or cuda
        is_freq_hybrid: if True, model takes sinogram input
        n_warmup: number of warmup forward passes before timing
    """
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_warmup:
                break
            inp = batch["sinogram" if is_freq_hybrid else "fbp"].to(device)
            _ = model(inp)

    slice_metrics: list[SliceMetrics] = []

    # Measure model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = param_bytes / (1024 ** 2)

    with torch.no_grad():
        for batch in test_loader:
            fbp_batch = batch["fbp"]
            sino_batch = batch["sinogram"]
            target_batch = batch["target"]

            # Time per-slice (process one at a time for accurate timing)
            for i in range(fbp_batch.shape[0]):
                inp = (sino_batch[i:i+1] if is_freq_hybrid else fbp_batch[i:i+1]).to(device)
                target = target_batch[i, 0].numpy()

                t0 = time.perf_counter()
                if is_freq_hybrid:
                    recon, _ = model(inp)
                else:
                    recon = model(inp)
                # Synchronize if GPU
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inf_ms = (time.perf_counter() - t0) * 1000

                pred = recon[0, 0].cpu().numpy()
                ssim, psnr, rmse = _compute_slice_metrics(pred, target)
                slice_metrics.append(SliceMetrics(ssim, psnr, rmse, inf_ms))

    ssims = [m.ssim for m in slice_metrics]
    psnrs = [m.psnr for m in slice_metrics]
    rmses = [m.rmse for m in slice_metrics]
    times = [m.inference_time_ms for m in slice_metrics]
    n = len(slice_metrics)
    total_time_s = sum(times) / 1000.0
    throughput = n / total_time_s if total_time_s > 0 else float("inf")

    return MethodResult(
        method_name=model_name,
        slice_metrics=slice_metrics,
        mean_ssim=float(np.mean(ssims)),
        mean_psnr=float(np.mean(psnrs)),
        mean_rmse=float(np.mean(rmses)),
        mean_inference_ms=float(np.mean(times)),
        throughput_slices_per_sec=throughput,
        n_parameters=n_params,
        model_size_mb=model_size_mb,
    )


def run_statistical_comparison(
    fbp_result: MethodResult,
    dl_results: list[MethodResult],
    n_bootstrap: int = 2000,
    n_permutations: int = 5000,
    seed: int = 42,
) -> list[MethodResult]:
    """Compute Wilcoxon, permutation test, bootstrap CI for each DL method vs FBP.

    Modifies results in-place and returns them.
    """
    fbp_ssims = np.array([m.ssim for m in fbp_result.slice_metrics], dtype=np.float64)

    for result in dl_results:
        dl_ssims = np.array([m.ssim for m in result.slice_metrics], dtype=np.float64)

        # Align lengths
        min_n = min(len(fbp_ssims), len(dl_ssims))
        fbp_aligned = fbp_ssims[:min_n]
        dl_aligned = dl_ssims[:min_n]

        if min_n < 2:
            continue

        diff = dl_aligned - fbp_aligned
        wil = paired_wilcoxon(dl_aligned, fbp_aligned)
        perm = paired_permutation_test(dl_aligned, fbp_aligned, n_permutations=n_permutations, random_state=seed)
        eff = rank_biserial_effect_size(dl_aligned, fbp_aligned)
        ci_low, ci_high = bootstrap_ci(diff, n_bootstrap=n_bootstrap, random_state=seed)

        result.ssim_vs_fbp_wilcoxon_p = wil["p_value"]
        result.ssim_vs_fbp_perm_p = perm["p_value"]
        result.ssim_vs_fbp_effect_r = eff
        result.ssim_vs_fbp_ci95_low = ci_low
        result.ssim_vs_fbp_ci95_high = ci_high

    return dl_results
