"""End-to-end CT reconstruction DL pipeline runner.

Executes the full measured pipeline:
  Stage 0: Load real OrganAMNIST CT data
  Stage 1: Simulate physics-accurate sinograms (Radon transform)
  Stage 2: Add realistic Poisson/Gaussian noise (quarter-dose CT)
  Stage 3: Train + evaluate 4 DL methods vs FBP baseline
  Stage 4: Statistical comparison (Wilcoxon, permutation, bootstrap CI)
  Stage 5: Generate comprehensive visualizations + CSV reports

All measurements saved to:
  results/
    training_curves.csv      - per-epoch train/val loss + SSIM/PSNR/RMSE per model
    inference_metrics.csv    - per-slice metrics per method
    method_summary.csv       - aggregate metrics + statistical tests
    ct_panel.png             - side-by-side CT slice comparison
    training_curves.png      - loss + SSIM training curves
    metrics_comparison.png   - bar charts of SSIM/PSNR/RMSE
    speed_quality_plot.png   - inference speed vs SSIM tradeoff
    radar_chart.png          - multi-metric radar
    freq_spectrum.png        - frequency domain analysis

NO DUMMY DATA. All CT slices are real (OrganAMNIST = LiTS liver CT scans).
All sinograms are real Radon transforms. All noise is simulated physics-accurate Poisson noise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset import CTReconDataset, compute_dataset_stats, make_theta, fbp_from_sinogram
from pipeline.losses import REDCNNLoss, UNetLoss, AttentionUNetLoss, FreqHybridLoss
from pipeline.train import train_model, TrainingResult
from pipeline.infer import run_fbp_inference, run_model_inference, run_statistical_comparison
from models.red_cnn import REDCNN
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.freq_hybrid_net import FreqHybridNet


# ---------------------------------------------------------------------------
# Visualization (inline to avoid circular imports)
# ---------------------------------------------------------------------------

def _plot_ct_panel(
    test_dataset: CTReconDataset,
    trained_models: dict,
    out_path: Path,
    n_cases: int = 6,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Plot CT slice panel: target | FBP | RED-CNN | U-Net | AttnUNet | FreqHybrid."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    method_names = ["Target", "FBP"] + list(trained_models.keys())
    n_methods = len(method_names)
    n_cases = min(n_cases, len(test_dataset))

    fig, axes = plt.subplots(n_cases, n_methods, figsize=(3 * n_methods, 3.5 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    def norm_display(img: np.ndarray) -> np.ndarray:
        lo, hi = img.min(), img.max()
        return np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1)

    for row_idx in range(n_cases):
        sample = test_dataset[row_idx]
        target = sample["target"][0].numpy()
        fbp = sample["fbp"][0].numpy()
        sino = sample["sinogram"].unsqueeze(0)

        images = [target, fbp]

        for model_name, (model, is_freq) in trained_models.items():
            model.eval()
            with torch.no_grad():
                inp = (sino if is_freq else sample["fbp"].unsqueeze(0)).to(device)
                if is_freq:
                    recon, _ = model(inp)
                else:
                    recon = model(inp)
                pred = recon[0, 0].cpu().numpy()
            images.append(pred)

        for col_idx, (img, name) in enumerate(zip(images, method_names)):
            ax = axes[row_idx, col_idx]
            ax.imshow(norm_display(img), cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"Case {row_idx+1}", fontsize=9)

    fig.suptitle("CT Reconstruction Comparison: Target vs FBP vs DL Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CT panel: {out_path}")


def _plot_training_curves(
    all_results: list[TrainingResult],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
    metrics = [
        ("val_loss", "Validation Loss", axes[0]),
        ("val_ssim", "Validation SSIM", axes[1]),
        ("val_psnr", "Validation PSNR (dB)", axes[2]),
    ]

    for res, color in zip(all_results, colors):
        epochs = [r.epoch for r in res.epoch_records]
        for attr, label, ax in metrics:
            values = [getattr(r, attr) for r in res.epoch_records]
            ax.plot(epochs, values, label=res.model_name, color=color, linewidth=2.0)

    for attr, label, ax in metrics:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Curves — All DL Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves: {out_path}")


def _plot_metrics_comparison(
    fbp_result,
    dl_results: list,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    all_results = [fbp_result] + dl_results
    names = [r.method_name for r in all_results]
    ssims = [r.mean_ssim for r in all_results]
    psnrs = [r.mean_psnr for r in all_results]
    rmses = [r.mean_rmse for r in all_results]

    colors = ["#95a5a6", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"][:len(all_results)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, ylabel, title in zip(
        axes,
        [ssims, psnrs, rmses],
        ["SSIM (↑)", "PSNR dB (↑)", "RMSE (↓)"],
        ["Structural Similarity", "Peak SNR", "Root Mean Square Error"],
    ):
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=1.0, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)

    plt.suptitle("Reconstruction Quality: FBP vs DL Methods (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Metrics comparison: {out_path}")


def _plot_speed_quality(
    fbp_result,
    dl_results: list,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    all_results = [fbp_result] + dl_results
    names = [r.method_name for r in all_results]
    ssims = [r.mean_ssim for r in all_results]
    times = [r.mean_inference_ms for r in all_results]
    colors = ["#95a5a6", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"][:len(all_results)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, ssim, t, color in zip(names, ssims, times, colors):
        ax.scatter(t, ssim, s=200, color=color, edgecolor="black", linewidth=1.5, zorder=3, label=name)
        ax.annotate(name, (t, ssim), textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Mean Inference Time per Slice (ms) — lower is faster", fontsize=11)
    ax.set_ylabel("Mean SSIM (↑)", fontsize=11)
    ax.set_title("Speed vs Quality Tradeoff: CT Reconstruction Methods", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Speed-quality plot: {out_path}")


def _plot_radar(
    fbp_result,
    dl_results: list,
    out_path: Path,
) -> None:
    """Radar chart comparing methods across normalized SSIM, PSNR, 1-RMSE, Speed."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_results = [fbp_result] + dl_results

    # Normalize metrics to [0, 1]
    ssims = np.array([r.mean_ssim for r in all_results])
    psnrs = np.array([r.mean_psnr for r in all_results])
    rmses = np.array([r.mean_rmse for r in all_results])
    times = np.array([r.mean_inference_ms for r in all_results])

    def norm_minmax(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if np.isclose(lo, hi):
            return np.ones_like(arr) * 0.5
        n = (arr - lo) / (hi - lo)
        return 1 - n if invert else n

    ssim_n = norm_minmax(ssims)
    psnr_n = norm_minmax(psnrs)
    rmse_n = norm_minmax(rmses, invert=True)  # lower is better
    speed_n = norm_minmax(times, invert=True)  # lower time = higher score

    labels = ["SSIM", "PSNR", "Accuracy\n(1-RMSE)", "Speed"]
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#95a5a6", "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"][:len(all_results)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for r, metrics_n, color in zip(all_results, zip(ssim_n, psnr_n, rmse_n, speed_n), colors):
        values = list(metrics_n) + [metrics_n[0]]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=r.method_name)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Radar: CT Reconstruction Methods", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Radar chart: {out_path}")


def _plot_freq_spectrum(test_dataset: CTReconDataset, out_path: Path) -> None:
    """Analyze frequency content of FBP vs target images."""
    import matplotlib.pyplot as plt

    n = min(20, len(test_dataset))
    fbp_spectra, tgt_spectra = [], []

    for i in range(n):
        sample = test_dataset[i]
        fbp = sample["fbp"][0].numpy()
        tgt = sample["target"][0].numpy()

        fbp_fft = np.abs(np.fft.fftshift(np.fft.fft2(fbp)))
        tgt_fft = np.abs(np.fft.fftshift(np.fft.fft2(tgt)))

        # Radial average
        h, w = fbp_fft.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy).astype(int)
        r_max = min(cx, cy)

        fbp_rad = np.array([fbp_fft[r == ri].mean() if np.any(r == ri) else 0 for ri in range(r_max)])
        tgt_rad = np.array([tgt_fft[r == ri].mean() if np.any(r == ri) else 0 for ri in range(r_max)])

        fbp_spectra.append(fbp_rad)
        tgt_spectra.append(tgt_rad)

    fbp_mean = np.mean(fbp_spectra, axis=0)
    tgt_mean = np.mean(tgt_spectra, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log-scale radial spectrum
    freq_axis = np.arange(len(fbp_mean))
    axes[0].semilogy(freq_axis, tgt_mean + 1e-8, label="Target (clean)", color="#2980b9", linewidth=2)
    axes[0].semilogy(freq_axis, fbp_mean + 1e-8, label="FBP (noisy)", color="#e74c3c", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Spatial Frequency (cycles/px)", fontsize=11)
    axes[0].set_ylabel("Mean Power Spectrum (log)", fontsize=11)
    axes[0].set_title("Radial Power Spectrum: Target vs FBP", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Ratio: FBP / Target (shows where FBP over/under-amplifies)
    ratio = (fbp_mean + 1e-8) / (tgt_mean + 1e-8)
    axes[1].plot(freq_axis, ratio, color="#8e44ad", linewidth=2)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Spatial Frequency (cycles/px)", fontsize=11)
    axes[1].set_ylabel("FBP / Target Spectrum Ratio", fontsize=11)
    axes[1].set_title("FBP Frequency Distortion (ratio > 1 = over-amplified)", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Frequency Domain Analysis: Why DL Correction Is Needed", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Frequency spectrum: {out_path}")


# ---------------------------------------------------------------------------
# Save CSV reports
# ---------------------------------------------------------------------------

def _save_training_csv(all_results: list[TrainingResult], out_path: Path) -> None:
    rows = []
    for res in all_results:
        for rec in res.epoch_records:
            row = {
                "model": res.model_name,
                "epoch": rec.epoch,
                "train_loss": rec.train_loss,
                "val_loss": rec.val_loss,
                "val_ssim": rec.val_ssim,
                "val_psnr": rec.val_psnr,
                "val_rmse": rec.val_rmse,
                "lr": rec.lr,
                "epoch_time_s": rec.epoch_time_s,
            }
            row.update({f"train_{k}": v for k, v in rec.train_loss_components.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  ✓ Training CSV: {out_path}")


def _save_inference_csv(fbp_result, dl_results: list, out_path: Path) -> None:
    rows = []
    for result in [fbp_result] + dl_results:
        for i, m in enumerate(result.slice_metrics):
            rows.append({
                "method": result.method_name,
                "slice_idx": i,
                "ssim": m.ssim,
                "psnr": m.psnr,
                "rmse": m.rmse,
                "inference_ms": m.inference_time_ms,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  ✓ Inference CSV: {out_path}")


def _save_summary_csv(fbp_result, dl_results: list, out_path: Path) -> None:
    rows = []
    for result in [fbp_result] + dl_results:
        row = {
            "method": result.method_name,
            "mean_ssim": result.mean_ssim,
            "mean_psnr": result.mean_psnr,
            "mean_rmse": result.mean_rmse,
            "mean_inference_ms": result.mean_inference_ms,
            "throughput_slices_per_sec": result.throughput_slices_per_sec,
            "n_parameters": result.n_parameters,
            "model_size_mb": result.model_size_mb,
            "ssim_vs_fbp_wilcoxon_p": result.ssim_vs_fbp_wilcoxon_p,
            "ssim_vs_fbp_perm_p": result.ssim_vs_fbp_perm_p,
            "ssim_vs_fbp_effect_r": result.ssim_vs_fbp_effect_r,
            "ssim_vs_fbp_ci95_low": result.ssim_vs_fbp_ci95_low,
            "ssim_vs_fbp_ci95_high": result.ssim_vs_fbp_ci95_high,
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  ✓ Summary CSV: {out_path}")


def _print_summary_table(fbp_result, dl_results: list) -> None:
    print("\n" + "=" * 95)
    print("FINAL RESULTS: CT RECONSTRUCTION — FBP vs DL METHODS")
    print("=" * 95)
    header = f"{'Method':<18} {'SSIM':>8} {'PSNR':>8} {'RMSE':>8} {'Time(ms)':>10} {'Params':>10} {'Wilcoxon p':>12}"
    print(header)
    print("-" * 95)
    for res in [fbp_result] + dl_results:
        p_str = f"{res.ssim_vs_fbp_wilcoxon_p:.4f}" if not np.isnan(res.ssim_vs_fbp_wilcoxon_p) else "baseline"
        print(
            f"{res.method_name:<18} {res.mean_ssim:>8.4f} {res.mean_psnr:>8.2f} {res.mean_rmse:>8.4f} "
            f"{res.mean_inference_ms:>10.2f} {res.n_parameters:>10,d} {p_str:>12}"
        )
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CT Reconstruction DL Pipeline")
    parser.add_argument("--data", default="../real_data/organamnist/raw/organamnist.npz",
                        help="Path to organamnist.npz")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Target image size (both H and W)")
    parser.add_argument("--n-angles", type=int, default=90,
                        help="Number of projection angles for sinogram simulation")
    parser.add_argument("--dose-fraction", type=float, default=0.25,
                        help="Dose fraction (0.25 = quarter-dose CT, realistic low-dose)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs per model")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Limit training samples (for quick-test mode)")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Limit test samples")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory for all results")
    parser.add_argument("--methods", nargs="+",
                        default=["red_cnn", "unet", "attention_unet", "freq_hybrid"],
                        help="Methods to train and evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-ch", type=int, default=32,
                        help="Base channels for U-Net models (32=lighter, 64=full)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  CT RECONSTRUCTION — END-TO-END DL PIPELINE")
    print(f"{'='*70}")
    print(f"  Device:       {device}")
    print(f"  Image size:   {args.image_size}×{args.image_size}")
    print(f"  Angles:       {args.n_angles}")
    print(f"  Dose:         {args.dose_fraction:.0%} (low-dose CT simulation)")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Methods:      {args.methods}")
    print(f"{'='*70}\n")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    data_path = PROJECT_ROOT / args.data

    # ---- Stage 0: Load data ----
    print("[ Stage 0 ] Loading real OrganAMNIST CT data...")
    t0 = time.perf_counter()
    train_ds = CTReconDataset(
        data_path, "train",
        target_size=args.image_size, n_angles=args.n_angles,
        dose_fraction=args.dose_fraction,
        max_samples=args.max_train, seed=args.seed,
    )
    val_ds = CTReconDataset(
        data_path, "val",
        target_size=args.image_size, n_angles=args.n_angles,
        dose_fraction=args.dose_fraction,
        max_samples=args.max_train // 4 if args.max_train else None, seed=args.seed,
    )
    test_ds = CTReconDataset(
        data_path, "test",
        target_size=args.image_size, n_angles=args.n_angles,
        dose_fraction=args.dose_fraction,
        max_samples=args.max_test, seed=args.seed,
    )
    load_time = time.perf_counter() - t0

    print(f"  Train: {len(train_ds)} slices | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Load time: {load_time*1000:.0f} ms")
    print(f"  Image shape: {args.image_size}×{args.image_size} | Angles: {args.n_angles}")
    print(f"  Sinogram shape: {args.n_angles}×{train_ds[0]['sinogram'].shape[2]} per slice")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ---- Stage 1+2: Sinogram simulation is done in dataset (verify here) ----
    sample = train_ds[0]
    sino_shape = tuple(sample["sinogram"].shape)
    fbp_shape = tuple(sample["fbp"].shape)
    target_shape = tuple(sample["target"].shape)
    print(f"\n  Sinogram: {sino_shape}  |  FBP: {fbp_shape}  |  Target: {target_shape}")
    print(f"  FBP pixel range: [{sample['fbp'].min():.3f}, {sample['fbp'].max():.3f}]")
    print(f"  Target pixel range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")

    # ---- Stage 3: FBP baseline ----
    print("\n[ Stage 3a ] Measuring FBP baseline...")
    fbp_result = run_fbp_inference(test_loader, device)
    print(f"  FBP: SSIM={fbp_result.mean_ssim:.4f} | PSNR={fbp_result.mean_psnr:.2f} | "
          f"RMSE={fbp_result.mean_rmse:.4f} | Time={fbp_result.mean_inference_ms:.3f}ms")

    # ---- Stage 3b: DL frequency spectrum analysis ----
    print("\n[ Freq Analysis ] Computing frequency power spectra...")
    _plot_freq_spectrum(test_ds, out_dir / "freq_spectrum.png")

    # ---- Stage 3c: Train and evaluate each DL method ----
    method_map = {
        "red_cnn": ("RED-CNN", REDCNN(n_channels=1, n_filters=64), REDCNNLoss(), False),
        "unet": ("U-Net", UNet(n_channels=1, n_classes=1, base_ch=args.base_ch), UNetLoss(), False),
        "attention_unet": ("AttentionUNet", AttentionUNet(n_channels=1, n_classes=1, base_ch=args.base_ch), AttentionUNetLoss(), False),
        "freq_hybrid": ("FreqHybrid", FreqHybridNet(
            n_angles=args.n_angles,
            n_detectors=train_ds[0]["sinogram"].shape[2],
            img_h=args.image_size, img_w=args.image_size,
        ), FreqHybridLoss(), True),
    }

    all_train_results: list = []
    dl_results: list = []
    trained_models: dict = {}

    for method_key in args.methods:
        if method_key not in method_map:
            print(f"  WARNING: Unknown method '{method_key}', skipping")
            continue

        display_name, model, loss_fn, is_freq = method_map[method_key]
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"  Training: {display_name} ({n_params:,} parameters)")
        print(f"{'='*50}")

        train_result = train_model(
            model=model,
            model_name=display_name,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            patience=args.patience,
            checkpoint_dir=checkpoint_dir,
            is_freq_hybrid=is_freq,
        )
        all_train_results.append(train_result)

        print(f"\n  Running test inference for {display_name}...")
        inf_result = run_model_inference(
            model=model,
            model_name=display_name,
            test_loader=test_loader,
            device=device,
            is_freq_hybrid=is_freq,
        )
        dl_results.append(inf_result)
        trained_models[display_name] = (model, is_freq)

        print(f"  {display_name}: SSIM={inf_result.mean_ssim:.4f} | PSNR={inf_result.mean_psnr:.2f} | "
              f"RMSE={inf_result.mean_rmse:.4f} | Time={inf_result.mean_inference_ms:.2f}ms | "
              f"Params={inf_result.n_parameters:,}")

    # ---- Stage 4: Statistical comparison ----
    print("\n[ Stage 4 ] Statistical comparison (vs FBP)...")
    dl_results = run_statistical_comparison(
        fbp_result, dl_results,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    # ---- Stage 5: Save everything ----
    print("\n[ Stage 5 ] Generating reports and visualizations...")

    _save_training_csv(all_train_results, out_dir / "training_curves.csv")
    _save_inference_csv(fbp_result, dl_results, out_dir / "inference_metrics.csv")
    _save_summary_csv(fbp_result, dl_results, out_dir / "method_summary.csv")

    if all_train_results:
        _plot_training_curves(all_train_results, out_dir / "training_curves.png")

    _plot_metrics_comparison(fbp_result, dl_results, out_dir / "metrics_comparison.png")
    _plot_speed_quality(fbp_result, dl_results, out_dir / "speed_quality_plot.png")
    _plot_radar(fbp_result, dl_results, out_dir / "radar_chart.png")

    if trained_models:
        _plot_ct_panel(
            test_ds, trained_models, out_dir / "ct_panel.png",
            n_cases=min(6, len(test_ds)), device=device,
        )

    # Save metadata JSON
    metadata = {
        "dataset": "OrganAMNIST (real CT slices from LiTS)",
        "data_path": str(data_path),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "image_size": args.image_size,
        "n_angles": args.n_angles,
        "dose_fraction": args.dose_fraction,
        "methods": args.methods,
        "device": str(device),
        "epochs": args.epochs,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    _print_summary_table(fbp_result, dl_results)
    print(f"\n✓ All results saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
