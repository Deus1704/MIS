import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ct_recon_dl.ct_recon.metrics import compute_metrics
from dataset_aapm import AAPMSinogramDataset
from lpd_model import LearnedPrimalDual


def evaluate_aapm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AAPM LPD Evaluation ===")
    print(f"Device: {device}")

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "real_data", "aapm_ldct")
    )
    target_root_dir = os.environ.get("AAPM_TARGET_ROOT_DIR")
    allow_fbp_target = os.environ.get("AAPM_ALLOW_FBP_TARGET", "1" if not target_root_dir else "0") == "1"

    # Load eval dataset
    eval_dataset = AAPMSinogramDataset(
        root_dir=root_dir,
        target_root_dir=target_root_dir,
        allow_fbp_target=allow_fbp_target,
        split="eval",
        image_size=512,
        max_samples=200,
        cache_to_ram=False,
        eval_split=0.1,
    )

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Get dimensions
    sample = eval_dataset[0]
    _, num_angles, num_detectors = sample["sinogram"].shape

    # Load model
    # Must match the checkpoint that was trained in train_aapm_lpd.py
    model = LearnedPrimalDual(
        num_iterations=5,
        image_size=512,
        num_angles=num_angles,
        num_detectors=num_detectors,
    ).to(device)

    model.eval()

    ckpt_path = "checkpoints/lpd_aapm_512.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint")
    else:
        print("WARNING: No checkpoint found!")
        return

    # Save a few qualitative side-by-side comparisons for inspection.
    vis_dir = "results/aapm_visuals"
    os.makedirs(vis_dir, exist_ok=True)

    # Evaluate
    lpd_stats = {"psnr": [], "ssim": [], "rmse": [], "time": []}
    fbp_stats = {"psnr": [], "ssim": [], "rmse": [], "time": []}

    print(f"\nEvaluating on {len(eval_dataset)} samples...")
    if target_root_dir:
        print(f"Target root: {os.path.abspath(target_root_dir)}")
    else:
        print("Target root: not set (set AAPM_TARGET_ROOT_DIR to use real paired ground truth)")
    if allow_fbp_target and not target_root_dir:
        print("WARNING: Falling back to FBP as a debug target. Metrics are not ground truth.")

    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            g = batch["sinogram"].to(device)
            f_true = batch["target"].squeeze().cpu().numpy()

            # LPD inference
            t0 = time.perf_counter()
            f_pred = model(g).cpu().squeeze().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            lpd_time = (time.perf_counter() - t0) * 1000
            lpd_stats["time"].append(lpd_time)

            f_pred = np.clip(f_pred, 0, 1)
            metrics_lpd = compute_metrics(f_pred, f_true, data_range=1.0)
            lpd_stats["psnr"].append(metrics_lpd["psnr"])
            lpd_stats["ssim"].append(metrics_lpd["ssim"])
            lpd_stats["rmse"].append(metrics_lpd["rmse"])

            # FBP baseline
            fbp_recon = batch["fbp"].squeeze().cpu().numpy()
            metrics_fbp = compute_metrics(fbp_recon, f_true, data_range=1.0)
            fbp_stats["psnr"].append(metrics_fbp["psnr"])
            fbp_stats["ssim"].append(metrics_fbp["ssim"])
            fbp_stats["rmse"].append(metrics_fbp["rmse"])

            if idx < 4:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                diff = np.abs(f_pred - fbp_recon)

                axes[0].imshow(fbp_recon, cmap="bone", vmin=0, vmax=1)
                axes[0].set_title("FBP")
                axes[0].axis("off")

                axes[1].imshow(f_pred, cmap="bone", vmin=0, vmax=1)
                axes[1].set_title("LPD")
                axes[1].axis("off")

                axes[2].imshow(diff, cmap="magma")
                axes[2].set_title("|LPD - FBP|")
                axes[2].axis("off")

                fig.suptitle(f"AAPM Phantom Sample {idx}")
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"sample_{idx:03d}.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)

            if idx % 20 == 0:
                print(f"  Processed {idx + 1}/{len(eval_dataset)}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS (AAPM Phantom Sinogram Data)")
    print("=" * 60)

    print("\nFBP (Baseline):")
    print(
        f"  PSNR: {np.mean(fbp_stats['psnr']):.2f} ± {np.std(fbp_stats['psnr']):.2f} dB"
    )
    print(f"  SSIM: {np.mean(fbp_stats['ssim']):.4f} ± {np.std(fbp_stats['ssim']):.4f}")
    print(f"  RMSE: {np.mean(fbp_stats['rmse']):.4f}")

    print("\nLPD (Learned):")
    print(
        f"  PSNR: {np.mean(lpd_stats['psnr']):.2f} ± {np.std(lpd_stats['psnr']):.2f} dB"
    )
    print(f"  SSIM: {np.mean(lpd_stats['ssim']):.4f} ± {np.std(lpd_stats['ssim']):.4f}")
    print(f"  RMSE: {np.mean(lpd_stats['rmse']):.4f}")
    print(f"  Time: {np.mean(lpd_stats['time']):.1f} ms/image")
    print("=" * 60)

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/aapm_metrics_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("Method,PSNR_Mean,PSNR_Std,SSIM_Mean,SSIM_Std,RMSE,InferenceTime_ms\n")
        f.write(
            f"FBP,{np.mean(fbp_stats['psnr']):.2f},{np.std(fbp_stats['psnr']):.2f},{np.mean(fbp_stats['ssim']):.4f},{np.std(fbp_stats['ssim']):.4f},{np.mean(fbp_stats['rmse']):.4f},0.0\n"
        )
        f.write(
            f"LPD,{np.mean(lpd_stats['psnr']):.2f},{np.std(lpd_stats['psnr']):.2f},{np.mean(lpd_stats['ssim']):.4f},{np.std(lpd_stats['ssim']):.4f},{np.mean(lpd_stats['rmse']):.4f},{np.mean(lpd_stats['time']):.1f}\n"
        )

    print(f"\nResults saved to {csv_path}")
    print(f"Visualization images saved to {vis_dir}")


if __name__ == "__main__":
    evaluate_aapm()
