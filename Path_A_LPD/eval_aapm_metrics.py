import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.transform import iradon

# Hook into existing repo metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ct_recon_dl.ct_recon.metrics import compute_metrics
from dataset_aapm import AAPMDataset
from lpd_model import LearnedPrimalDual

def evaluate_aapm_metrics():
    """
    Evaluates LPD vs FBP on the 1.8GB Phantom Evaluation Holdout set.
    Logs execution time (LPD Unrolled vs CPU FBP) and accuracy (SSIM/PSNR/RMSE).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"AAPM Phantom LPD Evaluation Script initiated. Target Device: {device}")
    
    # Validation constraints
    image_size = 512
    num_angles = 180
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'real_data', 'aapm_ldct'))
    
    # DataLoader pointing explicitly to Phantom_Eval split
    eval_dataset = AAPMDataset(root_dir=root_dir, split="eval", n_angles=num_angles, image_size=image_size, cache_to_ram=True)
    if len(eval_dataset) == 0:
        print("No evaluation data (Phantom) found. Please populate real_data/aapm_ldct/eval.")
        return
    
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    sample = eval_dataset[0]
    _, actual_angles, actual_detectors = sample['sinogram'].shape
    
    # Boot LPD and load theoretical weights
    model = LearnedPrimalDual(
        num_iterations=10, 
        image_size=image_size,
        num_angles=actual_angles,
        num_detectors=actual_detectors
    ).to(device)
    model.eval()
    
    ckpt_path = 'checkpoints/lpd_aapm_512.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("LPD Checkpoint loaded successfully.")
    else:
        print("WARNING: Checkpoint lpd_aapm_512.pth not found. Metrics will reflect untrained initialized weights.")

    lpd_stats = {'psnr': [], 'ssim': [], 'rmse': [], 'time': []}
    fbp_stats = {'psnr': [], 'ssim': [], 'rmse': [], 'time': []}
    
    print("\nExecuting Inference over the 1.8GB Phantom Eval Set...")
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            g = batch['sinogram'].to(device)
            f_true = batch['target'].squeeze().numpy()  # Extract the numpy target
            
            # 1. EVALUATE LPD (Unrolled Inference Speed & Metrics)
            t0 = time.perf_counter()
            f_pred = model(g).cpu().squeeze().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            lpd_time = (time.perf_counter() - t0) * 1000  # ms
            lpd_stats['time'].append(lpd_time)
            
            f_pred = np.clip(f_pred, 0, 1) # Force range match for metrics
            metrics_lpd = compute_metrics(f_pred, f_true, data_range=1.0)
            lpd_stats['psnr'].append(metrics_lpd['psnr'])
            lpd_stats['ssim'].append(metrics_lpd['ssim'])
            lpd_stats['rmse'].append(metrics_lpd['rmse'])
            
            # 2. EVALUATE FBP (Since we want to compare pure physics to Unrolled Deep Learning)
            t1 = time.perf_counter()
            fbp_recon = iradon(
                g.cpu().squeeze().numpy().T,
                theta=eval_dataset.theta,
                filter_name='ramp',
                circle=False,
            ).astype(np.float32)
            fbp_recon = (fbp_recon - fbp_recon.min()) / (fbp_recon.max() - fbp_recon.min() + 1e-8)
            fbp_recon = np.clip(fbp_recon, 0, 1)
            metrics_fbp = compute_metrics(fbp_recon, f_true, data_range=1.0)
            fbp_time = (time.perf_counter() - t1) * 1000
            fbp_stats['time'].append(fbp_time) 
            fbp_stats['ssim'].append(metrics_fbp['ssim'])
            fbp_stats['psnr'].append(metrics_fbp['psnr'])
            fbp_stats['rmse'].append(metrics_fbp['rmse'])
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(eval_loader)} slices...")

    # Aggregate & Output
    print("\n" + "="*50)
    print(f"FINAL METRICS (1.8GB Phantom Eval Test Set)")
    print("="*50)
    print("Classical Filtered Back Projection (FBP):")
    print(f"  PSNR: {np.mean(fbp_stats['psnr']):.2f} ± {np.std(fbp_stats['psnr']):.2f} dB")
    print(f"  SSIM: {np.mean(fbp_stats['ssim']):.4f} ± {np.std(fbp_stats['ssim']):.4f}")
    print(f"  RMSE: {np.mean(fbp_stats['rmse']):.4f}")
    print(f"  Avg Inference Time: ~{np.mean(fbp_stats['time']):.1f} ms/slice")
    
    print("\nLearned Primal-Dual (LPD):")
    print(f"  PSNR: {np.mean(lpd_stats['psnr']):.2f} ± {np.std(lpd_stats['psnr']):.2f} dB")
    print(f"  SSIM: {np.mean(lpd_stats['ssim']):.4f} ± {np.std(lpd_stats['ssim']):.4f}")
    print(f"  RMSE: {np.mean(lpd_stats['rmse']):.4f}")
    print(f"  Avg Inference Time: {np.mean(lpd_stats['time']):.1f} ± {np.std(lpd_stats['time']):.1f} ms/slice")
    print("="*50)

    # Dump a quick comparative CSV inside results folder
    os.makedirs('results', exist_ok=True)
    with open('results/aapm_metrics_comparison.csv', 'w') as f:
        f.write("Method,PSNR_Mean,PSNR_Std,SSIM_Mean,SSIM_Std,RMSE,InferenceTime_ms\n")
        f.write(f"FBP,{np.mean(fbp_stats['psnr']):.2f},{np.std(fbp_stats['psnr']):.2f},{np.mean(fbp_stats['ssim']):.4f},{np.std(fbp_stats['ssim']):.4f},{np.mean(fbp_stats['rmse']):.4f},{np.mean(fbp_stats['time']):.1f}\n")
        f.write(f"LPD,{np.mean(lpd_stats['psnr']):.2f},{np.std(lpd_stats['psnr']):.2f},{np.mean(lpd_stats['ssim']):.4f},{np.std(lpd_stats['ssim']):.4f},{np.mean(lpd_stats['rmse']):.4f},{np.mean(lpd_stats['time']):.1f}\n")

if __name__ == "__main__":
    evaluate_aapm_metrics()
