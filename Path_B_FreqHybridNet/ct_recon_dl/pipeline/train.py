"""Training loop for CT reconstruction DL models.

Provides per-epoch training + validation with full metric tracking.
All timings, losses, and image-quality metrics are recorded every epoch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_ssim: float
    val_psnr: float
    val_rmse: float
    lr: float
    epoch_time_s: float
    train_loss_components: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    model_name: str
    epoch_records: list[EpochRecord]
    best_val_ssim: float
    best_epoch: int
    total_train_time_s: float
    n_parameters: int
    final_model_path: str


def _compute_image_metrics(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute mean SSIM, PSNR, RMSE on a batch (CPU, numpy)."""
    preds = pred_batch.detach().cpu().numpy()
    targets = target_batch.detach().cpu().numpy()

    ssims, psnrs, rmses = [], [], []
    for p, t in zip(preds, targets):
        p_sq = p.squeeze()
        t_sq = t.squeeze()
        dr = float(t_sq.max() - t_sq.min())
        if dr < 1e-8:
            dr = 1.0
        ssims.append(structural_similarity(t_sq, p_sq, data_range=dr))
        psnrs.append(peak_signal_noise_ratio(t_sq, p_sq, data_range=dr))
        rmses.append(float(np.sqrt(np.mean((p_sq - t_sq) ** 2))))

    return float(np.mean(ssims)), float(np.mean(psnrs)), float(np.mean(rmses))


def train_model(
    model: nn.Module,
    model_name: str,
    loss_fn: Any,  # one of the loss classes from losses.py
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_dir: Path | None = None,
    is_freq_hybrid: bool = False,
) -> TrainingResult:
    """Full training loop with early stopping and metric tracking.

    Args:
        model: PyTorch model
        model_name: identifier string
        loss_fn: loss function (see pipeline/losses.py)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        device: cpu or cuda
        n_epochs: max epochs
        lr: initial learning rate
        weight_decay: L2 regularization
        patience: early stopping patience
        checkpoint_dir: directory to save best checkpoint
        is_freq_hybrid: if True, model returns (recon, filtered_sino)

    Returns:
        TrainingResult with per-epoch records
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_val_ssim = -1.0
    best_epoch = 0
    patience_counter = 0
    epoch_records: list[EpochRecord] = []

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(checkpoint_dir / f"{model_name}_best.pt") if checkpoint_dir else ""

    if best_model_path and Path(best_model_path).exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"  [{model_name}] Found existing checkpoint, skipping training.")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return TrainingResult(
            model_name=model_name,
            epoch_records=[],
            best_val_ssim=0.0,
            best_epoch=0,
            total_train_time_s=0.0,
            n_parameters=n_params,
            final_model_path=best_model_path,
        )

    total_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.perf_counter()

        # ---- Training phase ----
        model.train()
        train_losses: list[float] = []
        train_components: dict[str, list[float]] = {}

        for batch in train_loader:
            fbp = batch["fbp"].to(device)
            target = batch["target"].to(device)
            sino = batch["sinogram"].to(device)

            optimizer.zero_grad()

            if is_freq_hybrid:
                recon, filtered_sino = model(sino)
                loss_dict = loss_fn(recon, target, filtered_sino, sino)
            else:
                recon = model(fbp)
                loss_dict = loss_fn(recon, target)

            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k, v in loss_dict.items():
                if k != "total":
                    train_components.setdefault(k, []).append(v.item())

        # ---- Validation phase ----
        model.eval()
        val_losses: list[float] = []
        all_ssim, all_psnr, all_rmse = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                fbp = batch["fbp"].to(device)
                target = batch["target"].to(device)
                sino = batch["sinogram"].to(device)

                if is_freq_hybrid:
                    recon, filtered_sino = model(sino)
                    loss_dict = loss_fn(recon, target, filtered_sino, sino)
                else:
                    recon = model(fbp)
                    loss_dict = loss_fn(recon, target)

                val_losses.append(loss_dict["total"].item())

                # Image metrics on this batch
                # Clamp to [0,1] for metric computation
                recon_clamped = recon.clamp(0, 1)
                target_clamped = target.clamp(0, 1)
                ssim, psnr, rmse = _compute_image_metrics(recon_clamped, target_clamped)
                all_ssim.append(ssim)
                all_psnr.append(psnr)
                all_rmse.append(rmse)

        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        mean_ssim = float(np.mean(all_ssim))
        mean_psnr = float(np.mean(all_psnr))
        mean_rmse = float(np.mean(all_rmse))
        current_lr = float(scheduler.get_last_lr()[0])

        rec = EpochRecord(
            epoch=epoch,
            train_loss=mean_train_loss,
            val_loss=mean_val_loss,
            val_ssim=mean_ssim,
            val_psnr=mean_psnr,
            val_rmse=mean_rmse,
            lr=current_lr,
            epoch_time_s=epoch_time,
            train_loss_components={k: float(np.mean(v)) for k, v in train_components.items()},
        )
        epoch_records.append(rec)

        print(
            f"  [{model_name}] Epoch {epoch:3d}/{n_epochs} | "
            f"train={mean_train_loss:.4f} | val={mean_val_loss:.4f} | "
            f"SSIM={mean_ssim:.4f} | PSNR={mean_psnr:.2f} | RMSE={mean_rmse:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if mean_ssim > best_val_ssim:
            best_val_ssim = mean_ssim
            best_epoch = epoch
            patience_counter = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.perf_counter() - total_start

    # Load best weights
    if best_model_path and Path(best_model_path).exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"  [{model_name}] Loaded best checkpoint (epoch {best_epoch}, SSIM={best_val_ssim:.4f})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return TrainingResult(
        model_name=model_name,
        epoch_records=epoch_records,
        best_val_ssim=best_val_ssim,
        best_epoch=best_epoch,
        total_train_time_s=total_time,
        n_parameters=n_params,
        final_model_path=best_model_path,
    )
