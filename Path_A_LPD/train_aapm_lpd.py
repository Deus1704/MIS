import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_aapm import AAPMSinogramDataset
from lpd_model import LearnedPrimalDual


def train_aapm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AAPM LPD Training ===")
    print(f"Device: {device}")

    num_epochs = int(os.environ.get("AAPM_NUM_EPOCHS", "40"))
    learning_rate = float(os.environ.get("AAPM_LR", "1e-4"))
    train_max_samples = int(os.environ.get("AAPM_TRAIN_MAX_SAMPLES", "0")) or None
    eval_max_samples = int(os.environ.get("AAPM_EVAL_MAX_SAMPLES", "50"))

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "real_data", "aapm_ldct")
    )
    target_root_dir = os.environ.get("AAPM_TARGET_ROOT_DIR")
    allow_fbp_target = os.environ.get("AAPM_ALLOW_FBP_TARGET", "1" if not target_root_dir else "0") == "1"
    if not os.path.exists(root_dir):
        print(f"ERROR: Data path {root_dir} not found")
        return

    # Create datasets - reduced for CPU
    train_dataset = AAPMSinogramDataset(
        root_dir=root_dir,
        target_root_dir=target_root_dir,
        allow_fbp_target=allow_fbp_target,
        split="train",
        image_size=512,
        max_samples=train_max_samples,
        cache_to_ram=False,
        eval_split=0.1,
    )

    eval_dataset = AAPMSinogramDataset(
        root_dir=root_dir,
        target_root_dir=target_root_dir,
        allow_fbp_target=allow_fbp_target,
        split="eval",
        image_size=512,
        max_samples=eval_max_samples,
        cache_to_ram=True,
        eval_split=0.1,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    if target_root_dir:
        print(f"Target root: {os.path.abspath(target_root_dir)}")
    else:
        print("Target root: not set (set AAPM_TARGET_ROOT_DIR to use real paired ground truth)")
    if allow_fbp_target and not target_root_dir:
        print("WARNING: Falling back to FBP as a debug target. Training/eval metrics will not be ground truth.")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    sample = train_dataset[0]
    _, num_angles, num_detectors = sample["sinogram"].shape
    print(f"\nSinogram geometry: {num_angles} angles × {num_detectors} detectors")

    # Create model - reduced iterations for faster training
    model = LearnedPrimalDual(
        num_iterations=5,  # Reduced from 10
        image_size=512,
        num_angles=num_angles,
        num_detectors=num_detectors,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"\nStarting training")
    print(f"  epochs: {num_epochs}")
    print(f"  learning rate: {learning_rate}")
    if train_max_samples is None:
        print("  train samples: full available split")
    else:
        print(f"  train samples cap: {train_max_samples}")
    print(f"  eval samples cap: {eval_max_samples}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )

    best_eval_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            g = batch["sinogram"].to(device)
            f_true = batch["target"].to(device)

            optimizer.zero_grad()
            f_pred = model(g)
            loss = criterion(f_pred, f_true)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / num_batches
        print(f"--- Epoch {epoch + 1} Complete. Avg Loss: {avg_loss:.6f} ---")

        model.eval()
        eval_loss = 0.0
        num_eval = 0
        with torch.no_grad():
            for batch in eval_loader:
                g = batch["sinogram"].to(device)
                f_true = batch["target"].to(device)
                f_pred = model(g)
                loss = criterion(f_pred, f_true)
                eval_loss += loss.item()
                num_eval += 1

        avg_eval_loss = eval_loss / max(num_eval, 1)
        scheduler.step(avg_eval_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"    Eval loss: {avg_eval_loss:.6f} | LR: {current_lr:.2e}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_epoch = epoch + 1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/lpd_aapm_512_best.pth")
            print(f"    Saved new best checkpoint at epoch {best_epoch}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lpd_aapm_512.pth")
    print(f"\nModel saved to checkpoints/lpd_aapm_512.pth")
    if best_epoch > 0:
        print(f"Best checkpoint: epoch {best_epoch} with eval loss {best_eval_loss:.6f}")

    # Quick eval
    print("\n=== Quick Evaluation ===")
    model.eval()
    eval_loss = 0.0
    num_eval = 0

    with torch.no_grad():
        for batch in eval_loader:
            g = batch["sinogram"].to(device)
            f_true = batch["target"].to(device)
            f_pred = model(g)
            loss = criterion(f_pred, f_true)
            eval_loss += loss.item()
            num_eval += 1

    print(f"Eval Loss: {eval_loss / num_eval:.6f}")
    print("\n✓ Training complete! Run eval_aapm_metrics.py for full metrics.")


if __name__ == "__main__":
    train_aapm()
