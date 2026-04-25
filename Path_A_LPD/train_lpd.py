import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to path to import from the existing implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ct_recon_dl.pipeline.dataset import CTReconDataset
from lpd_model import LearnedPrimalDual

def main():
    # Hyperparameters
    num_iterations = 5   # LPD unrolled iterations
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 10
    image_size = 128
    num_angles = 180
    num_detectors = 180
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to real data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'organamnist', 'raw', 'organamnist.npz'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Real data not found at {data_path}")

    # DataLoaders using real organism data
    train_dataset = CTReconDataset(
        npz_path=data_path,
        split='train',
        target_size=image_size,
        n_angles=num_angles,
        max_samples=100,  # limit for training speed, adjust as necessary
        seed=42,
        verbose=True
    )
    
    # Update num_detectors based on actual simulated sinogram dimensions from dataset
    # We grab the first sample to figure this out dynamically
    sample = train_dataset[0]
    _, actual_angles, actual_detectors = sample['sinogram'].shape
    
    # Model
    model = LearnedPrimalDual(
        num_iterations=num_iterations,
        image_size=image_size,
        num_angles=actual_angles,
        num_detectors=actual_detectors
    ).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Parse dict items from CTReconDataset
            g = batch["sinogram"].to(device)
            f_true = batch["target"].to(device)
            
            # Forward pass
            f_pred = model(g)
            
            # Compute loss
            loss = criterion(f_pred, f_true)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
    print("Training Complete. Saving model...")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/lpd_model.pth')
    print("Model saved to checkpoints/lpd_model.pth")

if __name__ == "__main__":
    main()
