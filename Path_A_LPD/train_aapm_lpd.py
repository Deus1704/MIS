import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_aapm import AAPMDataset
import sys

# Load Primal Dual Architecture
from lpd_model import LearnedPrimalDual

def train_aapm():
    # Hardware Switch (We script this but you run this later on GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"AAPM Mayo Clinic LPD Training Script initiated. Target Device: {device}")
    
    # 512x512 Constants
    image_size = 512
    num_angles = 180
    num_detectors = 724  # Standard detector size for 512x512 Radon transform
    # The actual detector size depends on the skimage Radon output padding. Let's dynamically fetch it 
    # instead of hardcoding 724 if the array changes based on circle=False
    
    # Pathing
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'real_data', 'aapm_ldct'))
    if not os.path.exists(root_dir):
        print(f"WARNING: AAPM Data path {root_dir} not found. Did you run download_aapm.py?")
        return

    # DataLoader pointing to the Phantom_Train Split (~15,000 arrays)
    train_dataset = AAPMDataset(root_dir=root_dir, split="train", n_angles=num_angles, image_size=image_size, cache_to_ram=False)
    
    if len(train_dataset) == 0:
         print("No data. Exiting compilation. Please inject the AAPM NumPy arrays into real_data/aapm_ldct/train.")
         return
         
    sample_bundle = train_dataset[0]
    _, actual_angles, actual_detectors = sample_bundle['sinogram'].shape
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Batch 2 to prevent 512x512 VRAM explosion
    
    # Model instantiation
    # IMPORTANT: 512x512 spatial primitives alongside 10 iterations of Primal-Dual blocks requires massive GPU memory.
    # If hitting CUDA OOM on your GPU, lower num_iterations to 5.
    model = LearnedPrimalDual(
        num_iterations=10, 
        image_size=image_size,
        num_angles=actual_angles,
        num_detectors=actual_detectors
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print("\nStarting Grand Challenge Training Loop on 512x512 slices...")
    # The 1.8GB Phantom pushes ~15,000 training slices, so only ~10 epochs are needed vs the original 50 used for 2000 slices.
    num_epochs = 10 
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            g = batch["sinogram"].to(device)
            f_true = batch["target"].to(device)
            
            optimizer.zero_grad()
            f_pred = model(g)
            loss = criterion(f_pred, f_true)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}")
                
        print(f"--- Epoch {epoch+1} Complete. Average Loss: {epoch_loss / len(train_loader):.6f} ---")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/lpd_aapm_512.pth')
    print("AAPM 512x512 Model checkpoint saved to checkpoints/lpd_aapm_512.pth")

if __name__ == "__main__":
    train_aapm()
