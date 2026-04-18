import os

def download_aapm_dataset(output_dir="../real_data/aapm_ldct"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    
    print("=" * 60)
    print("AAPM Low Dose CT (LDCT) - 1.8GB 'Phantom' Split")
    print("=" * 60)
    print("WARNING: Direct open download is restricted by Medical Data Use Agreements.")
    print("To execute this pipeline using the ultra-light 1.8GB Phantom dataset:")
    print("1. Go to TCIA: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026")
    print("2. Click the blue 'DOWNLOAD (1.8GB)' button next to 'Images Phantom Object Only'.")
    print("3. Open the downloaded .tcia file in the NBIA Data Retriever.")
    print("4. This contains ~18,000 slices. Split them into your train/eval folders.")
    print("\nExpected Directory Structure Once Downloaded:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   └── Phantom_Train/ (Put ~15,000 Numpy slices here)")
    print(f"  └── eval/")
    print(f"      └── Phantom_Eval/  (Put ~3,000 Numpy slices here)")
    print("\nOnce populated, the dataset_aapm.py script will handle the conversion automatically.")

    # Generating mock structural files for the phantom
    import numpy as np
    
    print("\nGenerating mock structural files (512x512) to validate the pipeline architecture...")
    pat_dir = os.path.join(output_dir, "train", "Phantom_Train")
    os.makedirs(pat_dir, exist_ok=True)
    if len(os.listdir(pat_dir)) == 0:
        for slice_idx in range(5):
            np.save(os.path.join(pat_dir, f"slice_{slice_idx:03d}.npy"), np.random.rand(512, 512).astype(np.float32))
    
    eval_dir = os.path.join(output_dir, "eval", "Phantom_Eval")
    os.makedirs(eval_dir, exist_ok=True)
    if len(os.listdir(eval_dir)) == 0:
        for slice_idx in range(2):
            np.save(os.path.join(eval_dir, f"slice_{slice_idx:03d}.npy"), np.random.rand(512, 512).astype(np.float32))

    print(f"Target filesystem structurally prepped at: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_aapm_dataset()
