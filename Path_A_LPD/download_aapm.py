import os

def download_aapm_dataset(output_dir="../real_data/aapm_ldct"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    
    print("=" * 60)
    print("AAPM Low Dose CT (LDCT) - 50GB 'Lite' Split")
    print("=" * 60)
    print("WARNING: Direct open download is restricted by Medical Data Use Agreements.")
    print("To execute this pipeline while keeping the dataset under ~50GB:")
    print("1. Go to TCIA: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026")
    print("2. Open the NBIA Data Retriever.")
    print("3. DESELECT all patients EXCEPT the following:")
    print("   - L001, L002, L003 (For Training)")
    print("   - L010 (For Evaluation)")
    print("\nExpected Directory Structure Once Downloaded:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── L001/")
    print(f"  │   ├── L002/")
    print(f"  │   └── L003/")
    print(f"  └── eval/")
    print(f"      └── L010/")
    print("\nOnce populated, the dataset_aapm.py script will handle the conversion automatically.")

    # Generating mock structural files for 3 patients instead of 9
    import numpy as np
    
    print("\nGenerating mock structural files (512x512) to validate the pipeline architecture...")
    for PATIENT_ID in ["L001", "L002", "L003"]:
        pat_dir = os.path.join(output_dir, "train", PATIENT_ID)
        os.makedirs(pat_dir, exist_ok=True)
        if len(os.listdir(pat_dir)) == 0:
            for slice_idx in range(2):
                np.save(os.path.join(pat_dir, f"slice_{slice_idx:03d}.npy"), np.random.rand(512, 512).astype(np.float32))
    
    # 1 Eval patient
    eval_dir = os.path.join(output_dir, "eval", "L010")
    os.makedirs(eval_dir, exist_ok=True)
    if len(os.listdir(eval_dir)) == 0:
        for slice_idx in range(5):
            np.save(os.path.join(eval_dir, f"slice_{slice_idx:03d}.npy"), np.random.rand(512, 512).astype(np.float32))

    print(f"Target filesystem structurally prepped at: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_aapm_dataset()
