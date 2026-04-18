import os
import requests
from tqdm import tqdm

"""
AAPM Low Dose CT Grand Challenge (Mayo Clinic) Downloader

IMPORTANT: The AAPM dataset is hosted on The Cancer Imaging Archive (TCIA) under the collection:
"LDCT-and-Projection-data"

Access requires:
1. Registering for a TCIA account.
2. Signing the Mayo Clinic Data Use Agreement.
3. Using the NBIA Data Retriever or REST API with an API Token.

Below is the automated stub to pull the public manifest or via TCIAs REST API once you have your token.
"""

TCIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

def download_aapm_dataset(output_dir="../real_data/aapm_ldct", num_patients=10):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    
    print("=" * 60)
    print("AAPM Low Dose CT (LDCT) Grand Challenge Dataset")
    print("Provided by Mayo Clinic (512x512 Resolution, 3mm Slice Thickness)")
    print("=" * 60)
    print("WARNING: Direct open download is restricted by Medical Data Use Agreements.")
    print("To execute this pipeline, please download the LDCT-and-Projection-data from TCIA:")
    print("URL: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026")
    print("\nExpected Directory Structure Once Downloaded manually or via NBIA Retriever:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── L001/  (Patient 1 - DICOM slices)")
    print(f"  │   ├── ...")
    print(f"  │   └── L009/  (Patient 9)")
    print(f"  └── eval/")
    print(f"      └── L010/  (Patient 10 for Final Metrics)")
    print("\nOnce populated, the dataset_aapm.py script will handle the 512x512 Numpy conversion automatically.")
    print("Running a mock dataset generation just to stand up the filesystem for the models...")

    # For compilation/scripting validation, generate a few placeholder NumPy files if empty
    import numpy as np
    
    # 9 Training patients, simulating 216.8 slices on average (~2168 total)
    # We will just generate 2 dummy slices per patient so you can compile and dry-run without blowing up your disk with 200GB.
    print("\nGenerating mock structural files (512x512) to validate the pipeline architecture...")
    for i in range(1, 10):
        pat_dir = os.path.join(output_dir, "train", f"L{i:03d}")
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
