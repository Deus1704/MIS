# MIS Project: High-Fidelity CT Reconstruction

This repository contains two parallel approaches for CT image reconstruction from raw sinogram data:
- **Path A**: Learned Primal-Dual (LPD)
- **Path B**: FreqHybridNet (Pragmatic Backup)

## ⚠️ Data Setup Instructions

To keep this repository lightweight and respect Medical Data Use Agreements, **no dataset files (.npz, .npy) are stored in Git.** You must download them locally before running the models.

### 1. AAPM Low Dose CT Grand Challenge (Mayo Clinic Data)
Required for high-fidelity 512x512 testing in Path A.

1. Register for an account and sign the Mayo Clinic Data Use Agreement on [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026).
2. Run the preparation script to stand up the correct directory structure:
   ```bash
   python Path_A_LPD/download_aapm.py
   ```
3. Use the NBIA Data Retriever or REST API to pull ONLY the `L001`, `L002`, `L003` (Train) and `L010` (Eval) volumetric slices to keep the size under 50GB and place the numpy arrays within `real_data/aapm_ldct/`.

### 2. OrganAMNIST (Real CT slices)
Required for the fast training loops and Path B's FreqHybridNet pipeline.
1. Download the `organamnist.npz` file from the published MedMNIST / OrganAMNIST links.
2. Place the file precisely at: `Path_B_FreqHybridNet/real_data/organamnist/raw/organamnist.npz`

Once the data is populated, the `dataset_aapm.py` and `dataset.py` DataLoaders will automatically detect and cache it for execution.
