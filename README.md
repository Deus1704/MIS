# MIS Project: High-Fidelity CT Reconstruction

This repository contains two parallel approaches for CT image reconstruction from raw sinogram data:
- **Path A**: Learned Primal-Dual (LPD)
- **Path B**: FreqHybridNet (Pragmatic Backup)

## ⚠️ Data Setup Instructions

To keep this repository lightweight and respect Medical Data Use Agreements, **no dataset files (.npz, .npy) are stored in Git.** You must download them locally before running the models.

### 1. AAPM Low Dose CT Grand Challenge (Mayo Clinic Data)
Required for high-fidelity 512x512 testing in Path A.

To avoid massive file sizes (862GB), we are using the **1.8GB Phantom Object**:
1. Go to [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026).
2. Click **DOWNLOAD (1.8GB)** right next to "Images Phantom Object Only".
3. Open the downloaded file in the TCIA Data Retriever.
4. Convert the DICOM files to `.npy` arrays and place them in:
   - `real_data/aapm_ldct/train/Phantom_Train/`
   - `real_data/aapm_ldct/eval/Phantom_Eval/`

Run the prep script to build these folders:
```bash
python Path_A_LPD/download_aapm.py
```

### 2. OrganAMNIST (Real CT slices)
Required for the fast training loops and Path B's FreqHybridNet pipeline.
1. Download the `organamnist.npz` file from the published MedMNIST / OrganAMNIST links.
2. Place the file precisely at: `Path_B_FreqHybridNet/real_data/organamnist/raw/organamnist.npz`
