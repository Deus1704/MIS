import os
import numpy as np
import pydicom
from glob import glob

def convert_dicom_to_numpy(root_dir):
    print(f"Scanning {root_dir} for DICOM (.dcm) files...")
    dcm_files = glob(os.path.join(root_dir, '**', '*.dcm'), recursive=True)
    
    if not dcm_files:
        print("No .dcm files found. Already converted?")
        return

    print(f"Found {len(dcm_files)} DICOM slices. Converting to .npy...")
    for idx, dcm_path in enumerate(dcm_files):
        try:
            dicom = pydicom.dcmread(dcm_path)
            # The pixel array is typically 512x512
            image_array = dicom.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units (HU) optimally
            intercept = getattr(dicom, 'RescaleIntercept', 0)
            slope = getattr(dicom, 'RescaleSlope', 1)
            image_array = image_array * slope + intercept

            npy_path = dcm_path.replace('.dcm', '.npy')
            np.save(npy_path, image_array)
            
            # Clean up the original DCM to save disk space
            os.remove(dcm_path)
            
            if (idx + 1) % 500 == 0:
                print(f"  Converted {idx+1}/{len(dcm_files)} slices...")
                
        except Exception as e:
            print(f"Failed to convert {dcm_path}: {e}")

    print("DICOM to NumPy Fast-Conversion Complete!")

if __name__ == "__main__":
    convert_dicom_to_numpy("/teamspace/studios/this_studio/real_data/aapm_ldct")
