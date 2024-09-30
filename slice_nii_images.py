import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

ROOT_PATH = 'datasets/MICCAI_BraTS_2019_Data_Training'

OUTPUT_PATH = 'datasets/MICCAI_BraTS_2019_Data_Training/HGG_PNG'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def normalize_slice(slice_data):
    slice_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    return slice_normalized.astype(np.uint8)

for folder_name in os.listdir(os.path.join(ROOT_PATH, 'HGG')):

    nii_file = nib.load(os.path.join(ROOT_PATH, 'HGG', folder_name, f'{folder_name}_t1ce.nii'))
    nii_data = nii_file.get_fdata()

    os.makedirs(os.path.join(OUTPUT_PATH, folder_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'axial'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'coronal'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'sagittal'), exist_ok=True)

    j = 0

    for i in range(nii_data.shape[2]):
        axial_slice = nii_data[:, :, i]
        axial_slice_normalized = normalize_slice(axial_slice)

        axial_slice_normalized_resized = cv2.resize(axial_slice_normalized, (640, 640))

        if np.mean(axial_slice_normalized_resized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'axial/axial_slice_{j}.png'), axial_slice_normalized_resized, cmap='gray')
            j += 1

    j = 0

    for i in range(nii_data.shape[1]):
        coronal_slice = nii_data[:, i, :]
        coronal_slice_normalized = normalize_slice(coronal_slice)

        coronal_slice_normalized_resized = cv2.resize(coronal_slice_normalized, (640, 640))

        if np.mean(coronal_slice_normalized_resized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'coronal/coronal_slice_{j}.png'), coronal_slice_normalized_resized, cmap='gray')
            j += 1

    j = 0

    for i in range(nii_data.shape[0]):
        sagittal_slice = nii_data[i, :, :]
        sagittal_slice_normalized = normalize_slice(sagittal_slice)

        sagittal_slice_normalized_resized = cv2.resize(sagittal_slice_normalized, (640, 640))

        if np.mean(sagittal_slice_normalized_resized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'sagittal/sagittal_slice_{j}.png'), sagittal_slice_normalized_resized, cmap='gray')
            j += 1

print("All slices saved successfully!")
