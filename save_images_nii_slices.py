import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
import os

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii"
image_path_segmetation = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"

OUTPUT_PATH = r'C:\Users\leosn\Desktop\PIM\test_folder'

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

nii_segmentation_file = nib.load(image_path_segmetation)
nii_segmentation_data = nii_segmentation_file.get_fdata()

for i in range(nii_data.shape[2]):

    axial_slice = nii_data[:, :, i]
    axial_slice_segmentation = nii_segmentation_data[:, :, i]

    image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits = cv2.resize(image_8bits, (640, 640))

    image_8bits_segmentation = cv2.normalize(axial_slice_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits_segmentation = cv2.resize(image_8bits_segmentation, (640, 640))

    if np.mean(image_8bits_segmentation) != 0:
        cv2.imwrite(os.path.join(OUTPUT_PATH, f'{i}.png'), image_8bits)