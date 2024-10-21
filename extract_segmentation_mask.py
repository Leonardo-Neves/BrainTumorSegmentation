import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

from utils.digital_image_processing import DigitalImageProcessing


image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_003\BraTS20_Training_003_t1ce.nii"

image_segmentation_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_003\BraTS20_Training_003_seg.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

for i in range(nii_data.shape[1]):
    axial_slice = nii_data[:, i, :]

    axial_slice_8 = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axial_slice_8 = cv2.resize(axial_slice_8, (640, 640))

    plt.imsave(f'test_folder/slice2/slice_{i}.png', axial_slice_8, cmap='gray')

nii_segmentation_file = nib.load(image_segmentation_path)
nii_segmentation_data = nii_segmentation_file.get_fdata()

for i in range(nii_segmentation_data.shape[1]):
    axial_slice = nii_segmentation_data[:, i, :]

    axial_slice_8 = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axial_slice_8 = cv2.resize(axial_slice_8, (640, 640))

    axial_slice_8 = np.where(axial_slice_8 > 0, 255, 0)

    plt.imsave(f'test_folder/slice_segmentation2/slice_{i}.png', axial_slice_8, cmap='gray')

