import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii"
image_path_segmetation = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

nii_segmentation_file = nib.load(image_path_segmetation)
nii_segmentation_data = nii_segmentation_file.get_fdata()


slices = []
slices_segmentation = []

for i in range(nii_data.shape[2]):

    if i >= 30 and i <= 112:
        axial_slice = nii_data[:, :, i]
        axial_slice_segmentation = nii_segmentation_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        image_8bits = cv2.GaussianBlur(image_8bits, (5, 5), 0)

        image_8bits_segmentation = cv2.normalize(axial_slice_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits_segmentation = cv2.resize(image_8bits_segmentation, (640, 640))

        # Slice
        mask_non_zero_region = np.where(image_8bits == 0, 0, 1)
        mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        non_uniform_region = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        threshold = round(dip.otsuThresholding(pixels))

        mask = np.where(image_8bits >= threshold, 255, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Slice Segmentation

        image_8bits_segmentation = np.where(image_8bits_segmentation > 0, 255, 0)
        image_8bits_segmentation = cv2.normalize(image_8bits_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        slices.append(mask)
        slices_segmentation.append(image_8bits_segmentation)


def t(slices):
    mask_mean = np.mean(slices, axis=0).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean == 0, 0, 1)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
    pixels = non_uniform_region[mask_non_zero_region == 255]

    threshold = round(dip.otsuThresholding(pixels))

    mask_result = np.where(mask_mean >= threshold, 255, 0)
    # return cv2.normalize(mask_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mask_mean

cv2.imshow('result', t(slices))
cv2.imshow('seg', t(slices_segmentation))


# cv2.imshow('dif', t(slices_segmentation) - t(slices))

cv2.waitKey(0)  

