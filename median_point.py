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

        

        image_8bits_segmentation = cv2.normalize(axial_slice_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits_segmentation = cv2.resize(image_8bits_segmentation, (640, 640))

        # -------------------------- Slice --------------------------

        image_8bits = cv2.GaussianBlur(image_8bits, (5, 5), 0)

        # Selecting only the region of the brain
        mask_non_zero_region = np.where(image_8bits == 0, 0, 1)
        mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        non_uniform_region = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        # Histogram Equalization
        hist, bins = np.histogram(pixels, 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
        image_equalized = cdf_final[image_8bits]
        
        # Selecting only the region of the brain
        mask_non_zero_region = np.where(image_equalized == 0, 0, 1)
        mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        non_uniform_region = cv2.bitwise_and(image_equalized, image_equalized, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        # Applying Otsu's Thresholding
        threshold = round(dip.otsuThresholding(pixels))

        mask = np.where(image_8bits >= threshold, 255, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Applying Morphological Operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # -------------------------- Slice Segmentation --------------------------

        image_8bits_segmentation = np.where(image_8bits_segmentation > 0, 255, 0)
        image_8bits_segmentation = cv2.normalize(image_8bits_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        slices.append(mask)
        slices_segmentation.append(image_8bits_segmentation)