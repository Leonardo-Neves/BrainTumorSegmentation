import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
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


def a(slices):
    mask_mean = np.mean(slices, axis=0).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean == 0, 0, 1)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
    pixels = non_uniform_region[mask_non_zero_region == 255]

    # threshold = round(dip.otsuThresholding(pixels))

    # mask_result = np.where(mask_mean >= threshold, 255, 0)
    # mask_result = cv2.normalize(mask_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imshow('mask_mean', mask_mean)

    # Find the pick of intensity in the mean mask
    hot_point_mask = np.where(mask_mean == np.max(mask_mean), 255, 0)
    hot_point_mask = cv2.normalize(hot_point_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imshow('hot_point_mask', hot_point_mask)

    contours, _ = cv2.findContours(hot_point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawed_contours = np.zeros_like(hot_point_mask)
    drawed_contours = cv2.cvtColor(drawed_contours, cv2.COLOR_GRAY2BGR)

    centroids = []

    for countour in contours:
        M = cv2.moments(countour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        if cX != 0 and cY != 0:
            centroids.append([cX, cY])
        
    centroids_dataframe = pd.DataFrame(centroids, columns=['X', 'Y'])

    mean_x = int(centroids_dataframe['X'].mean())
    mean_y = int(centroids_dataframe['Y'].mean())


    # cv2.drawContours(drawed_contours, [countour], -1, (255, 255, 255), -1)

    # unique, counts = np.unique(pixels, return_counts=True)
    # plt.bar(unique, counts)
    # plt.show()

    cv2.imshow('drawed_contours', drawed_contours)

    cv2.waitKey(0)

def b(slices, FILTER_SIZE = 7):

    SIDE_FILTER = (FILTER_SIZE - 1) / 2

    groups = []

    for i in range(0, len(slices)):

        filter_left_side = [slices[i - j] for j in range(1, int(SIDE_FILTER) + 1) if (i - j) >= 0 and (i - j) <= len(slices)]
        filter_right_side = [slices[i + j] for j in range(1, int(SIDE_FILTER) + 1) if (i + j) >= 0 and (i + j) < len(slices)]

        groups.append(filter_left_side[::-1] + [slices[i]] + filter_right_side)

    for group in groups:
        mask_mean = np.mean(group, axis=0).astype(np.uint8)

        mask_non_zero_region = np.where(mask_mean == 0, 0, 1)
        mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        threshold = round(dip.otsuThresholding(pixels))

        mask_result = np.where(mask_mean >= threshold, 255, 0)
        mask_result = cv2.normalize(mask_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        cv2.imshow('mask_mean', mask_mean)
        cv2.imshow('mask_result', mask_result)

        cv2.waitKey(0)

def c(slices, FILTER_SIZE = 5):
    SIDE_FILTER = (FILTER_SIZE - 1) / 2

    groups = []

    for i in range(0, len(slices)):

        filter_left_side = [slices[i - j] for j in range(1, int(SIDE_FILTER) + 1) if (i - j) >= 0 and (i - j) <= len(slices)]
        filter_right_side = [slices[i + j] for j in range(1, int(SIDE_FILTER) + 1) if (i + j) >= 0 and (i + j) < len(slices)]

        groups.append(filter_left_side[::-1] + [slices[i]] + filter_right_side)

    for group in groups:
        mask_maximum = np.maximum.reduce(group)
        mask_minimum = np.minimum.reduce(group)
        
        cv2.imshow('mask_maximum', mask_maximum)
        cv2.imshow('mask_minimum', mask_minimum)
    
        cv2.waitKey(0)


a(slices)

# mask_mean, mask_result = a(slices)

# mask_mean_segmentation, mask_result_segmentation = a(slices_segmentation)

# cv2.imshow('mask_mean', mask_mean)
# cv2.imshow('mask_result', mask_result)

# cv2.imshow('mask_mean_segmentation', mask_mean_segmentation)
# cv2.imshow('mask_result_segmentation', mask_result_segmentation)

# cv2.waitKey(0)  

