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

    if i == 77:

        axial_slice = nii_data[:, :, i]
        axial_slice_segmentation = nii_segmentation_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        image_8bits_segmentation = cv2.normalize(axial_slice_segmentation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits_segmentation = cv2.resize(image_8bits_segmentation, (640, 640))

        slices.append(image_8bits)
        slices_segmentation.append(image_8bits_segmentation)

        mask = np.where(image_8bits == 0, 0, 1)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        non_uniform_region = cv2.bitwise_and(image_8bits, image_8bits, mask=mask)
        pixels = non_uniform_region[mask == 255]

        # unique, counts = np.unique(pixels, return_counts=True)
        # plt.bar(unique, counts)
        # plt.show()

        cv2.imshow('image_8bits', image_8bits)
        cv2.imshow('image_8bits_segmentation', image_8bits_segmentation)

        threshold = round(dip.otsuThresholding(pixels))

        mask = np.where(image_8bits >= threshold, 255, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow('mask', mask)
        
        cv2.waitKey(0)  

# max_matrix_maximum = np.maximum.reduce(slices)
# max_matrix_minimum = np.minimum.reduce(slices)

# slices_array = np.array(slices)
# mean_matrix = np.mean(slices_array, axis=0).astype(np.uint8)

# sliced_image = np.zeros_like(max_matrix_maximum)
# sliced_image[(max_matrix_maximum <= 170) & (max_matrix_maximum > 0)] = 255

# cv2.imshow('max_matrix_maximum', max_matrix_maximum)
# cv2.imshow('sliced_image', sliced_image)
# cv2.imshow('mean_matrix', mean_matrix)

# mask = np.where(max_matrix_maximum == 0, 0, 1)
# mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# non_uniform_region = cv2.bitwise_and(max_matrix_maximum, max_matrix_maximum, mask=mask)
# pixels = non_uniform_region[mask == 255]


# hist, bins = np.histogram(pixels, 256, [0, 256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# cdf_masked = np.ma.masked_equal(cdf, 0)
# cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
# cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
# image_equalized = cdf_final[max_matrix_maximum]

# # unique, counts = np.unique(pixels, return_counts=True)
# # plt.bar(unique, counts)
# # plt.show()


# cv2.imshow('image_equalized', image_equalized)

cv2.waitKey(0)  

