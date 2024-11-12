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
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        slices.append(mask)
    
mask_mean = np.mean(slices, axis=0).astype(np.uint8)

# Selecting only the region of the brain
mask_non_zero_region = np.where(mask_mean == 0, 0, 1)
mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
pixels = non_uniform_region[mask_non_zero_region == 255]

print('mean: ', np.mean(pixels))

# cv2.imshow('mask_mean', mask_mean)

# dip.plotImage3D(mask_mean)

# Find the pick of intensity in the mean mask
hot_point_mask = np.where(mask_mean == np.max(mask_mean), 255, 0)
hot_point_mask = cv2.normalize(hot_point_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Finding the median point
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

mean_centroid_x = int(centroids_dataframe['X'].mean())
mean_centroid_y = int(centroids_dataframe['Y'].mean())

print('mean_centroid_x: ', mean_centroid_x)
print('mean_centroid_y: ', mean_centroid_y)

# Filtering the countours using the median point

contours_drawn = []

for slice in slices:

    # cv2.imshow('slice', slice)

    contours_slice, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_distance = 0

    best_countour = None

    for countour in contours_slice:
        M = cv2.moments(countour)

        centroid_x, centroid_y = 0, 0

        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

        if centroid_x != 0 and centroid_y != 0:

            point1 = np.array([centroid_x, centroid_y])
            point2 = np.array([mean_centroid_x, mean_centroid_y])

            # Euclidean distance
            distance = np.abs(np.linalg.norm(point1 - point2))

            if distance < min_distance or min_distance == 0:
                min_distance = distance
                best_countour = countour

    drawed_contours = np.zeros_like(slice)
    # drawed_contours = cv2.cvtColor(drawed_contours, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(drawed_contours, [best_countour], -1, 255, -1)
    # cv2.circle(drawed_contours, (mean_centroid_x, mean_centroid_y), 5, (0, 255, 0), -1)

    contours_drawn.append(drawed_contours)


# ---------------------- Calculate the standard deviation of pixels between two points ----------------------

contours_drawn_maximum = np.maximum.reduce(contours_drawn)
# contours_drawn_maximum = cv2.normalize(contours_drawn_maximum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

contours_drawn_maximum_countours = dip.countoursInMask(contours_drawn_maximum)

contours_drawn_maximum_countour_drawn = np.zeros_like(contours_drawn_maximum)
cv2.drawContours(contours_drawn_maximum_countour_drawn, [contours_drawn_maximum_countours[0]], -1, 255, 1)

# Get the position of each pixel which is 255 intensity of the silhouette of the mask_mean
positions = np.column_stack(np.where(contours_drawn_maximum_countour_drawn == 255))

result = []

mask_positions = np.zeros_like(mask_mean)

for position in positions:
    pixel_intensities = dip.pixelsIntensityImaginaryLineBetweenTwoPoints(mask_mean, [mean_centroid_x, mean_centroid_y], position)
    pixel_intensities = [x for x in pixel_intensities if x != 0]
    
    mean = np.mean(pixel_intensities)
    std = np.std(pixel_intensities)

    # print(f'{position} - Mean: {np.mean(pixel_intensities)} Standard Deviation: {np.std(pixel_intensities)}', )

    if mean and std:
        mask_positions = dip.applyIntensityBelowImaginaryLineBetweenTwoPoints(mask_mean, mask_positions, 255, mean, std, [mean_centroid_x, mean_centroid_y], position)

    # result.append([position[0], position[1], f'{np.mean(pixel_intensities)}'.replace('.', ','), f'{np.std(pixel_intensities)}'.replace('.', ',')])


# result_dataframe = pd.DataFrame(result, columns=['X', 'Y', 'Mean', 'Standard Deviation'])

# result_dataframe.to_csv('result.csv', index=False, sep=';')

cv2.imshow('mask_positions', mask_positions)

cv2.waitKey(0)

            