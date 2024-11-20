import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing
from utils.frequency_domain import FrequencyDomain
from utils.spartial_domain import SpartialDomain

dip = DigitalImageProcessing()
fd = FrequencyDomain()
sd = SpartialDomain()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_003\BraTS20_Training_003_t1ce.nii"

def coronalSegmentation(image_path):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_coronal_border3.png', cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    cutoff_frequency = 40
    c = 1

    for i in range(nii_data.shape[1]):

        if i >= 150 and i <= 170: # BraTS20_Training_003_t1ce.nii
            axial_slice = nii_data[:, i, :]

            image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))

            # Gaussian High-Pass Filter
            padded_image = fd.padImage(image_8bits)
            H_u_v = fd.gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
            mask_gaussian, F_u_v = fd.filterImage(image_8bits, padded_image, H_u_v)
            image_8bits_filtered_gaussian = image_8bits + (c * mask_gaussian)
            image_8bits_filtered_gaussian = cv2.normalize(image_8bits_filtered_gaussian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Adaptive Histogram Equalization
            image_8bits_ahe = clahe.apply(image_8bits_filtered_gaussian)

            # Segmentation using Edge Detection
            sobel_x = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobel_x, sobel_y)
            sobel_combined = np.uint8(np.absolute(sobel_combined))

            _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

            # Close gaps
            kernel = np.ones((3, 3), np.uint8)
            closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            mask_non_zero_region = np.where(sobel_combined > 0, 255, 0)
            mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_non_zero_region = np.zeros_like(sobel_combined)
            cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

            region_of_interest = cv2.bitwise_and(sobel_combined, sobel_combined, mask=mask_non_zero_region)
            roi_values = region_of_interest[region_of_interest > 0]

            global_mean = np.mean(roi_values)

            _, mask = cv2.threshold(sobel_combined, global_mean, 255, cv2.THRESH_BINARY)

            processed_images.append(mask)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_mean_border = np.where(mask_mean_border > 0, 512, 0)

    mask_mean_without_border = mask_mean - mask_mean_border

    mask_mean_without_border = np.where(mask_mean_without_border < 0, 0, mask_mean_without_border).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean_without_border > 0, 255, 0)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    region_of_interest = cv2.bitwise_and(mask_mean_without_border, mask_mean_without_border, mask=mask_non_zero_region)
    roi_values = region_of_interest[region_of_interest > 0]

    masks = []

    for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        masks.append(sd.leoThreshold(mask_mean_without_border, mask_non_zero_region, i))

    masks = np.stack(masks)
    frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    for mask in masks:
        frequency_matrix += (mask == 255).astype(int)

    mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 2), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    erosion = cv2.erode(close, kernel, iterations = 1)

    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return dilation, mask_mean

def axialSegmentation(image_path):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_axial_border2.png', cv2.IMREAD_GRAYSCALE)

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    for i in range(nii_data.shape[2]):

        if i >= 61 and i <= 78: # BraTS20_Training_003_t1ce.nii
            axial_slice = nii_data[:, :, i]

            image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))

            # Gaussian High-Pass Filter
            padded_image = fd.padImage(image_8bits)
            H_u_v = fd.gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
            mask_gaussian, F_u_v = fd.filterImage(image_8bits, padded_image, H_u_v)
            image_8bits_filtered_gaussian = image_8bits + (c * mask_gaussian)
            image_8bits_filtered_gaussian = cv2.normalize(image_8bits_filtered_gaussian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Adaptive Histogram Equalization
            image_8bits_ahe = clahe.apply(image_8bits_filtered_gaussian)

            # Segmentation using Edge Detection
            sobel_x = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobel_x, sobel_y)
            sobel_combined = np.uint8(np.absolute(sobel_combined))

            _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

            # Close gaps
            kernel = np.ones((3, 3), np.uint8)
            closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            mask_non_zero_region = np.where(sobel_combined > 0, 255, 0)
            mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_non_zero_region = np.zeros_like(sobel_combined)
            cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

            region_of_interest = cv2.bitwise_and(sobel_combined, sobel_combined, mask=mask_non_zero_region)
            roi_values = region_of_interest[region_of_interest > 0]

            global_mean = np.mean(roi_values)

            _, mask = cv2.threshold(sobel_combined, global_mean, 255, cv2.THRESH_BINARY)

            processed_images.append(mask)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_mean_border = np.where(mask_mean_border > 0, 512, 0)

    mask_mean_without_border = mask_mean - mask_mean_border

    mask_mean_without_border = np.where(mask_mean_without_border < 0, 0, mask_mean_without_border).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean_without_border > 0, 255, 0)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    region_of_interest = cv2.bitwise_and(mask_mean_without_border, mask_mean_without_border, mask=mask_non_zero_region)
    roi_values = region_of_interest[region_of_interest > 0]

    masks = []

    for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        masks.append(sd.leoThreshold(mask_mean_without_border, mask_non_zero_region, i))

    masks = np.stack(masks)
    frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    for mask in masks:
        frequency_matrix += (mask == 255).astype(int)

    mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 2), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    erosion = cv2.erode(close, kernel, iterations = 1)

    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return dilation, mask_mean

def sagittalSegmentation(image_path):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_sagittal_border.png', cv2.IMREAD_GRAYSCALE)

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    for i in range(nii_data.shape[0]):

        if i >= 160 and i <= 178: # BraTS20_Training_003_t1ce.nii
            axial_slice = nii_data[i, :, :]

            image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))

            # Gaussian High-Pass Filter
            padded_image = fd.padImage(image_8bits)
            H_u_v = fd.gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
            mask_gaussian, F_u_v = fd.filterImage(image_8bits, padded_image, H_u_v)
            image_8bits_filtered_gaussian = image_8bits + (c * mask_gaussian)
            image_8bits_filtered_gaussian = cv2.normalize(image_8bits_filtered_gaussian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Adaptive Histogram Equalization
            image_8bits_ahe = clahe.apply(image_8bits_filtered_gaussian)

            # Segmentation using Edge Detection
            sobel_x = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobel_x, sobel_y)
            sobel_combined = np.uint8(np.absolute(sobel_combined))

            _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

            # Close gaps
            kernel = np.ones((3, 3), np.uint8)
            closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            mask_non_zero_region = np.where(sobel_combined > 0, 255, 0)
            mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_non_zero_region = np.zeros_like(sobel_combined)
            cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

            region_of_interest = cv2.bitwise_and(sobel_combined, sobel_combined, mask=mask_non_zero_region)
            roi_values = region_of_interest[region_of_interest > 0]

            global_mean = np.mean(roi_values)

            _, mask = cv2.threshold(sobel_combined, global_mean, 255, cv2.THRESH_BINARY)

            processed_images.append(mask)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_mean_border = np.where(mask_mean_border > 0, 512, 0)

    mask_mean_without_border = mask_mean - mask_mean_border

    mask_mean_without_border = np.where(mask_mean_without_border < 0, 0, mask_mean_without_border).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean_without_border > 0, 255, 0)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    region_of_interest = cv2.bitwise_and(mask_mean_without_border, mask_mean_without_border, mask=mask_non_zero_region)
    roi_values = region_of_interest[region_of_interest > 0]

    masks = []

    for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        masks.append(sd.leoThreshold(mask_mean_without_border, mask_non_zero_region, i))

    masks = np.stack(masks)
    frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    for mask in masks:
        frequency_matrix += (mask == 255).astype(int)

    mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 2), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    erosion = cv2.erode(close, kernel, iterations = 1)

    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return dilation, mask_mean

mask_segmentation_coronal, mask_mean_coronal = coronalSegmentation(image_path)

mask_segmentation_axial, mask_mean_axial = axialSegmentation(image_path)

mask_segmentation_sagittal, mask_mean_sagittal = sagittalSegmentation(image_path)

# cv2.imshow('mask_segmentation_coronal', mask_segmentation_coronal)
# cv2.imshow('mask_segmentation_axial', mask_segmentation_axial)
# cv2.imshow('mask_segmentation_sagittal', mask_segmentation_sagittal)

# cv2.waitKey(0)

contours_coronal, _ = cv2.findContours(mask_segmentation_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_axial, _ = cv2.findContours(mask_segmentation_axial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_sagittal, _ = cv2.findContours(mask_segmentation_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

similarities = []

for i, contour_coronal in enumerate(contours_coronal):
    for j, contour_sagittal in enumerate(contours_sagittal):
        try:
            area_contour_coronal = cv2.contourArea(contour_coronal)
            area_contour_sagittal = cv2.contourArea(contour_sagittal)

            similarity = cv2.matchShapes(contour_coronal, contour_sagittal, cv2.CONTOURS_MATCH_I1, 0.0)

            # Euclidean distance between two contours
            M_coronal = cv2.moments(contour_coronal)
            centroid_x_coronal, centroid_y_coronal = 0, 0
            if M_coronal["m00"] != 0:
                centroid_x_coronal = int(M_coronal["m10"] / M_coronal["m00"])
                centroid_y_coronal = int(M_coronal["m01"] / M_coronal["m00"])

            M_axial = cv2.moments(contour_sagittal)
            centroid_x_sagittal, centroid_y_sagittal = 0, 0
            if M_axial["m00"] != 0:
                centroid_x_sagittal = int(M_axial["m10"] / M_axial["m00"])
                centroid_y_sagittal = int(M_axial["m01"] / M_axial["m00"])

            point1 = np.array([centroid_x_coronal, centroid_y_coronal])
            point2 = np.array([centroid_x_sagittal, centroid_y_sagittal])

            distance = np.abs(np.linalg.norm(point1 - point2))

            similarities.append([similarity, i, j, area_contour_coronal, area_contour_sagittal, distance, centroid_x_coronal, centroid_y_coronal, centroid_x_sagittal, centroid_y_sagittal])

            # if i == 353 and j == 212:
            # if i == 353 and j == 71: coronal - sagittal
        except:
            pass


# ----------------------- Filtering contours between coronal and sagittal -----------------------

dataframe_similarity_coronal_axial = pd.DataFrame(similarities, columns=['Similarity', 'Coronal', 'Sagittal', 'Coronal_Area', 'Sagittal_Area', 'Euclidian_Distance', 'Coronal_X', 'Coronal_Y', 'Sagittal_X', 'Sagittal_Y'])

# Filter by max area
sorted_df = dataframe_similarity_coronal_axial.sort_values(by=['Coronal_Area', 'Sagittal_Area'], ascending=False)
first_rows = sorted_df.head(10)

# Filter by differences between areas
first_rows['Difference_Areas'] = np.where(first_rows['Coronal_Area'] >= first_rows['Sagittal_Area'], first_rows['Coronal_Area'] / first_rows['Sagittal_Area'], first_rows['Sagittal_Area'] / first_rows['Coronal_Area'])
first_rows = first_rows[first_rows['Difference_Areas'] <= 5]

# Filter by distance
filtered_df_coronal_sagittal = first_rows[(first_rows['Euclidian_Distance'] == first_rows['Euclidian_Distance'].min()) & (first_rows['Similarity'] == first_rows['Similarity'].min())]

# ----------------------- Filtering contours between coronal and axial -----------------------

filtered_df_coronal_axial = None

if not filtered_df_coronal_sagittal.empty:
    
    first_row = filtered_df_coronal_sagittal.iloc[0]

    point2 = np.array([first_row['Coronal_X'], first_row['Coronal_Y']])

    caracteristics = []

    for i, contour_axial in enumerate(contours_axial):
        try:
            # Euclidean distance between two contours
            M_axial = cv2.moments(contour_axial)
            centroid_x_axial, centroid_y_axial = 0, 0
            if M_coronal["m00"] != 0:
                centroid_x_axial = int(M_axial["m10"] / M_axial["m00"])
                centroid_y_axial = int(M_axial["m01"] / M_axial["m00"])

            if centroid_x_axial != 0 and centroid_y_axial != 0:

                point1 = np.array([centroid_x_axial, centroid_y_axial])

                delta = point2 - point1
                angle_radians = np.arctan2(delta[1], delta[0])
                angle_degrees = np.degrees(angle_radians)
                angle_degrees = angle_degrees % 360

                area_contour_axial = cv2.contourArea(contour_axial)
                area_contour_coronal = cv2.contourArea(contours_coronal[int(first_row['Coronal'])])

                similarity = cv2.matchShapes(contours_coronal[int(first_row['Coronal'])], contour_axial, cv2.CONTOURS_MATCH_I1, 0.0)

                caracteristics.append([int(first_row['Coronal']), i, angle_degrees, centroid_x_axial, centroid_y_axial, similarity, area_contour_axial, area_contour_coronal])
        except Exception as e:
            # print(e)
            pass

    dataframe_angle_axial_coronal = pd.DataFrame(caracteristics, columns=['Coronal', 'Axial', 'Angle', 'Axial_X', 'Axial_Y', 'Similarity', 'Axial_Area', 'Coronal_Area'])

    # Filter by max area
    sorted_df = dataframe_angle_axial_coronal.sort_values(by=['Coronal_Area', 'Axial_Area'], ascending=False)
    first_rows = sorted_df.head(10)

    # Filter by differences between areas
    first_rows['Difference_Areas'] = np.where(first_rows['Coronal_Area'] >= first_rows['Axial_Area'], first_rows['Coronal_Area'] / first_rows['Axial_Area'], first_rows['Axial_Area'] / first_rows['Coronal_Area'])
    first_rows = first_rows[first_rows['Difference_Areas'] <= 5]

    # Filter by angle
    filtered_df_coronal_axial = first_rows[(first_rows['Angle'] >= 85) & (first_rows['Angle'] <= 95)]

    # Adjustment on the shape of the countour
    first_row_filtered_df_coronal_sagittal = filtered_df_coronal_sagittal.iloc[0]
    first_row_filtered_df_coronal_axial = filtered_df_coronal_axial.iloc[0]

    kernel = np.ones((3, 3), np.uint8)

    mask_coronal = np.zeros_like(mask_mean_coronal)
    cv2.drawContours(mask_coronal, [contours_coronal[int(first_row_filtered_df_coronal_sagittal['Coronal'])]], -1, 255, -1)
    mask_coronal = cv2.morphologyEx(mask_coronal, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_coronal, contours, -1, 255, -1)

    mask_axial = np.zeros_like(mask_mean_axial)
    cv2.drawContours(mask_axial, [contours_axial[int(first_row_filtered_df_coronal_axial['Axial'])]], -1, 255, -1)
    mask_axial = cv2.morphologyEx(mask_axial, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_axial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_axial, contours, -1, 255, -1)

    mask_sagittal = np.zeros_like(mask_mean_sagittal)
    cv2.drawContours(mask_sagittal, [contours_sagittal[int(first_row_filtered_df_coronal_sagittal['Sagittal'])]], -1, 255, -1)
    mask_sagittal = cv2.morphologyEx(mask_sagittal, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_sagittal, contours, -1, 255, -1)

    cv2.imshow('mask_coronal', mask_coronal)

    cv2.imshow('mask_axial', mask_axial)

    cv2.imshow('mask_sagittal', mask_sagittal)

    cv2.waitKey(0)

    


  