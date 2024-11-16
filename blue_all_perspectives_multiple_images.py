import matplotlib.pyplot as plt
from ultralytics import YOLO
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

model = YOLO(r'C:\Users\leosn\Desktop\PIM\runs\obb\train11\weights\best.pt')

def coronalSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_coronal_border3.png', cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    processed_images_leo_threshold = []

    cutoff_frequency = 40
    c = 1

    for i in range(nii_data.shape[1]):

        if i >= initial_index and i <= end_index: # BraTS20_Training_003_t1ce.nii
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

            # Leo Thresholding
            mask_non_zero_region = np.where(image_8bits_ahe > 0, 255, 0).astype(np.uint8)

            masks = []

            for i in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37]:
                masks.append(sd.leoThreshold2(image_8bits_ahe, mask_non_zero_region, i))

            mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)

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

            processed_images_leo_threshold.append(mask_mean_leo_threshold)

    mask_mean_leo = np.mean(processed_images_leo_threshold, axis=0).astype(np.uint8)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    gradient = cv2.morphologyEx(mask_non_zero_region, cv2.MORPH_GRADIENT, kernel)
    diff = cv2.normalize(mask_non_zero_region - gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_mean_without_border = cv2.bitwise_and(mask_mean, mask_mean, mask=diff)

    masks = []

    for i in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37]:
    # for i in [3, 7, 15, 19, 23, 29, 35, 41, 47, 53]:
    # for i in [3, 5, 7, 9, 11, 13]:
        masks.append(sd.leoThreshold2(mask_mean_without_border, mask_non_zero_region, i))

    mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)

    ret3, mask_otsu = cv2.threshold(mask_mean_leo_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('mask_mean_leo_threshold', mask_mean_leo_threshold)

    # masks = np.stack(masks)
    # frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    # for mask in masks:
    #     frequency_matrix += (mask == 255).astype(int)

    # plt.imshow(frequency_matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar(label="Frequency of 255")
    # plt.title("Frequency Distribution of Pixel Value 255")
    

    # mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 4), 255, 0).astype(np.uint8)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    # erosion = cv2.erode(close, kernel, iterations = 1)

    # dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return mask_otsu, mask_mean, mask_mean_leo

def axialSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_axial_border2.png', cv2.IMREAD_GRAYSCALE)

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    processed_images_leo_threshold = []

    for i in range(nii_data.shape[2]):

        if i >= initial_index and i <= end_index: # BraTS20_Training_003_t1ce.nii
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

            # Leo Thresholding
            mask_non_zero_region = np.where(image_8bits_ahe > 0, 255, 0).astype(np.uint8)

            masks = []

            for i in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37]:
                masks.append(sd.leoThreshold2(image_8bits_ahe, mask_non_zero_region, i))

            mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)

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

            processed_images_leo_threshold.append(mask_mean_leo_threshold)

    mask_mean_leo = np.mean(processed_images_leo_threshold, axis=0).astype(np.uint8)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    gradient = cv2.morphologyEx(mask_non_zero_region, cv2.MORPH_GRADIENT, kernel)
    diff = cv2.normalize(mask_non_zero_region - gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_mean_without_border = cv2.bitwise_and(mask_mean, mask_mean, mask=diff)

    masks = []

    for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        masks.append(sd.leoThreshold2(mask_mean_without_border, mask_non_zero_region, i))

    mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)
    
    ret3, mask_otsu = cv2.threshold(mask_mean_leo_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)

    # masks = np.stack(masks)
    # frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    # for mask in masks:
    #     frequency_matrix += (mask == 255).astype(int)

    # mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 4), 255, 0).astype(np.uint8)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    # erosion = cv2.erode(close, kernel, iterations = 1)

    # dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return mask_otsu, mask_mean, mask_mean_leo

def sagittalSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    mask_mean_border = cv2.imread('images/mask_mean_sagittal_border.png', cv2.IMREAD_GRAYSCALE)

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

    processed_images_leo_threshold = []

    for i in range(nii_data.shape[0]):

        if i >= initial_index and i <= end_index: # BraTS20_Training_003_t1ce.nii
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

            # Leo Thresholding
            mask_non_zero_region = np.where(image_8bits_ahe > 0, 255, 0).astype(np.uint8)

            masks = []

            for i in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37]:
                masks.append(sd.leoThreshold2(image_8bits_ahe, mask_non_zero_region, i))

            mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)

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

            processed_images_leo_threshold.append(mask_mean_leo_threshold)

    mask_mean_leo = np.mean(processed_images_leo_threshold, axis=0).astype(np.uint8)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    mask_non_zero_region = np.where(mask_mean > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_non_zero_region = np.zeros_like(mask_mean)
    cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    gradient = cv2.morphologyEx(mask_non_zero_region, cv2.MORPH_GRADIENT, kernel)
    diff = cv2.normalize(mask_non_zero_region - gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_mean_without_border = cv2.bitwise_and(mask_mean, mask_mean, mask=diff)

    masks = []

    for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        masks.append(sd.leoThreshold2(mask_mean_without_border, mask_non_zero_region, i))

    mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)
    
    ret3, mask_otsu = cv2.threshold(mask_mean_leo_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)

    # masks = np.stack(masks)
    # frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

    # for mask in masks:
    #     frequency_matrix += (mask == 255).astype(int)

    # mask_result = np.where(frequency_matrix >= round(np.max(frequency_matrix) / 4), 255, 0).astype(np.uint8)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # close = cv2.morphologyEx(mask_result, cv2.MORPH_CLOSE, kernel)

    # erosion = cv2.erode(close, kernel, iterations = 1)

    # dilation = cv2.dilate(erosion, kernel, iterations = 1)

    return mask_otsu, mask_mean, mask_mean_leo

def loadContours(path):

    contours_gathered = []

    for image_name in os.listdir(path):

        image_path = os.path.join(path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contours_gathered.append(contour)

    return contours_gathered

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

ORIGINAL_SEGMENTATION_MASKS_OVERPOSED = r'C:\Users\leosn\Desktop\PIM\datasets\original_segmentation_mask_overposed'

OUTPUT_SEGMENTATION_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\results_segmentation'

dataframe_slices = pd.read_csv('ellipses_indexes_mask_segmentation.csv', sep=';')

coronal_contours_to_be_deleted = loadContours(r'C:\Users\leosn\Desktop\PIM\images\contour\coronal')

axial_contours_to_be_deleted = loadContours(r'C:\Users\leosn\Desktop\PIM\images\contour\axial')

def convertXYWHRToX1Y1X2Y2(xywhr):
    x, y, w, h, r = xywhr[0], xywhr[1], xywhr[2], xywhr[3], xywhr[4]

    x1 = x - w / 2
    y1 = y - h / 2

    x2 = x + w / 2
    y2 = y + h / 2

    return int(x1), int(y1), int(x2), int(y2)

def convertXYWHRToX1Y1X2Y2X3Y3X4Y4(xywhr):
    x, y, w, h, r = xywhr[0], xywhr[1], xywhr[2], xywhr[3], xywhr[4]

    x1, y1 = int(x - (w / 2)), int(y - (h / 2))
    x2, y2 = int(x + (w / 2)), int(y - (h / 2))
    x3, y3 = int(x + (w / 2)), int(y + (h / 2))
    x4, y4 = int(x - (w / 2)), int(y + (h / 2))

    return int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)

def getMaskBasedOnBoundingBoxPosition(mask_segmentation, mask_mean_leo_thresholding, box):

    x1, y1, x2, y2, x3, y3, x4, y4 = convertXYWHRToX1Y1X2Y2X3Y3X4Y4(box)

    contours, _ = cv2.findContours(mask_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(mask_segmentation)

    for i, contour in enumerate(contours):

        M = cv2.moments(contour)

        centroid_x, centroid_y = 0, 0
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

        if centroid_x >= x1 and centroid_x <= x3 and centroid_y >= y1 and centroid_y <= y3:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    ret3, mask_otsu = cv2.threshold(mask_mean_leo_thresholding[y1:y4, x1:x2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    crop = np.sum([mask[y1:y4, x1:x2], mask_otsu], axis=0)

    crop = np.where(crop > 255, 255, crop).astype(np.uint8)

    mask[y1:y4, x1:x2] = crop

    return mask

dataframe = []

for folder_name in os.listdir(ROOT_PATH):

    print(folder_name)

    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_t1ce.nii')

    dataframe_slices_filtered = dataframe_slices.loc[dataframe_slices['Folder'] == folder_name]

    row = dataframe_slices_filtered.iloc[0]

    if dataframe_slices_filtered.shape[0] == 0:
        print(1)
        break

    # ------------------------------- Loading Original Segmentation Masks -------------------------------

    original_mask_coronal = cv2.imread(os.path.join(ORIGINAL_SEGMENTATION_MASKS_OVERPOSED, folder_name, f'{folder_name}_coronal.png'), cv2.IMREAD_GRAYSCALE)
    original_mask_axial = cv2.imread(os.path.join(ORIGINAL_SEGMENTATION_MASKS_OVERPOSED, folder_name, f'{folder_name}_axial.png'), cv2.IMREAD_GRAYSCALE)
    original_mask_sagittal = cv2.imread(os.path.join(ORIGINAL_SEGMENTATION_MASKS_OVERPOSED, folder_name, f'{folder_name}_sagittal.png'), cv2.IMREAD_GRAYSCALE)

    # ------------------------------- Segmentation section -------------------------------

    mask_segmentation_coronal, mask_mean_coronal, mask_mean_leo_thresholding_coronal = coronalSegmentation(image_path, row['Coronal_Initial_Index'], row['Coronal_End_Index'])

    mask_segmentation_axial, mask_mean_axial, mask_mean_leo_thresholding_axial = axialSegmentation(image_path, row['Axial_Initial_Index'], row['Axial_End_Index'])

    mask_segmentation_sagittal, mask_mean_sagittal, mask_mean_leo_thresholding_sagittal = sagittalSegmentation(image_path, row['Sagittal_Initial_Index'], row['Sagittal_End_Index'])

    # cv2.imshow('mask_segmentation_coronal', mask_segmentation_coronal)
    # cv2.imshow('mask_segmentation_axial', mask_segmentation_axial)
    # cv2.imshow('mask_segmentation_sagittal', mask_segmentation_sagittal)

    # cv2.imshow('mask_mean_coronal', mask_mean_coronal)
    # cv2.imshow('mask_mean_axial', mask_mean_axial)
    # cv2.imshow('mask_mean_sagittal', mask_mean_sagittal)

    # cv2.waitKey(0)

    # cv2.imwrite(f'{folder_name}_t1ce_coronal.png', mask_segmentation_coronal)
    # cv2.imwrite(f'{folder_name}_t1ce_axial.png', mask_segmentation_axial)
    # cv2.imwrite(f'{folder_name}_t1ce_sagittal.png', mask_segmentation_sagittal)

    mask_mean_coronal_rgb = cv2.cvtColor(mask_mean_coronal, cv2.COLOR_GRAY2RGB)
    mask_mean_axial_rgb = cv2.cvtColor(mask_mean_axial, cv2.COLOR_GRAY2RGB)
    mask_mean_sagittal_rgb = cv2.cvtColor(mask_mean_sagittal, cv2.COLOR_GRAY2RGB)

    results_coronal = model([mask_mean_coronal_rgb], stream=True, imgsz=(640, 800))
    results_axial = model([mask_mean_axial_rgb], stream=True, imgsz=(640, 800))
    results_sagittal = model([mask_mean_sagittal_rgb], stream=True, imgsz=(640, 800))

    boxes_coronal = [result.obb.xywhr.cpu().numpy()[0] for result in results_coronal if len(result.obb.xywhr) > 0]
    boxes_axial = [result.obb.xywhr.cpu().numpy()[0] for result in results_axial if len(result.obb.xywhr) > 0]
    boxes_sagittal = [result.obb.xywhr.cpu().numpy()[0] for result in results_sagittal if len(result.obb.xywhr) > 0]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

    if len(boxes_coronal) > 0:

        mask = getMaskBasedOnBoundingBoxPosition(mask_segmentation_coronal, mask_mean_leo_thresholding_coronal, boxes_coronal[0])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask_mean_coronal_rgb, contours, -1, (0, 255, 0), 1)

        cv2.drawContours(mask, contours, -1, 255, -1)

        original_contours, _ = cv2.findContours(original_mask_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_original_mask = 0

        larger_area = 0
        index_larger_area_original_mask = 0
        for i, countour in enumerate(original_contours):

            area = cv2.contourArea(countour)

            area_original_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_original_mask = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_original_mask = i

        mask_segmented_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_mask = 0

        larger_area = 0
        index_larger_area_mask_segmented = 0
        for i, countour in enumerate(mask_segmented_contours):
            area = cv2.contourArea(countour)

            area_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_mask_segmented = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_mask_segmented = i

        similarity = cv2.matchShapes(original_contours[index_larger_area_original_mask], mask_segmented_contours[index_larger_area_mask_segmented], cv2.CONTOURS_MATCH_I1, 0.0)

        difference_between_masks = original_mask_coronal - mask

        percentage = (np.sum(difference_between_masks == 255) / np.sum(original_mask_coronal == 255)) * 100

        os.makedirs(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name), exist_ok=True)

        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_mask_drawned_coronal.png'), mask_mean_coronal_rgb)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_segmented_mask_coronal.png'), mask)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_original_mask_coronal.png'), original_mask_coronal)

        dataframe.append([folder_name, 'Coronal', "True", f'{similarity}'.replace('.', ','), area_original_mask, area_mask, f'{np.sum(difference_between_masks == 255)}'.replace('.', ','), f'{percentage}'.replace('.', ',')])

        # cv2.imshow('mask coronal', mask_mean_coronal_rgb)
    else:
        dataframe.append([folder_name, 'Coronal', "False", 0, 0, 0, 0, 0])   

    if len(boxes_axial) > 0:

        mask = getMaskBasedOnBoundingBoxPosition(mask_segmentation_axial, mask_mean_leo_thresholding_axial, boxes_axial[0])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask_mean_axial_rgb, contours, -1, (0, 255, 0), 1)

        cv2.drawContours(mask, contours, -1, 255, -1)

        original_contours, _ = cv2.findContours(original_mask_axial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_original_mask = 0

        larger_area = 0
        index_larger_area_original_mask = 0
        for i, countour in enumerate(original_contours):

            area = cv2.contourArea(countour)

            area_original_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_original_mask = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_original_mask = i

        mask_segmented_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_mask = 0

        larger_area = 0
        index_larger_area_mask_segmented = 0
        for i, countour in enumerate(mask_segmented_contours):
            area = cv2.contourArea(countour)

            area_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_mask_segmented = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_mask_segmented = i

        similarity = cv2.matchShapes(original_contours[index_larger_area_original_mask], mask_segmented_contours[index_larger_area_mask_segmented], cv2.CONTOURS_MATCH_I1, 0.0)

        difference_between_masks = original_mask_axial - mask

        percentage = (np.sum(difference_between_masks == 255) / np.sum(original_mask_axial == 255)) * 100

        os.makedirs(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name), exist_ok=True)

        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_mask_drawned_axial.png'), mask_mean_axial_rgb)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_segmented_mask_axial.png'), mask)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_original_mask_axial.png'), original_mask_axial)

        dataframe.append([folder_name, 'Axial', "True", f'{similarity}'.replace('.', ','), area_original_mask, area_mask, f'{np.sum(difference_between_masks == 255)}'.replace('.', ','), f'{percentage}'.replace('.', ',')])

        # cv2.imshow('mask axial', mask_mean_axial_rgb)
    else:
        dataframe.append([folder_name, 'Axial', "False", 0, 0, 0, 0, 0])   

    if len(boxes_sagittal) > 0:

        mask = getMaskBasedOnBoundingBoxPosition(mask_segmentation_sagittal, mask_mean_leo_thresholding_sagittal, boxes_sagittal[0])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask_mean_sagittal_rgb, contours, -1, (0, 255, 0), 1)

        cv2.drawContours(mask, contours, -1, 255, -1)

        original_contours, _ = cv2.findContours(original_mask_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_original_mask = 0

        larger_area = 0
        index_larger_area_original_mask = 0
        for i, countour in enumerate(original_contours):

            area = cv2.contourArea(countour)

            area_original_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_original_mask = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_original_mask = i

        mask_segmented_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_mask = 0

        larger_area = 0
        index_larger_area_mask_segmented = 0
        for i, countour in enumerate(mask_segmented_contours):
            area = cv2.contourArea(countour)

            area_mask += area

            if larger_area == 0:
                larger_area = area
                index_larger_area_mask_segmented = i
            else:
                if area > larger_area:
                    larger_area = area
                    index_larger_area_mask_segmented = i

        similarity = cv2.matchShapes(original_contours[index_larger_area_original_mask], mask_segmented_contours[index_larger_area_mask_segmented], cv2.CONTOURS_MATCH_I1, 0.0)

        difference_between_masks = original_mask_sagittal - mask

        percentage = (np.sum(difference_between_masks == 255) / np.sum(original_mask_sagittal == 255)) * 100

        os.makedirs(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name), exist_ok=True)

        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_mask_drawned_sagittal.png'), mask_mean_sagittal_rgb)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_segmented_mask_sagittal.png'), mask)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTATION_PATH, folder_name, f'{folder_name}_original_mask_sagittal.png'), original_mask_sagittal)

        dataframe.append([folder_name, 'Sagittal', "True", f'{similarity}'.replace('.', ','), area_original_mask, area_mask, f'{np.sum(difference_between_masks == 255)}'.replace('.', ','), f'{percentage}'.replace('.', ',')])

        # cv2.imshow('mask sagittal', mask_mean_sagittal_rgb)
    else:
        dataframe.append([folder_name, 'Sagittal', "False", 0, 0, 0, 0, 0])   
    

    cv2.waitKey(0)

dataframe = pd.DataFrame(dataframe, columns=['Image', 'Perspective', 'Segmented', 'Similarity', 'Area Original Mask', 'Area Mask', 'Difference Between Masks', 'Percentage Over Original Mask Total Area'])

dataframe.to_csv('results_segmentation.csv', index=False, sep=';')


        


  