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

def coronalSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

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

    return mask_mean

def axialSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

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

    return mask_mean

def sagittalSegmentation(image_path, initial_index, end_index):

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    cutoff_frequency = 40
    c = 1

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

    processed_images = []

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

    return mask_mean

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

OUTPUT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean'

dataframe_slices = pd.read_csv('ellipses_indexes_mask_segmentation.csv', sep=';')

for folder_name in os.listdir(ROOT_PATH):

    print(folder_name)

    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_t1ce.nii')

    # Selecting the row to the index of the image
    image_index = int(folder_name.split('_')[-1])

    dataframe_slices_filtered = dataframe_slices.loc[dataframe_slices['Folder'] == folder_name]

    row = dataframe_slices_filtered.iloc[0]

    if dataframe_slices_filtered.shape[0] == 0:
        break

    # ------------------------------- Segmentation section -------------------------------

    mask_mean_coronal = coronalSegmentation(image_path, row['Coronal_Initial_Index'], row['Coronal_End_Index'])

    mask_mean_axial = axialSegmentation(image_path, row['Axial_Initial_Index'], row['Axial_End_Index'])

    mask_mean_sagittal = sagittalSegmentation(image_path, row['Sagittal_Initial_Index'], row['Sagittal_End_Index'])

    # os.makedirs(os.path.join(OUTPUT_PATH, folder_name), exist_ok=True)

    cv2.imwrite(os.path.join(OUTPUT_PATH, f'{folder_name}_coronal.png'), mask_mean_coronal)
    cv2.imwrite(os.path.join(OUTPUT_PATH, f'{folder_name}_axial.png'), mask_mean_axial)
    cv2.imwrite(os.path.join(OUTPUT_PATH, f'{folder_name}_sagittal.png'), mask_mean_sagittal)