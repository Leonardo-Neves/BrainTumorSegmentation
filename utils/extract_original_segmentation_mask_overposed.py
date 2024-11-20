from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

def getMasksAxial(nii_data):
    masks = []

    initial_index = 0
    end_index = 0

    for i in range(nii_data.shape[2]):
        
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_8bits = cv2.resize(image_8bits, (800, 640))

        mask_non_zero_region = np.where(image_8bits > 0, 255, 0).astype(np.uint8)
        region_of_interest = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        unique, counts = np.unique(roi_values, return_counts=True)

        df = pd.DataFrame(data={'Intensity': unique, 'Frequency': counts})

        mask = np.zeros_like(mask_non_zero_region, dtype=np.uint8)

        if not df.empty or df.shape[0] > 0:
            sorted_df_first_3_rows = df.sort_values(by=['Frequency'], ascending=False).head(3)

            intensities = sorted_df_first_3_rows['Intensity'].values

            if 255 in intensities and 63 in intensities and 127 in intensities:
                mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                mask = mask1 + mask2

            elif 255 in intensities and 63 in intensities:
                
                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_63_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 63)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_63_intensity['Frequency'].values[0]:
                    mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                    mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    mask = mask1 + mask2

                elif row_with_255_intensity['Frequency'].values[0] > row_with_63_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    
            elif 255 in intensities and 127 in intensities:

                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_127_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 127)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                elif row_with_255_intensity['Frequency'].values[0] > row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 127, 255, 0).astype(np.uint8)
            elif 63 in intensities and 127 in intensities:
                mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                
        if np.mean(mask) != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if initial_index == 0:
                initial_index = i
            else:
                end_index = i

            masks.append(mask)

    return masks, initial_index, end_index

def getMasksCoronal(nii_data):
    masks = []

    initial_index = 0
    end_index = 0

    for i in range(nii_data.shape[1]):
        
        axial_slice = nii_data[:, i, :]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_8bits = cv2.resize(image_8bits, (800, 640))

        mask_non_zero_region = np.where(image_8bits > 0, 255, 0).astype(np.uint8)
        region_of_interest = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        unique, counts = np.unique(roi_values, return_counts=True)

        df = pd.DataFrame(data={'Intensity': unique, 'Frequency': counts})

        mask = np.zeros_like(mask_non_zero_region, dtype=np.uint8)

        if not df.empty or df.shape[0] > 0:
            sorted_df_first_3_rows = df.sort_values(by=['Frequency'], ascending=False).head(3)

            intensities = sorted_df_first_3_rows['Intensity'].values

            if 255 in intensities and 63 in intensities and 127 in intensities:
                mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                mask = mask1 + mask2

            elif 255 in intensities and 63 in intensities:
                
                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_63_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 63)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_63_intensity['Frequency'].values[0]:
                    mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                    mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    mask = mask1 + mask2

                elif row_with_255_intensity['Frequency'].values[0] > row_with_63_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)

            elif 255 in intensities and 127 in intensities:

                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_127_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 127)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                elif row_with_255_intensity['Frequency'].values[0] > row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 127, 255, 0).astype(np.uint8)
            elif 63 in intensities and 127 in intensities:
                mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                
        if np.mean(mask) != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if initial_index == 0:
                initial_index = i
            else:
                end_index = i

            masks.append(mask)

    return masks, initial_index, end_index

def getMasksSagittal(nii_data):
    masks = []

    initial_index = 0
    end_index = 0

    for i in range(nii_data.shape[0]):
        
        axial_slice = nii_data[i, :, :]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_8bits = cv2.resize(image_8bits, (800, 640))

        mask_non_zero_region = np.where(image_8bits > 0, 255, 0).astype(np.uint8)
        region_of_interest = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        unique, counts = np.unique(roi_values, return_counts=True)

        df = pd.DataFrame(data={'Intensity': unique, 'Frequency': counts})

        mask = np.zeros_like(mask_non_zero_region, dtype=np.uint8)

        if not df.empty or df.shape[0] > 0:
            sorted_df_first_3_rows = df.sort_values(by=['Frequency'], ascending=False).head(3)

            intensities = sorted_df_first_3_rows['Intensity'].values

            if 255 in intensities and 63 in intensities and 127 in intensities:
                mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                mask = mask1 + mask2

            elif 255 in intensities and 63 in intensities:
                
                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_63_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 63)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_63_intensity['Frequency'].values[0]:
                    mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                    mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    mask = mask1 + mask2

                elif row_with_255_intensity['Frequency'].values[0] > row_with_63_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)

            elif 255 in intensities and 127 in intensities:

                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_127_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 127)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                elif row_with_255_intensity['Frequency'].values[0] > row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 127, 255, 0).astype(np.uint8)
            elif 63 in intensities and 127 in intensities:
                mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                
        if np.mean(mask) != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if initial_index == 0:
                initial_index = i
            else:
                end_index = i

            masks.append(mask)

    return masks, initial_index, end_index

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

OUTPUT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\original_segmentation_mask_overposed'

for folder_name in os.listdir(ROOT_PATH):
    print(folder_name)

    index = int(folder_name.split('_')[-1])

    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_seg.nii')

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    os.makedirs(os.path.join(OUTPUT_PATH, folder_name), exist_ok=True)

    # Axial

    masks_axial, initial_index_axial, end_index_axial = getMasksAxial(nii_data)

    mask_axial = np.sum(masks_axial, axis=0)
    mask_axial = np.where(mask_axial > 0, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask_axial = cv2.morphologyEx(mask_axial, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_axial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_axial = np.zeros_like(mask_axial)
    cv2.drawContours(mask_axial, contours, -1, 255, -1)

    cv2.imwrite(os.path.join(OUTPUT_PATH, folder_name, f'{folder_name}_axial.png'), mask_axial)

    # Coronal

    masks_coronal, initial_index_coronal, end_index_coronal = getMasksCoronal(nii_data)

    mask_coronal = np.sum(masks_coronal, axis=0)
    mask_coronal = np.where(mask_coronal > 0, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask_coronal = cv2.morphologyEx(mask_coronal, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_coronal = np.zeros_like(mask_coronal)
    cv2.drawContours(mask_coronal, contours, -1, 255, -1)

    cv2.imwrite(os.path.join(OUTPUT_PATH, folder_name, f'{folder_name}_coronal.png'), mask_coronal)

    # Sagittal
    
    masks_sagittal, initial_index_sagittal, end_index_sagittal = getMasksSagittal(nii_data)

    mask_sagittal = np.sum(masks_sagittal, axis=0)
    mask_sagittal = np.where(mask_sagittal > 0, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask_sagittal = cv2.morphologyEx(mask_sagittal, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_sagittal = np.zeros_like(mask_sagittal)
    cv2.drawContours(mask_sagittal, contours, -1, 255, -1)

    cv2.imwrite(os.path.join(OUTPUT_PATH, folder_name, f'{folder_name}_sagittal.png'), mask_sagittal)
    