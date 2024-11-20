from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

# image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"

def getMasksAxial(nii_data, index):
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

def generateMaskOverposed(masks):
    mask_overposed = np.sum(masks, axis=0).astype(np.uint8)
    mask_overposed = np.where(mask_overposed > 255, 255, mask_overposed).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask_overposed, cv2.MORPH_OPEN, kernel)

def getEllipseFromMask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = np.vstack(contours)
    
    drawed_contours = np.zeros_like(mask, dtype=np.uint8)

    if len(all_points) >= 5:
        unified_ellipse = cv2.fitEllipse(all_points)
        cv2.ellipse(drawed_contours, unified_ellipse, 255, 2)

        return unified_ellipse
    else:
        return []

dataframe = []

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

for folder_name in os.listdir(ROOT_PATH):
    print(folder_name)

    index = int(folder_name.split('_')[-1])

    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_seg.nii')

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    masks_axial, initial_index_axial, end_index_axial = getMasksAxial(nii_data, index)
    mask_overposed_axial = generateMaskOverposed(masks_axial)
    unified_ellipse_axial = getEllipseFromMask(mask_overposed_axial)

    masks_coronal, initial_index_coronal, end_index_coronal = getMasksCoronal(nii_data)
    mask_overposed_coronal = generateMaskOverposed(masks_coronal)
    unified_ellipse_coronal = getEllipseFromMask(mask_overposed_coronal)

    masks_sagittal, initial_index_sagittal, end_index_sagittal = getMasksSagittal(nii_data)
    mask_overposed_sagittal = generateMaskOverposed(masks_sagittal)
    unified_ellipse_sagittal = getEllipseFromMask(mask_overposed_sagittal)

    dataframe.append(
        [
            folder_name, 
            initial_index_axial, 
            end_index_axial, 
            unified_ellipse_axial[0][0], 
            unified_ellipse_axial[0][1], 
            unified_ellipse_axial[1][0], 
            unified_ellipse_axial[1][1], 
            unified_ellipse_axial[2],
            initial_index_coronal, 
            end_index_coronal, 
            unified_ellipse_coronal[0][0], 
            unified_ellipse_coronal[0][1], 
            unified_ellipse_coronal[1][0], 
            unified_ellipse_coronal[1][1], 
            unified_ellipse_coronal[2],
            initial_index_sagittal, 
            end_index_sagittal, 
            unified_ellipse_sagittal[0][0], 
            unified_ellipse_sagittal[0][1], 
            unified_ellipse_sagittal[1][0], 
            unified_ellipse_sagittal[1][1], 
            unified_ellipse_sagittal[2]
        ]
    )

    # print(chr(27) + "[2J")

dataframe = pd.DataFrame(dataframe, columns=
    [
        'Folder', 
        'Axial_Initial_Index', 
        'Axial_End_Index', 
        'Axial_Center_X', 
        'Axial_Center_Y', 
        'Axial_Width', 
        'Axial_Height', 
        'Axial_Angle',
        'Coronal_Initial_Index', 
        'Coronal_End_Index', 
        'Coronal_Center_X', 
        'Coronal_Center_Y', 
        'Coronal_Width', 
        'Coronal_Height', 
        'Coronal_Angle',
        'Sagittal_Initial_Index', 
        'Sagittal_End_Index', 
        'Sagittal_Center_X', 
        'Sagittal_Center_Y', 
        'Sagittal_Width', 
        'Sagittal_Height', 
        'Sagittal_Angle'
    ]
)

dataframe.to_csv('ellipses_indexes_mask_segmentation_val.csv', index=False, sep=';')
  
# masks = np.stack(processed_images)
# frequency_matrix = np.zeros_like(processed_images[0], dtype=np.int32)

# for mask in masks:
#     frequency_matrix += (mask == 255).astype(int)

# plt.imshow(frequency_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar(label="Frequency of 255")
# plt.title("Frequency Distribution of Pixel Value 255")

# image = np.ones_like(processed_images[0])

# def on_trackbar(val):
#     pass

# cv2.namedWindow('Image Window')

# cv2.createTrackbar('Brightness', 'Image Window', 0, 100, on_trackbar)

# while True:
    
#     brightness = cv2.getTrackbarPos('Brightness', 'Image Window')

#     mask_result = np.where(frequency_matrix >= brightness, 255, 0)
#     mask_result = cv2.normalize(mask_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     cv2.imshow('Image Window', mask_result)

#     # Break the loop if the user presses the 'ESC' key
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC key
#         break

# cv2.destroyAllWindows()
