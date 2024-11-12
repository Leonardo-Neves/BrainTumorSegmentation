from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

# image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_326\BraTS20_Training_326_seg.nii"

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

for folder_name in os.listdir(ROOT_PATH):
    print(folder_name)


    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_seg.nii')

    nii_file = nib.load(image_path)
    nii_data = nii_file.get_fdata()

    masks = []

    initial_index = 0
    end_index = 0

    for i in range(nii_data.shape[2]):
            
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_8bits = cv2.resize(image_8bits, (800, 640))

        cv2.imshow('image_8bits', image_8bits)

        mask_non_zero_region = np.where(image_8bits > 0, 255, 0).astype(np.uint8)
        region_of_interest = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        unique, counts = np.unique(roi_values, return_counts=True)

        df = pd.DataFrame(data={'Intensity': unique, 'Frequency': counts})

        mask = np.zeros_like(mask_non_zero_region, dtype=np.uint8)

        if not df.empty or df.shape[0] > 0:
            sorted_df_first_3_rows = df.sort_values(by=['Frequency'], ascending=False).head(3)

            print(sorted_df_first_3_rows)

            intensities = sorted_df_first_3_rows['Intensity'].values

            if 255 in intensities and 63 in intensities and 127 in intensities:
                print(2)
                mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)

                mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                mask = mask1 + mask2
            elif 255 in intensities and 63 in intensities:
                print(1)
                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_63_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 63)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_63_intensity['Frequency'].values[0]:
                    mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                    mask2 = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    mask = mask1 + mask2

                    print('1 - 1')
                elif row_with_255_intensity['Frequency'].values[0] > row_with_63_intensity['Frequency'].values[0]:
                    # mask1 = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                    mask = np.where(image_8bits == 63, 255, 0).astype(np.uint8)
                    # mask = mask1 + mask2

                    print('1 - 2')
            elif 255 in intensities and 127 in intensities:
                print(3)
                row_with_255_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 255)]
                row_with_127_intensity = sorted_df_first_3_rows[(sorted_df_first_3_rows['Intensity'] == 127)]

                if row_with_255_intensity['Frequency'].values[0] < row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits >= 210, 255, 0).astype(np.uint8)
                elif row_with_255_intensity['Frequency'].values[0] > row_with_127_intensity['Frequency'].values[0]:
                    mask = np.where(image_8bits == 127, 255, 0).astype(np.uint8)
            elif 63 in intensities and 127 in intensities:
                print(4)
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

            
        cv2.imshow('mask', mask)

        cv2.waitKey(0)