import pandas as pd
import numpy as np
import cv2
import nibabel as nib
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

    mask_mean_border = cv2.imread('images/mask_mean_coronal_border3.png', cv2.IMREAD_GRAYSCALE)

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

            # cv2.imshow('image_8bits_ahe', image_8bits_ahe)

            mask_non_zero_region = np.where(image_8bits_ahe > 0, 255, 0).astype(np.uint8)

            mask = sd.leoThreshold(image_8bits_ahe, mask_non_zero_region, 19)

            mask2 = sd.leoThreshold2(image_8bits_ahe, mask_non_zero_region, 19)

            masks = []

            # for i in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37]:
            # for i in [3, 7, 15, 19, 23, 29, 35, 41, 47, 53]:
            # for i in [3, 5, 7, 9, 11, 13]:
            #     masks.append(sd.leoThreshold(image_8bits_ahe, mask_non_zero_region, i))

            # mask_mean_leo_threshold = np.mean(masks, axis=0).astype(np.uint8)

            cv2.imshow('mask', mask)

            cv2.imshow('mask2', mask2)

            cv2.waitKey(0)

            processed_images.append(mask)

    mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

    cv2.imshow('mask_mean', mask_mean)

    cv2.waitKey(0)

    return mask_mean



ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'

dataframe_slices = pd.read_excel('slices.xlsx').dropna()

for folder_name in os.listdir(ROOT_PATH):

    image_path = os.path.join(ROOT_PATH, folder_name, f'{folder_name}_t1ce.nii')

    # Selecting the row to the index of the image
    image_index = int(folder_name.split('_')[-1])



    dataframe_slices_filtered = dataframe_slices.loc[dataframe_slices['Index'] == image_index]

    row = dataframe_slices_filtered.iloc[0]

    if dataframe_slices_filtered.shape[0] == 0:
        print(1)
        break

    # ------------------------------- Segmentation section -------------------------------

    mask_mean_coronal = coronalSegmentation(image_path, row['Coronal_Initial'], row['Coronal_End'])













