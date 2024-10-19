from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.stats import zscore
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing
from utils.frequency_domain import FrequencyDomain

dip = DigitalImageProcessing()

fd = FrequencyDomain()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

slices = []

cutoff_frequency = 40

# Pre-Processing
for i in range(nii_data.shape[2]):
    axial_slice = nii_data[:, :, i]

    image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits = cv2.resize(image_8bits, (640, 640))

    if np.mean(image_8bits) > 0:
        
        # Gaussian High-Pass Filter
        padded_image = fd.padImage(image_8bits)
        H_u_v = fd.gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
        mask_butterworth, F_u_v = fd.filterImage(image_8bits, padded_image, H_u_v)
        c = 1
        image_8bits_filtered_butterworth = image_8bits + (c * mask_butterworth)
        image_8bits_filtered_butterworth = cv2.normalize(image_8bits_filtered_butterworth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        slices.append(image_8bits_filtered_butterworth)