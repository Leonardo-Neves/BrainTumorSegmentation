from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

processed_images = []

for i in range(nii_data.shape[2]):

    if i >= 30 and i <= 112:
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        cv2.imshow('image_8bits', image_8bits)

        # hist = cv2.calcHist([image_8bits], [0], None, [256], [0, 256])

        # print([e[0]for e in hist])

        

        mask = np.where(image_8bits == 255, 255, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow('mask', mask)

        cv2.waitKey(0)

        processed_images.append(mask)

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
