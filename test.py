import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

slices = []

for i in range(nii_data.shape[2]):

    if i >= 30 and i <= 112:
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        # --------------------- Shading correction ---------------------
        background = cv2.GaussianBlur(image_8bits, (51, 51), 0)
        corrected_image = cv2.subtract(image_8bits, background)
        corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --------------------- Sharpening with Laplacian kernel ---------------------

        height, width = image_8bits.shape
        padded_image = np.zeros((2 * height, 2 * width), dtype=np.uint8)
        padded_image[:height, :width] = image_8bits

        F_u_v = np.fft.fftshift(np.fft.fft2(padded_image))

        # h_x_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Sobel
        h_x_y = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) # Laplacian
        h_x_y_padded = np.pad(h_x_y, ((0, 1), (0, 1)), mode='constant')
        H_u_v = np.fft.fftshift(np.fft.fft2(h_x_y_padded, s=F_u_v.shape))

        fft_filtered = F_u_v * H_u_v

        mask_laplacian_kernal = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))

        mask_laplacian_kernal = mask_laplacian_kernal[:height, :width]

        mask_laplacian_kernal[mask_laplacian_kernal < 0] = 0

        mask_laplacian_kernal = cv2.normalize(mask_laplacian_kernal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("image_8bits", image_8bits)

        cv2.imshow("mask_laplacian_kernal", mask_laplacian_kernal)

        result = image_8bits + (1 * mask_laplacian_kernal)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("result", result)

        cv2.waitKey(0)
