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

def gaussianHighpassFilter(cutoff_frequency, shape = (0, 0)):
 
    rows, cols = shape[0], shape[1]
    center_row, center_col = rows // 2, cols // 2

    # New implementation
    gaussian_filter = np.ones((rows, cols), np.float32)
    center = (rows // 2, cols // 2)
    i, j = np.ogrid[:rows, :cols]

    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
    gaussian_filter = 1 - np.exp(-(distance**2) / (2 * (cutoff_frequency ** 2)))

    return gaussian_filter

def butterworthHighpassFilter(cutoff_frequency, order, shape = (0, 0)):
    rows, cols = shape[0], shape[1]
    center_row, center_col = rows // 2, cols // 2

    # New implementation
    butterworth_filter = np.ones((rows, cols), np.float32)
    center = (rows // 2, cols // 2)
    i, j = np.ogrid[:rows, :cols]

    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
    butterworth_filter = 1 / (1 + (cutoff_frequency / distance)**(2 * order))

    return butterworth_filter

# def idealHighpassFilter(cutoff_frequency, shape = (0, 0)):

#     rows, cols = shape[0], shape[1]
#     center_row, center_col = rows // 2, cols // 2

#     ideal_filter = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
#             if distance > cutoff_frequency:
#                 ideal_filter[i, j] = 1

#     return ideal_filter

def idealHighpassFilter(cutoff_frequency, shape=(0, 0)):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # Create meshgrid for row and column indices
    x = np.arange(0, rows) - center_row
    y = np.arange(0, cols) - center_col
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Compute the distance from the center for all points
    distance = np.sqrt(X**2 + Y**2)

    # Apply the cutoff frequency threshold
    ideal_filter = np.where(distance > cutoff_frequency, 1, 0)

    return ideal_filter

def filterImage(image, padded_image, kernel):

    # 1° f(x, y) as M x N, pad the image, P = 2M and Q = 2N
    padded_image = padded_image

    # 2° Compute the Fourier transform and center the spectrum
    F_u_v = np.fft.fftshift(np.fft.fft2(padded_image))

    # 3° Creating Sobel kernel
    H_u_v = kernel

    # 4° Applying the filter to the image
    fft_filtered = F_u_v * H_u_v

    # 5° Apply Inverse Fourier Transform to obtain the filtered image
    filtered_image = np.abs(np.fft.ifft2(fft_filtered))

    # 6° Removing pad from the image
    height, width = image.shape
    filtered_image = filtered_image[:height, :width]

    return filtered_image

def padImage(image):
    
    height, width = image.shape
    padded_image = np.zeros((2 * height, 2 * width), dtype=np.uint8)
    padded_image[:height, :width] = image

    return padded_image

cutoff_frequency = 40
order = 2

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

        padded_image = padImage(image_8bits)

        # H_u_v = butterworthHighpassFilter(cutoff_frequency, order, padded_image.shape)
        H_u_v = gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
        # H_u_v = idealHighpassFilter(cutoff_frequency, padded_image.shape)

        mask_laplacian_kernal = filterImage(image_8bits, padded_image, H_u_v)

        cv2.imshow("image_8bits", image_8bits)

        cv2.imshow("mask_laplacian_kernal", mask_laplacian_kernal)
        
        c = 1

        result = image_8bits + (c * mask_laplacian_kernal)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("result", result)

        cv2.waitKey(0)
