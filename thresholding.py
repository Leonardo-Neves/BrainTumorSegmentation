from scipy.ndimage import convolve
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

    return filtered_image, F_u_v

def padImage(image):
    
    height, width = image.shape
    padded_image = np.zeros((2 * height, 2 * width), dtype=np.uint8)
    padded_image[:height, :width] = image

    return padded_image

def normalizeZeroToOne(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

cutoff_frequency = 40
order = 2

for i in range(nii_data.shape[2]):

    if i >= 30 and i <= 112:
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        # --------------------- Shading correction ---------------------
        # background = cv2.GaussianBlur(image_8bits, (51, 51), 0)
        # corrected_image = cv2.subtract(image_8bits, background)
        # corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --------------------- Sharpening with Laplacian kernel ---------------------

        padded_image = padImage(image_8bits)

        # H_u_v1 = butterworthHighpassFilter(cutoff_frequency, order, padded_image.shape)
        # H_u_v1 = idealHighpassFilter(cutoff_frequency, padded_image.shape)
        H_u_v1 = gaussianHighpassFilter(cutoff_frequency, padded_image.shape)

        mask_butterworth, F_u_v = filterImage(image_8bits, padded_image, H_u_v1)

        c = 1

        image_8bits_filtered_butterworth = image_8bits + (c * mask_butterworth)

        image_8bits_filtered_butterworth = cv2.normalize(image_8bits_filtered_butterworth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("butterworthHighpassFilter", image_8bits_filtered_butterworth)

        blur = cv2.GaussianBlur(image_8bits_filtered_butterworth, (5, 5) ,0)

        # Selecting only the region of the brain
        ret3, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_non_zero_region = np.zeros_like(mask_otsu)
        cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

        non_uniform_region = cv2.bitwise_and(blur, blur, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        # unique, counts = np.unique(pixels, return_counts=True)
        # plt.bar(unique, counts)
        # plt.show()

        threshold = round(dip.otsuThresholding(pixels))

        mask_otsu = np.where(blur >= threshold, 255, 0)
        mask_otsu = cv2.normalize(mask_otsu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("THRESH_OTSU", mask_otsu)

        cv2.waitKey(0)

