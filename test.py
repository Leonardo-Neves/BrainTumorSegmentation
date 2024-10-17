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

def laplacianFilterSpatialDomain(image):
    # laplacian_kernel = np.array([[1,  1,  1],
    #                              [1, -8,  1],
    #                              [1,  1,  1]])

    laplacian_kernel = np.array([[0,  1,  0],
                                 [1, -4,  1],
                                 [0,  1,  0]])                                 

    return convolve(image, laplacian_kernel)

def laplacianFilter(shape = (0, 0)):
    rows, cols = shape[0], shape[1]
    center_row, center_col = rows // 2, cols // 2

    # New implementation
    laplacian_filter = np.ones((rows, cols), np.float32)
    center = (rows // 2, cols // 2)
    i, j = np.ogrid[:rows, :cols]

    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
    laplacian_filter = (((-4) * np.pi) ** 2) * (distance ** 2)

    return laplacian_filter

def filterImageLaplacianFilter(image, padded_image):

    padded_image = padded_image

    F_u_v = np.fft.fftshift(np.fft.fft2(padded_image))

    H_u_v = laplacianFilter(padded_image.shape)

    # fft_filtered = F_u_v - (H_u_v * F_u_v)
    fft_filtered = (1 - H_u_v) * F_u_v

    fft_filtered = np.abs(np.fft.ifft2(fft_filtered))

    height, width = image.shape
    fft_filtered = fft_filtered[:height, :width]

    cv2.imshow("fft_filtered", fft_filtered)

    return cv2.normalize(fft_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


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

def section38Book(image_8bits):

    smoothed_image = cv2.GaussianBlur(image_8bits, (5, 5), 0)
    sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = np.uint8(gradient_magnitude)

    laplacian = cv2.Laplacian(image_8bits, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    combined_laplacian_gradient = laplacian * gradient_magnitude

    laplacian[laplacian < 0] = 0

    result = image_8bits + (1 * laplacian)

    cv2.imshow("gradient_magnitude", gradient_magnitude)

    cv2.imshow("laplacian", laplacian)

    cv2.imshow("combined_laplacian_gradient", combined_laplacian_gradient)

    cv2.imshow("image_8bits", image_8bits)

    cv2.imshow("result", result)

    cv2.waitKey(0)

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

        # padded_image = padImage(image_8bits)

        # H_u_v = butterworthHighpassFilter(cutoff_frequency, order, padded_image.shape)
        # H_u_v = gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
        # H_u_v = idealHighpassFilter(cutoff_frequency, padded_image.shape)

        # H_u_v = laplacianFilter(padded_image.shape)

        # laplacian = laplacianFilterSpatialDomain(image_8bits)

        # mask_laplacian_kernal, F_u_v = filterImage(image_8bits, padded_image, H_u_v)

        # cv2.imshow("image_8bits", image_8bits)

        # cv2.imshow("mask_laplacian_kernal", mask_laplacian_kernal)
        

        # mask_laplacian_kernal = cv2.normalize(mask_laplacian_kernal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        c = 1

    
        smoothed_image = cv2.GaussianBlur(image_8bits, (5, 5), 0)
        sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        gradient_magnitude = np.uint8(gradient_magnitude)




        cv2.imshow("gradient_magnitude", gradient_magnitude)

        

        laplacian = cv2.Laplacian(image_8bits, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        cv2.imshow("laplacian", laplacian)


        combined_laplacian_gradient = laplacian * gradient_magnitude

        cv2.imshow("combined_laplacian_gradient", combined_laplacian_gradient)

        # # combined_laplacian_gradient = cv2.normalize(combined_laplacian_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        laplacian[laplacian < 0] = 0

        result = image_8bits + (1 * laplacian)

        cv2.imshow("image_8bits", image_8bits)

        cv2.imshow("result", result)

        cv2.waitKey(0)

