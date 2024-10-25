from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from skimage.filters import threshold_niblack, threshold_sauvola
import mahotas

from utils.digital_image_processing import DigitalImageProcessing
from utils.frequency_domain import FrequencyDomain
from utils.spartial_domain import SpartialDomain

dip = DigitalImageProcessing()
fd = FrequencyDomain()
sd = SpartialDomain()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_003\BraTS20_Training_003_t1ce.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

mask_mean_border = cv2.imread('mask_mean_coronal_border.png', cv2.IMREAD_GRAYSCALE)

slices = []

def manual_adaptive_threshold_with_mask(img, mask, max_value, method, threshold_type, block_size, C):
    # Ensure the image is in grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Create an output image initialized to zeros (black)
    output = np.zeros_like(img)

    # Step 2: Apply the mask to extract the region of interest
    region_of_interest = cv2.bitwise_and(img, img, mask=mask)

    # Step 3: Calculate the mean filter using a block size (neighborhood size) on the region of interest
    mean_filter = cv2.boxFilter(region_of_interest, ddepth=-1, ksize=(block_size, block_size), normalize=True)

    # Step 4: Apply thresholding only to the region of interest
    thresholded = region_of_interest - mean_filter + C  # Calculate pixel-wise threshold

    # Apply binary threshold based on the chosen type
    if threshold_type == cv2.THRESH_BINARY:
        adaptive_result = np.where(thresholded > 0, max_value, 0)
    elif threshold_type == cv2.THRESH_BINARY_INV:
        adaptive_result = np.where(thresholded > 0, 0, max_value)

    # Ensure the result is in the correct format
    adaptive_result = adaptive_result.astype(np.uint8)

    # Step 5: Place the thresholded pixels back into the output image
    output[mask > 0] = adaptive_result[mask > 0]

    # Step 6: Keep the original image in areas where the mask is zero
    output[mask == 0] = img[mask == 0]

    return output

def laplacian_of_gaussian(img, sigma=1.0):
    """
    Apply Laplacian of Gaussian (LoG) to an image.
    
    Parameters:
    - img: Input grayscale image.
    - sigma: Standard deviation for Gaussian kernel.
    
    Returns:
    - LoG edge-detected image.
    """
    # Step 1: Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Step 2: Apply Laplacian operator
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Step 3: Find zero-crossings (edges)
    zero_crossing = np.zeros_like(laplacian)
    zero_crossing[np.where(np.diff(np.sign(laplacian), axis=0))] = 255
    zero_crossing[np.where(np.diff(np.sign(laplacian), axis=1))] = 255
    
    return zero_crossing

def normalizeZeroToOne(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def getMeanCentroid(mask_mean):
    # Find the pick of intensity in the mean mask
    hot_point_mask = np.where(mask_mean == np.max(mask_mean), 255, 0)
    hot_point_mask = cv2.normalize(hot_point_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Finding the median point
    contours, _ = cv2.findContours(hot_point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours[0].shape != (1, 1, 2):

        centroids = []

        for countour in contours:
            M = cv2.moments(countour)

            cX, cY = 0, 0

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            if cX != 0 and cY != 0:
                centroids.append([cX, cY])
            
        centroids_dataframe = pd.DataFrame(centroids, columns=['X', 'Y'])

        mean_centroid_x = int(centroids_dataframe['X'].mean())
        mean_centroid_y = int(centroids_dataframe['Y'].mean())

        return mean_centroid_x, mean_centroid_y
    else:
        return contours[0][0][0][0], contours[0][0][0][1]

cutoff_frequency = 40
c = 1

clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(12, 12))

processed_images = []

for i in range(nii_data.shape[1]):

    if i >= 150 and i <= 170: # BraTS20_Training_003_t1ce.nii
        axial_slice = nii_data[:, i, :]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_8bits = cv2.resize(image_8bits, (800, 640))

        # cv2.imshow('image_8bits', image_8bits)

        padded_image = fd.padImage(image_8bits)

        # Gaussian High-Pass Filter
        H_u_v = fd.gaussianHighpassFilter(cutoff_frequency, padded_image.shape)
        mask_gaussian, F_u_v = fd.filterImage(image_8bits, padded_image, H_u_v)
        image_8bits_filtered_gaussian = image_8bits + (c * mask_gaussian)
        image_8bits_filtered_gaussian = cv2.normalize(image_8bits_filtered_gaussian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Adaptive Histogram Equalization
        image_8bits_ahe = clahe.apply(image_8bits_filtered_gaussian)

        # cv2.imshow('image_8bits_ahe', image_8bits_ahe)

        # Selecting only the region of the brain
        ret3, mask_otsu = cv2.threshold(image_8bits_ahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_non_zero_region = np.zeros_like(mask_otsu)
        cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

        # cv2.imshow('mask_non_zero_region', mask_non_zero_region)

        # Otsu's Thresholding
        region_of_interest = cv2.bitwise_and(image_8bits_ahe, image_8bits_ahe, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        otsu_threshold_value = cv2.threshold(roi_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        thresholded_region = cv2.threshold(region_of_interest, otsu_threshold_value, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('mask_otsu', thresholded_region)

        # ADAPTIVE_THRESH_MEAN_C

        th2 = cv2.adaptiveThreshold(region_of_interest, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th2)

        window_size = 9
        thresh_niblack = threshold_niblack(image_8bits_ahe, window_size=window_size, k=0.001)
        binary_niblack = image_8bits_ahe > thresh_niblack

        # cv2.imshow('binary_niblack', binary_niblack.astype(np.uint8) * 255)

        window_size = 87
        thresh_sauvola = threshold_sauvola(region_of_interest, window_size=window_size)

        binary_sauvola = region_of_interest > thresh_sauvola

        # cv2.imshow('binary_sauvola', binary_sauvola.astype(np.uint8) * 255)

        window_size = 5
        contrast_threshold = 20
        bernsen_result = mahotas.thresholding.bernsen(image_8bits_ahe, window_size, contrast_threshold)

        # cv2.imshow('bernsen_result', bernsen_result.astype(np.uint8)) 

        blur = cv2.GaussianBlur(image_8bits_ahe, (7, 7) ,0)

        sigma = 0.001
        log_edges = laplacian_of_gaussian(blur, sigma)

        # cv2.imshow('log_edges', log_edges)

        # blurred_image = cv2.GaussianBlur(image_8bits_ahe, (3, 3), 1.4)

        edges = cv2.Canny(image_8bits_ahe, 0, 180)

        # cv2.imshow('edges', edges)

        sobel_x = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_8bits_ahe, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_combined = np.uint8(np.absolute(sobel_combined))

        # cv2.imshow('sobel_combined', sobel_combined)

        _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
        
        # cv2.imshow('binary_image', binary_image)

        # Close gaps
        kernel = np.ones((3, 3), np.uint8)
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


        mask_non_zero_region = np.where(sobel_combined > 0, 255, 0)
        mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_non_zero_region = np.zeros_like(sobel_combined)
        cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

        region_of_interest = cv2.bitwise_and(sobel_combined, sobel_combined, mask=mask_non_zero_region)
        roi_values = region_of_interest[region_of_interest > 0]

        global_mean = np.mean(roi_values)

        _, mask = cv2.threshold(sobel_combined, global_mean, 255, cv2.THRESH_BINARY)

        # cv2.imshow('mask', mask)

        # cv2.imshow('closed_image', closed_image)

        # cv2.waitKey(0)

        processed_images.append(mask)

mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

cv2.imshow('mask_mean', mask_mean)

mask_mean_without_border = mask_mean - mask_mean_border

cv2.imshow('mask_mean_without_border', mask_mean_without_border)

mask_non_zero_region = np.where(mask_mean_without_border > 0, 255, 0)
mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

contours, _ = cv2.findContours(mask_non_zero_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask_non_zero_region = np.zeros_like(mask_mean)
cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

region_of_interest = cv2.bitwise_and(mask_mean_without_border, mask_mean_without_border, mask=mask_non_zero_region)
roi_values = region_of_interest[region_of_interest > 0]

mask_leo = sd.leoThreshold(mask_mean_without_border, mask_non_zero_region, 9)

cv2.imshow('mask_leo', mask_leo)

# non_uniform_region = cv2.bitwise_and(mask_mean_without_border, mask_mean_without_border, mask=mask_non_zero_region)
# pixels = non_uniform_region[mask_non_zero_region == 255]

# unique, counts = np.unique(pixels, return_counts=True)
# plt.bar(unique, counts)
# plt.show()

# otsu_threshold_value = cv2.threshold(roi_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
# thresholded_region = cv2.threshold(region_of_interest, otsu_threshold_value, 255, cv2.THRESH_BINARY)[1]

# kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(thresholded_region, kernel,iterations = 1)

# cv2.imshow('erosion coronal', erosion)

cv2.waitKey(0)

# opening = cv2.morphologyEx(mask_mean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
# mask_mean = cv2.GaussianBlur(mask_mean, (7, 7) ,0)

# edges = cv2.Canny(mask_mean, 50, 180)

# cv2.imshow('edges', edges)

# cv2.waitKey(0)

# # Otsu's Thresholding
# mask_non_zero_region = np.where(mask_mean > 0, 255, 0)
# mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
# pixels = non_uniform_region[mask_non_zero_region == 255]

# threshold = round(dip.otsuThresholding(pixels))

# mask_otsu = np.where(mask_mean >= threshold, 255, 0)
# mask_otsu = cv2.normalize(mask_otsu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Mean Centroid

# mean_centroid_x, mean_centroid_y = getMeanCentroid(mask_mean)

# # Filtering the countours using the median point
# contours_drawn = []

# for slice in processed_images:

#     contours_slice, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     min_distance = 0

#     best_countour = None
#     best_centroid = None

#     for countour in contours_slice:
#         M = cv2.moments(countour)

#         centroid_x, centroid_y = 0, 0

#         if M["m00"] != 0:
#             centroid_x = int(M["m10"] / M["m00"])
#             centroid_y = int(M["m01"] / M["m00"])

#         if centroid_x != 0 and centroid_y != 0:

#             point1 = np.array([centroid_x, centroid_y])
#             point2 = np.array([mean_centroid_x, mean_centroid_y])

#             # Euclidean distance
#             distance = np.abs(np.linalg.norm(point1 - point2))

#             if distance < min_distance or min_distance == 0:
#                 min_distance = distance
#                 best_countour = countour
#                 best_centroid = [centroid_x, centroid_y]

#     area = cv2.contourArea(best_countour)

#     # Pixel units
#     if area >= 100:
#         drawed_contours = np.zeros_like(slice)
#         cv2.drawContours(drawed_contours, [best_countour], -1, 255, -1)
#         contours_drawn.append([best_countour, drawed_contours, best_centroid, area])

# contours_drawed = [contours[1] for contours in contours_drawn]

# if len(contours_drawed) > 0:

#     masks = np.stack(contours_drawed)
#     frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

#     for mask in masks:
#         frequency_matrix += (mask == 255).astype(int)

#     plt.imshow(frequency_matrix, cmap='hot', interpolation='nearest')
#     plt.colorbar(label="Frequency of 255")
#     plt.title("Frequency Distribution of Pixel Value 255")
#     plt.show()

# image = np.ones_like(mask_mean)

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
