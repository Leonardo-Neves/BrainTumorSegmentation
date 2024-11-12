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
order = 2

processed_images = []

for i in range(nii_data.shape[2]):

    if i >= 30 and i <= 112:
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        padded_image = padImage(image_8bits)

        # Gaussian High-Pass Filter
        H_u_v = gaussianHighpassFilter(cutoff_frequency, padded_image.shape)

        mask_butterworth, F_u_v = filterImage(image_8bits, padded_image, H_u_v)

        c = 1

        image_8bits_filtered_butterworth = image_8bits + (c * mask_butterworth)

        image_8bits_filtered_butterworth = cv2.normalize(image_8bits_filtered_butterworth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blur = cv2.GaussianBlur(image_8bits_filtered_butterworth, (5, 5) ,0)

        # Selecting only the region of the brain
        ret3, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_non_zero_region = np.zeros_like(mask_otsu)
        cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

        non_uniform_region = cv2.bitwise_and(blur, blur, mask=mask_non_zero_region)
        pixels = non_uniform_region[mask_non_zero_region == 255]

        # Otsu's Thresholding
        threshold = round(dip.otsuThresholding(pixels))

        mask_otsu = np.where(blur >= threshold, 255, 0)
        mask_otsu = cv2.normalize(mask_otsu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        processed_images.append(mask_otsu)

mask_mean = np.mean(processed_images, axis=0).astype(np.uint8)

# Otsu's Thresholding
mask_non_zero_region = np.where(mask_mean > 0, 255, 0)
mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

non_uniform_region = cv2.bitwise_and(mask_mean, mask_mean, mask=mask_non_zero_region)
pixels = non_uniform_region[mask_non_zero_region == 255]

threshold = round(dip.otsuThresholding(pixels))

mask_otsu = np.where(mask_mean >= threshold, 255, 0)
mask_otsu = cv2.normalize(mask_otsu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Mean Centroid

mean_centroid_x, mean_centroid_y = getMeanCentroid(mask_mean)

# Filtering the countours using the median point
contours_drawn = []

for slice in processed_images:

    contours_slice, _ = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_distance = 0

    best_countour = None
    best_centroid = None

    for countour in contours_slice:
        M = cv2.moments(countour)

        centroid_x, centroid_y = 0, 0

        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

        if centroid_x != 0 and centroid_y != 0:

            point1 = np.array([centroid_x, centroid_y])
            point2 = np.array([mean_centroid_x, mean_centroid_y])

            # Euclidean distance
            distance = np.abs(np.linalg.norm(point1 - point2))

            if distance < min_distance or min_distance == 0:
                min_distance = distance
                best_countour = countour
                best_centroid = [centroid_x, centroid_y]

    area = cv2.contourArea(best_countour)

    # Pixel units
    if area >= 100:
        drawed_contours = np.zeros_like(slice)
        cv2.drawContours(drawed_contours, [best_countour], -1, 255, -1)
        contours_drawn.append([best_countour, drawed_contours, best_centroid, area])

mask_result = np.zeros_like(mask_mean)
# mask_result = contours_drawn[0][1]

# for i in range(1, len(contours_drawn)):
#     mask_result = cv2.bitwise_and(mask_result, contours_drawn[i][1])

# for i in range(1, len(contours_drawn)):
    
#     actual_contour = contours_drawn[i][0]
#     actual_contour_drawed = contours_drawn[i][1]
#     actual_contour_centroid = np.array(contours_drawn[i][2])
#     actual_contour_area = contours_drawn[i][3]

#     previous_contour = contours_drawn[i-1][0]
#     previous_contour_drawed = contours_drawn[i-1][1]
#     previous_contour_centroid = np.array(contours_drawn[i-1][2])
#     previous_contour_area = contours_drawn[i-1][3]

#     # mask_result = cv2.bitwise_and(contours_drawn[i], contours_drawn[i-1])

#     similarity = cv2.matchShapes(actual_contour, previous_contour, cv2.CONTOURS_MATCH_I1, 0.0)

#     distance = np.abs(np.linalg.norm(actual_contour_centroid - previous_contour_centroid))

#     if np.mean(mask_result) > 0:

#         similarity = cv2.matchShapes(actual_contour, previous_contour, cv2.CONTOURS_MATCH_I1, 0.0)


#         similarity_with_mask_result = cv2.matchShapes(mask_result, actual_contour, previous_contour, cv2.CONTOURS_MATCH_I1, 0.0)

#     else:

#         similarity = cv2.matchShapes(actual_contour, previous_contour, cv2.CONTOURS_MATCH_I1, 0.0)


#     if similarity <= 1:
#         if np.mean(mask_result) > 0:
#             # mask_result = cv2.bitwise_and(actual_contour_drawed, previous_contour_drawed)
#             and_op = cv2.bitwise_and(actual_contour_drawed, previous_contour_drawed)
#             mask_result = cv2.bitwise_and(mask_result, and_op)
#         else:
#             mask_result = cv2.bitwise_and(actual_contour_drawed, previous_contour_drawed)
#     elif similarity > 1:
#         if actual_contour_area > previous_contour_area:
#             mask_result = cv2.bitwise_and(mask_result, actual_contour_drawed)
#         else:
#             mask_result = cv2.bitwise_and(mask_result, previous_contour_drawed)
        

#     print(f'actual_contour_centroid: {actual_contour_centroid} previous_contour_centroid: {previous_contour_centroid} actual_contour_area: {actual_contour_area} previous_contour_area: {previous_contour_area} similarity: {similarity} distance: {distance}')

#     cv2.imshow('contours_drawn[i]', actual_contour_drawed)
#     cv2.imshow('contours_drawn[i-1]', previous_contour_drawed)
#     cv2.imshow('mask_result', mask_result)
#     cv2.waitKey(0)

contours_drawed = [contours[1] for contours in contours_drawn]

masks = np.stack(contours_drawed)
frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

for mask in masks:
    frequency_matrix += (mask == 255).astype(int)

plt.imshow(frequency_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label="Frequency of 255")
plt.title("Frequency Distribution of Pixel Value 255")

image = np.ones_like(mask_mean)

def on_trackbar(val):
    pass

cv2.namedWindow('Image Window')

cv2.createTrackbar('Brightness', 'Image Window', 0, 100, on_trackbar)

while True:
    
    brightness = cv2.getTrackbarPos('Brightness', 'Image Window')

    mask_result = np.where(frequency_matrix >= brightness, 255, 0)
    mask_result = cv2.normalize(mask_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow('Image Window', mask_result)

    # Break the loop if the user presses the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
