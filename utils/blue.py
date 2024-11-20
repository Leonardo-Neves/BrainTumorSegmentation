from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii"

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

    if i >= 40 and i <= 82: # BraTS20_Training_001_t1ce.nii
    # if i >= 38 and i <= 70: # BraTS20_Training_002_t1ce.nii
    # if i >= 61 and i <= 78: # BraTS20_Training_003_t1ce.nii
        axial_slice = nii_data[:, :, i]

        image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_8bits = cv2.resize(image_8bits, (640, 640))

        # cv2.imshow('image_8bits', image_8bits)

        padded_image = padImage(image_8bits)

        # Gaussian High-Pass Filter
        H_u_v = gaussianHighpassFilter(cutoff_frequency, padded_image.shape)

        mask_butterworth, F_u_v = filterImage(image_8bits, padded_image, H_u_v)

        c = 1

        image_8bits_filtered_butterworth = image_8bits + (c * mask_butterworth)

        image_8bits_filtered_butterworth = cv2.normalize(image_8bits_filtered_butterworth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blur = cv2.GaussianBlur(image_8bits_filtered_butterworth, (5, 5) ,0)

        # cv2.imshow('blur', blur)

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

        # cv2.imshow('mask_otsu', mask_otsu)

        # cv2.waitKey(0)

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

contours_drawed = [contours[1] for contours in contours_drawn]

masks = np.stack(contours_drawed)
frequency_matrix = np.zeros_like(mask_mean, dtype=np.int32)

for mask in masks:
    frequency_matrix += (mask == 255).astype(int)

plt.imshow(frequency_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label="Frequency of 255")
plt.title("Frequency Distribution of Pixel Value 255")
plt.show()

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


# ---------------------- Pre Processment ----------------------

# # Selecting only the region of the brain
# ret3, mask_otsu = cv2.threshold(image_8bits_ahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# mask_non_zero_region = np.zeros_like(mask_otsu)
# cv2.drawContours(mask_non_zero_region, contours, -1, 255, -1)

# # cv2.imshow('mask_non_zero_region', mask_non_zero_region)

# # Otsu's Thresholding
# region_of_interest = cv2.bitwise_and(image_8bits_ahe, image_8bits_ahe, mask=mask_non_zero_region)
# roi_values = region_of_interest[region_of_interest > 0]

# otsu_threshold_value = cv2.threshold(roi_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
# thresholded_region = cv2.threshold(region_of_interest, otsu_threshold_value, 255, cv2.THRESH_BINARY)[1]

# # cv2.imshow('mask_otsu', thresholded_region)

# # ADAPTIVE_THRESH_MEAN_C

# th2 = cv2.adaptiveThreshold(region_of_interest, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# # cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th2)

# window_size = 9
# thresh_niblack = threshold_niblack(image_8bits_ahe, window_size=window_size, k=0.001)
# binary_niblack = image_8bits_ahe > thresh_niblack

# # cv2.imshow('binary_niblack', binary_niblack.astype(np.uint8) * 255)

# window_size = 87
# thresh_sauvola = threshold_sauvola(region_of_interest, window_size=window_size)

# binary_sauvola = region_of_interest > thresh_sauvola

# # cv2.imshow('binary_sauvola', binary_sauvola.astype(np.uint8) * 255)

# window_size = 5
# contrast_threshold = 20
# bernsen_result = mahotas.thresholding.bernsen(image_8bits_ahe, window_size, contrast_threshold)

# # cv2.imshow('bernsen_result', bernsen_result.astype(np.uint8)) 

# blur = cv2.GaussianBlur(image_8bits_ahe, (7, 7) ,0)

# sigma = 0.001
# log_edges = laplacian_of_gaussian(blur, sigma)

# # cv2.imshow('log_edges', log_edges)

# # blurred_image = cv2.GaussianBlur(image_8bits_ahe, (3, 3), 1.4)

# edges = cv2.Canny(image_8bits_ahe, 0, 180)

# # cv2.imshow('edges', edges)

# ---------------------- Segmentation ----------------------

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


# ---------------------- blue_all_perspectives_multiple_images old ----------------------

# similarities = []

# for i, contour_coronal in enumerate(contours_coronal):
#     for j, contour_sagittal in enumerate(contours_sagittal):
#         try:
#             area_contour_coronal = cv2.contourArea(contour_coronal)
#             area_contour_sagittal = cv2.contourArea(contour_sagittal)

#             similarity = cv2.matchShapes(contour_coronal, contour_sagittal, cv2.CONTOURS_MATCH_I1, 0.0)

#             # Euclidean distance between two contours
#             M_coronal = cv2.moments(contour_coronal)
#             centroid_x_coronal, centroid_y_coronal = 0, 0
#             if M_coronal["m00"] != 0:
#                 centroid_x_coronal = int(M_coronal["m10"] / M_coronal["m00"])
#                 centroid_y_coronal = int(M_coronal["m01"] / M_coronal["m00"])

#             M_axial = cv2.moments(contour_sagittal)
#             centroid_x_sagittal, centroid_y_sagittal = 0, 0
#             if M_axial["m00"] != 0:
#                 centroid_x_sagittal = int(M_axial["m10"] / M_axial["m00"])
#                 centroid_y_sagittal = int(M_axial["m01"] / M_axial["m00"])

#             point1 = np.array([centroid_x_coronal, centroid_y_coronal])
#             point2 = np.array([centroid_x_sagittal, centroid_y_sagittal])

#             distance = np.abs(np.linalg.norm(point2 - point1))

#             d = np.sqrt((centroid_x_coronal - centroid_x_sagittal) ** 2 + (centroid_y_coronal - centroid_y_sagittal) ** 2)

#             x = np.sqrt((centroid_x_coronal - centroid_x_sagittal) ** 2)
#             y = np.sqrt((centroid_y_coronal - centroid_y_sagittal) ** 2)

#             similarities.append([similarity, i, j, area_contour_coronal, area_contour_sagittal, distance, d, x, y, np.abs(centroid_x_coronal - centroid_x_sagittal), np.abs(centroid_y_coronal - centroid_y_sagittal), centroid_x_coronal, centroid_y_coronal, centroid_x_sagittal, centroid_y_sagittal])

#         except:
#             pass

# ----------------------- Filtering contours between coronal and sagittal -----------------------

# dataframe_similarity_coronal_axial = pd.DataFrame(similarities, columns=['Similarity', 'Coronal', 'Sagittal', 'Coronal_Area', 'Sagittal_Area', 'Euclidian_Distance', 'Euclidian_Distance2', 'X', 'Y', 'X_Distance', 'Y_Distance', 'Coronal_X', 'Coronal_Y', 'Sagittal_X', 'Sagittal_Y'])

# dataframe_similarity_coronal_axial = dataframe_similarity_coronal_axial.loc[(dataframe_similarity_coronal_axial['Coronal_Area'] != 0) & (dataframe_similarity_coronal_axial['Sagittal_Area'] != 0)]

# sorted_df = dataframe_similarity_coronal_axial.sort_values(by=['Euclidian_Distance'], ascending=True)

# rows_area_above_100 = sorted_df[(sorted_df['Coronal_Area'] >= 100) & (sorted_df['Sagittal_Area'] >= 100)]

# filtered_df_coronal_sagittal = rows_area_above_100[rows_area_above_100['Euclidian_Distance'] <= 70]

# filtered_df_coronal_sagittal = filtered_df_coronal_sagittal[filtered_df_coronal_sagittal['Y_Distance'] == filtered_df_coronal_sagittal['Y_Distance'].min()]

# print(filtered_df_coronal_sagittal)

# mask_coronal = np.zeros_like(mask_mean_coronal)
# mask_sagittal = np.zeros_like(mask_mean_sagittal)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# for i, row in filtered_df_coronal_sagittal.iterrows():

#     cv2.drawContours(mask_coronal, [contours_coronal[int(row['Coronal'])]], -1, 255, -1)
#     mask_coronal = cv2.morphologyEx(mask_coronal, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(mask_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(mask_coronal, contours, -1, 255, -1)
    
    

    
#     cv2.drawContours(mask_sagittal, [contours_sagittal[int(row['Sagittal'])]], -1, 255, -1)
#     mask_sagittal = cv2.morphologyEx(mask_sagittal, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(mask_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(mask_sagittal, contours, -1, 255, -1)
    
    

# mask_coronal = cv2.cvtColor(mask_coronal, cv2.COLOR_GRAY2BGR)
# mask_sagittal = cv2.cvtColor(mask_sagittal, cv2.COLOR_GRAY2BGR)

# for i, row in filtered_df_coronal_sagittal.iterrows():
#     cv2.putText(mask_coronal, f'EC: {str(row['Euclidian_Distance'])[:6]}', (int(row['Coronal_X']), int(row['Coronal_Y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
#     cv2.putText(mask_sagittal, f'EC: {str(row['Euclidian_Distance'])[:6]}', (int(row['Sagittal_X']), int(row['Sagittal_Y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

# cv2.imshow('mask_coronal', mask_coronal)

# cv2.imshow('mask_sagittal', mask_sagittal)

# plt.show()

# cv2.waitKey(0)

# break

# ----------------------- Filtering contours between coronal and axial -----------------------

# filtered_df_coronal_axial = None

# print(filtered_df_coronal_sagittal)

# print(dataframe_similarity_coronal_axial)

# if not filtered_df_coronal_sagittal.empty and filtered_df_coronal_sagittal.shape[0] > 0:
    
#     first_row = filtered_df_coronal_sagittal.iloc[0]

#     point2 = np.array([first_row['Coronal_X'], first_row['Coronal_Y']])

#     caracteristics = []

#     for i, contour_axial in enumerate(contours_axial):
#         try:
#             # Euclidean distance between two contours
#             M_axial = cv2.moments(contour_axial)
#             centroid_x_axial, centroid_y_axial = 0, 0
#             if M_coronal["m00"] != 0:
#                 centroid_x_axial = int(M_axial["m10"] / M_axial["m00"])
#                 centroid_y_axial = int(M_axial["m01"] / M_axial["m00"])

#             if centroid_x_axial != 0 and centroid_y_axial != 0:

#                 point1 = np.array([centroid_x_axial, centroid_y_axial])

#                 delta = point2 - point1
#                 angle_radians = np.arctan2(delta[1], delta[0])
#                 angle_degrees = np.degrees(angle_radians)
#                 angle_degrees = angle_degrees % 360

#                 area_contour_axial = cv2.contourArea(contour_axial)
#                 area_contour_coronal = cv2.contourArea(contours_coronal[int(first_row['Coronal'])])

#                 similarity = cv2.matchShapes(contours_coronal[int(first_row['Coronal'])], contour_axial, cv2.CONTOURS_MATCH_I1, 0.0)

#                 caracteristics.append([int(first_row['Coronal']), i, angle_degrees, centroid_x_axial, centroid_y_axial, similarity, area_contour_axial, area_contour_coronal])
#         except Exception as e:
#             # print(e)
#             pass

#     dataframe_angle_axial_coronal = pd.DataFrame(caracteristics, columns=['Coronal', 'Axial', 'Angle', 'Axial_X', 'Axial_Y', 'Similarity', 'Axial_Area', 'Coronal_Area'])

#     # Filter by max area
#     sorted_df = dataframe_angle_axial_coronal.sort_values(by=['Coronal_Area', 'Axial_Area'], ascending=False)
#     first_rows = sorted_df.head(10)

#     # Filter by differences between areas
#     first_rows['Difference_Areas'] = np.where(first_rows['Coronal_Area'] >= first_rows['Axial_Area'], first_rows['Coronal_Area'] / first_rows['Axial_Area'], first_rows['Axial_Area'] / first_rows['Coronal_Area'])
#     first_rows = first_rows[first_rows['Difference_Areas'] <= 5]

#     # Filter by angle
#     filtered_df_coronal_axial = first_rows[(first_rows['Angle'] >= 85) & (first_rows['Angle'] <= 95)]

#     # Adjustment on the shape of the countour
#     first_row_filtered_df_coronal_sagittal = filtered_df_coronal_sagittal.iloc[0]
#     first_row_filtered_df_coronal_axial = filtered_df_coronal_axial.iloc[0]

#     kernel = np.ones((3, 3), np.uint8)

#     mask_coronal = np.zeros_like(mask_mean_coronal)
#     cv2.drawContours(mask_coronal, [contours_coronal[int(first_row_filtered_df_coronal_sagittal['Coronal'])]], -1, 255, -1)
#     mask_coronal = cv2.morphologyEx(mask_coronal, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(mask_coronal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(mask_coronal, contours, -1, 255, -1)

#     mask_axial = np.zeros_like(mask_mean_axial)
#     cv2.drawContours(mask_axial, [contours_axial[int(first_row_filtered_df_coronal_axial['Axial'])]], -1, 255, -1)
#     mask_axial = cv2.morphologyEx(mask_axial, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(mask_axial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(mask_axial, contours, -1, 255, -1)

#     mask_sagittal = np.zeros_like(mask_mean_sagittal)
#     cv2.drawContours(mask_sagittal, [contours_sagittal[int(first_row_filtered_df_coronal_sagittal['Sagittal'])]], -1, 255, -1)
#     mask_sagittal = cv2.morphologyEx(mask_sagittal, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(mask_sagittal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(mask_sagittal, contours, -1, 255, -1)

#     cv2.imshow('mask_coronal', mask_coronal)

#     cv2.imshow('mask_axial', mask_axial)

#     cv2.imshow('mask_sagittal', mask_sagittal)

#     cv2.waitKey(0)

