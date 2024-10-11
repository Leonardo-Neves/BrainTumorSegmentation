import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_ubyte
from skimage.feature import peak_local_max
import os

import scipy.ndimage as ndimage

from utils.digital_image_processing import DigitalImageProcessing


dip = DigitalImageProcessing()

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingDataSliced\BraTS20_Training_001\axial\slices'

image_paths = [
    'images/192.jpg',
    'images/193.jpg',
    'images/194.jpg',
    'images/195.jpg',
    'images/196.jpg',
    'images/197.jpg',
]

processed_images = []

for image_name in os.listdir(ROOT_PATH):

    image = np.array(cv2.imread(os.path.join(ROOT_PATH, image_name), cv2.IMREAD_GRAYSCALE))

    height, width = image.shape

    image = np.where(image < 25, 0, image)

    image_equalized = cv2.equalizeHist(image)

    cv2.imshow(f'image_equalized', image_equalized)

    

    image_mdc = dip.MDC(image_equalized)

    mask = np.where(image_equalized == 0, 0, 1)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    



    mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    
    cv2.imshow(f'mask_closing', mask_closing)

    # gradient = cv2.morphologyEx(mask_closing, cv2.MORPH_GRADIENT, np.ones((51, 51), np.uint8))

    # combined_mask = cv2.bitwise_or(mask_closing, gradient)

    # inverted_gradient = cv2.bitwise_not(gradient)

    # mask_result = cv2.bitwise_and(mask_closing, mask_closing, mask=gradient)
    # mask_result = mask_closing - mask_result

    # difference_mask_result = mask_closing - mask_result

    # image_without_skull = cv2.bitwise_and(image_mdc, image_mdc, mask=mask_result)

    # cv2.imshow('image_without_skull', image_without_skull)

    cv2.waitKey(0)  

#     processed_images.append(image_without_skull)

max_matrix_maximum = np.maximum.reduce(processed_images)

max_matrix_minimum = np.minimum.reduce(processed_images)

# Intensity Level Slicing
# lower_intensity = 100
upper_intensity = 170

sliced_image = np.zeros_like(max_matrix_maximum)
sliced_image[(max_matrix_maximum <= upper_intensity) & (max_matrix_maximum > 0)] = 255

mask = np.where(max_matrix_maximum > 0, 255, 0)
mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

result_mask = mask - sliced_image

result_mask = cv2.erode(result_mask, np.ones((3, 3), np.uint8), iterations=1)

contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, [contours[147]], -1, 255, 2)

for i, contour in enumerate(contours):

    print(contours[i].shape)
