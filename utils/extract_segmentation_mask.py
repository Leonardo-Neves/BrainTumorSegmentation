import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import pandas as pd

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

from sklearn.mixture import GaussianMixture

image_segmentation_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_003\BraTS20_Training_003_seg.nii"

nii_segmentation_file = nib.load(image_segmentation_path)
nii_segmentation_data = nii_segmentation_file.get_fdata()

def on_trackbar(val):
    pass

cv2.namedWindow('Image Window')

cv2.createTrackbar('index', 'Image Window', 0, nii_segmentation_data.shape[2], on_trackbar)

gmm = GaussianMixture(n_components=3, random_state=0) 

while True:
    
    i = cv2.getTrackbarPos('index', 'Image Window')

    axial_slice = nii_segmentation_data[:, :, i]

    image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_8bits = cv2.resize(image_8bits, (800, 640))

    cv2.imshow('image_8bits', image_8bits)

    mask_non_zero_region = np.where(image_8bits > 0, 255, 0)
    mask_non_zero_region = cv2.normalize(mask_non_zero_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    non_uniform_region = cv2.bitwise_and(image_8bits, image_8bits, mask=mask_non_zero_region)
    pixels = non_uniform_region[mask_non_zero_region == 255]

    threshold = round(dip.otsuThresholding(pixels))

    mask_otsu = np.where(image_8bits >= threshold, 255, 0)
    mask_otsu = cv2.normalize(mask_otsu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    mask_seg = np.where(image_8bits >= 220, 255, 0).astype(np.uint8)

    print(np.mean(pixels))
    
    cv2.imshow('mask_seg', mask_seg if np.mean(pixels) <= 170 else np.zeros_like(mask_seg))
    cv2.imshow('Image Window', mask_otsu)

    # Break the loop if the user presses the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()

