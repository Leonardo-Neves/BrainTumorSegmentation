import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import warnings

warnings.filterwarnings("ignore")

ROOT_PATH = 'datasets/MICCAI_BraTS_2020_Data_Training/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

OUTPUT_PATH = 'datasets/MICCAI_BraTS_2020_Data_Training/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingDataSliced'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def preProcessingPhase(image):
    # Normalizing the image to 8 bits and rezising it
    image_8bits = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits = cv2.resize(image_8bits, (640, 640))

    # Selecting only the region of the brain
    mask = np.where(image_8bits == 0, 0, 1)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    non_uniform_region = cv2.bitwise_and(image_8bits, image_8bits, mask=mask)
    pixels = non_uniform_region[mask == 255]
    
    # Histogram Equalization
    hist, bins = np.histogram(pixels, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
    image_equalized = cdf_final[image_8bits]

    # Filtering images to remove paper and salt noise in spatial domain
    image = np.array(image_equalized)

    filtered_image_without_pepper_salt = np.zeros_like(image)

    def medianFilter(matrix_neighborhood):
        return np.median(matrix_neighborhood)

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            neighborhood = np.array([[image[i-1, j-1], image[i-1, j],   image[i-1, j+1]],
                                    [image[i, j-1],   image[i-1, j-1], image[i, j+1]],
                                    [image[i+1, j-1], image[i+1, j],   image[i+1, j+1]]])

            filtered_image_without_pepper_salt[i, j]= medianFilter(neighborhood)

    return filtered_image_without_pepper_salt

def preProcessingPhaseSegmentationMask(image):
    image_8bits = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_8bits = cv2.resize(image_8bits, (640, 640))

    return np.where(image_8bits > 0, 255, 0)

for folder_name in tqdm(os.listdir(ROOT_PATH)):

    nii_file = nib.load(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_t1ce.nii'))
    nii_data = nii_file.get_fdata()

    nii_segmentation_file = nib.load(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_seg.nii'))
    nii_segmentation_data = nii_segmentation_file.get_fdata()

    os.makedirs(os.path.join(OUTPUT_PATH, folder_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'axial'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'axial', 'slices'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'axial', 'slices_segmentation'), exist_ok=True)

    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'coronal'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'coronal', 'slices'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'coronal', 'slices_segmentation'), exist_ok=True)

    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'sagittal'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'sagittal', 'slices'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, folder_name, 'sagittal', 'slices_segmentation'), exist_ok=True)

    j = 0

    for i in range(nii_data.shape[2]):
        axial_slice = nii_data[:, :, i]
        axial_slice_segmentation = nii_segmentation_data[:, :, i]

        axial_slice_normalized = preProcessingPhase(axial_slice)
        axial_slice_segmentation_normalized = preProcessingPhaseSegmentationMask(axial_slice_segmentation)

        if np.mean(axial_slice_segmentation_normalized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'axial/slices/axial_slice_{j}.png'), axial_slice_normalized, cmap='gray')
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'axial/slices_segmentation/axial_slice_{j}.png'), axial_slice_segmentation_normalized, cmap='gray')
            j += 1

    j = 0

    for i in range(nii_data.shape[1]):
        coronal_slice = nii_data[:, i, :]
        coronal_slice_segmentation = nii_segmentation_data[:, i, :]

        coronal_slice_normalized = preProcessingPhase(coronal_slice)
        coronal_slice_segmentation_normalized = preProcessingPhaseSegmentationMask(coronal_slice_segmentation)

        if np.mean(coronal_slice_normalized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'coronal/slices/coronal_slice_{j}.png'), coronal_slice_normalized, cmap='gray')
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'coronal/slices_segmentation/coronal_slice_{j}.png'), coronal_slice_segmentation_normalized, cmap='gray')
            j += 1

    j = 0

    for i in range(nii_data.shape[0]):
        sagittal_slice = nii_data[i, :, :]
        sagittal_slice_segmentation = nii_segmentation_data[i, :, :]

        sagittal_slice_normalized = preProcessingPhase(sagittal_slice)
        sagittal_slice_segmentation_normalized = preProcessingPhaseSegmentationMask(sagittal_slice_segmentation)

        if np.mean(sagittal_slice_normalized) != 0:
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'sagittal/slices/sagittal_slice_{j}.png'), sagittal_slice_normalized, cmap='gray')
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, f'sagittal/slices_segmentation/sagittal_slice_{j}.png'), sagittal_slice_segmentation_normalized, cmap='gray')
            j += 1

print("All slices saved successfully!")
