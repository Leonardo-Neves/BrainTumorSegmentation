import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()

image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2020_Data_Training\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii"

nii_file = nib.load(image_path)
nii_data = nii_file.get_fdata()

for i in range(nii_data.shape[2]):
    axial_slice = nii_data[:, :, i]

    axial_slice_8 = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axial_slice_8 = cv2.resize(axial_slice_8, (640, 640))

    if i == 77:

        # Selecting only the region of the brain
        mask = np.where(axial_slice_8 == 0, 0, 1)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        non_uniform_region = cv2.bitwise_and(axial_slice_8, axial_slice_8, mask=mask)
        pixels = non_uniform_region[mask == 255]

        # mean_value = np.mean(pixels)
        # min_value = np.min(pixels)
        # max_value = np.max(pixels)
        # unique, counts = np.unique(pixels, return_counts=True)

        # print("Mean pixel value:", mean_value)
        # print("Min pixel value:", min_value)
        # print("Max pixel value:", max_value)
        # print("Pixel value distribution (Intensity : Frequency):", dict(zip(unique, counts)))

        # plt.bar(unique, counts)
        # plt.show()
        
        # Histogram Equalization
        hist, bins = np.histogram(pixels, 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
        image_equalized = cdf_final[axial_slice_8]

        # non_uniform_region = cv2.bitwise_and(image_equalized, image_equalized, mask=mask)
        # pixels = non_uniform_region[mask == 255]
        # unique, counts = np.unique(pixels, return_counts=True)
        # plt.bar(unique, counts)
        # plt.show()

        # Filtering images to remove paper and salt noise
        image = np.array(image_equalized)

        filtered_image = np.zeros_like(image)

        def medianFilter(matrix_neighborhood):
            return np.median(matrix_neighborhood)

        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighborhood = np.array([[image[i-1, j-1], image[i-1, j],   image[i-1, j+1]],
                                        [image[i, j-1],   image[i-1, j-1], image[i, j+1]],
                                        [image[i+1, j-1], image[i+1, j],   image[i+1, j+1]]])

                filtered_image[i, j]= medianFilter(neighborhood)

        cv2.imshow(f'image_equalized', image_equalized)
        cv2.imshow(f'filtered_image', filtered_image)

        plt.show()
        cv2.waitKey(0) 

