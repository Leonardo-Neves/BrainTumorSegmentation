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

j = 0

for i in range(nii_data.shape[2]):
    axial_slice = nii_data[:, :, i]

    # print(np.max(axial_slice))
    # print(np.min(axial_slice))

    # axial_slice = (axial_slice / np.max(axial_slice)) * 255

    axial_slice_8 = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # # if np.mean(axial_slice) != 0:

    axial_slice_8 = cv2.resize(axial_slice_8, (640, 640))

    # # cv2.imshow(f'image {i}', cv2.equalizeHist(axial_slice_resized))

    if i == 77:

        mask = np.where(axial_slice_8 == 0, 0, 1)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # cv2.imshow(f'oliginal', axial_slice_8)
        
        non_uniform_region = cv2.bitwise_and(axial_slice_8, axial_slice_8, mask=mask)

        pixels = non_uniform_region[mask == 255]

        mean_value = np.mean(pixels)
        min_value = np.min(pixels)
        max_value = np.max(pixels)
        unique, counts = np.unique(pixels, return_counts=True)

        # print("Mean pixel value:", mean_value)
        # print("Min pixel value:", min_value)
        # print("Max pixel value:", max_value)
        # print("Pixel value distribution (Intensity : Frequency):", dict(zip(unique, counts)))

        plt.bar(unique, counts)
        plt.show()
        
        hist, bins = np.histogram(pixels, 256, [0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
        image_equalized = cdf_final[axial_slice_8]

        non_uniform_region = cv2.bitwise_and(image_equalized, image_equalized, mask=mask)
        pixels = non_uniform_region[mask == 255]
        unique, counts = np.unique(pixels, return_counts=True)
        plt.bar(unique, counts)
        plt.show()

        cv2.imshow(f'image_equalized', image_equalized)

        plt.show()
        cv2.waitKey(0) 


        # brain_region = cv2.bitwise_and(axial_slice, axial_slice, mask=mask)
        
        # non_zero_pixels = axial_slice[axial_slice > 0]

        # brain_region_equalized = cv2.equalizeHist(brain_region)

        # image_dataframe = dip.probabilityIntensity(non_zero_pixels)

        
    #     (h, w) = axial_slice_resized.shape[:2]

    #     center = (w // 2, h // 2)

    #     rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1)

    #     rotated_image = cv2.warpAffine(axial_slice_resized, rotation_matrix, (w, h))

    #     rotated_image = cv2.equalizeHist(rotated_image)


        

       

    j += 1
