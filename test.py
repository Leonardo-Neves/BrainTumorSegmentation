import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

from utils.digital_image_processing import DigitalImageProcessing

dip = DigitalImageProcessing()




image_path = r"C:\Users\leosn\Desktop\PIM\datasets\MICCAI_BraTS_2019_Data_Training\HGG\BraTS19_2013_2_1\BraTS19_2013_2_1_t1ce.nii"

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

        # mask = np.where(axial_slice == 0, 0, 1)
        # mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # brain_region = cv2.bitwise_and(axial_slice, axial_slice, mask=mask)
        
        # non_zero_pixels = axial_slice[axial_slice > 0]

        # brain_region_equalized = cv2.equalizeHist(brain_region)

        # image_dataframe = dip.probabilityIntensity(non_zero_pixels)

        # plt.bar(image_dataframe['Intensity'], image_dataframe['Probability'])
        # plt.show()

    #     (h, w) = axial_slice_resized.shape[:2]

    #     center = (w // 2, h // 2)

    #     rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1)

    #     rotated_image = cv2.warpAffine(axial_slice_resized, rotation_matrix, (w, h))

    #     rotated_image = cv2.equalizeHist(rotated_image)


        # cv2.imshow(f'image {i}', brain_region_equalized)

        # cv2.waitKey(0) 

    j += 1
