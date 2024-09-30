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

    # axial_slice = (axial_slice / np.max(axial_slice)) * 255

    axial_slice = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if np.mean(axial_slice) != 0:

        axial_slice_resized = cv2.resize(axial_slice, (640, 640))

        cv2.imshow(f'image {j}', cv2.equalizeHist(axial_slice_resized))

        cv2.waitKey(0) 

        j += 1
