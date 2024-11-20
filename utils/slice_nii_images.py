import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import warnings
import concurrent.futures as futures

warnings.filterwarnings("ignore")

ROOT_PATH = 'datasets/MICCAI_BraTS_2020_Data_Training/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

OUTPUT_PATH = 'datasets/MICCAI_BraTS2020_TrainingDataSliced'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def threadProcessment(folder_name):
    print(folder_name)
    try:
        global ROOT_PATH
        global OUTPUT_PATH

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
        
        started = False
        index_started_axial = 0
        index_ended_axial = 0

        for i in range(nii_data.shape[2]):
            axial_slice = nii_data[:, :, i]
            
            image_8bits = cv2.normalize(axial_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))

            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, 'axial', 'slices', f'{i}.png'), image_8bits, cmap='gray')

        started = False
        index_started_coronal = 0
        index_ended_coronal = 0

        for i in range(nii_data.shape[1]):
            coronal_slice = nii_data[:, i, :]
            
            image_8bits = cv2.normalize(coronal_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))

            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, 'coronal', 'slices', f'{i}.png'), image_8bits, cmap='gray')
        
        started = False
        index_started_sagittal = 0
        index_ended_sagittal = 0

        for i in range(nii_data.shape[0]):
            sagittal_slice = nii_data[i, :, :]

            image_8bits = cv2.normalize(sagittal_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_8bits = cv2.rotate(image_8bits, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_8bits = cv2.resize(image_8bits, (800, 640))
            
            plt.imsave(os.path.join(OUTPUT_PATH, folder_name, 'sagittal', 'slices', f'{i}.png'), image_8bits, cmap='gray')
        
        return 1
    except Exception as e:
        print(e)
    
counter = 0

with futures.ThreadPoolExecutor(30) as executor:
            
    future_to_get_bv = {executor.submit(threadProcessment, folder_name): folder_name for i, folder_name in enumerate(os.listdir(ROOT_PATH))}

    for future in futures.as_completed(future_to_get_bv):
        number = future.result()

        counter += number

        print("Processed folders:", counter)

print("All slices saved successfully!")
