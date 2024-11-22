import numpy as np
import pandas as pd
import cv2
import os

import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\results_segmentation'

num_classes = 2

jaccard = MulticlassJaccardIndex(num_classes=2)

panoptic_quality_metric = PanopticQuality(things=[1], stuffs=[0])

dataframe = []

for folder_name in os.listdir(ROOT_PATH):

    print(f'{folder_name} start')

    
    iou_score_axial = 0
    panoptic_quality_score_axial = 0
    dice_coefficient_axial = 0

    iou_score_coronal = 0
    panoptic_quality_score_coronal = 0
    dice_coefficient_coronal = 0

    iou_score_sagittal = 0
    panoptic_quality_score_sagittal = 0
    dice_coefficient_sagittal = 0

    # -------------------------- Axial --------------------------

    if os.path.exists(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_axial.png')):
        
        original_mask_axial = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_axial.png'), cv2.IMREAD_GRAYSCALE)
        segmented_mask_axial = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_segmented_mask_axial.png'), cv2.IMREAD_GRAYSCALE)

        # Aggregated Jaccard Index

        original_mask_axial_binary = torch.tensor(original_mask_axial, dtype=torch.int) // 255
        segmented_mask_axial_binary = torch.tensor(segmented_mask_axial, dtype=torch.int) // 255

        iou_score_axial = jaccard(segmented_mask_axial_binary, original_mask_axial_binary)

        # Panoptic Quality

        ground_truth_instance = (torch.tensor(original_mask_axial) == 255).long()
        prediction_instance = (torch.tensor(segmented_mask_axial) == 255).long()

        ground_truth_tensor = torch.stack((ground_truth_instance, ground_truth_instance), dim=-1) 
        prediction_tensor = torch.stack((prediction_instance, prediction_instance), dim=-1)

        panoptic_quality_metric.update(prediction_tensor, ground_truth_tensor)
        panoptic_quality_score_axial = panoptic_quality_metric.compute()

        # Dice 

        mask_original_binary = (original_mask_axial > 0).astype(np.uint8)
        mask_segmented_binary = (segmented_mask_axial > 0).astype(np.uint8)

        intersection = np.sum(mask_original_binary & mask_segmented_binary)
        union = np.sum(mask_original_binary) + np.sum(mask_segmented_binary)

        dice_coefficient_axial = (2 * intersection) / union if union > 0 else 1.0

    # -------------------------- Coronal --------------------------

    if os.path.exists(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_coronal.png')):

        original_mask_coronal = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_coronal.png'), cv2.IMREAD_GRAYSCALE)
        segmented_mask_coronal = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_segmented_mask_coronal.png'), cv2.IMREAD_GRAYSCALE)

        # Aggregated Jaccard Index

        original_mask_coronal_binary = torch.tensor(original_mask_coronal, dtype=torch.int) // 255
        segmented_mask_coronal_binary = torch.tensor(segmented_mask_coronal, dtype=torch.int) // 255

        iou_score_coronal = jaccard(segmented_mask_coronal_binary, original_mask_coronal_binary)

        # Panoptic Quality

        ground_truth_instance = (torch.tensor(original_mask_coronal) == 255).long()
        prediction_instance = (torch.tensor(segmented_mask_coronal) == 255).long()

        ground_truth_tensor = torch.stack((ground_truth_instance, ground_truth_instance), dim=-1) 
        prediction_tensor = torch.stack((prediction_instance, prediction_instance), dim=-1)

        panoptic_quality_metric.update(prediction_tensor, ground_truth_tensor)
        panoptic_quality_score_coronal = panoptic_quality_metric.compute()

        # Dice 

        mask_original_binary = (original_mask_coronal > 0).astype(np.uint8)
        mask_segmented_binary = (segmented_mask_coronal > 0).astype(np.uint8)

        intersection = np.sum(mask_original_binary & mask_segmented_binary)
        union = np.sum(mask_original_binary) + np.sum(mask_segmented_binary)

        dice_coefficient_coronal = (2 * intersection) / union if union > 0 else 1.0

    # -------------------------- Sagittal --------------------------

    if os.path.exists(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_sagittal.png')):

        original_mask_sagittal = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_original_mask_sagittal.png'), cv2.IMREAD_GRAYSCALE)
        segmented_mask_sagittal = cv2.imread(os.path.join(ROOT_PATH, folder_name, f'{folder_name}_segmented_mask_sagittal.png'), cv2.IMREAD_GRAYSCALE)

        # Aggregated Jaccard Index

        original_mask_sagittal_binary = torch.tensor(original_mask_sagittal, dtype=torch.int) // 255
        segmented_mask_sagittal_binary = torch.tensor(segmented_mask_sagittal, dtype=torch.int) // 255

        iou_score_sagittal = jaccard(segmented_mask_sagittal_binary, original_mask_sagittal_binary)

        # Panoptic Quality

        ground_truth_instance = (torch.tensor(original_mask_sagittal) == 255).long()
        prediction_instance = (torch.tensor(segmented_mask_sagittal) == 255).long()

        ground_truth_tensor = torch.stack((ground_truth_instance, ground_truth_instance), dim=-1) 
        prediction_tensor = torch.stack((prediction_instance, prediction_instance), dim=-1)

        panoptic_quality_metric.update(prediction_tensor, ground_truth_tensor)
        panoptic_quality_score_sagittal = panoptic_quality_metric.compute()

        # Dice 

        mask_original_binary = (original_mask_sagittal > 0).astype(np.uint8)
        mask_segmented_binary = (segmented_mask_sagittal > 0).astype(np.uint8)

        intersection = np.sum(mask_original_binary & mask_segmented_binary)
        union = np.sum(mask_original_binary) + np.sum(mask_segmented_binary)

        dice_coefficient_sagittal = (2 * intersection) / union if union > 0 else 1.0

    dataframe.append([
        folder_name, 
        0 if iou_score_axial == 0 else iou_score_axial.item(), 
        0 if iou_score_coronal == 0 else iou_score_coronal.item(), 
        0 if iou_score_sagittal == 0 else iou_score_sagittal.item(), 
        0 if panoptic_quality_score_axial == 0 else panoptic_quality_score_axial.item(), 
        0 if panoptic_quality_score_coronal == 0 else panoptic_quality_score_coronal.item(), 
        0 if panoptic_quality_score_sagittal == 0 else panoptic_quality_score_sagittal.item(), 
        0 if dice_coefficient_axial == 0 else dice_coefficient_axial, 
        0 if dice_coefficient_coronal == 0 else dice_coefficient_coronal, 
        0 if dice_coefficient_sagittal == 0 else dice_coefficient_sagittal
    ])

    print(f'{folder_name} end')

dataframe = pd.DataFrame(dataframe, columns=['Image', 'Jaccard Index Axial', 'Jaccard Index Coronal', 'Jaccard Index Sagittal', 'Panoptic Quality Axial', 'Panoptic Quality Coronal', 'Panoptic Quality Sagittal', 'Dice Coefficient Axial', 'Dice Coefficient Coronal', 'Dice Coefficient Sagittal'])

dataframe.to_csv('metrics.csv', sep=';', index=False)