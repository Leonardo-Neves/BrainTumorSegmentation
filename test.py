import pandas as pd
import cv2

image_path = r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean\train\BraTS20_Training_001\sagittal.png'

image = cv2.imread(image_path)

dataframe = pd.read_csv('ellipses_indexes_mask_segmentation.csv', sep=';')

row = dataframe.iloc[0]

ellipse_caracteristics = [(row['Sagittal_Center_X'], row['Sagittal_Center_Y']), (row['Sagittal_Width'], row['Sagittal_Height']), row['Sagittal_Angle']]

cv2.ellipse(image, ellipse_caracteristics, 255, 2)

cv2.imshow('image', image)

cv2.waitKey(0)