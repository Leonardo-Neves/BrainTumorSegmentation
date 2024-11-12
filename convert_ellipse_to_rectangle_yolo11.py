import numpy as np
import pandas as pd

import cv2
import os
import math

def ellipse_to_rectangle(ellipse, image_width, image_height):
    # Ellipse parameters: (cx, cy, a, b, theta)
    cx, cy, a, b, theta = ellipse

    width = math.sqrt((a * math.cos(theta))**2 + (b * math.sin(theta))**2)
    height = math.sqrt((a * math.sin(theta))**2 + (b * math.cos(theta))**2)

    return (cx/image_width, cy/image_height, width/image_width, height/image_height)

dataframe = pd.read_csv('annotations.csv', sep=';')

image_width = 800
image_height = 640

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\tumor_object_detection\images'

OUTPUT_TXT_FILES = r'C:\Users\leosn\Desktop\PIM\datasets\tumor_object_detection\labels'

for i, row in dataframe.iterrows():
    ellipse = [row['Center_X'], row['Center_Y'], row['Major_Axis'], row['Minor_Axis'], row['Angle']]
    
    rectangle = ellipse_to_rectangle(ellipse, image_width, image_height)

    # image = cv2.imread(os.path.join(ROOT_PATH, f"{row['Image']}.png"), cv2.IMREAD_GRAYSCALE)

    x_center, y_center, width, height = int(rectangle[0] * image_width), int(rectangle[1] * image_height), int(rectangle[2] * image_width), int(rectangle[3] * image_height)

    x1, y1 = x_center - (width / 2), y_center - (height / 2)
    x2, y2 = x_center + (width / 2), y_center - (height / 2)
    x3, y3 = x_center + (width / 2), y_center + (height / 2)
    x4, y4 = x_center - (width / 2), y_center + (height / 2)

    # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

    # cv2.imshow('Image', image)

    # cv2.waitKey(0)

    file_value = f'0 {x1 / image_width} {y1 / image_height} {x2 / image_width} {y2 / image_height} {x3 / image_width} {y3 / image_height} {x4 / image_width} {y4 / image_height}'

    file = open(os.path.join(OUTPUT_TXT_FILES, f"{row['Image']}.txt"), 'w')
    file.write(file_value)
    file.close()

    
    
