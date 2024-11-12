import json
import cv2
import numpy as np
import pandas as pd

from pycocotools.mask import decode, frPyObjects
from pycocotools.coco import COCO

with open('instances_default.json') as f:
    data = json.load(f)

cocoo = COCO('instances_default.json')

dataframe = []

for annotation in data['annotations']:

    for image in data['images']:
        if annotation['image_id'] == image['id']:

            ann_ids = cocoo.getAnnIds(imgIds=image['id'])
            anns = cocoo.loadAnns(ann_ids)

            rleObjs = [frPyObjects(obj["segmentation"], 800, 640) for obj in anns]
            mask = decode(rleObjs)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:  # Need at least 5 points to fit an ellipse
                    ellipse = cv2.fitEllipse(largest_contour)
                    center, axes, angle = ellipse[0], ellipse[1], ellipse[2]

                    dataframe.append([image['file_name'].replace('.png', ''), center[0], center[1], axes[0], axes[1], angle])

dataframe = pd.DataFrame(dataframe, columns=['Image', 'Center_X', 'Center_Y', 'Major_Axis', 'Minor_Axis', 'Angle'])

row = dataframe.iloc[0]

image = cv2.imread(r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean\{}.png'.format(row["Image"]), cv2.IMREAD_GRAYSCALE)

cv2.ellipse(image, [(int(row["Center_X"]), int(row["Center_Y"])), (int(row["Major_Axis"]), int(row["Minor_Axis"])), row["Angle"]], 255, 1)

cv2.imshow('Image', image)

cv2.waitKey(0)


# dataframe.to_csv('annotations.csv', index=False, sep=';')