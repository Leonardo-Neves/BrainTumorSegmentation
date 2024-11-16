import cv2
import numpy as np

image = cv2.imread(r"C:\Users\leosn\Desktop\PIM\datasets\results_segmentation\BraTS20_Training_013\BraTS20_Training_013_segmented_mask_axial.png", cv2.IMREAD_GRAYSCALE)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Image", image)

cv2.waitKey(0)

