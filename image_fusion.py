import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_ubyte

from utils.digital_image_processing import DigitalImageProcessing


# dip = DigitalImageProcessing()

image1 = cv2.imread('images/192.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('images/193.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('images/194.jpg', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('images/195.jpg', cv2.IMREAD_GRAYSCALE)
image5 = cv2.imread('images/196.jpg', cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread('images/197.jpg', cv2.IMREAD_GRAYSCALE)

# image = (image1 + image2 + image3 + image4 + image5 + image6) / 6

# image = image.astype(np.int8)



max_matrix = np.maximum.reduce([image1, image2, image3, image4, image5, image6])

cv2.imshow('Image', max_matrix) 

cv2.waitKey(0)  

