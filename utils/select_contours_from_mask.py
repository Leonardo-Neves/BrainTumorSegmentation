import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
import cv2
import os

from utils.digital_image_processing import DigitalImageProcessing
from utils.frequency_domain import FrequencyDomain
from utils.spartial_domain import SpartialDomain

dip = DigitalImageProcessing()
fd = FrequencyDomain()
sd = SpartialDomain()

ROOT_PATH = r'C:\Users\leosn\Desktop\Slices\coronal'

for image_name in os.listdir(ROOT_PATH):

    if image_name.endswith('.png'):
        print(image_name)

        image_path = os.path.join(ROOT_PATH, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        cv2.imshow('image', image)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):

            mask = np.zeros_like(image)

            cv2.drawContours(mask, [contour], -1, 255, -1)
            print(i)
            cv2.imshow('contour', mask)

            cv2.waitKey(0)

