import tensorflow as tf
from tensorflow import keras

import cv2
import pandas as pd
import os

model = tf.keras.models.load_model('model')

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean'

for image_name in os.listdir(ROOT_PATH):

    image_axial = cv2.imread(os.path.join(ROOT_PATH, image_name), cv2.IMREAD_GRAYSCALE)

    image_axial_inference = tf.expand_dims(image_axial.reshape((640, 800, 1)), 0)

    predictions_axial = model.predict(image_axial_inference)
    cv2.ellipse(image_axial, [(int(predictions_axial[0][0]), int(predictions_axial[0][1])), (int(predictions_axial[0][2]), int(predictions_axial[0][3])), int(predictions_axial[0][4])], 255, 1)

    cv2.imshow('image_axial', image_axial)

    cv2.waitKey(0)

