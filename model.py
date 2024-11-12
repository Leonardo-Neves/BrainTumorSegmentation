import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

DATASET_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean'

images_axial = []
images_coronal = []
images_sagittal = []

dataframe = pd.read_csv('annotations.csv', sep=';')

X_values = []
Y_values = []

for i, row in dataframe.iterrows():
    X_values.append(cv2.imread(r'C:\Users\leosn\Desktop\PIM\datasets\mask_mean\{}.png'.format(row["Image"]), cv2.IMREAD_GRAYSCALE))
    Y_values.append([row['Center_X'], row['Center_Y'], row['Major_Axis'], row['Minor_Axis'], row['Angle']])

# for folder_name in os.listdir(DATASET_PATH):

#     image_axial = cv2.imread(os.path.join(DATASET_PATH, folder_name, f'axial.png'), cv2.IMREAD_GRAYSCALE)
#     image_coronal = cv2.imread(os.path.join(DATASET_PATH, folder_name, f'coronal.png'), cv2.IMREAD_GRAYSCALE)
#     image_sagittal = cv2.imread(os.path.join(DATASET_PATH, folder_name, f'sagittal.png'), cv2.IMREAD_GRAYSCALE)

#     images_axial.append(image_axial)
#     images_coronal.append(image_coronal)
#     images_sagittal.append(image_sagittal)

# dataframe = pd.read_csv('ellipses_indexes_mask_segmentation.csv', sep=';')

# y_label_axial = []
# y_label_coronal = []
# y_label_sagital = []

# for i, row in dataframe.iterrows():
#     y_label_axial.append([row['Axial_Center_X'], row['Axial_Center_Y'], row['Axial_Width'], row['Axial_Height'], row['Axial_Angle']])
#     y_label_coronal.append([row['Coronal_Center_X'], row['Coronal_Center_Y'], row['Coronal_Width'], row['Coronal_Height'], row['Coronal_Angle']])
#     y_label_sagital.append([row['Sagittal_Center_X'], row['Sagittal_Center_Y'], row['Sagittal_Width'], row['Sagittal_Height'], row['Sagittal_Angle']])

X_values = np.array(X_values)
Y_values = np.array(Y_values)

print('X_values shape:', X_values.shape)
print('Y_values shape:', Y_values.shape)

size_x = int(len(X_values) * 0.8)
size_y = int(len(Y_values) * 0.8)

X_train = X_values[:size_x]
X_val = X_values[size_x:]

Y_train = Y_values[:size_y]
Y_val = Y_values[size_y:]

print(f'X_train: {len(X_train)} X_val: {len(X_val)}')
print(f'Y_train: {len(Y_train)} Y_val: {len(Y_val)}')

image_size = (640, 800)

inputs = keras.Input(shape=(640, 800, 1))

x = layers.Rescaling(1.0 / 255)(inputs)

# Model 1

# x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Flatten()(x)
# # x = layers.Dense(32, activation='relu')(x)
# # x = layers.Dropout(0.5)(x)
# x = layers.Dense(16, activation='relu')(x)
# x = layers.Dense(8, activation='relu')(x)
# outputs = layers.Dense(5, activation='linear')(x)

# Model 2


x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
# x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
# x = layers.Dense(32, activation='relu')(x)
# x = layers.Dropout(0.5)(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
outputs = layers.Dense(5, activation='linear')(x)

model = keras.Model(inputs, outputs)

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

epochs = 30

callbacks = [
    # keras.callbacks.ModelCheckpoint("epoch_{epoch}.keras"),
    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
]

history = model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=(X_val, Y_val), callbacks=callbacks)

model.save('model')

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
