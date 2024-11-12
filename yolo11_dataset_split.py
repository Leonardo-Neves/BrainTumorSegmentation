import os
import random
import shutil

ROOT_PATH = r'C:\Users\leosn\Desktop\PIM\datasets\tumor_object_detection'

labels = [filename for filename in os.listdir(os.path.join(ROOT_PATH, 'labels')) if filename.endswith('.txt')]
dataset_size = len(labels)

random.shuffle(labels)

size = int(dataset_size * 0.8)

print(f'Size: {size}')

for i, filename_label in enumerate(labels):

    filename_image = f"{filename_label.split('.')[0]}.png"
    image_path = os.path.join(ROOT_PATH, 'images', filename_image)

    if i <= size:
        shutil.move(os.path.join(ROOT_PATH, 'images', filename_image), os.path.join(ROOT_PATH, 'images', 'train', filename_image))
        shutil.move(os.path.join(ROOT_PATH, 'labels', filename_label), os.path.join(ROOT_PATH, 'labels', 'train', filename_label))
    elif i > size:
        shutil.move(os.path.join(ROOT_PATH, 'images', filename_image), os.path.join(ROOT_PATH, 'images', 'val', filename_image))
        shutil.move(os.path.join(ROOT_PATH, 'labels', filename_label), os.path.join(ROOT_PATH, 'labels', 'val', filename_label))
