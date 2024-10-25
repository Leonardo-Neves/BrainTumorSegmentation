import numpy as np
import cv2

class SpartialDomain:

    def leoThreshold(self, image, mask_image, kernel_size=3):

        region_of_interest = cv2.bitwise_and(image, image, mask=mask_image)
        roi_values = np.array(region_of_interest[region_of_interest > 0])

        global_mean = np.mean(roi_values)
        global_std = np.std(roi_values)

        sides = int((kernel_size - 1) / 2)

        padded_image = np.pad(image, pad_width=sides, mode='constant', constant_values=0)

        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighbor = np.array(padded_image[i-sides:i+(sides + 1), j-sides:j+(sides + 1)])
                neighbor_mean = np.mean(neighbor)

                if neighbor_mean > global_mean:
                    global_mean = neighbor_mean
                    global_std = np.std(neighbor)

        return np.where((image >= (global_mean - global_std)) & (image <= (global_mean + global_std)), 255, 0).astype(np.uint8)
    
                    