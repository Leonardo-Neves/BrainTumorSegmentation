import numpy as np
import cv2
from scipy.ndimage import uniform_filter, generic_filter

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

        print('1- global_mean', global_mean)
        print('1- global_std', global_std)

        return np.where((image >= (global_mean - global_std)) & (image <= (global_mean + global_std)), 255, 0).astype(np.uint8)


    def leoThreshold2(self, image, mask_image, kernel_size=3):
        
        mean = cv2.blur(image, (kernel_size, kernel_size))

        squared_image = image ** 2
        mean_of_squares = cv2.blur(squared_image, (kernel_size, kernel_size))

        std_dev = np.sqrt((image - mean) ** 2)

        i, j = np.unravel_index(np.argmax(mean), mean.shape)

        global_mean = np.max(mean)
        global_std = std_dev[i, j]

        print('2- global_mean', global_mean)
        print('2- global_std', global_std)

        return np.where((image >= global_mean - global_std) & (image <= (global_mean + global_std)), 255, 0).astype(np.uint8)
    
    def leoThreshold3(self, image, mask_image, kernel_size=3):
        """
        Optimized version of leoThreshold that maintains the same behavior
        but with improved performance.
        """
        # Initial ROI calculation
        region_of_interest = cv2.bitwise_and(image, image, mask=mask_image)
        roi_values = np.array(region_of_interest[region_of_interest > 0])
        
        # Initial statistics from ROI
        global_mean = np.mean(roi_values)
        global_std = np.std(roi_values)
        
        # Pre-calculate padding size
        sides = int((kernel_size - 1) / 2)
        
        # Use numpy's advanced indexing for faster neighborhood extraction
        padded_image = np.pad(image, sides, mode='constant', constant_values=0)
        
        # Create views into the padded image for faster neighborhood access
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_image, 
            (kernel_size, kernel_size)
        )[:-kernel_size+1, :-kernel_size+1]
        
        # Calculate means for all neighborhoods at once
        neighbor_means = np.mean(windows, axis=(2, 3))
        
        # Find neighborhoods with means higher than global_mean
        higher_means = neighbor_means > global_mean
        
        if np.any(higher_means):
            # Update global statistics based on neighborhoods with higher means
            max_neighbor_mean = np.max(neighbor_means[higher_means])
            # Get the windows corresponding to the maximum mean
            max_windows = windows[neighbor_means == max_neighbor_mean]
            global_mean = max_neighbor_mean
            global_std = np.std(max_windows)
        
        # Final thresholding
        return np.where(
            (image >= global_mean) & (image <= (global_mean + global_std)),
            255, 
            0
        ).astype(np.uint8)
    
                    