from collections import Counter
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DigitalImageProcessing:

    def probabilityIntensity(self, image):
        row_vector = image.flatten()
        width, height = image.shape

        counts = Counter(row_vector)
        count = counts.items() # [(intensity, quantity)]

        rows = []

        for intensity, quantity in counts.items():

            # Probability = n_k / MN
            probability = quantity / (width * height)
            rows.append([intensity, quantity, probability])


        return pd.DataFrame(rows, columns = ['Intensity', 'Quantity', 'Probability'])
    
    def cumulativeDistribution(self, df):
        
        df = df.sort_values(by='Intensity').reset_index(drop=True)

        df['Cumulative Probability'] = df['Probability'].cumsum()

        return df
    
    def histogramEqualization(self, image):
        """
        Perform histogram equalization on a grayscale image.

        Parameters:
        image (numpy.ndarray): Grayscale input image.

        Returns:
        numpy.ndarray: Histogram equalized image.
        """
        # Check if the image is grayscale
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale.")

        # Get the histogram of the image
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # Compute the cumulative distribution function (CDF)
        cdf = hist.cumsum()

        # Normalize the CDF
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        # Mask to avoid division by zero
        cdf_masked = np.ma.masked_equal(cdf, 0)

        # Perform histogram equalization
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')

        # Map the original image pixels to equalized values
        image_equalized = cdf_final[image]

        return image_equalized
    
    def MDC(self, mdc_image):
        width, height = mdc_image.shape

        # print('mdc_image.shape', mdc_image.shape)

        threshold = 0

        left = [0, 0]
        right = [0, 0]

        top = [0, 0]
        bottom = [0, 0]

        height_half = int(height/2)
        width_half = int(width/2)

        # Left
        for i in range(0, width_half):
            
            if mdc_image[height_half-1, i] > threshold:
                left = [i, height_half-1]
                break
        
        # Right
        for i in reversed(range(width_half, width)):

            if mdc_image[height_half-1, i] > threshold:
                right = [i, height_half-1]
                break

        # Top
        for i in range(0, height_half):
            if mdc_image[i, width_half-1] > threshold:
                top = [width_half-1, i]
                break

        # Bottom
        for i in reversed(range(height_half, height)):
            if mdc_image[i, width_half-1] > threshold:
                bottom = [width_half-1, i]
                break

        radius = 1
        color = 255
        thickness = 2

        DISPLACEMENT_LEFT = 30
        DISPLACEMENT_RIGHT = 30
        DISPLACEMENT_TOP = 30
        DISPLACEMENT_BOTTOM = 30

        # left[0] = left[0] + DISPLACEMENT_LEFT
        # right[0] = right[0] - DISPLACEMENT_RIGHT
        # top[1] = top[1] + DISPLACEMENT_TOP
        # bottom[1] = bottom[1] - DISPLACEMENT_BOTTOM

        cv2.circle(mdc_image, (left[0], left[1]), radius, color, thickness)
        cv2.circle(mdc_image, (right[0], right[1]), radius, color, thickness)
        cv2.circle(mdc_image, (top[0], top[1]), radius, color, thickness)
        cv2.circle(mdc_image, (bottom[0], bottom[1]), radius, color, thickness)

        # print(left, right, top, bottom)

        left, right, top, bottom = left[0], right[0], top[1], bottom[1]
        
        center = ((left + right) // 2, (top + bottom) // 2)
        axes = ((right - left) // 2, (bottom - top) // 2)

        mask = np.zeros_like(mdc_image)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        masked_image = cv2.bitwise_and(mdc_image, mdc_image, mask=mask)

        return masked_image
    
    def plotImage3D(self, image):
        x = np.arange(0, image.shape[1], 1)
        y = np.arange(0, image.shape[0], 1)
        x, y = np.meshgrid(x, y)

        z = image

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(x, y, z, color='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')

        plt.show()