import numpy as np

class FrequencyDomain:

    def butterworthHighpassFilter(self, cutoff_frequency, order, shape = (0, 0)):
        rows, cols = shape[0], shape[1]
        center_row, center_col = rows // 2, cols // 2

        # New implementation
        butterworth_filter = np.ones((rows, cols), np.float32)
        center = (rows // 2, cols // 2)
        i, j = np.ogrid[:rows, :cols]

        distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
        butterworth_filter = 1 / (1 + (cutoff_frequency / distance)**(2 * order))

        return butterworth_filter

    def idealHighpassFilter(self, cutoff_frequency, shape=(0, 0)):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2

        # Create meshgrid for row and column indices
        x = np.arange(0, rows) - center_row
        y = np.arange(0, cols) - center_col
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Compute the distance from the center for all points
        distance = np.sqrt(X**2 + Y**2)

        # Apply the cutoff frequency threshold
        ideal_filter = np.where(distance > cutoff_frequency, 1, 0)

        return ideal_filter

    def gaussianHighpassFilter(self, cutoff_frequency, shape = (0, 0)):
    
        rows, cols = shape[0], shape[1]
        center_row, center_col = rows // 2, cols // 2

        # New implementation
        gaussian_filter = np.ones((rows, cols), np.float32)
        center = (rows // 2, cols // 2)
        i, j = np.ogrid[:rows, :cols]

        distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
        gaussian_filter = 1 - np.exp(-(distance**2) / (2 * (cutoff_frequency ** 2)))

        return gaussian_filter

    def filterImage(self, image, padded_image, kernel):

        # 1° f(x, y) as M x N, pad the image, P = 2M and Q = 2N
        padded_image = padded_image

        # 2° Compute the Fourier transform and center the spectrum
        F_u_v = np.fft.fftshift(np.fft.fft2(padded_image))

        # 3° Creating Sobel kernel
        H_u_v = kernel

        # 4° Applying the filter to the image
        fft_filtered = F_u_v * H_u_v

        # 5° Apply Inverse Fourier Transform to obtain the filtered image
        filtered_image = np.abs(np.fft.ifft2(fft_filtered))

        # 6° Removing pad from the image
        height, width = image.shape
        filtered_image = filtered_image[:height, :width]

        return filtered_image, F_u_v

    def padImage(self, image):
        
        height, width = image.shape
        padded_image = np.zeros((2 * height, 2 * width), dtype=np.uint8)
        padded_image[:height, :width] = image

        return padded_image
