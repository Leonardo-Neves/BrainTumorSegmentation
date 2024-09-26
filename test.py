import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# Create or load a grayscale image (example random image)
image = np.random.random((100, 100))

neighborhood_size = 5
local_max = ndimage.maximum_filter(image, size=neighborhood_size) == image

background = (image == 0)
eroded_background = ndimage.binary_erosion(background, structure=np.ones((3, 3)))
detected_peaks = local_max ^ eroded_background

# Visualize the result
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(detected_peaks, cmap='gray')
plt.title('Regional Maxima')

plt.show()
