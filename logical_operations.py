import cv2
import numpy as np

dimension = [400, 500]

b1 = np.zeros(dimension)
b1 = cv2.rectangle(b1, (60, 120), (320, 340), (255, 255, 255), -1)

b2 = np.zeros(dimension)
b2 = cv2.rectangle(b2, (160, 30), (480, 250), (255, 255, 255), -1)

def AND_operation(matrix1, matrix2, dimension):
  return [[matrix1[i][j] & matrix2[i][j] for j in range(dimension[1])] for i in range(dimension[0])]

# B1 and B2
and_operation = cv2.bitwise_and(b1, b2) # OpenCV version

# B1 or B2
or_operation = cv2.bitwise_or(b1, b2)

# B1 and [NOT (B2)]
not_b2 = cv2.bitwise_not(b2)
b1andnotb2_operation = cv2.bitwise_and(b1, not_b2)

# B1 xor B2
b1xorb2_operation = cv2.bitwise_xor(b1, b2)


image1 = cv2.imread('images/196.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('images/197.jpg', cv2.IMREAD_GRAYSCALE) 

not_b2 = cv2.bitwise_not(image1)
b1andnotb2_operation = cv2.bitwise_and(image2, not_b2)



def medianFilter(matrix_neighborhood):
    return np.median(matrix_neighborhood)

image = np.array(image2)

filtered_image = np.zeros_like(image)


for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        neighborhood = np.array([[image[i-1, j-1], image[i-1, j],   image[i-1, j+1]],
                                [image[i, j-1],   image[i-1, j-1], image[i, j+1]],
                                [image[i+1, j-1], image[i+1, j],   image[i+1, j+1]]])

        filtered_image[i, j]= medianFilter(neighborhood)



cv2.imshow('image', filtered_image)

cv2.waitKey(0)  
