import numpy as np
import cv2


area1 = 230
area2 = 281

# Mean between two areas
arr1 = np.array([area1, area2])

arr1 = arr1 / np.max(arr1)

print('arr1: ', arr1)

print('mean: ', np.mean(arr1))
print('std: ', np.std(arr1))

# contour1 = contours1[0]
# contour2 = contours2[0]

# # Step 5: Compare the shapes using cv2.matchShapes()
# similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

# if area1 > area2:
#     print('pass')
# elif