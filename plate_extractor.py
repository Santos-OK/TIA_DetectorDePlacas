import cv2
import numpy as np

# 0. Read image
img = cv2.imread("./placa4.jpg", cv2.IMREAD_UNCHANGED)

# 1. Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Convert to binary through a threshold
binary_img = cv2.adaptiveThreshold(
    gray_img, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=21, 
    C=9
)

# 3. Connected components
connectivity = 8
objects, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity)


# 4. Select areas
areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, objects)]

filtered_areas = [a for a in areas if a > 50]
average = np.mean(filtered_areas)

big_areas_id = []
for i in range(1, objects):
    if stats[i, cv2.CC_STAT_AREA] > average:
        big_areas_id.append(i)

probable_areas = sorted(big_areas_id, key=lambda label_id: stats[label_id, cv2.CC_STAT_AREA], reverse = True)

if len(probable_areas) >= 2:
    placa_label_id = probable_areas[2]  # El segundo más grande
elif len(probable_areas) == 1:
    placa_label_id = probable_areas[0]  # Único objeto grande
else:
    placa_label_id = None

mask_placa = np.zeros(binary_img.shape, dtype=np.uint8)

if placa_label_id is not None:
    mask_placa[labels == placa_label_id] = 255

cv2.imshow('mask', mask_placa)
cv2.imshow('Image', img)
cv2.imshow('Grayscale', gray_img)
cv2.imshow('Binary', binary_img)
cv2.waitKey(0)  