import cv2
import time

img = cv2.imread("./landscape.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not open or find the image.")
    exit(0)

filtered_img = img.copy()

size = 3
offset = int(size/2)
rows, cols = img.shape

t_inicial = time.perf_counter()

for i in range(offset, rows-offset):
    for j in range(offset, cols-offset):

        suma = 0
        for k in range(-offset, offset+1):
            for l in range(-offset, offset+1):
                suma += int(img[i+k][j+l])

        p = int(suma/(size*size))
        filtered_img[i][j] = p

t_final = time.perf_counter()
print("Tiempo aplicando el filtro: ", t_final - t_inicial, " segundos.")

cv2.imshow("Image", img)
cv2.imshow("Mean Filter", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


