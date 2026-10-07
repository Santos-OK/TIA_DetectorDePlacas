import cv2
import matplotlib.pyplot as plt
import numpy as np

# Imagen
f1 = np.array(
    [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 2, 2, 1, 0],
        [1, 2, 5, 5, 2, 1],
        [0, 2, 5, 4, 3, 1],
        [0, 0, 1, 3, 2, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)

print("--- Matriz Original f1(x,y) ---")
print(f1)

# Histograma
hist = cv2.calcHist([f1], [0], None, [6], [0, 6]).flatten()

print("\nCantidades histograma: ")
for i, count in enumerate(hist):
    print(f"Intensidad {i}: {int(count)} píxeles")

_, manualThreshold = cv2.threshold(f1, 3, 255, cv2.THRESH_BINARY)

# Umbralización k = 3
print("\n--- Umbralización manual k = 3 ---")
print(manualThreshold)

#Umbralización promedio
def k_promedio(img, tol = 1e-2):
    p = float(np.mean(img))

    while True:

        i1 = img[img > p]
        i2 = img[img <= p]

        if i1.size > 0:
            u1 = float(np.mean(i1))
        else:
            u1 = 0.0

        if i2.size > 0:
            u2 = float(np.mean(i2))
        else:
            u2 = 0.0

        newP = (u1+u2)/2.0

        if abs(newP - p) < tol:
            break

        p = newP
    return p

averageK =  k_promedio(f1)

_, averageThreshold = cv2.threshold(f1, averageK, 255, cv2.THRESH_BINARY)

# Umbralización k promedio
print("\n--- Umbralización Promedio ---")
print(averageK)
print(averageThreshold)

f1_openCV = (f1 * (255//5))

# Umbralización k Otsu
kOtsu, otsuThreshold = cv2.threshold(f1_openCV, 0, 255, cv2.THRESH_OTSU)
kOtsuReal = float(kOtsu) * (5.0 / 255.0)

print("\n--- Umbralización Otsu ---")
print(kOtsuReal)
print(otsuThreshold)

# Visualización del Histograma
plt.figure(figsize=(6, 4))
plt.bar(range(6), hist, color="gray", edgecolor="black")
plt.title("Histograma de la Imagen f1(x,y)")
plt.xlabel("Intensidad I")
plt.ylabel("Frecuencia (N° de Píxeles)")
plt.xticks(range(6))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()