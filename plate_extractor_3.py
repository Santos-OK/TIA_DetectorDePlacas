import cv2
import numpy as np

# 0. Leer imagen
img = cv2.imread("./placa4.jpg", cv2.IMREAD_UNCHANGED)

# 1. Convertir a escala de grises y reducir ruido
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# El filtro bilateral suaviza la textura del auto pero mantiene los bordes afilados
bfilter = cv2.bilateralFilter(gray_img, 11, 17, 17)

# 2. Detección de bordes y Cierre Morfológico
# Canny extrae los bordes fuertes (el marco de la placa y las letras)
edges = cv2.Canny(bfilter, 30, 200)

# Un kernel pequeño conecta líneas interrumpidas del marco sin fusionarlo con la defensa
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary_img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 3 y 4. Extraer objetos (Contornos) y seleccionar la placa
# findContours es ideal para identificar las formas rectangulares cerradas
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

placa_contour = None
max_area_valida = 0

for cnt in contours:
    # Obtener el rectángulo delimitador de cada contorno
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect_ratio = float(w) / h

    # Filtrar por proporción geométrica de una placa (aprox 2.0 a 5.0)
    # y descartar ruido (áreas muy pequeñas) o la imagen entera (áreas inmensas)
    if 2.0 <= aspect_ratio <= 5.0 and 5000 < area < 80000:
        if area > max_area_valida:
            max_area_valida = area
            placa_contour = cnt

# Crear la máscara binaria en negro
mask_placa = np.zeros(gray_img.shape, dtype=np.uint8)

if placa_contour is not None:
    # 5. Rellenar huecos (Dibujar el contorno sólido en la máscara)
    # Al pasar -1 en el grosor, OpenCV rellena el interior del contorno de blanco (255)
    cv2.drawContours(mask_placa, [placa_contour], -1, 255, -1)

    # Dibujar un rectángulo verde en la imagen original para comprobar el resultado
    x, y, w, h = cv2.boundingRect(placa_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    print(f"Placa encontrada con área: {max_area_valida} y Aspect Ratio: {float(w) / h:.2f}")
else:
    print("No se encontró ninguna placa con las proporciones dadas.")

# Mostrar resultados
cv2.imshow('1. Grayscale', gray_img)
cv2.imshow('2. Canny Edges', edges)
cv2.imshow('3. Binary Morph', binary_img)
cv2.imshow('4. Mask Placa', mask_placa)
cv2.imshow('5. Resultado', img)

cv2.waitKey(0)
cv2.destroyAllWindows()