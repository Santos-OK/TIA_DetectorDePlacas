import cv2
import numpy as np

def detectar_placa_diapositiva(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavizado ligero para evitar ruido en el umbral
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Imagen Binaria (Otsu encuentra automáticamente el umbral óptimo)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    img_area = gray.shape[0] * gray.shape[1]
    
    mejor_placa = None
    mejor_score = 0
    
    for i in range(1, num_labels):
        area_obj = stats[i, cv2.CC_STAT_AREA]
        # Descartar ruido muy pequeño o el fondo entero
        if not (0.005 * img_area < area_obj < 0.60 * img_area):
            continue
            
        # Aislar el objeto
        obj_mask = np.zeros(gray.shape, dtype=np.uint8)
        obj_mask[labels == i] = 255
        
        # 3. Rellenar huecos (encontrando el contorno externo y rellenándolo)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        
        # 4. Obtener contorno y Envoltura Convexa (EC)
        area_rellena = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        area_ec = cv2.contourArea(hull)
        
        if area_ec == 0:
            continue
            
        # Regla de la diapositiva: A(O) ≈ A(EC) (Solidez cercana a 1.0)
        solidez = area_rellena / area_ec
        
        # Verificar proporción rectangular básica usando el rectángulo rotado mínimo
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        
        # Las placas mexicanas tienen proporción ~2:1 (ej. 30x15 cm), aceptamos rango amplio
        if solidez > 0.88 and (1.4 <= aspect <= 3.2):
            if area_rellena > mejor_score:
                mejor_score = area_rellena
                mejor_placa = hull

    resultado = img.copy()
    if mejor_placa is not None:
        # 5. Aproximar a 4 esquinas para la transformación de perspectiva
        peri = cv2.arcLength(mejor_placa, True)
        esquinas = cv2.approxPolyDP(mejor_placa, 0.04 * peri, True)
        
        cv2.drawContours(resultado, [mejor_placa], -1, (0, 255, 0), 3)
        for pt in esquinas:
            cv2.circle(resultado, tuple(pt[0]), 8, (0, 0, 255), -1)
        print(f"Placa detectada con solidez alta. Esquinas encontradas: {len(esquinas)}")
    else:
        print("No se encontró placa con los criterios.")
        
    return resultado

# Prueba con tus imágenes
res = detectar_placa_diapositiva("./placa4.jpg")
cv2.imshow("Deteccion", res)
cv2.waitKey(0)
cv2.destroyAllWindows()