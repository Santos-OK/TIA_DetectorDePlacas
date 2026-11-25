import cv2
import numpy as np

def ordenar_esquinas(pts):
    """Ordena 4 puntos en: [arriba-izq, arriba-der, abajo-der, abajo-izq]."""
    pts = pts.reshape((4, 2)).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # La suma (x + y) es mínima arriba-izq y máxima abajo-der
    suma = pts.sum(axis=1)
    rect[0] = pts[np.argmin(suma)]
    rect[2] = pts[np.argmax(suma)]

    # La diferencia (y - x) es mínima arriba-der y máxima abajo-izq
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def enderezar_placa(img, esquinas):
    """Aplica transformación de perspectiva para dejar la placa recta."""
    rect = ordenar_esquinas(esquinas)
    (tl, tr, br, bl) = rect

    # Calcular alto de la nueva imagen y forzar la proporción real de una placa mexicana
    # (30.5 x 15.2 cm ≈ 2:1). Así el carácter no sale aplastado si la foto es de lado.
    alto_izq = np.linalg.norm(bl - tl)
    alto_der = np.linalg.norm(br - tr)
    max_alto = max(int(max(alto_izq, alto_der)), 150)
    max_ancho = 2 * max_alto

    dst = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    placa_recta = cv2.warpPerspective(img, M, (max_ancho, max_alto))
    return placa_recta

def buscar_mejor_candidato(binary_img, aspect_min=1.5, aspect_max=2.8, solidez_min=0.88, rectang_min=0.0):
    """Busca componentes conectados que cumplan A(O) ≈ A(EC) y proporción de placa."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    img_h, img_w = binary_img.shape
    img_area = img_h * img_w

    mejor_hull = None
    mejor_score = 0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        area_obj = stats[i, cv2.CC_STAT_AREA]

        # 1. Filtrar por área relativa (ni ruido diminuto ni toda la imagen)
        if not (0.015 * img_area < area_obj < 0.65 * img_area):
            continue

        # Evitar objetos pegados a los 4 bordes de la foto (suele ser la carrocería)
        bordes_tocados = (x == 0) + (y == 0) + (x + w_box >= img_w - 1) + (y + h_box >= img_h - 1)
        if bordes_tocados >= 2:
            continue

        # Aislar objeto y rellenar huecos (letras internas)
        obj_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        obj_mask[labels == i] = 255
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # 2. Envoltura Convexa (EC) y Solidez: A(O) ≈ A(EC)
        area_rellena = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        area_ec = cv2.contourArea(hull)
        if area_ec == 0:
            continue

        solidez = area_rellena / area_ec

        # 3. Proporción rectangular usando rectángulo rotado (soporta placas inclinadas)
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)

        # Placas mexicanas tienen relación ~2:1 (en foto de lado se ve más "cuadrada",
        # por eso en el respaldo se acepta un rango mas amplio)
        # Rectangularidad: qué tanto llena el objeto a su rectángulo mínimo
        rectang = area_rellena / (w * h)
        if solidez > solidez_min and rectang > rectang_min and (aspect_min <= aspect <= aspect_max):
            # Preferir objetos grandes y muy rectangulares
            score = area_rellena * solidez * rectang
            if score > mejor_score:
                mejor_score = score
                mejor_hull = hull

    return mejor_hull

def detectar_y_extraer_placa(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Intento 1: Umbral global de Otsu (ideal para autos oscuros como placa3 y placa4)
    _, bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mejor_hull = buscar_mejor_candidato(bin_otsu)

    # Intento 2 (Respaldo): Umbral adaptativo + cierre (autos claros / sombras, ej. PYC-58-77).
    # Se prueban varios parámetros y se acepta una proporción más "cuadrada" por la perspectiva.
    if mejor_hull is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for bloque, c in [(51, -5), (101, -5), (75, -8), (51, -10)]:
            bin_adap = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloque, c
            )
            bin_adap = cv2.morphologyEx(bin_adap, cv2.MORPH_CLOSE, kernel)
            mejor_hull = buscar_mejor_candidato(bin_adap, aspect_min=1.05, aspect_max=2.8, solidez_min=0.88, rectang_min=0.80)
            if mejor_hull is not None:
                break

    resultado_vis = img.copy()
    placa_enderezada = None

    if mejor_hull is not None:
        # Obtener exactamente 4 esquinas
        peri = cv2.arcLength(mejor_hull, True)
        esquinas = cv2.approxPolyDP(mejor_hull, 0.03 * peri, True)

        # Si por los bordes redondeados dio != 4 puntos, usar las 4 esquinas del rectángulo mínimo
        if len(esquinas) != 4:
            rect = cv2.minAreaRect(mejor_hull)
            esquinas = np.int32(cv2.boxPoints(rect)).reshape((4, 1, 2))

        # Dibujar contorno verde y esquinas rojas
        cv2.drawContours(resultado_vis, [esquinas], -1, (0, 255, 0), 3)
        for pt in esquinas:
            cv2.circle(resultado_vis, tuple(pt[0]), 8, (0, 0, 255), -1)

        # Enderezar perspectiva para enviarla al lector
        placa_enderezada = enderezar_placa(img, esquinas)
        print(f"[OK] Placa detectada y enderezada en {img_path}")
    else:
        print(f"[X] No se encontró placa en {img_path}")

    return resultado_vis, placa_enderezada

def detectar_con_pasos(img_path):
    """Ejecuta la detección guardando las imágenes intermedias para la GUI."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    vis_det, placa_recta = detectar_y_extraer_placa(img_path)

    pasos = {
        "1. Imagen Original": img,
        "2. Escala de Grises": gray,
        "3. Binarización (Otsu)": bin_otsu,
        "4. Detección y Envoltura Convexa": vis_det,
        "5. Placa Enderezada (Perspectiva)": placa_recta,
    }
    return vis_det, placa_recta, pasos

if __name__ == "__main__":
    # Prueba con tus 3 fotos principales
    for archivo in ["./placa2.jpg", "./placa5.jpg", "./placa4.jpg"]:
        vis, placa_recta = detectar_y_extraer_placa(archivo)
        cv2.imshow(f"Deteccion - {archivo}", vis)
        if placa_recta is not None:
            cv2.imshow(f"Enderezada - {archivo}", placa_recta)
        cv2.waitKey(0)
        cv2.destroyAllWindows()