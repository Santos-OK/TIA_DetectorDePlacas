import os
import cv2
import numpy as np

TAMANO_CHAR = (28, 28)
CARPETA_PLANTILLAS = "./plantillas"

# Caracteres válidos en placas de México (A-Z sin I, Ñ, O, Q + 0-9)
LETRAS_VALIDAS = "ABCDEFGHJKLMNPRSTUVWXYZ"
NUMEROS_VALIDOS = "0123456789"
CARACTERES_MEXICO = NUMEROS_VALIDOS + LETRAS_VALIDAS


def preparar_vector_28x28(recorte_binario):
    """Redimensiona a 28x28, aplana a 1D y normaliza a vector unitario."""
    redim = cv2.resize(recorte_binario, TAMANO_CHAR, interpolation=cv2.INTER_AREA)
    vector = redim.flatten().astype(np.float32)
    mag = np.linalg.norm(vector)
    if mag > 0:
        vector = vector / mag
    return vector


def cargar_plantillas(carpeta=None):
    """
    Carga plantillas reales desde ./plantillas/<CHAR>.png. Devuelve {char: [vectores]}.
    Si falta algún carácter válido, genera un respaldo sintético.
    """
    carpeta = carpeta or CARPETA_PLANTILLAS
    os.makedirs(carpeta, exist_ok=True)
    plantillas = {}

    for char in CARACTERES_MEXICO:
        vectores = []
        for nombre in sorted(os.listdir(carpeta)):
            base, ext = os.path.splitext(nombre)
            if ext.lower() == ".png" and (base == char or base.startswith(char + "_")):
                img_real = cv2.imread(os.path.join(carpeta, nombre), cv2.IMREAD_GRAYSCALE)
                if img_real is not None:
                    vectores.append(preparar_vector_28x28(img_real))

        if not vectores:
            # Respaldo sintético mientras recolectas el carácter real
            img_char = np.zeros((80, 60), dtype=np.uint8)
            cv2.putText(img_char, char, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3)
            coords = cv2.findNonZero(img_char)
            x, y, w, h = cv2.boundingRect(coords)
            vectores.append(preparar_vector_28x28(img_char[y:y + h, x:x + w]))

        plantillas[char] = vectores

    return plantillas


def segmentar_caracteres(img_placa_recortada):
    """Extrae la imagen binaria inversa y las cajas (x, y, w, h) de cada carácter."""
    # Canal V (máximo de B,G,R) en vez de gris: la tinta es negra en los 3 canales,
    # mientras que una sombra azulada sigue teniendo un canal alto, así que no se confunde con letra.
    gray = img_placa_recortada.max(axis=2)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    alto_placa, ancho_placa = blur.shape

    # Black-hat: resalta trazos oscuros DELGADOS (letras) e ignora sombras/franjas anchas
    # que un umbral global o adaptativo pegaría a los caracteres.
    k = int(alto_placa * 0.22) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, bin_inv = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    candidatos = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        ratio_alto = h / alto_placa
        aspecto = w / float(h)

        if (0.30 <= ratio_alto <= 0.75) and (0.15 <= aspecto <= 0.85) and area > 100:
            candidatos.append((x, y, w, h))

    # Los caracteres de una placa miden casi lo mismo: descartar los que se
    # alejan de la altura mediana
    if candidatos:
        med = np.median([c[3] for c in candidatos])
        candidatos = [c for c in candidatos if abs(c[3] - med) <= 0.2 * med]

    candidatos = _separar_y_normalizar(bin_inv, candidatos)
    candidatos.sort(key=lambda item: item[0])
    return bin_inv, candidatos


def _separar_y_normalizar(bin_inv, candidatos):
    """
    1) Parte bloques demasiado anchos (dos caracteres pegados) en el valle de la proyección vertical.
    2) Todas las cajas usan la misma banda vertical (altura mediana) para quitar rebabas de sombra.
    """
    if not candidatos:
        return candidatos
    med_y = int(np.median([c[1] for c in candidatos]))
    med_h = int(np.median([c[3] for c in candidatos]))
    ancho_tipico = np.percentile([c[2] for c in candidatos], 40)

    resultado = []
    for (x, y, w, h) in candidatos:
        n = int(round(w / ancho_tipico))
        cortes = [x]
        if n >= 2:
            proy = (bin_inv[med_y:med_y + med_h, x:x + w] > 0).sum(axis=0)
            for k in range(1, n):
                esperado = int(k * w / n)
                margen = max(3, int(0.25 * w / n))
                ini, fin = max(1, esperado - margen), min(w - 1, esperado + margen)
                cortes.append(x + ini + int(np.argmin(proy[ini:fin])))
        cortes.append(x + w)

        for a, b in zip(cortes[:-1], cortes[1:]):
            banda = bin_inv[med_y:med_y + med_h, a:b]
            cols = np.where(banda.sum(axis=0) > 0)[0]
            if len(cols) == 0:
                continue
            x0, x1 = a + cols[0], a + cols[-1] + 1
            resultado.append((x0, med_y, x1 - x0, med_h))
    return resultado


def recolectar_plantillas(img_placa_recortada):
    """
    Herramienta interactiva: muestra cada carácter detectado y lo guarda en ./plantillas/
    al presionar su tecla correspondiente (presiona ESC para saltar).
    """
    os.makedirs(CARPETA_PLANTILLAS, exist_ok=True)
    bin_inv, candidatos = segmentar_caracteres(img_placa_recortada)

    for (x, y, w, h) in candidatos:
        char_crop = bin_inv[y:y + h, x:x + w]
        char_28 = cv2.resize(char_crop, TAMANO_CHAR, interpolation=cv2.INTER_AREA)

        # Mostrar ampliado para verlo bien
        preview = cv2.resize(char_28, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Presiona la tecla de este caracter (ESC para omitir)", preview)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            continue

        letra = chr(key).upper()
        if letra in CARACTERES_MEXICO:
            ruta = os.path.join(CARPETA_PLANTILLAS, f"{letra}.png")
            cv2.imwrite(ruta, char_28)
            print(f"Guardada plantilla real: {ruta}")

    cv2.destroyAllWindows()


def extraer_y_leer_placa(img_placa_recortada, plantillas=None):
    """Lee la placa usando producto punto contra las plantillas mexicanas."""
    bin_inv, candidatos = segmentar_caracteres(img_placa_recortada)
    if plantillas is None:
        plantillas = cargar_plantillas()

    texto_placa = ""
    vis = img_placa_recortada.copy()

    for (x, y, w, h) in candidatos:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        vec_char = preparar_vector_28x28(bin_inv[y:y + h, x:x + w])

        mejor_letra = "?"
        mayor_similitud = -1.0

        for letra, variantes in plantillas.items():
            for vec_plantilla in variantes:
                similitud = np.dot(vec_char, vec_plantilla)
                if similitud > mayor_similitud:
                    mayor_similitud = similitud
                    mejor_letra = letra

        texto_placa += mejor_letra
        cv2.putText(vis, mejor_letra, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return texto_placa, bin_inv, vis

def leer_con_pasos(img_placa_recortada, plantillas=None):
    """Ejecuta la lectura guardando las etapas intermedias de caracteres para la GUI."""
    gray_v = img_placa_recortada.max(axis=2)
    blur = cv2.GaussianBlur(gray_v, (3, 3), 0)
    k = int(blur.shape[0] * 0.22) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    texto, bin_inv, vis_chars = extraer_y_leer_placa(img_placa_recortada, plantillas)

    pasos_lectura = {
        "6. Filtro Black-Hat (Aísla tinta)": blackhat,
        "7. Binaria Inversa (Caracteres)": bin_inv,
        "8. Segmentación y Producto Punto": vis_chars,
    }
    return texto, vis_chars, pasos_lectura

if __name__ == "__main__":
    from plate_detector import detectar_y_extraer_placa

    _, placa_recta = detectar_y_extraer_placa("./placa4.jpg")
    resultado, _, vis = extraer_y_leer_placa(placa_recta)
    print(f"Placa leída: {resultado}")
    cv2.imshow("Resultado", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()