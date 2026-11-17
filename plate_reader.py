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

def cargar_plantillas():
    """
    Carga plantillas reales desde ./plantillas/<CHAR>.png.
    Si falta algún carácter válido, genera un respaldo sintético temporal.
    """
    os.makedirs(CARPETA_PLANTILLAS, exist_ok=True)
    plantillas = {}
    
    for char in CARACTERES_MEXICO:
        ruta_png = os.path.join(CARPETA_PLANTILLAS, f"{char}.png")
        
        if os.path.exists(ruta_png):
            # Cargar imagen real guardada (28x28 en escala de grises)
            img_real = cv2.imread(ruta_png, cv2.IMREAD_GRAYSCALE)
            plantillas[char] = preparar_vector_28x28(img_real)
        else:
            # Respaldo sintético mientras recolectas el carácter real
            img_char = np.zeros((80, 60), dtype=np.uint8)
            cv2.putText(img_char, char, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3)
            coords = cv2.findNonZero(img_char)
            x, y, w, h = cv2.boundingRect(coords)
            plantillas[char] = preparar_vector_28x28(img_char[y:y+h, x:x+w])
            
    return plantillas

def segmentar_caracteres(img_placa_recortada):
    """Extrae la imagen binaria inversa y las cajas (x, y, w, h) de cada carácter."""
    gray = cv2.cvtColor(img_placa_recortada, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    alto_placa, _ = bin_inv.shape
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
            
    candidatos.sort(key=lambda item: item[0])
    return bin_inv, candidatos

def recolectar_plantillas(img_placa_recortada):
    """
    Herramienta interactiva: muestra cada carácter detectado y lo guarda en ./plantillas/
    al presionar su tecla correspondiente (presiona ESC para saltar).
    """
    os.makedirs(CARPETA_PLANTILLAS, exist_ok=True)
    bin_inv, candidatos = segmentar_caracteres(img_placa_recortada)
    
    for (x, y, w, h) in candidatos:
        char_crop = bin_inv[y:y+h, x:x+w]
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

def extraer_y_leer_placa(img_placa_recortada):
    """Lee la placa usando producto punto contra las plantillas mexicanas."""
    bin_inv, candidatos = segmentar_caracteres(img_placa_recortada)
    plantillas = cargar_plantillas()
    
    texto_placa = ""
    vis = img_placa_recortada.copy()
    
    for (x, y, w, h) in candidatos:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        vec_char = preparar_vector_28x28(bin_inv[y:y+h, x:x+w])
        
        mejor_letra = "?"
        mayor_similitud = -1.0
        
        for letra, vec_plantilla in plantillas.items():
            similitud = np.dot(vec_char, vec_plantilla)
            if similitud > mayor_similitud:
                mayor_similitud = similitud
                mejor_letra = letra
                
        texto_placa += mejor_letra

    return texto_placa, bin_inv, vis

if __name__ == "__main__":
    img = cv2.imread("./placa4.jpg")
    h, w, _ = img.shape
    placa_recortada = img[int(h*0.42):int(h*0.85), int(w*0.14):int(w*0.69)]

    placa_recortada = cv2.imread("./placa5.jpg")
    
    # Descomenta la siguiente línea UNA VEZ para guardar los caracteres reales de placa4.jpg:
    #recolectar_plantillas(placa_recortada)
    
    resultado, _, vis = extraer_y_leer_placa(placa_recortada)
    print(f"Placa leída: {resultado}")
    cv2.imshow("Resultado", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()