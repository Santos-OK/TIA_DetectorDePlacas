"""
Crea las plantillas REALES en ./plantillas/<CHAR>.png (28x28, igual que tu versión anterior).

Modos:
  python crear_plantillas.py            -> automático: usa ETIQUETAS (foto -> texto conocido).
                                           Si en alguna foto no coincide el número de caracteres,
                                           pasa solo a modo manual para esa foto.
  python crear_plantillas.py manual placa3.jpg
                                        -> manual: detecta y endereza la placa, te muestra cada
                                           carácter y guardas con la tecla correspondiente
                                           (ESC para omitir), como antes.
"""
import os
import sys
import cv2
from plate_detector import detectar_y_extraer_placa
from plate_reader import (segmentar_caracteres, recolectar_plantillas,
                          CARPETA_PLANTILLAS, TAMANO_CHAR)

# foto -> texto de la placa SIN guiones
ETIQUETAS = {
    "./placa.jpg":  "PYC5877",
    "./placa3.jpg": "A04SCB8",
    "./placa4.jpg": "HWY953B",
    "./placa6.jpg": "HYN051A"
}


def guardar(crop_bin, char, carpeta):
    """Guarda 28x28 como <CHAR>.png"""
    crop = cv2.resize(crop_bin, TAMANO_CHAR, interpolation=cv2.INTER_AREA)
    ruta = os.path.join(carpeta, f"{char}.png")
    n = 2
    while os.path.exists(ruta):
        ruta = os.path.join(carpeta, f"{char}_{n}.png")
        n += 1
    ok = cv2.imwrite(ruta, crop)
    print(f"   {'guardada' if ok else 'ERROR al guardar'}: {os.path.abspath(ruta)}")


def modo_manual(archivo):
    _, placa = detectar_y_extraer_placa(archivo)
    if placa is None:
        print(f"[X] No se detectó placa en {archivo}")
        return
    recolectar_plantillas(placa)      # tu herramienta interactiva de siempre


def modo_automatico(etiquetas=ETIQUETAS, carpeta=CARPETA_PLANTILLAS):
    os.makedirs(carpeta, exist_ok=True)
    print("Carpeta de plantillas:", os.path.abspath(carpeta))
    for archivo, texto in etiquetas.items():
        if not os.path.exists(archivo):
            print(f"[X] No existe {os.path.abspath(archivo)} (corre el script desde la carpeta de tus fotos)")
            continue
        _, placa = detectar_y_extraer_placa(archivo)
        if placa is None:
            print(f"[X] {archivo}: no se detectó la placa")
            continue
        bin_inv, cajas = segmentar_caracteres(placa)
        if len(cajas) != len(texto):
            print(f"[!] {archivo}: se segmentaron {len(cajas)} caracteres y se esperaban {len(texto)} "
                  f"({texto}). Paso a modo manual para esta foto.")
            recolectar_plantillas(placa)
            continue
        print(f"[OK] {archivo}: {texto}")
        for (x, y, w, h), char in zip(cajas, texto):
            guardar(bin_inv[y:y+h, x:x+w], char, carpeta)
    print("Plantillas ahora:", sorted(os.listdir(carpeta)))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "manual":
        modo_manual(sys.argv[2])
    else:
        modo_automatico()