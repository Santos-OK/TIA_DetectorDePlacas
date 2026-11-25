import sys
import cv2
from plate_detector import detectar_y_extraer_placa
from plate_reader import extraer_y_leer_placa

def leer_placa(ruta):
    """Foto completa -> (texto, imagen con detección, placa recta, imagen con caracteres)."""
    vis_det, placa_recta = detectar_y_extraer_placa(ruta)
    if placa_recta is None:
        return None, vis_det, None, None
    texto, _, vis_chars = extraer_y_leer_placa(placa_recta)
    return texto, vis_det, placa_recta, vis_chars

if __name__ == "__main__":
    archivos = sys.argv[1:] or ["./placa5.jpg", "./placa3.jpg", "./placa4.jpg"]
    for archivo in archivos:
        texto, vis_det, placa, vis_chars = leer_placa(archivo)
        print(f"{archivo}: {texto}")
        cv2.imshow(f"Deteccion - {archivo}", vis_det)
        if vis_chars is not None:
            cv2.imshow(f"Lectura - {archivo}", vis_chars)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
