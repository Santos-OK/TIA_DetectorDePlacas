import sys
import cv2
import numpy as np

# ---------------------------------------------------------------
# Parámetros ajustables (puntos de partida, se afinan con las fotos)
# ---------------------------------------------------------------
MIN_AREA_RATIO = 0.01   # área mínima del objeto, como fracción de la imagen
MAX_AREA_RATIO = 0.30   # área máxima del objeto, como fracción de la imagen
ASPECT_MIN = 1.2        # ancho / alto mínimo de la caja delimitadora
ASPECT_MAX = 3.5        # ancho / alto máximo de la caja delimitadora
MIN_FILL_RATIO = 0.40   # área del objeto / área de su caja


def get_binary(img):
    # Grises + binaria adaptativa
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=9,
    )
    return gray_img, binary_img


def find_candidates(binary_img, connectivity=8):
    """ Regresa la info de componentes y la lista de ids que parecen placa."""
    objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=connectivity
    )

    img_h, img_w = binary_img.shape
    img_area = img_h * img_w

    candidates = []
    for i in range(1, objects):  # el 0 es el fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 1. Rango de área
        area_ratio = area / img_area
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue

        # 2. Relación de aspecto
        aspect = w / h
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # 3. Relación de relleno
        fill = area / (w * h)
        if fill < MIN_FILL_RATIO:
            continue

        # 4. No tocar el borde de la imagen
        if x == 0 or y == 0 or x + w >= img_w or y + h >= img_h:
            continue

        candidates.append(i)

    return labels, stats, candidates


def pick_largest(stats, candidates):
    """el objeto con más píxeles, pero entre candidatos."""
    best_id = None
    best_area = 0
    for i in candidates:
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_id = i
    return best_id


def draw_candidates(img, stats, candidates, chosen_id):
    """Dibuja todos los candidatos en azul y el elegido en verde."""
    vis = img.copy()
    for i in candidates:
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        color = (0, 255, 0) if i == chosen_id else (255, 0, 0)
        thickness = 4 if i == chosen_id else 2
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def extract_plate_mask(img):
    gray_img, binary_img = get_binary(img)
    labels, stats, candidates = find_candidates(binary_img)
    plate_id = pick_largest(stats, candidates)

    mask_placa = np.zeros(binary_img.shape, dtype=np.uint8)
    if plate_id is not None:
        mask_placa[labels == plate_id] = 255

    return gray_img, binary_img, labels, stats, candidates, plate_id, mask_placa


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./placa2.jpg"

    # 0. Leer imagen
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("No se pudo abrir la imagen:", path)
        sys.exit(1)

    gray_img, binary_img, labels, stats, candidates, plate_id, mask_placa = \
        extract_plate_mask(img)

    print(f"Candidatos que pasaron los filtros: {len(candidates)}")
    if plate_id is None:
        print("No se encontró placa.")
    else:
        print(f"Placa elegida: label {plate_id}, "
              f"área {stats[plate_id, cv2.CC_STAT_AREA]} px")

    cv2.imshow("Candidatos (azul) y elegido (verde)",
               draw_candidates(img, stats, candidates, plate_id))
    cv2.imshow("mask", mask_placa)
    cv2.imshow("Binary", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()