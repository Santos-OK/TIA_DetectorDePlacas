# Sistema de Detección y Reconocimiento de Placas Vehiculares (MX)

Aplicación de visión artificial desarrollada en Python que localiza, endereza y lee placas vehiculares mexicanas a partir de fotografías, utilizando técnicas clásicas de procesamiento de imágenes (binarización, componentes conectados, envoltura convexa, transformación de perspectiva, morfología Black-Hat y reconocimiento mediante producto punto de vectores normalizados).

---

## Requisitos del Sistema

### 1. Versión de Python
* **Python 3.11 – 3.14** (Recomendado: **Python 3.12**).
* *Nota:* La instalación debe incluir **Tkinter.**

### 2. Librerías Necesarias
El proyecto requiere únicamente 3 paquetes externos de Python:
* **`opencv-python`** (`>= 4.8.0`): Procesamiento de imágenes, filtros morfológicos, detección de contornos y transformación de perspectiva.
* **`numpy`** (`>= 1.26.0`): Operaciones matriciales, normalización de vectores y cálculo de producto punto.
* **`pillow`** (`>= 10.0.0`, recomendado `>= 13.0.0`): Conversión de matrices OpenCV para su visualización dentro de la interfaz gráfica Tkinter.

Instalación rápida desde la terminal:
```bash
pip install opencv-python numpy pillow
```
## Requisitos y Formato de las Imágenes

Para obtener una detección y lectura óptima, las fotografías deben cumplir con las siguientes características:

* **Formatos soportados:** `.jpg`, `.jpeg`, `.png`, `.bmp` (3 canales de color BGR/RGB).
* **Estándar de placa:** Placas vehiculares de México (proporción aproximada `2:1`, caracteres alfanuméricos `0-9` y `A-Z` excluyendo `I`, `Ñ`, `O`, `Q`).
* **Escala y encuadre:** La placa debe ocupar entre el **1.5% y el 65%** del área total de la fotografía y no estar cortada por los bordes de la imagen.
* **Ángulo e iluminación:** Soporta inclinaciones laterales (el sistema corrige la perspectiva automáticamente) y sombras parciales, siempre que los caracteres tengan suficiente resolución en píxeles para distinguirse con claridad.

---

## Estructura del Proyecto

Asegúrate de mantener exactamente estos nombres de archivo en la misma carpeta:

```text
proyecto_placas/
├── gui_app.py             # Interfaz gráfica principal (Tkinter) con botón "Show Process"
├── main.py                # Ejecución por consola / línea de comandos
├── plate_detector.py      # Módulo 1: Localización de placa, envoltura convexa y perspectiva
├── plate_reader.py        # Módulo 2: Filtro Black-Hat, segmentación y producto punto
├── crear_plantillas.py    # Herramienta para generar plantillas reales (28x28 px)
├── plantillas/            # Carpeta con las imágenes binarias de referencia (<CHAR>.png)
└── README.md              # Este archivo de documentación
```

> **Nota importante sobre la carpeta `plantillas/`:** 
> Si la carpeta `./plantillas/` ya contiene las imágenes `.png` de referencia pre-generadas, puedes ejecutar directamente la interfaz gráfica (`gui_app.py`) o la consola (`main_2.py`) sin necesidad de correr primero `crear_plantillas.py`. Solo es necesario ejecutar el generador si la carpeta está vacía o si deseas agregar nuevas variantes tipográficas.

---

## Instrucciones de Uso

### Paso 1: Ejecutar la Interfaz Gráfica (Recomendado)
Inicia la aplicación visual con Tkinter:
```bash
python gui_app.py
```
1. Haz clic en **"Subir Foto de Placa"** y selecciona una imagen de tu equipo.
2. El panel izquierdo mostrará la placa detectada con su contorno verde y el panel derecho mostrará el recorte enderezado junto con el texto leído.
3. Haz clic en **"Show Process"** para abrir la ventana emergente con el desglose visual de las 8 etapas intermedias del algoritmo.

### Paso 2: Ejecución por Consola (Opcional)
Si deseas probar el flujo directamente desde la terminal sin abrir la interfaz:
```bash
python main.py ./placa4.jpg
```

### Paso 3: Generar o Ampliar las Plantillas de Caracteres (Solo si `plantillas/` está vacía)
Para crear la base de plantillas de `28x28` píxeles en la carpeta `./plantillas/`:
```bash
python crear_plantillas.py
```
* *(Opcional)* Para agregar caracteres de una foto nueva de forma interactiva:
  ```bash
  python crear_plantillas.py manual ruta_de_tu_foto.jpg
  ```
