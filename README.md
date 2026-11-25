# Sistema de Detección y Reconocimiento de Placas Vehiculares (MX)

Aplicación de visión artificial desarrollada en Python que localiza, endereza y lee placas vehiculares mexicanas a partir de fotografías, utilizando técnicas clásicas de procesamiento de imágenes (binarización, componentes conectados, envoltura convexa, transformación de perspectiva, morfología Black-Hat y reconocimiento mediante producto punto de vectores normalizados).

---

## Requisitos del Sistema

### 1. Versión de Python
* **Python 3.11 – 3.14** (Recomendado: **Python 3.12**).
* *Nota:* Asegúrate de que tu instalación de Python incluya **Tkinter** (viene incluido por defecto en los instaladores oficiales de Windows y macOS; en Linux se instala con `sudo apt install python3-tk`).

### 2. Librerías Necesarias
El proyecto requiere únicamente 3 paquetes externos de Python:
* **`opencv-python`** (`>= 4.8.0`): Procesamiento de imágenes, filtros morfológicos, detección de contornos y transformación de perspectiva.
* **`numpy`** (`>= 1.26.0`): Operaciones matriciales, normalización de vectores y cálculo de producto punto.
* **`pillow`** (`>= 10.0.0`, recomendado `>= 13.0.0`): Conversión de matrices OpenCV para su visualización dentro de la interfaz gráfica Tkinter.

Instalación rápida desde la terminal:
```bash
pip install opencv-python numpy pillow