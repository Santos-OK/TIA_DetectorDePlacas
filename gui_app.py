import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from plate_detector import detectar_con_pasos
from plate_reader import leer_con_pasos, cargar_plantillas


class DetectorPlacasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector y Lector de Placas Vehiculares (MX)")
        self.root.geometry("950x650")
        self.root.configure(bg="#1e272e")

        # Cargar plantillas una sola vez al iniciar
        self.plantillas = cargar_plantillas()
        self.pasos_actuales = {}

        self._construir_ui()

    def _construir_ui(self):
        # --- 1. ENCABEZADO (Nombre del programa) ---
        header = tk.Frame(self.root, bg="#0fbcf9", height=65)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="SISTEMA DE DETECCIÓN Y RECONOCIMIENTO DE PLACAS",
            font=("Segoe UI", 16, "bold"),
            bg="#0fbcf9",
            fg="#1e272e",
        ).pack(pady=16)

        # --- 2. BARRA DE CONTROLES (Subir foto y Show Process) ---
        controles = tk.Frame(self.root, bg="#1e272e", pady=15)
        controles.pack(fill=tk.X)

        btn_subir = tk.Button(
            controles,
            text="📂  Subir Foto de Placa",
            font=("Segoe UI", 11, "bold"),
            bg="#05c46b",
            fg="white",
            activebackground="#0be881",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            cursor="hand2",
            command=self.subir_y_procesar,
        )
        btn_subir.pack(side=tk.LEFT, padx=(40, 15))

        self.btn_proceso = tk.Button(
            controles,
            text="🔍  Show Process",
            font=("Segoe UI", 11, "bold"),
            bg="#3c40c6",
            fg="white",
            activebackground="#575fcf",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.mostrar_proceso,
        )
        self.btn_proceso.pack(side=tk.LEFT)

        # --- 3. ÁREA PRINCIPAL (Imagen cargada + Resultado leído) ---
        cuerpo = tk.Frame(self.root, bg="#1e272e")
        cuerpo.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Panel izquierdo: Foto original / detección
        panel_izq = tk.LabelFrame(
            cuerpo,
            text=" Vista Previa de la Imagen ",
            font=("Segoe UI", 10, "bold"),
            bg="#2f3640",
            fg="#d2dae2",
            bd=2,
        )
        panel_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        self.lbl_imagen = tk.Label(
            panel_izq,
            text="Sube una imagen (.jpg, .png) para comenzar",
            font=("Segoe UI", 11),
            bg="#2f3640",
            fg="#808e9b",
        )
        self.lbl_imagen.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel derecho: Placa recortada y texto leído
        panel_der = tk.LabelFrame(
            cuerpo,
            text=" Resultado del Reconocimiento ",
            font=("Segoe UI", 10, "bold"),
            bg="#2f3640",
            fg="#d2dae2",
            bd=2,
            width=340,
        )
        panel_der.pack(side=tk.RIGHT, fill=tk.Y)
        panel_der.pack_propagate(False)

        tk.Label(
            panel_der,
            text="Placa Segmentada:",
            font=("Segoe UI", 10),
            bg="#2f3640",
            fg="#d2dae2",
        ).pack(pady=(25, 8))

        self.lbl_recorte = tk.Label(
            panel_der,
            text="---",
            bg="#1e272e",
            fg="#808e9b",
            width=36,
            height=7,
        )
        self.lbl_recorte.pack(padx=15)

        tk.Label(
            panel_der,
            text="Placa Leída:",
            font=("Segoe UI", 11),
            bg="#2f3640",
            fg="#d2dae2",
        ).pack(pady=(30, 8))

        # Cuadro estilo placa metálica para el texto leído
        self.lbl_texto_placa = tk.Label(
            panel_der,
            text="-------",
            font=("Consolas", 28, "bold"),
            bg="#f5f6fa",
            fg="#1e272e",
            relief=tk.RIDGE,
            bd=4,
            padx=15,
            pady=10,
        )
        self.lbl_texto_placa.pack(padx=15)

    def _cv2_a_tk(self, img_cv2, max_w, max_h):
        """Convierte una imagen de OpenCV (BGR o Escala de Grises) a formato Tkinter."""
        if len(img_cv2.shape) == 2:
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        h, w, _ = img_rgb.shape
        escala = min(max_w / w, max_h / h)
        nuevo_w, nuevo_h = max(1, int(w * escala)), max(1, int(h * escala))

        img_pil = Image.fromarray(img_rgb).resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img_pil)

    def subir_y_procesar(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar fotografía del vehículo",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not ruta:
            return

        try:
            vis_det, placa_recta, pasos_det = detectar_con_pasos(ruta)
            self.pasos_actuales = dict(pasos_det)

            # Mostrar imagen principal con contorno detectado
            foto_tk = self._cv2_a_tk(vis_det, 520, 460)
            self.lbl_imagen.configure(image=foto_tk, text="")
            self.lbl_imagen.image = foto_tk

            if placa_recta is None:
                self.lbl_recorte.configure(image="", text="No se detectó placa", width=36, height=7)
                self.lbl_texto_placa.configure(text="NO DETECTADA", fg="#ff3f34")
                self.btn_proceso.configure(state=tk.NORMAL)
                return

            # Leer caracteres y guardar sus pasos
            texto, vis_chars, pasos_lec = leer_con_pasos(placa_recta, self.plantillas)
            self.pasos_actuales.update(pasos_lec)

            # Mostrar recorte con letras detectadas
            recorte_tk = self._cv2_a_tk(vis_chars, 290, 145)
            self.lbl_recorte.configure(image=recorte_tk, text="", width=290, height=145)
            self.lbl_recorte.image = recorte_tk

            # Mostrar texto final de la placa
            self.lbl_texto_placa.configure(text=texto if texto else "SIN LECTURA", fg="#1e272e")
            self.btn_proceso.configure(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error al procesar", str(e))

    def mostrar_proceso(self):
        """Abre una ventana emergente con la cuadrícula de todas las etapas intermedias."""
        if not self.pasos_actuales:
            return

        ventana_proc = tk.Toplevel(self.root)
        ventana_proc.title("Desglose del Proceso de Visión Artificial")
        ventana_proc.geometry("1120x680")
        ventana_proc.configure(bg="#1e272e")

        tk.Label(
            ventana_proc,
            text="Etapas Intermedias del Algoritmo",
            font=("Segoe UI", 14, "bold"),
            bg="#1e272e",
            fg="#0fbcf9",
        ).pack(pady=12)

        grid_frame = tk.Frame(ventana_proc, bg="#1e272e")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        columnas = 4
        idx = 0
        for titulo, img_paso in self.pasos_actuales.items():
            if img_paso is None:
                continue

            fila = idx // columnas
            col = idx % columnas

            tarjeta = tk.Frame(grid_frame, bg="#2f3640", bd=1, relief=tk.SOLID, padx=8, pady=8)
            tarjeta.grid(row=fila, column=col, padx=10, pady=10, sticky="nsew")

            tk.Label(
                tarjeta,
                text=titulo,
                font=("Segoe UI", 9, "bold"),
                bg="#2f3640",
                fg="#f5f6fa",
            ).pack(pady=(0, 6))

            img_tk = self._cv2_a_tk(img_paso, 230, 180)
            lbl = tk.Label(tarjeta, image=img_tk, bg="#1e272e")
            lbl.image = img_tk
            lbl.pack()

            idx += 1

        for c in range(columnas):
            grid_frame.columnconfigure(c, weight=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorPlacasApp(root)
    root.mainloop()