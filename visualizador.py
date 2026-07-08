"""
visualizador.py — Renderizado del esqueleto, ángulos y HUD en pantalla.

Dibuja sobre el frame de la cámara:
- Esqueleto conectado con colores por nivel de riesgo
- Ángulos articulares en los nodos clave
- Panel HUD semitransparente con scores RULA
- Barra de riesgo visual
- Indicador de FPS
"""

import cv2
import numpy as np
import time
from config import (
    SKELETON_CONEXIONES, TORSO_CONEXIONES,
    COLOR_VERDE, COLOR_AMARILLO, COLOR_NARANJA, COLOR_ROJO,
    COLOR_BLANCO, COLOR_GRIS, COLOR_HUD_FONDO,
    KEYPOINT_NOMBRES,
)


class Visualizador:
    """
    Renderiza el análisis de postura sobre el frame de la cámara.
    Incluye esqueleto, ángulos, panel HUD y barra de riesgo.
    """

    # Mapeo color_key → color BGR
    COLORES_RIESGO = {
        'verde': COLOR_VERDE,
        'amarillo': COLOR_AMARILLO,
        'naranja': COLOR_NARANJA,
        'rojo': COLOR_ROJO,
    }

    def __init__(self):
        self.fps_anterior = time.time()
        self.fps_valor = 0
        self.fps_buffer = []

    def _obtener_color_riesgo(self, resultado):
        """Obtiene el color BGR basado en el nivel de riesgo RULA."""
        if resultado is None:
            return COLOR_GRIS
        return self.COLORES_RIESGO.get(resultado['color_key'], COLOR_GRIS)

    def _calcular_fps(self):
        """Calcula FPS con promedio móvil."""
        ahora = time.time()
        dt = ahora - self.fps_anterior
        self.fps_anterior = ahora
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)
        if len(self.fps_buffer) > 30:
            self.fps_buffer.pop(0)
        self.fps_valor = np.mean(self.fps_buffer) if self.fps_buffer else 0
        return self.fps_valor

    def _dibujar_esqueleto(self, frame, keypoints_2d, color_riesgo):
        """Dibuja el esqueleto con keypoints y conexiones."""
        if keypoints_2d is None:
            return

        # Dibujar conexiones (líneas)
        for idx1, idx2 in SKELETON_CONEXIONES:
            if idx1 in keypoints_2d and idx2 in keypoints_2d:
                pt1 = (int(keypoints_2d[idx1][0]), int(keypoints_2d[idx1][1]))
                pt2 = (int(keypoints_2d[idx2][0]), int(keypoints_2d[idx2][1]))

                # Color diferente para el torso
                if (idx1, idx2) in TORSO_CONEXIONES or (idx2, idx1) in TORSO_CONEXIONES:
                    color_linea = color_riesgo
                    grosor = 3
                else:
                    color_linea = color_riesgo
                    grosor = 2

                cv2.line(frame, pt1, pt2, color_linea, grosor, cv2.LINE_AA)

        # Dibujar keypoints (círculos)
        for idx, (x, y, conf) in keypoints_2d.items():
            centro = (int(x), int(y))
            # Puntos principales más grandes
            if idx in {0, 5, 6, 7, 8, 9, 10, 11, 12}:
                cv2.circle(frame, centro, 6, color_riesgo, -1, cv2.LINE_AA)
                cv2.circle(frame, centro, 6, COLOR_BLANCO, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, centro, 4, color_riesgo, -1, cv2.LINE_AA)

    def _dibujar_angulos(self, frame, keypoints_2d, resultado):
        """Muestra los ángulos principales en los nodos articulares."""
        if resultado is None or keypoints_2d is None:
            return

        angulos = resultado['angulos']
        color = self._obtener_color_riesgo(resultado)

        # Ángulo del cuello (cerca de la nariz)
        if 0 in keypoints_2d and 'cuello_flexion' in angulos:
            pos = (int(keypoints_2d[0][0]) + 15, int(keypoints_2d[0][1]) - 10)
            texto = f"{int(angulos['cuello_flexion'])}"
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLOR_BLANCO, 2, cv2.LINE_AA)
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

        # Ángulo del codo izquierdo
        if 7 in keypoints_2d and 'antebrazo_izq' in angulos:
            pos = (int(keypoints_2d[7][0]) + 12, int(keypoints_2d[7][1]) - 8)
            texto = f"{int(angulos['antebrazo_izq'])}"
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLOR_BLANCO, 2, cv2.LINE_AA)
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

        # Ángulo del codo derecho
        if 8 in keypoints_2d and 'antebrazo_der' in angulos:
            pos = (int(keypoints_2d[8][0]) + 12, int(keypoints_2d[8][1]) - 8)
            texto = f"{int(angulos['antebrazo_der'])}"
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLOR_BLANCO, 2, cv2.LINE_AA)
            cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

    def _dibujar_hud(self, frame, resultado):
        """Dibuja el panel HUD semitransparente con scores RULA."""
        h, w = frame.shape[:2]

        # Dimensiones del panel
        panel_w = 280
        panel_h = 260
        margen = 10
        x0, y0 = margen, margen

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      COLOR_HUD_FONDO, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Borde del panel
        color_borde = self._obtener_color_riesgo(resultado)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      color_borde, 2, cv2.LINE_AA)

        if resultado is None:
            cv2.putText(frame, "Sin deteccion", (x0 + 15, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GRIS, 1, cv2.LINE_AA)
            return

        # Título
        score = resultado['score_final']
        texto_titulo = f"RULA Score: {score}"
        cv2.putText(frame, texto_titulo, (x0 + 15, y0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_borde, 2, cv2.LINE_AA)

        # Nivel de acción
        cv2.putText(frame, resultado['texto'], (x0 + 15, y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_borde, 1, cv2.LINE_AA)

        # Línea separadora
        cv2.line(frame, (x0 + 10, y0 + 72), (x0 + panel_w - 10, y0 + 72),
                 COLOR_GRIS, 1)

        # Scores individuales
        scores = resultado['scores']
        nombres_display = [
            ('Cuello', 'cuello'),
            ('Tronco', 'tronco'),
            ('Brazo sup.', 'brazo_superior'),
            ('Antebrazo', 'antebrazo'),
            ('Muneca', 'muneca'),
        ]

        y_offset = y0 + 95
        for nombre_display, key in nombres_display:
            valor = scores.get(key, 0)
            # Nombre del segmento
            cv2.putText(frame, nombre_display, (x0 + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLANCO, 1, cv2.LINE_AA)
            # Valor
            color_val = self._color_por_score_individual(valor)
            cv2.putText(frame, str(valor), (x0 + 160, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_val, 2, cv2.LINE_AA)

            # Barra mini
            barra_x = x0 + 185
            barra_ancho = 80
            barra_alto = 8
            barra_y = y_offset - 10
            cv2.rectangle(frame, (barra_x, barra_y),
                          (barra_x + barra_ancho, barra_y + barra_alto),
                          COLOR_GRIS, 1)
            fill_w = int(barra_ancho * min(valor / 4.0, 1.0))
            cv2.rectangle(frame, (barra_x, barra_y),
                          (barra_x + fill_w, barra_y + barra_alto),
                          color_val, -1)

            y_offset += 25

        # Línea separadora
        cv2.line(frame, (x0 + 10, y_offset - 5),
                 (x0 + panel_w - 10, y_offset - 5), COLOR_GRIS, 1)

        # Barra de riesgo general
        y_offset += 10
        cv2.putText(frame, "Riesgo:", (x0 + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLANCO, 1, cv2.LINE_AA)
        barra_x = x0 + 90
        barra_ancho = 170
        barra_alto = 14
        barra_y = y_offset - 11
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + barra_ancho, barra_y + barra_alto),
                      COLOR_GRIS, 1)
        fill_w = int(barra_ancho * min(score / 7.0, 1.0))
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + fill_w, barra_y + barra_alto),
                      color_borde, -1)

    # Etiquetas y fiabilidad mostradas en pantalla por tipo de vista.
    # 'angulo' es la única vista marcada como no fiable: el ratio cae en una
    # zona intermedia ambigua entre frontal y lateral. Esta clasificación viene
    # de evaluador_rula._detectar_vista, fuente única de verdad (ver
    # resultado['vista'] en evaluar()), para no duplicar la lógica aquí.
    ETIQUETAS_VISTA = {'frontal': 'FRENTE', 'lateral': 'LADO', 'angulo': 'ANGULO'}
    VISTAS_FIABLES = {'frontal', 'lateral'}

    def _dibujar_vista(self, frame, resultado):
        """Muestra el ángulo de cámara detectado y la fiabilidad sagital."""
        if resultado is None or 'vista' not in resultado:
            return

        etiqueta = self.ETIQUETAS_VISTA.get(resultado['vista'])
        if etiqueta is None:
            return
        fiable = resultado['vista'] in self.VISTAS_FIABLES

        w = frame.shape[1]
        color = COLOR_VERDE if fiable else COLOR_AMARILLO
        texto = f"Vista: {etiqueta}"
        cv2.putText(frame, texto, (w - 175, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if not fiable:
            cv2.putText(frame, "postura sagital incierta", (w - 235, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_AMARILLO, 1, cv2.LINE_AA)

    def _dibujar_fps(self, frame):
        """Muestra el indicador de FPS en la esquina inferior izquierda."""
        h, w = frame.shape[:2]
        fps = self._calcular_fps()
        texto = f"FPS: {int(fps)}"
        cv2.putText(frame, texto, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VERDE, 1, cv2.LINE_AA)

    def _dibujar_instrucciones(self, frame):
        """Muestra instrucciones de teclas en la esquina inferior derecha."""
        h, w = frame.shape[:2]
        cv2.putText(frame, "C: calibrar erguido  |  Q: salir", (w - 280, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRIS, 1, cv2.LINE_AA)

    @staticmethod
    def _color_por_score_individual(score):
        """Retorna color basado en un score individual (1-4+)."""
        if score <= 1:
            return COLOR_VERDE
        elif score <= 2:
            return COLOR_AMARILLO
        elif score <= 3:
            return COLOR_NARANJA
        else:
            return COLOR_ROJO

    def dibujar(self, frame, keypoints_2d, resultado):
        """
        Renderiza todo el análisis sobre el frame.

        Args:
            frame: Imagen BGR de OpenCV (se modifica in-place).
            keypoints_2d: dict de keypoints 2D de la Fase 1.
            resultado: dict de evaluación RULA de la Fase 3.
        """
        color_riesgo = self._obtener_color_riesgo(resultado)

        # Capa 1: Esqueleto
        self._dibujar_esqueleto(frame, keypoints_2d, color_riesgo)

        # Capa 2: Ángulos en articulaciones
        self._dibujar_angulos(frame, keypoints_2d, resultado)

        # Capa 3: Panel HUD
        self._dibujar_hud(frame, resultado)

        # Capa 3.5: Indicador de vista de cámara
        self._dibujar_vista(frame, resultado)

        # Capa 4: FPS
        self._dibujar_fps(frame)

        # Capa 5: Instrucciones
        self._dibujar_instrucciones(frame)
