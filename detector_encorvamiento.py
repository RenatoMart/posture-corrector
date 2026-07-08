"""
detector_encorvamiento.py — Detección de encorvamiento en vista lateral por silueta.

Problema que resuelve
----------------------
De costado, cuando la cadera queda oculta tras el escritorio, el detector 2D
la fabrica justo debajo del hombro. El vector hombro→cadera sale entonces
vertical y el tronco parece SIEMPRE erguido, aunque la persona esté encorvada.

Solución
--------
Cuando la pelvis no es visible, la única información del tronco que queda en la
imagen es el CONTORNO de la espalda alta. Al encorvarse, ese contorno se inclina
hacia adelante. Este módulo:

    1. Recorta un ROI del torso a partir de los keypoints (hombro/oreja).
    2. Extrae bordes con Canny dentro del ROI.
    3. Para cada fila, toma el punto del contorno de la espalda más cercano a
       la posición esperada (robusto frente a bordes de ropa internos y al
       respaldo de la silla, que quedan más lejos).
    4. Ajusta una recta a esos puntos y mide su inclinación respecto a la
       vertical, en la dirección de "hacia adelante" (la que mira la cara).
    5. Calibra ese ángulo contra la postura MÁS erguida observada de la propia
       persona (línea base), y reporta solo el exceso (el encorvamiento real).

El resultado se entrega como un ángulo de flexión de tronco en grados,
comparable con el que produce el evaluador RULA, para alimentar el score de
tronco cuando la cadera es fabricada.
"""

import cv2
import numpy as np
from config import (
    ENCORVADO_CANNY_LOW, ENCORVADO_CANNY_HIGH, ENCORVADO_BLUR_KSIZE,
    ENCORVADO_ROI_ESCALA_X, ENCORVADO_ROI_ARRIBA, ENCORVADO_ROI_ABAJO,
    ENCORVADO_OFFSET_ESPALDA, ENCORVADO_MIN_FILAS, ENCORVADO_RESIDUO_MAX,
    ENCORVADO_MARGEN_GRADOS, ENCORVADO_GANANCIA, ENCORVADO_SUAVIZADO,
    ENCORVADO_CALIB_ALPHA_SUBIDA, ENCORVADO_CALIB_ALPHA_BAJADA,
)


class AnalizadorSiluetaLateral:
    """
    Estima la inclinación del tronco en vista lateral a partir del contorno
    de la espalda (detección de bordes), calibrada contra la postura erguida.
    """

    def __init__(self):
        self._angulo_suavizado = None   # EMA del ángulo crudo medido (grados con signo)
        self._baseline = None           # ángulo de la postura más erguida observada

    def recalibrar(self):
        """Reinicia el suavizado y la línea base de postura erguida (tecla 'C')."""
        self._angulo_suavizado = None
        self._baseline = None

    # ------------------------------------------------------------------ #
    # Utilidades geométricas
    # ------------------------------------------------------------------ #

    @staticmethod
    def _direccion_frente(keypoints_2d, hombro_x):
        """
        Determina hacia qué lado (en x) mira la persona: +1 = derecha, -1 = izquierda.

        La nariz queda por delante de la oreja/hombro; su posición respecto al
        hombro indica la dirección de la mirada y, por tanto, dónde está la
        espalda (lado opuesto). Devuelve None si no hay referencia de cara.
        """
        for idx in (0, 1, 2):  # nariz, ojos
            if idx in keypoints_2d:
                dx = keypoints_2d[idx][0] - hombro_x
                if abs(dx) > 1e-3:
                    return 1.0 if dx > 0 else -1.0
        return None

    @staticmethod
    def _escala_torso(keypoints_2d, hombro_idx, hombro_xy):
        """
        Escala del torso en píxeles = distancia hombro→oreja del lado visible
        (o hombro→nariz como respaldo). Sirve para dimensionar el ROI de forma
        invariante a la distancia a la cámara.
        """
        oreja_idx = 3 if hombro_idx == 5 else 4
        for idx in (oreja_idx, 3, 4, 0):
            if idx in keypoints_2d:
                p = keypoints_2d[idx]
                d = np.hypot(p[0] - hombro_xy[0], p[1] - hombro_xy[1])
                if d > 5:
                    return d
        return None

    # ------------------------------------------------------------------ #
    # Extracción del contorno de la espalda
    # ------------------------------------------------------------------ #

    def _contorno_espalda(self, bordes, x0, y0, esperado_x_local, back_dir, ancho_roi):
        """
        Recorre el mapa de bordes fila por fila y toma, por cada fila, el punto
        del contorno de la espalda más cercano a la posición esperada.

        Args:
            bordes: mapa binario de Canny del ROI (HxW).
            x0, y0: origen del ROI en el frame completo.
            esperado_x_local: x esperado del contorno (coord. local del ROI).
            back_dir: +1 si la espalda está a la derecha del hombro, -1 si a la izq.
            ancho_roi: ancho del ROI en px.

        Returns:
            (xs, ys): arrays de coordenadas del contorno en el frame completo,
            o (None, None) si no hay suficientes puntos.
        """
        alto, ancho = bordes.shape
        xs, ys = [], []

        # Banda de búsqueda en el lado de la espalda respecto al hombro.
        # Se aceptan bordes entre 0.15 y 1.4 de la escala para descartar tanto
        # bordes internos pegados al centro como el respaldo de la silla lejano.
        if back_dir > 0:
            x_min_local = max(0, int(esperado_x_local - 0.5 * ancho_roi))
            x_max_local = ancho - 1
        else:
            x_min_local = 0
            x_max_local = min(ancho - 1, int(esperado_x_local + 0.5 * ancho_roi))

        for fila in range(alto):
            columnas = np.nonzero(bordes[fila, x_min_local:x_max_local + 1])[0]
            if columnas.size == 0:
                continue
            columnas = columnas + x_min_local
            # Punto del contorno = borde más cercano a la posición esperada
            idx_mejor = int(np.argmin(np.abs(columnas - esperado_x_local)))
            xs.append(x0 + columnas[idx_mejor])
            ys.append(y0 + fila)

        if len(ys) < ENCORVADO_MIN_FILAS:
            return None, None
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #

    def analizar(self, frame, keypoints_2d, lado):
        """
        Mide el encorvamiento del tronco en vista lateral.

        Args:
            frame: imagen BGR de OpenCV (numpy HxWx3). Sin dibujos encima.
            keypoints_2d: dict {idx: (x, y, conf)} de la Fase 1.
            lado: 'izq' o 'der' — el lado visible/fiable en vista lateral.

        Returns:
            float: ángulo de flexión de tronco en grados (>= 0), ya descontada
            la línea base de postura erguida; o None si no hay señal fiable
            (en cuyo caso el evaluador mantiene su medición por keypoints).
        """
        if frame is None or keypoints_2d is None:
            return None

        hombro_idx = 5 if lado == 'izq' else 6
        if hombro_idx not in keypoints_2d:
            return None

        hombro = keypoints_2d[hombro_idx]
        hombro_xy = (hombro[0], hombro[1])

        back_dir_frente = self._direccion_frente(keypoints_2d, hombro[0])
        if back_dir_frente is None:
            return None
        back_dir = -back_dir_frente  # la espalda está en el lado opuesto a la cara

        escala = self._escala_torso(keypoints_2d, hombro_idx, hombro_xy)
        if escala is None:
            return None

        alto_f, ancho_f = frame.shape[:2]

        # --- Definir ROI del torso alrededor del hombro ---
        half_w = ENCORVADO_ROI_ESCALA_X * escala
        x0 = int(max(0, hombro[0] - half_w))
        x1 = int(min(ancho_f, hombro[0] + half_w))
        y0 = int(max(0, hombro[1] - ENCORVADO_ROI_ARRIBA * escala))
        y1 = int(min(alto_f, hombro[1] + ENCORVADO_ROI_ABAJO * escala))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None

        roi = frame[y0:y1, x0:x1]
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        k = ENCORVADO_BLUR_KSIZE | 1  # forzar impar
        gris = cv2.GaussianBlur(gris, (k, k), 0)
        bordes = cv2.Canny(gris, ENCORVADO_CANNY_LOW, ENCORVADO_CANNY_HIGH)

        # Posición esperada del contorno de la espalda (coord. local del ROI)
        esperado_x_local = (hombro[0] - x0) + back_dir * ENCORVADO_OFFSET_ESPALDA * escala

        xs, ys = self._contorno_espalda(
            bordes, x0, y0, esperado_x_local, back_dir, x1 - x0
        )
        if xs is None:
            return None

        # --- Ajuste robusto de recta x = a·y + b ---
        # Rechazo de outliers: primer ajuste, descartar puntos con residuo alto,
        # reajustar. Si el residuo sigue siendo alto, la señal no es fiable.
        a, b = np.polyfit(ys, xs, 1)
        residuos = np.abs(xs - (a * ys + b))
        mantener = residuos < (2.0 * np.median(residuos) + 1.0)
        if np.count_nonzero(mantener) >= ENCORVADO_MIN_FILAS:
            a, b = np.polyfit(ys[mantener], xs[mantener], 1)
            residuos = np.abs(xs[mantener] - (a * ys[mantener] + b))
        if np.sqrt(np.mean(residuos ** 2)) > ENCORVADO_RESIDUO_MAX:
            return None  # contorno demasiado ruidoso (fondo/ropa/silla)

        # --- Inclinación con signo respecto a la vertical ---
        # a = dx/dy. El contorno va de arriba (hombro) hacia abajo. Si la parte
        # alta queda por delante (dirección de la cara) de la parte baja, hay
        # inclinación hacia adelante = encorvamiento.
        # x_arriba - x_abajo = a·(y_arriba - y_abajo) = -a·span; "hacia adelante"
        # es la dirección de la cara (back_dir_frente).
        inclinacion_adelante = (-a) * back_dir_frente  # px de avance por px de altura
        angulo = np.degrees(np.arctan(inclinacion_adelante))

        # --- Suavizado temporal (EMA) ---
        if self._angulo_suavizado is None:
            self._angulo_suavizado = angulo
        else:
            s = ENCORVADO_SUAVIZADO
            self._angulo_suavizado = s * angulo + (1 - s) * self._angulo_suavizado
        ang = self._angulo_suavizado

        # --- Calibración: línea base = postura más erguida (menor ángulo) ---
        if self._baseline is None:
            self._baseline = ang
        elif ang < self._baseline:
            # postura más erguida: adoptar rápido
            self._baseline += (ang - self._baseline) * ENCORVADO_CALIB_ALPHA_SUBIDA
        else:
            # más encorvado: deriva lentísima (no erosiona la referencia)
            self._baseline += (ang - self._baseline) * ENCORVADO_CALIB_ALPHA_BAJADA

        # Exceso de inclinación sobre la postura erguida, con zona muerta, y
        # escalado a flexión de tronco efectiva (el contorno de la espalda alta
        # subestima la flexión de tronco completa).
        exceso = ang - self._baseline - ENCORVADO_MARGEN_GRADOS
        return float(max(0.0, exceso) * ENCORVADO_GANANCIA)
