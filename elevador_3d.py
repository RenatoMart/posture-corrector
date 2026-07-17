"""
elevador_3d.py — Fase 2: Ensamblado de la pose 3D.

Con YOLO esta fase era compleja: había que ESTIMAR la profundidad (Z) a partir
del escorzo de los segmentos, con proporciones antropométricas, modelo de cámara
pinhole y calibración relativa. Todo eso era un rodeo para suplir que YOLO no
daba profundidad, y era justo lo que fallaba al encorvarse.

Con MediaPipe la profundidad viene medida (world landmarks en 3D real). Por eso
esta fase se reduce a:

    1. Tomar la pose 3D en cm que entrega el detector.
    2. Añadir dos puntos virtuales que usa el evaluador RULA:
         17 = cuello (punto medio entre hombros)
         18 = centro de cadera (punto medio entre caderas)
    3. Suavizado temporal (media móvil exponencial) para estabilizar el HUD.

Convención de ejes (heredada del evaluador): X = horizontal, Y = vertical con
el sentido hacia ABAJO positivo (como la gravedad), Z = profundidad.
"""

import numpy as np
from config import (
    SUAVIZADO_ALPHA, SUAVIZADO_ONEEURO,
    ONEEURO_3D_MIN_CUTOFF, ONEEURO_3D_BETA, ONEEURO_3D_D_CUTOFF,
)
from filtro_oneeuro import EstabilizadorPose


class Elevador3D:
    """
    Ensambla la pose 3D real de MediaPipe para el evaluador RULA.

    Ya no estima profundidad: los world landmarks vienen en 3D. Solo agrega los
    puntos virtuales de cuello/cadera y aplica suavizado temporal.
    """

    def __init__(self, altura_sujeto_cm=None):
        """
        Args:
            altura_sujeto_cm: ignorado (ya no se escala por antropometría;
                se conserva en la firma por compatibilidad con Postura.py).
        """
        self.historial = {}          # último valor suavizado por keypoint (modo EMA)
        self.alpha = SUAVIZADO_ALPHA
        # Filtro One-Euro para la pose 3D (más estable que la EMA de alpha fijo)
        self.estabilizador = (
            EstabilizadorPose(
                min_cutoff=ONEEURO_3D_MIN_CUTOFF,
                beta=ONEEURO_3D_BETA,
                d_cutoff=ONEEURO_3D_D_CUTOFF,
            ) if SUAVIZADO_ONEEURO else None
        )

    def recalibrar(self):
        """
        Reinicia el suavizado temporal (tecla 'C').

        Con la pose 3D real ya no hay una "línea base de postura erguida" que
        calibrar (eso era propio de la estimación por escorzo). Se limpia el
        historial para que el suavizado no arrastre la postura anterior.
        """
        self.historial = {}
        if self.estabilizador is not None:
            self.estabilizador.reset()

    def elevar(self, keypoints_world, frame_shape=None):
        """
        Ensambla la pose 3D lista para el evaluador.

        Args:
            keypoints_world: dict {coco_idx: np.array([X, Y, Z]) en cm} de la
                Fase 1 (world landmarks de MediaPipe).
            frame_shape: ignorado (se conserva por compatibilidad de firma).

        Returns:
            dict {idx: np.array([X, Y, Z])} en cm, con los índices COCO 0-16 más
            los virtuales 17 (cuello) y 18 (centro de cadera); o None.
        """
        if not keypoints_world:
            return None

        # Copia para no mutar la entrada
        keypoints_3d = {idx: pos.astype(float).copy()
                        for idx, pos in keypoints_world.items()}

        # --- Puntos virtuales que consume el evaluador ---
        # 17 = cuello (punto medio entre hombros)
        if 5 in keypoints_3d and 6 in keypoints_3d:
            keypoints_3d[17] = (keypoints_3d[5] + keypoints_3d[6]) / 2
        # 18 = centro de cadera (punto medio entre caderas)
        if 11 in keypoints_3d and 12 in keypoints_3d:
            keypoints_3d[18] = (keypoints_3d[11] + keypoints_3d[12]) / 2

        # --- Suavizado temporal ---
        # MediaPipe ya suaviza, pero un filtro extra estabiliza los ángulos y el
        # HUD. One-Euro (por defecto) se adapta a la velocidad: quita el temblor
        # en reposo sin añadir retraso al moverse.
        if self.estabilizador is not None:
            return self.estabilizador(keypoints_3d)

        # Fallback: EMA de alpha fijo. Si un punto salta mucho (>40 cm) entre
        # frames se acelera la adaptación para no arrastrar un valor viejo.
        for idx, pos in keypoints_3d.items():
            if idx in self.historial:
                dist = np.linalg.norm(pos - self.historial[idx])
                alpha_eff = min(self.alpha + 0.5, 0.95) if dist > 40 else self.alpha
                keypoints_3d[idx] = alpha_eff * pos + (1 - alpha_eff) * self.historial[idx]
            self.historial[idx] = keypoints_3d[idx].copy()

        return keypoints_3d
