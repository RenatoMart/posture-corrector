"""
detector_2d.py — Fase 1: Extracción 2D Ultrarrápida con YOLOv8-Pose.

Utiliza el modelo YOLOv8-Nano-Pose para extraer los 17 keypoints COCO
de la persona detectada en cada frame. Optimizado para velocidad en
tiempo real sin necesidad de GPU.

Mejoras:
    - Preprocesamiento CLAHE para mejorar detección en baja iluminación
    - Detección parcial: funciona con solo el torso superior visible
    - Estimación de caderas cuando no son visibles en el frame
    - Detección con 1 solo hombro (vista lateral de cámara)
    - Estimación del hombro faltante para vistas no frontales
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import (
    YOLO_MODELO, YOLO_CONFIANZA, KEYPOINT_CONFIANZA_MIN, KEYPOINT_NOMBRES,
    PREPROCESAMIENTO_ACTIVO, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)


# Confianza asignada a keypoints estimados (no detectados directamente)
CONF_ESTIMADA = 0.25


class Detector2D:
    """
    Detector de pose 2D basado en YOLOv8-Pose.

    Extrae las coordenadas (x, y) en píxeles de los 17 keypoints COCO
    de la persona con mayor confianza en el frame.

    Soporta:
    - Detección parcial (solo torso superior visible)
    - Vista lateral (1 solo hombro visible)
    - Estimación de hombro y caderas faltantes
    - Tracking de persona principal entre frames
    """

    RATIO_TORSO_HOMBROS = 1.4

    # Frames sin detección antes de resetear el tracker
    _MAX_FRAMES_PERDIDOS = 15
    # Distancia máxima en px para seguir considerando la misma persona
    _MAX_DIST_TRACKING = 250

    def __init__(self, modelo=YOLO_MODELO, confianza=YOLO_CONFIANZA):
        """
        Args:
            modelo: Nombre del modelo YOLOv8-Pose (se descarga automáticamente).
            confianza: Umbral mínimo de confianza para detección de persona.
        """
        self.modelo = YOLO(modelo)
        self.confianza = confianza

        # Crear objeto CLAHE reutilizable (más eficiente que recrearlo cada frame)
        if PREPROCESAMIENTO_ACTIVO:
            self.clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_SIZE,
            )
            print(f"[Detector2D] Preprocesamiento CLAHE activado "
                  f"(clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE})")
        else:
            self.clahe = None

        # Estado del tracker de persona principal
        self._centroide_rastreado = None  # (cx, cy) en píxeles del frame anterior
        self._frames_perdidos = 0

        print(f"[Detector2D] Modelo '{modelo}' cargado correctamente.")

    def _preprocesar_frame(self, frame):
        """
        Mejora el frame para detección en condiciones de iluminación difíciles.
        
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) solo
        en el canal de luminosidad (espacio LAB) para mejorar contraste sin
        distorsionar colores. Coste: ~1-2ms por frame.
        
        Args:
            frame: Imagen BGR de OpenCV (numpy array HxWx3).
            
        Returns:
            Frame preprocesado con contraste mejorado.
        """
        if self.clahe is None:
            return frame

        # Convertir a LAB para ecualizar solo luminosidad
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Aplicar CLAHE al canal de luminosidad
        l_eq = self.clahe.apply(l_channel)

        # Reconstruir imagen
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        resultado = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        return resultado

    def _seleccionar_persona(self, resultados):
        """
        Selecciona el índice de la persona principal en los tensores de YOLO.

        Estrategia:
        - Sin historial (primer frame o tras pérdida): elegir la persona con mayor
          bounding box (más grande = más cercana a la cámara).
        - Con historial activo y múltiples personas: seguir a la más cercana al
          centroide registrado en el frame anterior (tracking por centroide).
        - Si la persona rastreada se aleja más de _MAX_DIST_TRACKING px, se
          reinicia el tracker y se vuelve a elegir por tamaño.

        Returns:
            int: índice en los tensores de YOLO, o None si no hay detecciones.
        """
        tiene_boxes = (
            resultados[0].boxes is not None
            and resultados[0].boxes.data.shape[0] > 0
        )
        tiene_kps = (
            resultados[0].keypoints is not None
            and resultados[0].keypoints.data.shape[0] > 0
        )

        if not tiene_boxes or not tiene_kps:
            self._frames_perdidos += 1
            if self._frames_perdidos >= self._MAX_FRAMES_PERDIDOS:
                self._centroide_rastreado = None
            return None

        boxes_np = resultados[0].boxes.xyxy.cpu().numpy()  # (n, 4)
        n = boxes_np.shape[0]

        centroides = np.column_stack([
            (boxes_np[:, 0] + boxes_np[:, 2]) / 2,
            (boxes_np[:, 1] + boxes_np[:, 3]) / 2,
        ])
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])

        if self._centroide_rastreado is not None and n > 1:
            # Tracking activo con múltiples personas: seguir la más cercana
            dists = np.linalg.norm(
                centroides - np.array(self._centroide_rastreado), axis=1
            )
            idx_min = int(np.argmin(dists))
            if dists[idx_min] <= self._MAX_DIST_TRACKING:
                idx = idx_min
            else:
                # La persona rastreada desapareció: reiniciar con la más grande
                idx = int(np.argmax(areas))
                self._centroide_rastreado = None
        else:
            # Sin historial o una sola persona: la más grande (más cercana)
            idx = int(np.argmax(areas))

        self._centroide_rastreado = (float(centroides[idx, 0]), float(centroides[idx, 1]))
        self._frames_perdidos = 0
        return idx

    def _estimar_hombro_faltante(self, keypoints, frame_shape):
        """
        Estima el hombro faltante cuando la cámara está en ángulo lateral.
        
        Usa la orientación inferida del cuerpo (orejas, nariz, cadera visible)
        para estimar la posición del hombro oculto. Se asigna confianza baja
        para que el evaluador pondere correctamente.
        
        Args:
            keypoints: dict con al menos 1 hombro detectado.
            frame_shape: tuple (alto, ancho, canales).
            
        Returns:
            dict actualizado con hombro estimado.
        """
        alto, ancho = frame_shape[:2]

        tiene_izq = 5 in keypoints
        tiene_der = 6 in keypoints

        if tiene_izq and tiene_der:
            return keypoints  # No falta ninguno

        # El hombro visible
        hombro_visible_idx = 5 if tiene_izq else 6
        hombro_faltante_idx = 6 if tiene_izq else 5
        hv = keypoints[hombro_visible_idx]

        # Estimar ancho de hombros: usar orejas si están disponibles,
        # o usar una proporción del tamaño de la persona en el frame
        ancho_estimado = 0

        # Método 1: Usar distancia entre orejas × 2.2 como proxy
        if 3 in keypoints and 4 in keypoints:
            dist_orejas = abs(keypoints[3][0] - keypoints[4][0])
            ancho_estimado = dist_orejas * 2.2
        # Método 2: Usar distancia oreja-nariz × 3 como proxy
        elif 0 in keypoints and (3 in keypoints or 4 in keypoints):
            oreja_idx = 3 if 3 in keypoints else 4
            dist_nariz_oreja = abs(keypoints[0][0] - keypoints[oreja_idx][0])
            ancho_estimado = dist_nariz_oreja * 3.0
        # Método 3: Usar la cadera del mismo lado si está disponible
        elif (hombro_visible_idx == 5 and 11 in keypoints):
            # La distancia hombro-cadera da una referencia de escala
            dist_torso = abs(keypoints[5][1] - keypoints[11][1])
            ancho_estimado = dist_torso * 0.6  # Hombros ≈ 60% del largo del torso
        elif (hombro_visible_idx == 6 and 12 in keypoints):
            dist_torso = abs(keypoints[6][1] - keypoints[12][1])
            ancho_estimado = dist_torso * 0.6

        # Si no se pudo estimar, usar un valor por defecto basado en el frame
        if ancho_estimado < 20:
            ancho_estimado = ancho * 0.15  # ~15% del ancho del frame

        # Para vista lateral, el hombro oculto está detrás (reducir separación)
        # En vista lateral los hombros se ven más juntos
        separacion = ancho_estimado * 0.3  # Solo 30% de la separación real

        # Determinar dirección: el hombro faltante va hacia el lado opuesto
        if hombro_visible_idx == 5:  # Falta el derecho
            hombro_x = hv[0] + separacion
        else:  # Falta el izquierdo
            hombro_x = hv[0] - separacion

        # Misma altura Y que el hombro visible
        hombro_y = hv[1]

        # Clamp dentro del frame
        hombro_x = max(5, min(hombro_x, ancho - 5))

        keypoints[hombro_faltante_idx] = (float(hombro_x), float(hombro_y), CONF_ESTIMADA)

        return keypoints

    def _estimar_caderas(self, keypoints, frame_shape):
        """
        Estima posición de caderas cuando no son visibles en el frame.
        
        Usa la posición de los hombros y proporciones antropométricas para
        estimar dónde estarían las caderas. Se asigna una confianza baja
        para que el sistema sepa que son estimaciones.
        
        Args:
            keypoints: dict {idx: (x, y, conf)} con keypoints detectados.
            frame_shape: tuple (alto, ancho, canales) del frame.
            
        Returns:
            dict actualizado con caderas estimadas (si faltaban).
        """
        alto, ancho = frame_shape[:2]

        if 5 not in keypoints or 6 not in keypoints:
            return keypoints

        hombro_izq = keypoints[5]
        hombro_der = keypoints[6]

        # Calcular ancho de hombros en píxeles
        ancho_hombros = abs(hombro_izq[0] - hombro_der[0])

        if ancho_hombros < 5:
            # Vista muy lateral: hombros casi superpuestos
            # Usar la altura como referencia para el largo del torso
            if 0 in keypoints:  # Usar nariz como referencia de escala
                dist_cabeza_hombro = abs(keypoints[0][1] - hombro_izq[1])
                largo_torso = dist_cabeza_hombro * 2.0
            else:
                largo_torso = alto * 0.25  # Estimación por defecto
        else:
            # Estimar longitud del torso en píxeles
            largo_torso = ancho_hombros * self.RATIO_TORSO_HOMBROS

        # Estimar posición de caderas: debajo de los hombros
        centro_hombros_x = (hombro_izq[0] + hombro_der[0]) / 2
        centro_hombros_y = (hombro_izq[1] + hombro_der[1]) / 2

        # Para vista lateral, las caderas están casi alineadas con los hombros
        ancho_caderas = max(ancho_hombros * 0.80, 10)

        cadera_y = min(centro_hombros_y + largo_torso, alto - 5)
        cadera_izq_x = centro_hombros_x - ancho_caderas / 2
        cadera_der_x = centro_hombros_x + ancho_caderas / 2

        if 11 not in keypoints:
            keypoints[11] = (float(cadera_izq_x), float(cadera_y), CONF_ESTIMADA)

        if 12 not in keypoints:
            keypoints[12] = (float(cadera_der_x), float(cadera_y), CONF_ESTIMADA)

        return keypoints

    def detectar(self, frame):
        """
        Detecta keypoints 2D de la persona principal en el frame.

        Funciona con cámara frontal, lateral o en ángulo.
        Solo requiere al menos 1 hombro visible para iniciar la detección.

        Args:
            frame: Imagen BGR de OpenCV (numpy array HxWx3).

        Returns:
            dict {idx: (x, y, confianza)} con los keypoints detectados,
            o None si no se detecta una persona con suficientes puntos.
        """
        # Preprocesar frame para mejorar detección en baja luz
        frame_procesado = self._preprocesar_frame(frame)

        # Inferencia YOLOv8-Pose (verbose=False para no llenar la consola)
        resultados = self.modelo(frame_procesado, conf=self.confianza, verbose=False)

        if len(resultados) == 0:
            self._frames_perdidos += 1
            if self._frames_perdidos >= self._MAX_FRAMES_PERDIDOS:
                self._centroide_rastreado = None
            return None

        # Seleccionar la persona principal con tracking por centroide
        # kps.data shape: (num_personas, 17, 3) donde 3 = (x, y, conf)
        persona_idx = self._seleccionar_persona(resultados)
        if persona_idx is None:
            return None

        kps = resultados[0].keypoints
        persona = kps.data[persona_idx].cpu().numpy()

        keypoints = {}
        for idx in range(17):
            x, y, conf = persona[idx]
            if conf >= KEYPOINT_CONFIANZA_MIN:
                keypoints[idx] = (float(x), float(y), float(conf))

        # Punto mínimo: al menos 1 hombro visible
        # (permite detección desde cualquier ángulo de cámara)
        tiene_hombro = 5 in keypoints or 6 in keypoints
        if not tiene_hombro:
            return None

        # Si falta un hombro, estimarlo (vista lateral)
        if 5 not in keypoints or 6 not in keypoints:
            keypoints = self._estimar_hombro_faltante(keypoints, frame.shape)

        # Si faltan caderas, estimarlas desde los hombros
        if 11 not in keypoints or 12 not in keypoints:
            keypoints = self._estimar_caderas(keypoints, frame.shape)

        return keypoints

    def obtener_bbox(self, frame):
        """
        Obtiene el bounding box de la persona detectada.

        Returns:
            tuple (x1, y1, x2, y2) o None
        """
        frame_procesado = self._preprocesar_frame(frame)
        resultados = self.modelo(frame_procesado, conf=self.confianza, verbose=False)

        if len(resultados) == 0 or resultados[0].boxes is None:
            return None

        boxes = resultados[0].boxes
        if boxes.data.shape[0] == 0:
            return None

        box = boxes.xyxy[0].cpu().numpy()
        return tuple(box.astype(int))
