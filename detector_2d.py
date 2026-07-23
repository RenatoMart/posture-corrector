"""
detector_2d.py — Fase 1: Extracción de pose con MediaPipe Pose Landmarker.

Reemplaza a YOLOv8-Pose. La diferencia clave para este proyecto:

    - YOLO daba 17 keypoints SOLO en 2D (x, y). La profundidad (encorvarse,
      cabeza adelantada) había que estimarla con geometría por escorzo, que es
      frágil y fallaba justo cuando la persona estaba encorvada.
    - MediaPipe da 33 landmarks CON profundidad real: los "world landmarks" son
      coordenadas 3D en metros, con origen en el centro de las caderas. Así la
      postura sagital se mide en 3D real y directo, en cualquier vista.

Usa la API moderna de MediaPipe (`mediapipe.tasks`), ya que la antigua
`mp.solutions.pose` fue retirada. El modelo `.task` se descarga solo la primera
vez (igual que YOLO descargaba su `.pt`).

Salida de `detectar()`:
    - keypoints_2d:   {coco_idx: (x_px, y_px, visibilidad)} para dibujar el
                      esqueleto y clasificar la vista de cámara.
    - keypoints_world:{coco_idx: np.array([X, Y, Z]) en cm} con la pose 3D real
                      (X derecha, Y abajo, Z profundidad), lista para el evaluador.

Los 33 landmarks de MediaPipe se mapean a los 17 índices COCO que ya usaba el
resto del sistema (evaluador_rula, visualizador, config), por lo que esos
módulos siguen funcionando sin cambios.
"""

import os
import sys
import contextlib
import urllib.request

# --- Silenciar los avisos informativos de MediaPipe / TensorFlow Lite ---------
# NO son errores: son mensajes del motor C++ (XNNPACK, "feedback manager", absl).
# Estas variables deben fijarse ANTES de importar mediapipe para tener efecto.
os.environ.setdefault('GLOG_minloglevel', '2')      # oculta INFO y WARNING del C++
os.environ.setdefault('GLOG_logtostderr', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # oculta logs de TensorFlow Lite

import warnings
# Aviso de deprecación de protobuf (inofensivo, depende de la versión instalada)
warnings.filterwarnings('ignore', category=UserWarning, module=r'google\.protobuf.*')
warnings.filterwarnings('ignore', message=r'.*SymbolDatabase\.GetPrototype.*')

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:  # Bajar también la verbosidad del logger absl que usa MediaPipe
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass

from config import (
    MP_MODEL_COMPLEXITY, MP_MODELOS, MP_USAR_GPU,
    MP_MIN_DET_CONF, MP_MIN_TRACK_CONF, MP_MIN_PRESENCE_CONF,
    MP_PRESENCE_MIN, MP_PRESENCE_MIN_3D,
    PREPROCESAMIENTO_ACTIVO, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
)


@contextlib.contextmanager
def _silenciar_stderr_c():
    """
    Redirige temporalmente el descriptor de stderr (nivel del sistema) a nulo.

    Se usa solo mientras se carga el modelo de MediaPipe, donde el motor C++
    escupe unos avisos informativos (XNNPACK, feedback manager, absl) que no se
    pueden filtrar desde Python. Si la carga fallara de verdad, MediaPipe lanza
    una excepción Python, que NO se silencia.
    """
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, ValueError):
        # stderr sin descriptor real (p.ej. redirigido en un IDE): no hacer nada
        yield
        return
    with open(os.devnull, 'w') as devnull:
        guardado = os.dup(fd)
        try:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), fd)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(guardado, fd)
            os.close(guardado)


# =============================================================================
# Mapeo MediaPipe (33 landmarks) → COCO (17 keypoints)
# Índices MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# =============================================================================
_MP_A_COCO = {
    0: 0,    # nariz
    2: 1,    # ojo izquierdo
    5: 2,    # ojo derecho
    7: 3,    # oreja izquierda
    8: 4,    # oreja derecha
    11: 5,   # hombro izquierdo
    12: 6,   # hombro derecho
    13: 7,   # codo izquierdo
    14: 8,   # codo derecho
    15: 9,   # muñeca izquierda
    16: 10,  # muñeca derecha
    23: 11,  # cadera izquierda
    24: 12,  # cadera derecha
    25: 13,  # rodilla izquierda
    26: 14,  # rodilla derecha
    27: 15,  # tobillo izquierdo
    28: 16,  # tobillo derecho
}

# Factor metros → centímetros (los world landmarks vienen en metros)
_M_A_CM = 100.0

# Directorio del proyecto (para guardar el modelo .task junto al código)
_DIR = os.path.dirname(os.path.abspath(__file__))


class Detector2D:
    """
    Detector de pose basado en MediaPipe Pose Landmarker (Tasks API).

    Extrae la pose de la persona más prominente en el frame y la entrega en dos
    representaciones alineadas por índice COCO: 2D en píxeles (para dibujar y
    clasificar la vista) y 3D real en cm (world landmarks, para el evaluador).

    MediaPipe rastrea automáticamente a una sola persona entre frames (modo
    VIDEO), por lo que ya no hace falta el tracking por centroide del detector
    anterior.
    """

    def __init__(self, modelo=None, confianza=MP_MIN_DET_CONF, usar_gpu=MP_USAR_GPU):
        """
        Args:
            modelo: ruta a un .task concreto; si es None, se elige según
                MP_MODEL_COMPLEXITY y se descarga automáticamente si falta.
            confianza: confianza mínima de detección de la persona.
            usar_gpu: intentar el delegado GPU de MediaPipe. Si el build no lo
                soporta (caso de Windows, GPU deshabilitada en la librería) cae
                automáticamente a CPU. Se puede forzar con `python Postura.py --gpu`.
        """
        ruta_modelo = modelo or self._asegurar_modelo(MP_MODEL_COMPLEXITY)

        self.landmarker, self.delegado = self._crear_landmarker(
            ruta_modelo, confianza, usar_gpu
        )
        self.presence_min = MP_PRESENCE_MIN          # filtro para los puntos 2D (dibujo/vista)
        self.presence_min_3d = MP_PRESENCE_MIN_3D    # filtro (bajo) para los world landmarks 3D

        # Timestamp monótono creciente (ms) que exige el modo VIDEO
        self._ts_ms = 0

        # CLAHE reutilizable para mejorar detección en baja iluminación
        if PREPROCESAMIENTO_ACTIVO:
            self.clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_SIZE,
            )
            print(f"[Detector2D] Preprocesamiento CLAHE activado "
                  f"(clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE})")
        else:
            self.clahe = None

        print(f"[Detector2D] MediaPipe Pose Landmarker cargado "
              f"(modelo={os.path.basename(ruta_modelo)}, delegado={self.delegado}, "
              f"presencia_min={self.presence_min}).")

    @staticmethod
    def _crear_landmarker(ruta_modelo, confianza, usar_gpu):
        """
        Crea el PoseLandmarker con el delegado pedido (GPU o CPU).

        Si se pide GPU pero el build de MediaPipe no lo soporta (típico en
        Windows: "GPU processing is disabled in build flags"), se captura el
        error y se reintenta con CPU, avisando por consola. Nunca deja el
        sistema sin detector por pedir GPU.

        Returns:
            tuple (landmarker, nombre_delegado_usado)
        """
        Delegate = mp_python.BaseOptions.Delegate

        def _construir(delegate):
            opciones = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ruta_modelo, delegate=delegate),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=confianza,
                min_pose_presence_confidence=MP_MIN_PRESENCE_CONF,
                min_tracking_confidence=MP_MIN_TRACK_CONF,
                output_segmentation_masks=False,
            )
            # La creación del modelo emite avisos informativos del C++; se silencian.
            with _silenciar_stderr_c():
                return mp_vision.PoseLandmarker.create_from_options(opciones)

        if usar_gpu:
            print("[Detector2D] Intentando acelerar por GPU...")
            try:
                landmarker = _construir(Delegate.GPU)
                print("[Detector2D] GPU ACTIVADA.")
                return landmarker, 'GPU'
            except Exception as e:
                motivo = str(e).splitlines()[0] if str(e) else type(e).__name__
                print("[Detector2D] GPU no disponible en este equipo "
                      f"({motivo}). Usando CPU.")

        return _construir(Delegate.CPU), 'CPU'

    @staticmethod
    def _asegurar_modelo(complejidad):
        """
        Devuelve la ruta local del modelo .task, descargándolo si no existe.
        """
        nombre, url = MP_MODELOS.get(complejidad, MP_MODELOS[1])
        ruta = os.path.join(_DIR, nombre)
        if not os.path.exists(ruta):
            print(f"[Detector2D] Descargando modelo '{nombre}' (una sola vez)...")
            urllib.request.urlretrieve(url, ruta)
            print(f"[Detector2D] Modelo guardado en {ruta}")
        return ruta

    def _preprocesar_frame(self, frame):
        """
        Mejora el frame para condiciones de iluminación difíciles.

        Aplica CLAHE solo en el canal de luminosidad (espacio LAB) para mejorar
        el contraste sin distorsionar colores. Coste ~1-2 ms por frame.
        """
        if self.clahe is None:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self.clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def detectar(self, frame):
        """
        Detecta la pose de la persona principal en el frame.

        Args:
            frame: imagen BGR de OpenCV (numpy HxWx3).

        Returns:
            tuple (keypoints_2d, keypoints_world):
                keypoints_2d:    dict {coco_idx: (x_px, y_px, visibilidad)}
                keypoints_world: dict {coco_idx: np.array([X, Y, Z]) en cm}
            Ambos None si no se detecta una persona con al menos un hombro.
        """
        frame_proc = self._preprocesar_frame(frame)

        # MediaPipe Tasks espera un mp.Image en RGB
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # El modo VIDEO exige timestamps estrictamente crecientes (ms)
        self._ts_ms += 33
        resultado = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        if not resultado.pose_landmarks:
            return None, None

        alto, ancho = frame.shape[:2]
        lm2d = resultado.pose_landmarks[0]
        lmw = (resultado.pose_world_landmarks[0]
               if resultado.pose_world_landmarks else None)

        keypoints_2d = {}
        keypoints_world = {}

        for mp_idx, coco_idx in _MP_A_COCO.items():
            p = lm2d[mp_idx]
            presencia = float(p.presence)
            vis = float(p.visibility)  # se guarda para que el evaluador clasifique la vista

            # --- Punto 2D (para dibujar el esqueleto y clasificar la vista) ---
            # Se filtra por PRESENCIA: no tiene sentido dibujar un punto que está
            # FUERA del encuadre (quedaría en una posición sin significado).
            if presencia >= self.presence_min:
                px = float(p.x * ancho)
                py = float(p.y * alto)
                keypoints_2d[coco_idx] = (px, py, vis)

            # --- Punto 3D (world landmark) en cm: X derecha, Y abajo, Z prof. ---
            # MediaPipe estima el ESQUELETO 3D COMPLETO aunque el punto esté fuera
            # de cuadro (lo infiere por proporciones humanas). Por eso el 3D usa un
            # umbral MUCHO más bajo: así se conservan las caderas/tronco inferidos
            # cuando la persona está muy cerca y solo se le ve cabeza y hombros,
            # y se puede seguir midiendo la postura (RULA + clasificador ML).
            if lmw is not None and presencia >= self.presence_min_3d:
                w = lmw[mp_idx]
                keypoints_world[coco_idx] = np.array([
                    w.x * _M_A_CM,
                    w.y * _M_A_CM,
                    w.z * _M_A_CM,
                ])

        # Punto mínimo: al menos un hombro visible (igual que el sistema anterior)
        if 5 not in keypoints_2d and 6 not in keypoints_2d:
            return None, None

        if not keypoints_world:
            keypoints_world = None

        return keypoints_2d, keypoints_world

    def obtener_bbox(self, frame):
        """
        Bounding box aproximado de la persona a partir de los landmarks 2D.

        Returns:
            tuple (x1, y1, x2, y2) o None.
        """
        keypoints_2d, _ = self.detectar(frame)
        if not keypoints_2d:
            return None

        xs = [p[0] for p in keypoints_2d.values()]
        ys = [p[1] for p in keypoints_2d.values()]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def cerrar(self):
        """Libera los recursos de MediaPipe."""
        self.landmarker.close()
