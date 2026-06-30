"""
config.py — Configuración central del sistema de análisis de postura.

Contiene:
- Índices y nombres de los 17 keypoints COCO
- Conexiones del esqueleto para visualización
- Proporciones antropométricas (fracción de altura total)
- Parámetros de cámara, alarma y detección
"""

# =============================================================================
# KEYPOINTS COCO (17 puntos) — Índices estándar de YOLOv8-Pose
# =============================================================================
KEYPOINT_NOMBRES = {
    0: 'nariz',
    1: 'ojo_izq',
    2: 'ojo_der',
    3: 'oreja_izq',
    4: 'oreja_der',
    5: 'hombro_izq',
    6: 'hombro_der',
    7: 'codo_izq',
    8: 'codo_der',
    9: 'muneca_izq',
    10: 'muneca_der',
    11: 'cadera_izq',
    12: 'cadera_der',
    13: 'rodilla_izq',
    14: 'rodilla_der',
    15: 'tobillo_izq',
    16: 'tobillo_der',
}

# Índices rápidos por nombre (para acceso legible)
IDX = {v: k for k, v in KEYPOINT_NOMBRES.items()}

# =============================================================================
# CONEXIONES DEL ESQUELETO — Pares de keypoints para dibujar líneas
# =============================================================================
SKELETON_CONEXIONES = [
    # Cara
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Brazos
    (5, 7), (7, 9),    # Brazo izquierdo
    (6, 8), (8, 10),   # Brazo derecho
    # Torso
    (5, 6),             # Hombros
    (5, 11), (6, 12),   # Laterales del torso
    (11, 12),           # Caderas
    # Piernas
    (11, 13), (13, 15), # Pierna izquierda
    (12, 14), (14, 16), # Pierna derecha
]

# Conexiones que pertenecen al torso (para colorear diferente)
TORSO_CONEXIONES = {(5, 6), (5, 11), (6, 12), (11, 12)}

# =============================================================================
# PROPORCIONES ANTROPOMÉTRICAS — Fracción de la altura total del sujeto
# Basadas en datos antropométricos estándar (Drillis & Contini, 1966)
# =============================================================================
PROPORCIONES_CORPORALES = {
    'hombro_codo': 0.186,       # Longitud del brazo superior
    'codo_muneca': 0.146,       # Longitud del antebrazo
    'hombro_cadera': 0.288,     # Longitud del tronco
    'cadera_rodilla': 0.245,    # Longitud del muslo
    'rodilla_tobillo': 0.246,   # Longitud de la espinilla
    'ancho_hombros': 0.259,     # Ancho entre hombros
}

# Segmentos corporales para propagación de profundidad (padre, hijo, nombre)
SEGMENTOS_CADENA = [
    (5, 7, 'hombro_codo'),       # Hombro izq → Codo izq
    (7, 9, 'codo_muneca'),       # Codo izq → Muñeca izq
    (6, 8, 'hombro_codo'),       # Hombro der → Codo der
    (8, 10, 'codo_muneca'),      # Codo der → Muñeca der
    (11, 13, 'cadera_rodilla'),  # Cadera izq → Rodilla izq
    (13, 15, 'rodilla_tobillo'), # Rodilla izq → Tobillo izq
    (12, 14, 'cadera_rodilla'),  # Cadera der → Rodilla der
    (14, 16, 'rodilla_tobillo'), # Rodilla der → Tobillo der
]

# =============================================================================
# PARÁMETROS DE CÁMARA
# =============================================================================
CAMERA_INDEX = 0                # Índice de la cámara (0 = principal)
CAMERA_FOV_GRADOS = 60         # Campo de visión estimado (grados)

# =============================================================================
# PARÁMETROS DEL SUJETO
# =============================================================================
ALTURA_SUJETO_CM = 170          # Altura del sujeto en centímetros

# =============================================================================
# PARÁMETROS DE DETECCIÓN
# =============================================================================
YOLO_MODELO = 'yolov8n-pose.pt'  # Modelo nano (más rápido, 6.4MB)
YOLO_CONFIANZA = 0.5             # Confianza mínima de detección de persona
KEYPOINT_CONFIANZA_MIN = 0.3     # Confianza mínima por keypoint individual

# =============================================================================
# PREPROCESAMIENTO DE IMAGEN (mejora de iluminación)
# =============================================================================
PREPROCESAMIENTO_ACTIVO = True       # Activar/desactivar filtros de imagen
CLAHE_CLIP_LIMIT = 2.0               # Límite de contraste CLAHE (mayor = más contraste)
CLAHE_TILE_SIZE = (8, 8)             # Tamaño de grilla para ecualización adaptativa

# =============================================================================
# PARÁMETROS DE ALARMA
# =============================================================================
ALARMA_FRECUENCIA_HZ = 1000     # Frecuencia del tono (Hz)
ALARMA_DURACION_MS = 300         # Duración del tono (ms)
UMBRAL_FRAMES_ALARMA = 30       # Frames consecutivos antes de activar alarma
UMBRAL_RULA_ALARMA = 5          # Score RULA mínimo para considerar riesgo

# =============================================================================
# PARÁMETROS DE SUAVIZADO TEMPORAL
# =============================================================================
SUAVIZADO_ALPHA = 0.4            # Factor de suavizado exponencial (0-1, mayor = más reactivo)

# =============================================================================
# COLORES BGR PARA VISUALIZACIÓN
# =============================================================================
COLOR_VERDE = (0, 200, 100)      # Postura aceptable (RULA 1-2)
COLOR_AMARILLO = (0, 220, 255)   # Investigar (RULA 3-4)
COLOR_NARANJA = (0, 140, 255)    # Cambio pronto (RULA 5-6)
COLOR_ROJO = (0, 0, 255)         # Cambio inmediato (RULA 7+)
COLOR_BLANCO = (255, 255, 255)
COLOR_GRIS = (180, 180, 180)
COLOR_HUD_FONDO = (30, 30, 30)   # Fondo del panel HUD
