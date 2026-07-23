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
# PARÁMETROS DE DETECCIÓN — MediaPipe Pose (BlazePose)
# =============================================================================
# MediaPipe entrega 33 landmarks CON profundidad real (world landmarks en metros,
# origen en el centro de las caderas). Eso resuelve la limitación central del
# sistema anterior: la postura sagital (encorvarse / cabeza adelantada) ahora se
# mide en 3D real en CUALQUIER vista, sin estimar profundidad por escorzo.
# Los 33 landmarks se mapean a los 17 índices COCO (ver detector_2d._MP_A_COCO)
# para que el resto del pipeline (evaluador, visualizador) siga igual.
MP_MODEL_COMPLEXITY = 1          # 0 = lite (rápido) / 1 = full (equilibrado) / 2 = heavy (más preciso).
                                 # En CPU (MediaPipe en Windows es SIEMPRE CPU, ver MP_USAR_GPU),
                                 # 'full' da el mejor equilibrio precisión/FPS. 'heavy' es hasta ~3x
                                 # más lento y en un portátil ahoga los FPS. Si aún vas justo, baja a 0
                                 # (lite); si te sobra CPU y quieres máxima precisión, sube a 2 (heavy).
MP_USAR_GPU = False              # Intentar acelerar por GPU (delegado GPU de MediaPipe).
                                 # OJO: la rueda de MediaPipe para Windows en Python viene compilada
                                 # SOLO para CPU (GPU disabled in build flags), así que en Windows esto
                                 # cae automáticamente a CPU. Útil solo en Linux o en un build con GPU.
                                 # Se puede forzar por línea de comandos: python Postura.py --gpu
MP_MIN_DET_CONF = 0.5            # Confianza mínima para detectar a la persona
MP_MIN_TRACK_CONF = 0.5          # Confianza mínima para seguir a la persona entre frames
MP_MIN_PRESENCE_CONF = 0.5       # Confianza mínima de presencia de la pose
MP_PRESENCE_MIN = 0.5            # Presencia mínima por landmark para usarlo EN 2D (0-1). Se filtra
                                 # por PRESENCIA (¿está el punto en el encuadre?) y NO por visibilidad.
                                 # Afecta SOLO a los puntos 2D que se dibujan y a la clasificación de
                                 # vista: no tiene sentido dibujar un punto que está fuera de cuadro.
                                 # La visibilidad se guarda igual y el evaluador la usa para clasificar
                                 # la vista (frente/lado).
MP_PRESENCE_MIN_3D = 0.0        # Presencia mínima por landmark para usarlo EN 3D (world landmark).
                                 # Se pone MUY bajo (0 = conservar todos) a propósito: MediaPipe estima
                                 # el ESQUELETO 3D COMPLETO aunque un punto esté FUERA de cuadro (lo
                                 # infiere por proporciones humanas). Conservar esos puntos inferidos
                                 # permite medir el tronco/caderas cuando la persona está MUY CERCA de
                                 # la cámara y solo se le ve cabeza, hombros y parte del torso. Súbelo
                                 # (p.ej. 0.1-0.2) si notas ruido por inferencias malas.

# Modelos de MediaPipe Tasks (archivo .task, se descarga solo la primera vez).
# Cada complejidad usa un bundle distinto (lite / full / heavy).
MP_MODELOS = {
    0: ('pose_landmarker_lite.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'),
    1: ('pose_landmarker_full.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_full/float16/latest/pose_landmarker_full.task'),
    2: ('pose_landmarker_heavy.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'),
}
CONF_FABRICADO = 0.26            # Visibilidad <= a esto = landmark poco fiable (oculto).
                                 # Usado por el evaluador para clasificar la vista de cámara.

# =============================================================================
# CLASIFICACIÓN DE VISTA DE CÁMARA (frontal / ángulo / lateral)
# =============================================================================
# Determina si el tronco/cuello se miden con ambos lados (frontal) o con un solo
# lado real (lateral/ángulo), para no contaminar la medición con keypoints
# fabricados del lado oculto. Se combinan dos señales robustas e independientes
# de las caderas (que suelen estar fabricadas al estar sentado en un escritorio):
#
#  1) Asimetría de confianza entre lados del cuerpo. En perfil, todo un lado
#     (ojo, oreja, hombro, cadera) queda ocluido y YOLO le da baja confianza.
#  2) Separación de hombros normalizada por el alto de la cabeza (de hombros a
#     nariz/oreja). De frente los hombros se ven anchos; de costado se juntan.
VISTA_ASIMETRIA_LATERAL = 0.45   # conf_lado_debil / conf_lado_fuerte < esto => lateral
VISTA_SPREAD_FRONTAL = 1.40      # (ancho_hombros / alto_cabeza) > esto => frontal
VISTA_SPREAD_LATERAL = 0.70      # (ancho_hombros / alto_cabeza) < esto => lateral
VISTA_NARIZ_OFFSET_LATERAL = 0.60  # |x_nariz - x_centro_hombros| / alto_cabeza > esto => lateral.
                                   # De frente la nariz queda centrada entre los hombros aunque
                                   # gires la cabeza; de costado queda desplazada hacia el hombro
                                   # visible AUNQUE mires a la cámara. Esta señal evita que el
                                   # sistema exija mirar a la cámara para reconocer la vista lateral.
VISTA_HISTERESIS_FRAMES = 4      # frames consistentes requeridos para cambiar de vista (anti-parpadeo)

# =============================================================================
# PREPROCESAMIENTO DE IMAGEN (mejora de iluminación)
# =============================================================================
PREPROCESAMIENTO_ACTIVO = True       # Activar/desactivar filtros de imagen
CLAHE_CLIP_LIMIT = 2.0               # Límite de contraste CLAHE (mayor = más contraste)
CLAHE_TILE_SIZE = (8, 8)             # Tamaño de grilla para ecualización adaptativa

# =============================================================================
# ELEVACIÓN SAGITAL — Detección de inclinación frontal
# (encorvarse / cabeza adelantada cuando se mira de frente a la cámara)
# =============================================================================
# Con una sola cámara frontal, encorvarse es un movimiento en profundidad (Z)
# que no se ve en X-Y. Estos parámetros controlan cómo se estima esa
# profundidad a partir del "escorzo" (acortamiento aparente) del torso y cuello.
ANCHO_HOMBROS_MIN_PX = 45        # Ancho mínimo de hombros (px) para tratar la vista como frontal/ángulo
ESCORZO_DEADZONE = 0.90          # Zona muerta: solo se cuenta escorzo si el segmento se acorta > 10%
                                 # (evita falsos positivos por variación corporal entre personas)
ESCORZO_DZ_MAX_FRAC = 0.85       # Límite de profundidad estimada (fracción de la longitud real) para estabilidad

# --- Calibración relativa (línea base personal) ---
# En lugar de comparar contra proporciones antropométricas absolutas (que
# varían entre personas y dependen de acertar ALTURA_SUJETO_CM y el FOV), el
# escorzo del cuello/torso se mide contra la MEJOR postura observada de la
# propia persona en la sesión (ratio segmento/ancho_hombros, invariante a la
# distancia a la cámara). Siéntate erguido al iniciar, o presiona 'C' para
# recalibrar la línea base en cualquier momento.
CALIB_ALPHA_SUBIDA = 0.05        # Velocidad de adopción de una postura MÁS erguida (~20 frames)
CALIB_ALPHA_BAJADA = 0.0005      # Deriva lentísima hacia abajo (tolera cambios de persona/encuadre)

# =============================================================================
# ENCORVAMIENTO EN VISTA LATERAL — Detección por silueta (bordes Canny)
# =============================================================================
# De costado, cuando la cadera queda oculta tras el escritorio, el detector la
# fabrica justo debajo del hombro, por lo que el tronco siempre parece vertical
# y el encorvamiento NO se detecta. La única señal que queda es el contorno de
# la espalda alta: al encorvarse, esa silueta se inclina hacia adelante.
# Aquí se extrae ese contorno con detección de bordes (Canny) dentro de un ROI
# acotado por los keypoints (hombro/oreja) y se mide su inclinación respecto a
# la vertical, calibrada contra la postura más erguida de la propia persona.
ENCORVADO_ACTIVO = True          # Activar la detección de encorvamiento por silueta
ENCORVADO_CANNY_LOW = 40         # Umbral bajo de Canny
ENCORVADO_CANNY_HIGH = 120       # Umbral alto de Canny
ENCORVADO_BLUR_KSIZE = 5         # Desenfoque gaussiano previo (reduce ruido de bordes)
ENCORVADO_ROI_ESCALA_X = 1.4     # Ancho del ROI = escala_torso × esto (a cada lado del hombro)
ENCORVADO_ROI_ARRIBA = 0.5       # ROI hacia arriba desde el hombro (× escala_torso)
ENCORVADO_ROI_ABAJO = 1.6        # ROI hacia abajo desde el hombro (× escala_torso)
ENCORVADO_OFFSET_ESPALDA = 0.6   # Posición esperada del contorno de la espalda (× escala) desde el hombro
ENCORVADO_MIN_FILAS = 12         # Mínimo de filas con contorno válido para confiar en la medición
ENCORVADO_RESIDUO_MAX = 14.0     # Desviación máxima (px) del contorno respecto a la recta ajustada
ENCORVADO_MARGEN_GRADOS = 4.0    # Zona muerta: solo cuenta inclinación > margen sobre la línea base
ENCORVADO_GANANCIA = 1.8         # Factor contorno→flexión de tronco. La inclinación del contorno de
                                 # la espalda ALTA subestima la flexión de tronco completa; esta
                                 # ganancia la escala para alimentar los umbrales de _score_tronco.
ENCORVADO_SUAVIZADO = 0.35       # Suavizado temporal (EMA) del ángulo medido (0-1, mayor = más reactivo)
ENCORVADO_CALIB_ALPHA_SUBIDA = 0.05    # Adopción de postura más erguida (baja el ángulo base)
ENCORVADO_CALIB_ALPHA_BAJADA = 0.0005  # Deriva lentísima al encorvarse (no erosiona la línea base)

# =============================================================================
# PARÁMETROS DE ALARMA
# =============================================================================
ALARMA_FRECUENCIA_HZ = 1000     # Frecuencia del tono (Hz)
ALARMA_DURACION_MS = 300         # Duración del tono (ms)
UMBRAL_FRAMES_ALARMA = 45       # Frames consecutivos en riesgo antes de activar alarma
UMBRAL_FRAMES_LIBERA = 12       # Frames consecutivos en buena postura para liberar la alarma (histéresis)
UMBRAL_RULA_ALARMA = 5          # Score RULA mínimo para considerar riesgo

# =============================================================================
# PARÁMETROS DE SUAVIZADO TEMPORAL
# =============================================================================
SUAVIZADO_ALPHA = 0.4            # Factor de suavizado exponencial (0-1, mayor = más reactivo)
                                 # Solo se usa si SUAVIZADO_ONEEURO = False.

# --- Filtro One-Euro (anti-jitter) -------------------------------------------
# Suavizado adaptativo: quita el temblor de los landmarks cuando estás quieto y
# NO añade retraso al moverte (a diferencia de la EMA de alpha fijo de arriba).
# Ver filtro_oneeuro.py. Sube beta si notas "lag" al moverte; baja min_cutoff
# (y/o beta) si aún tiemblan los puntos estando quieto.
SUAVIZADO_ONEEURO = True         # True = One-Euro (recomendado) / False = EMA clásica

# Pose 3D (world landmarks en cm) → ángulos RULA y HUD.
# Defaults calibrados: ~65% menos temblor en reposo con ~0.5 cm de retraso al
# moverse (imperceptible). Sube beta para menos lag; baja min_cutoff/beta para
# aún menos temblor.
ONEEURO_3D_MIN_CUTOFF = 0.8      # Hz. Menor = más estable en reposo
ONEEURO_3D_BETA = 0.05           # Mayor = reacciona antes al movimiento
ONEEURO_3D_D_CUTOFF = 1.0        # Hz. Cutoff de la derivada (dejar en 1.0)

# Pose 2D (píxeles) → esqueleto dibujado en pantalla (escala distinta al 3D)
ONEEURO_2D_MIN_CUTOFF = 1.0      # Hz
ONEEURO_2D_BETA = 0.03           # px: velocidades típicas de decenas–cientos px/s
ONEEURO_2D_D_CUTOFF = 1.0        # Hz

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
