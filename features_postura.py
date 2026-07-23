"""
features_postura.py — Extracción de características de postura para el modelo ML.

Este módulo es el CORAZÓN del detector de encorvamiento por Machine Learning y
la pieza más importante de todo el flujo: la MISMA función que se usa aquí debe
usarse al recolectar datos (`recolectar_datos.py`), al entrenar
(`entrenar_modelo.py`) y al predecir en vivo (`clasificador_postura.py`). Si el
cálculo difiere entre esas etapas, el modelo recibe números distintos a los que
aprendió y falla en silencio. Por eso vive en un solo lugar.

Idea
----
No se entrena sobre píxeles. Se entrena sobre ~13 ÁNGULOS Y PROPORCIONES que se
derivan de la pose 3D real de MediaPipe (world landmarks, ya en cm y centrados en
la cadera). El Random Forest aprende "erguido vs encorvado" sobre esos números.

Todas las features se normalizan por la ESCALA DEL TORSO (ancho de hombros) para
que den igual a distinta distancia de la cámara, y se apoyan en la proyección al
plano sagital (se descarta X) para tolerar que la persona esté de frente o de
costado.

Convención de ejes (heredada del proyecto): X = horizontal (derecha +),
Y = vertical con ABAJO positivo, Z = profundidad. Índices COCO habituales más los
virtuales 17 = cuello (medio de hombros) y 18 = centro de cadera.
"""

import numpy as np

# Orden CANÓNICO de las columnas. entrenar_modelo.py y clasificador_postura.py
# dependen de este orden exacto. No reordenar sin reentrenar el modelo.
FEATURE_NAMES = [
    'tronco_flexion_sagital',   # inclinación del tronco adelante/atrás (plano sagital)
    'tronco_flexion_3d',        # inclinación 3D total del tronco (incluye lateral)
    'tronco_lean_lateral',      # inclinación lateral del tronco (plano frontal)
    'cuello_flexion_vs_tronco', # flexión del cuello RELATIVA al eje del tronco
    'cuello_flexion_vertical',  # flexión del cuello respecto a la vertical
    'craneovertebral',          # ángulo craneovertebral aprox. (cabeza adelantada)
    'cabeza_adelantada_z',      # desplazamiento en profundidad cabeza vs hombros
    'hombros_protraccion_z',    # hombros hacia adelante (redondeo de hombros)
    'curvatura_toracica',       # joroba alta: cuánto se aleja el cuello de la recta cadera-cabeza
    'cabeza_caida_y',           # hundimiento vertical de la cabeza respecto a hombros
    'nariz_vs_orejas_y',        # mirar hacia abajo (nariz baja respecto a orejas)
    'altura_torso_rel',         # compresión vertical del torso (se acorta al encorvarse)
    'hombro_asimetria_z',       # torsión: diferencia de profundidad entre hombros
]

N_FEATURES = len(FEATURE_NAMES)

_VERTICAL_ARRIBA = np.array([0.0, -1.0, 0.0])  # Y crece hacia abajo


def _angulo(v1, v2):
    """Ángulo en grados entre dos vectores 3D (0 si alguno es casi nulo)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _sagital(v):
    """Proyección al plano sagital: descarta la componente lateral (X)."""
    return np.array([0.0, v[1], v[2]])


def _frontal(v):
    """Proyección al plano frontal: descarta la profundidad (Z)."""
    return np.array([v[0], v[1], 0.0])


def _cabeza(kp3d):
    """
    Punto de referencia de la cabeza, con la misma prioridad que usa el
    evaluador RULA: orejas > ojos > una oreja > nariz. Devuelve None si no hay.
    """
    if 3 in kp3d and 4 in kp3d:
        return (kp3d[3] + kp3d[4]) / 2      # medio entre orejas
    if 1 in kp3d and 2 in kp3d:
        return (kp3d[1] + kp3d[2]) / 2      # medio entre ojos
    for idx in (3, 4, 0):
        if idx in kp3d:
            return kp3d[idx].astype(float)
    return None


def features_validos(kp3d):
    """
    ¿Hay pose suficiente para calcular features fiables?

    Se exige el cuello y el centro de cadera virtuales (17, 18), al menos un
    hombro real y alguna referencia de cabeza. La herramienta de recolección
    solo guarda una fila cuando esto es True, para no meter ruido al dataset.
    """
    if kp3d is None:
        return False
    if 17 not in kp3d or 18 not in kp3d:
        return False
    if 5 not in kp3d and 6 not in kp3d:
        return False
    return _cabeza(kp3d) is not None


def _escala_torso(kp3d, cuello, cabeza):
    """
    Escala del torso en cm para normalizar: ancho de hombros si están ambos;
    si no, distancia cuello→cabeza como respaldo. Nunca devuelve 0.
    """
    if 5 in kp3d and 6 in kp3d:
        d = np.linalg.norm(kp3d[5] - kp3d[6])
        if d > 1e-3:
            return d
    d = np.linalg.norm(cabeza - cuello)
    return d if d > 1e-3 else 1.0


def extraer_features(kp3d, kp2d=None):
    """
    Calcula el vector de características de postura a partir de la pose 3D.

    Args:
        kp3d: dict {idx: np.array([X, Y, Z])} en cm, con virtuales 17/18
              (la salida de Elevador3D.elevar()).
        kp2d: dict {idx: (x, y, vis)} opcional. Reservado por si en el futuro se
              agregan features 2D; hoy no se usa (se acepta por compatibilidad).

    Returns:
        dict {nombre_feature: valor_float} con las claves de FEATURE_NAMES, o
        None si la pose no es suficiente (features_validos() == False).
    """
    if not features_validos(kp3d):
        return None

    cuello = kp3d[17].astype(float)          # medio de hombros
    cadera = kp3d[18].astype(float)          # centro de cadera
    cabeza = _cabeza(kp3d)
    escala = _escala_torso(kp3d, cuello, cabeza)

    f = {}

    # --- TRONCO -------------------------------------------------------------
    vec_tronco = cuello - cadera             # apunta hacia arriba (Y negativo)
    f['tronco_flexion_sagital'] = _angulo(_sagital(vec_tronco), _VERTICAL_ARRIBA)
    f['tronco_flexion_3d'] = _angulo(vec_tronco, _VERTICAL_ARRIBA)
    f['tronco_lean_lateral'] = _angulo(_frontal(vec_tronco), _VERTICAL_ARRIBA)

    # --- CUELLO / CABEZA ----------------------------------------------------
    vec_cuello = cabeza - cuello             # del cuello a la cabeza
    # Flexión del cuello respecto a la vertical (mirar hacia abajo la sube)
    f['cuello_flexion_vertical'] = _angulo(_sagital(vec_cuello), _VERTICAL_ARRIBA)
    # Flexión del cuello RELATIVA al tronco: aísla el cuello de la inclinación
    # global del tronco (lo correcto biomecánicamente).
    f['cuello_flexion_vs_tronco'] = _angulo(_sagital(vec_cuello), _sagital(vec_tronco))

    # Ángulo craneovertebral aprox.: inclinación de la línea cuello→cabeza
    # respecto a la HORIZONTAL en el plano sagital. Baja cuando la cabeza se
    # adelanta (forward head posture). Referencia clínica de cabeza adelantada.
    horizontal_sag = np.array([0.0, 0.0, 1.0])  # eje de profundidad
    f['craneovertebral'] = _angulo(_sagital(vec_cuello), horizontal_sag)

    # Cabeza adelantada en profundidad: la cabeza se mueve hacia la cámara
    # respecto a los hombros. Normalizado por la escala (signo lo aprende el RF).
    f['cabeza_adelantada_z'] = float((cabeza[2] - cuello[2]) / escala)

    # Hundimiento vertical de la cabeza (encorvarse baja la cabeza).
    f['cabeza_caida_y'] = float((cabeza[1] - cuello[1]) / escala)

    # Mirar hacia abajo: la nariz cae respecto a la línea de las orejas.
    if 0 in kp3d and 3 in kp3d and 4 in kp3d:
        orejas_y = (kp3d[3][1] + kp3d[4][1]) / 2
        f['nariz_vs_orejas_y'] = float((kp3d[0][1] - orejas_y) / escala)
    else:
        f['nariz_vs_orejas_y'] = 0.0

    # --- HOMBROS ------------------------------------------------------------
    # Protracción (hombros hacia adelante): profundidad media de hombros vs cadera.
    f['hombros_protraccion_z'] = float((cuello[2] - cadera[2]) / escala)

    # Asimetría de profundidad entre hombros (torsión del torso).
    if 5 in kp3d and 6 in kp3d:
        f['hombro_asimetria_z'] = float((kp3d[5][2] - kp3d[6][2]) / escala)
    else:
        f['hombro_asimetria_z'] = 0.0

    # --- CURVATURA / COMPRESIÓN --------------------------------------------
    # Joroba alta: distancia perpendicular del cuello a la recta cadera→cabeza.
    # En una espalda recta, cadera-cuello-cabeza están casi alineados; al
    # encorvarse el cuello se separa de esa recta hacia adelante.
    eje = cabeza - cadera
    norma_eje = np.linalg.norm(eje)
    if norma_eje > 1e-6:
        proy = np.dot(cuello - cadera, eje) / norma_eje
        punto_recta = cadera + (proy / norma_eje) * eje
        f['curvatura_toracica'] = float(np.linalg.norm(cuello - punto_recta) / escala)
    else:
        f['curvatura_toracica'] = 0.0

    # Compresión vertical del torso: la altura vertical (Y) del torso se acorta
    # cuando la persona se encorva/hunde. Normalizada por la escala.
    f['altura_torso_rel'] = float(abs(cuello[1] - cadera[1]) / escala)

    return f


def vector_features(kp3d, kp2d=None):
    """
    Igual que extraer_features pero devuelve un np.array en el orden canónico
    de FEATURE_NAMES (lo que consume scikit-learn). None si la pose no sirve.
    """
    f = extraer_features(kp3d, kp2d)
    if f is None:
        return None
    return np.array([f[n] for n in FEATURE_NAMES], dtype=float)
