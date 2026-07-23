"""
recolectar_datos.py — Captura etiquetada de datos para entrenar el detector ML.

Este es el "botón de entrenamiento" del que hablabas. Abre la webcam, corre la
misma pose de MediaPipe que el sistema real, y te deja ETIQUETAR tu postura en
vivo con el teclado. Cada frame etiquetado se guarda como una fila de números
(las features de `features_postura.py`) en un CSV. No guarda imágenes.

Cómo se usa
-----------
    python recolectar_datos.py

Controles:
    G  → grabar postura ERGUIDA  (label 0). Se queda grabando hasta que pauses.
    B  → grabar postura ENCORVADA (label 1). Idem.
    1  → marcar vista de FRENTE (frontal)   \
    2  → marcar vista de COSTADO (lateral)   > se guardan en la columna 'vista'
    3  → marcar vista en ANGULO             /
    0  → volver a vista AUTOMÁTICA (la detecta el evaluador)
    ESPACIO → PAUSA (deja de grabar; la cámara sigue viéndose).
    Z  → deshacer: borra la última fila guardada.
    C  → recalibrar el suavizado de la pose.
    Q  → salir.

La "vista" no es una feature de entrenamiento: sirve para asegurarte de juntar
datos balanceados de frente y de costado, y para analizar después. Marcarla a
mano (1/2/3) es más fiable que dejársela al detector automático.

Flujo típico:
    1. Siéntate DERECHO, pulsa G, quédate erguido moviéndote un poco (gira, acércate,
       aléjate) ~15-20 s para juntar variedad. Pulsa ESPACIO.
    2. Siéntate ENCORVADO, pulsa B, adopta distintos grados de encorvamiento
       ~15-20 s. Pulsa ESPACIO.
    3. Repite en otra distancia / de costado / con otra ropa / otra persona.
    Meta orientativa: 300-600 filas por clase, repartidas en varias "sesiones".

El archivo `datos_postura.csv` se va llenando (se AÑADE, no se sobreescribe), así
que puedes cerrar y volver otro día: los datos se acumulan. Cada corrida marca
sus filas con un identificador de SESIÓN (fecha-hora), que luego el entrenador
usa para validar sin trampas.
"""

import os
import csv
import datetime

import cv2
import numpy as np

from detector_2d import Detector2D
from elevador_3d import Elevador3D
from features_postura import FEATURE_NAMES, vector_features, features_validos
from config import CAMERA_INDEX, ALTURA_SUJETO_CM

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos_postura.csv')

WINDOW = 'Recoleccion de datos de postura'

# --- Cámara robusta (mismo criterio que Postura.py) -------------------------
# DirectShow es más fiable para la webcam integrada en Windows y evita quedar
# "pegado" a una cámara virtual (DroidCam).
_CAM_BACKEND = cv2.CAP_DSHOW
_MAX_CAMARAS = 6


def _abrir_camara(indice):
    """Abre una cámara por índice (DirectShow), configurada a 640x480.
    Devuelve el VideoCapture o None si no abre / no entrega imagen."""
    cap = cv2.VideoCapture(indice, _CAM_BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _listar_camaras():
    """Índices de cámara disponibles (0.._MAX_CAMARAS-1)."""
    disponibles = []
    for idx in range(_MAX_CAMARAS):
        cap = cv2.VideoCapture(idx, _CAM_BACKEND)
        if cap.isOpened():
            disponibles.append(idx)
        cap.release()
    return disponibles

# Columnas del CSV = features + metadatos. 'vista' y 'sesion' NO son features de
# entrenamiento; sirven para analizar y para el split por sesión.
COLUMNAS = FEATURE_NAMES + ['vista', 'sesion', 'label']

# Etiquetas
LABEL_ERGUIDO = 0
LABEL_ENCORVADO = 1
NOMBRE_LABEL = {LABEL_ERGUIDO: 'ERGUIDO', LABEL_ENCORVADO: 'ENCORVADO'}
COLOR_LABEL = {LABEL_ERGUIDO: (0, 220, 0), LABEL_ENCORVADO: (0, 80, 255)}


def _clasificar_vista(evaluador, kp2d):
    """Etiqueta la vista (frontal/lateral/angulo) reutilizando el evaluador."""
    try:
        vista, _ = evaluador._detectar_vista(kp2d)
        return vista
    except Exception:
        return 'na'


def main():
    print("=" * 60)
    print("  RECOLECTOR DE DATOS DE POSTURA (etiquetado en vivo)")
    print("=" * 60)

    detector = Detector2D()
    elevador = Elevador3D(altura_sujeto_cm=ALTURA_SUJETO_CM)

    # Evaluador solo para etiquetar la vista (opcional). Si falla, se ignora.
    try:
        from evaluador_rula import EvaluadorRULA
        evaluador = EvaluadorRULA()
    except Exception:
        evaluador = None

    # Sesión = marca de tiempo de esta corrida (para el split por grupos)
    sesion = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Abrir CSV en modo "añadir"; escribir cabecera solo si el archivo es nuevo
    archivo_nuevo = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    fcsv = open(CSV_PATH, 'a', newline='', encoding='utf-8')
    writer = csv.writer(fcsv)
    if archivo_nuevo:
        writer.writerow(COLUMNAS)
        fcsv.flush()

    # Contar cuántas filas ya había por clase (de sesiones previas)
    conteo = {LABEL_ERGUIDO: 0, LABEL_ENCORVADO: 0}
    conteo_sesion = {LABEL_ERGUIDO: 0, LABEL_ENCORVADO: 0}
    if not archivo_nuevo:
        try:
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                for fila in csv.DictReader(f):
                    lbl = int(fila['label'])
                    conteo[lbl] = conteo.get(lbl, 0) + 1
        except Exception:
            pass

    camaras = _listar_camaras()
    print(f"[Cámara] Cámaras detectadas (índices): {camaras if camaras else 'ninguna'}")
    indice_camara = CAMERA_INDEX
    cap = _abrir_camara(indice_camara)
    if cap is None and camaras:
        indice_camara = camaras[0]
        print(f"[Cámara] El índice {CAMERA_INDEX} no respondió; "
              f"usando índice {indice_camara}.")
        cap = _abrir_camara(indice_camara)
    if cap is None:
        print("[ERROR] No se pudo abrir ninguna cámara. "
              "Cierra DroidCam/Zoom/Teams y reintenta.")
        fcsv.close()
        return
    print(f"[Cámara] Usando índice {indice_camara}. (N = cambiar de cámara)")

    # Ventana redimensionable con un tamaño inicial cómodo (video 640x480 + panel
    # de menú de 122 px). Puedes agrandarla arrastrando las esquinas.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 760, 715)

    print(f"[OK] Grabando en: {CSV_PATH}")
    print(f"     Sesión: {sesion}")
    print("     G=erguido  B=encorvado  ESPACIO=pausa  Z=deshacer  Q=salir")
    print("-" * 60)

    modo = None          # None = pausa; 0 = grabando erguido; 1 = grabando encorvado
    ultima_fila_pos = None  # posición en el archivo de la última fila (para deshacer)
    # Vista marcada A MANO con las teclas 1/2/3. None = automática (la detecta el
    # evaluador). Marcarla tú es más fiable y asegura balance frontal/costado.
    vista_manual = None
    NOMBRE_VISTA_MANUAL = {'frontal': 'FRENTE', 'lateral': 'COSTADO', 'angulo': 'ANGULO'}

    fallos_lectura = 0
    MAX_FALLOS = 60      # ~2 s a 30 FPS antes de rendirse (no morir por un hipo)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Un frame fallido NO cierra el recolector; solo se rinde tras muchos.
            fallos_lectura += 1
            if fallos_lectura >= MAX_FALLOS:
                print("[ERROR] La cámara dejó de entregar imagen.")
                break
            cv2.waitKey(30)
            continue
        fallos_lectura = 0
        frame = cv2.flip(frame, 1)

        kp2d, kp_world = detector.detectar(frame)
        kp3d = elevador.elevar(kp_world) if kp_world is not None else None
        valido = features_validos(kp3d)

        # --- Grabar si estamos en modo activo y la pose es válida ---
        if modo is not None and valido:
            vec = vector_features(kp3d)
            if vec is not None:
                # Vista: la marcada a mano tiene prioridad; si no, la automática.
                if vista_manual is not None:
                    vista = vista_manual
                else:
                    vista = _clasificar_vista(evaluador, kp2d) if evaluador else 'na'
                fila = list(vec) + [vista, sesion, modo]
                ultima_fila_pos = fcsv.tell()
                writer.writerow(fila)
                fcsv.flush()
                conteo[modo] += 1
                conteo_sesion[modo] += 1

        # --- HUD: barra de estado compacta SOBRE el video ---
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
        if modo is None:
            estado, color = "PAUSA", (200, 200, 200)
        else:
            estado, color = f"GRABANDO {NOMBRE_LABEL[modo]}", COLOR_LABEL[modo]
            cv2.circle(frame, (w - 24, 22), 9, (0, 0, 255), -1)   # punto REC
        cv2.putText(frame, estado, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        pose_txt = "pose OK" if valido else "SIN POSE"
        pose_col = (0, 220, 0) if valido else (0, 0, 255)
        cv2.putText(frame, pose_txt, (12, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, pose_col, 2)

        # Indicador de VISTA (manual o automática)
        if vista_manual is not None:
            vista_txt = f"VISTA: {NOMBRE_VISTA_MANUAL[vista_manual]}"
            vista_col = (0, 220, 255)          # amarillo = fijada a mano
        else:
            auto = _clasificar_vista(evaluador, kp2d) if (evaluador and valido) else 'na'
            vista_txt = f"VISTA: auto ({auto})"
            vista_col = (200, 200, 200)
        cv2.putText(frame, vista_txt, (150, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, vista_col, 2)

        # --- Panel de MENÚ debajo del video (así siempre se ve completo) ---
        panel = np.zeros((122, w, 3), dtype=np.uint8)
        totales = (f"Total  erguido: {conteo[LABEL_ERGUIDO]}    "
                   f"encorvado: {conteo[LABEL_ENCORVADO]}    "
                   f"(esta sesion  +{conteo_sesion[LABEL_ERGUIDO]} / "
                   f"+{conteo_sesion[LABEL_ENCORVADO]})")
        cv2.putText(panel, totales, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.line(panel, (12, 36), (w - 12, 36), (70, 70, 70), 1)
        cv2.putText(panel, "G = erguido     B = encorvado     ESPACIO = pausa",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1)
        cv2.putText(panel, "1 = frente    2 = costado    3 = angulo    0 = auto",
                    (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        cv2.putText(panel, "Z = deshacer     C = recalibrar     Q = salir",
                    (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)

        lienzo = cv2.vconcat([frame, panel])
        cv2.imshow(WINDOW, lienzo)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('g'), ord('G')):
            modo = LABEL_ERGUIDO
        elif key in (ord('b'), ord('B')):
            modo = LABEL_ENCORVADO
        elif key == ord(' '):
            modo = None
        elif key == ord('1'):
            vista_manual = 'frontal'
            print("[Vista] Fijada a mano: FRENTE")
        elif key == ord('2'):
            vista_manual = 'lateral'
            print("[Vista] Fijada a mano: COSTADO")
        elif key == ord('3'):
            vista_manual = 'angulo'
            print("[Vista] Fijada a mano: ANGULO")
        elif key == ord('0'):
            vista_manual = None
            print("[Vista] Automática (la detecta el evaluador)")
        elif key in (ord('c'), ord('C')):
            elevador.recalibrar()
        elif key in (ord('n'), ord('N')):
            # Cambiar de cámara SIN tocar la activa (probar solo otros índices).
            # Si ninguna otra funciona, se conserva la actual y sigue corriendo.
            nuevo_cap = None
            nuevo_indice = indice_camara
            for salto in range(1, _MAX_CAMARAS):
                candidato = (indice_camara + salto) % _MAX_CAMARAS
                if candidato == indice_camara:
                    continue
                nuevo_cap = _abrir_camara(candidato)
                if nuevo_cap is not None:
                    nuevo_indice = candidato
                    break
            if nuevo_cap is not None:
                cap.release()
                cap = nuevo_cap
                indice_camara = nuevo_indice
                fallos_lectura = 0
                modo = None                 # pausa la grabación al cambiar fuente
                elevador.recalibrar()
                print(f"[Camara] Cambiada a indice {indice_camara} (grabacion en PAUSA).")
            else:
                print("[Camara] No hay otra camara disponible; "
                      f"se mantiene la actual (indice {indice_camara}).")
        elif key in (ord('z'), ord('Z')):
            # Deshacer la última fila: truncar el archivo en la posición guardada
            if ultima_fila_pos is not None:
                fcsv.flush()
                fcsv.seek(ultima_fila_pos)
                fcsv.truncate()
                fcsv.seek(0, os.SEEK_END)
                # Descontar del contador (de la clase que estaba grabando)
                if modo is not None and conteo[modo] > 0:
                    conteo[modo] -= 1
                    conteo_sesion[modo] -= 1
                ultima_fila_pos = None
                print("[Deshacer] Última fila eliminada.")

    cap.release()
    cv2.destroyAllWindows()
    detector.cerrar()
    fcsv.close()

    print("-" * 60)
    print(f"[OK] Guardado en {CSV_PATH}")
    print(f"     Total acumulado -> erguido: {conteo[LABEL_ERGUIDO]}  "
          f"encorvado: {conteo[LABEL_ENCORVADO]}")
    if min(conteo.values()) < 200:
        print("     [Sugerencia] Junta al menos ~300 por clase, en varias "
              "sesiones/distancias, antes de entrenar.")


if __name__ == '__main__':
    main()
