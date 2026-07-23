"""
Postura.py — Sistema de Análisis de Postura Ergonómica en Tiempo Real.

Arquitectura de 3 Fases:
    Fase 1: Extracción de pose con MediaPipe Pose (33 landmarks, 3D real)
    Fase 2: Ensamblado de la pose 3D (puntos virtuales + suavizado)
    Fase 3: Evaluación ergonómica RULA (Rapid Upper Limb Assessment)

Uso:
    python Postura.py
    
    - Se activa la cámara de la laptop
    - Presiona 'Q' para salir
    - Presiona 'C' sentado erguido para recalibrar tu línea base de postura
    - El panel HUD muestra el score RULA en tiempo real
    - Alarma sonora si se detecta riesgo sostenido
"""

import argparse

import cv2
import winsound
import threading
from detector_2d import Detector2D
from elevador_3d import Elevador3D
from evaluador_rula import EvaluadorRULA
from visualizador import Visualizador
from filtro_oneeuro import EstabilizadorPose
from clasificador_postura import ClasificadorPostura
from config import (
    CAMERA_INDEX, ALTURA_SUJETO_CM, MP_USAR_GPU,
    ALARMA_FRECUENCIA_HZ, ALARMA_DURACION_MS,
    UMBRAL_FRAMES_ALARMA, UMBRAL_FRAMES_LIBERA, UMBRAL_RULA_ALARMA,
    SUAVIZADO_ONEEURO, ONEEURO_2D_MIN_CUTOFF, ONEEURO_2D_BETA, ONEEURO_2D_D_CUTOFF,
)


def emitir_alarma_async():
    """Emite alarma sonora en un hilo separado (no bloquea el video)."""
    def _beep():
        winsound.Beep(ALARMA_FRECUENCIA_HZ, ALARMA_DURACION_MS)
    threading.Thread(target=_beep, daemon=True).start()


# Backend DirectShow: en Windows es mucho más fiable para la webcam integrada
# (evita que OpenCV se quede "pegado" a una cámara virtual como DroidCam).
_CAM_BACKEND = cv2.CAP_DSHOW
_MAX_CAMARAS = 6   # cuántos índices sondear al buscar/cambiar de cámara


def _abrir_camara(indice):
    """Abre una cámara por índice con backend DirectShow.

    Devuelve el VideoCapture ya configurado (640x480) o None si no se pudo
    abrir / no entrega frames.
    """
    cap = cv2.VideoCapture(indice, _CAM_BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Verifica que realmente entregue imagen (algunas cámaras "abren" pero no dan frame).
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _listar_camaras():
    """Devuelve la lista de índices de cámara disponibles (0.._MAX_CAMARAS-1)."""
    disponibles = []
    for idx in range(_MAX_CAMARAS):
        cap = cv2.VideoCapture(idx, _CAM_BACKEND)
        if cap.isOpened():
            disponibles.append(idx)
        cap.release()
    return disponibles


def _parsear_args():
    """Lee las opciones de línea de comandos."""
    p = argparse.ArgumentParser(
        description="Análisis de postura ergonómica (RULA) en tiempo real.")
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument(
        '--gpu', dest='usar_gpu', action='store_true', default=None,
        help="Forzar el uso de GPU (si el build de MediaPipe lo soporta; en "
             "Windows suele caer a CPU automáticamente).")
    grupo.add_argument(
        '--cpu', dest='usar_gpu', action='store_false',
        help="Forzar el uso de CPU.")
    return p.parse_args()


def main():
    args = _parsear_args()
    usar_gpu = MP_USAR_GPU if args.usar_gpu is None else args.usar_gpu
    # ==========================================================================
    # INICIALIZACIÓN DE MÓDULOS
    # ==========================================================================
    print("=" * 60)
    print("  SISTEMA DE ANÁLISIS DE POSTURA ERGONÓMICA")
    print("  Arquitectura: MediaPipe Pose → 3D real → RULA")
    print("=" * 60)
    print()

    print("[1/4] Inicializando detector de pose (MediaPipe Pose)...")
    detector = Detector2D(usar_gpu=usar_gpu)
    print(f"      >>> PROCESANDO CON: {detector.delegado} <<<")

    print("[2/4] Inicializando ensamblador 3D...")
    elevador = Elevador3D(altura_sujeto_cm=ALTURA_SUJETO_CM)

    print("[3/4] Inicializando evaluador RULA...")
    evaluador = EvaluadorRULA()

    print("[4/4] Inicializando visualizador...")
    visualizador = Visualizador()

    # Clasificador ML de encorvamiento (Random Forest entrenado con tus datos).
    # Si no existe modelo_postura.joblib, queda 'no disponible' y el sistema
    # sigue funcionando solo con RULA (no rompe nada).
    clasificador = ClasificadorPostura()

    # Estabilizador anti-jitter de la pose 2D (los puntos que se dibujan en
    # pantalla). Solo suaviza x,y (n_coords=2); la visibilidad se conserva.
    estabilizador_2d = (
        EstabilizadorPose(
            min_cutoff=ONEEURO_2D_MIN_CUTOFF,
            beta=ONEEURO_2D_BETA,
            d_cutoff=ONEEURO_2D_D_CUTOFF,
            n_coords=2,
        ) if SUAVIZADO_ONEEURO else None
    )

    # ==========================================================================
    # ACTIVAR CÁMARA
    # ==========================================================================
    print()
    camaras = _listar_camaras()
    print(f"[Cámara] Cámaras detectadas (índices): {camaras if camaras else 'ninguna'}")

    # Se intenta primero el índice de config; si no da, la primera disponible.
    indice_camara = CAMERA_INDEX
    print(f"[Cámara] Abriendo cámara (índice {indice_camara})...")
    cap = _abrir_camara(indice_camara)
    if cap is None and camaras:
        indice_camara = camaras[0]
        print(f"[Cámara] El índice {CAMERA_INDEX} no respondió; "
              f"usando índice {indice_camara}.")
        cap = _abrir_camara(indice_camara)

    if cap is None:
        print("[ERROR] No se pudo acceder a ninguna cámara.")
        print("  → Verifica que no esté siendo usada por otra aplicación")
        print("    (cierra DroidCam/Zoom/Teams) y vuelve a intentar.")
        return

    print(f"[OK] Cámara activada correctamente (índice {indice_camara}).")
    print()
    print("  Controles:")
    print("    Q  → Salir")
    print("    C  → Recalibrar postura erguida (presiona sentado derecho)")
    print("    N  → Cambiar a la siguiente cámara (integrada / DroidCam / etc.)")
    print()
    print("-" * 60)

    # ==========================================================================
    # VARIABLES DE CONTROL
    # ==========================================================================
    frames_riesgo = 0        # Frames consecutivos con RULA alto
    frames_ok = 0            # Frames consecutivos con buena postura (para histéresis)
    alarma_activa = False    # Evitar alarmas repetitivas
    fallos_lectura = 0       # Frames fallidos seguidos (para no morir por un hipo)
    MAX_FALLOS = 60          # ~2 s a 30 FPS antes de rendirse

    # ==========================================================================
    # BUCLE PRINCIPAL
    # ==========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            # Un frame fallido NO cierra el programa: puede ser un hipo temporal
            # (cambio de cámara, USB, etc.). Solo se rinde tras muchos seguidos.
            fallos_lectura += 1
            if fallos_lectura >= MAX_FALLOS:
                print("[ERROR] La cámara dejó de entregar imagen. "
                      "Pulsa N la próxima vez para cambiar de cámara.")
                break
            cv2.waitKey(30)
            continue
        fallos_lectura = 0

        # Espejo horizontal (más natural para el usuario)
        frame = cv2.flip(frame, 1)

        # ==================================================================
        # FASE 1: Extracción de pose (MediaPipe Pose) — 2D + 3D real
        # ==================================================================
        keypoints_2d, keypoints_world = detector.detectar(frame)

        # Anti-jitter de los puntos 2D antes de usarlos (dibujo, silueta, vista)
        if estabilizador_2d is not None and keypoints_2d is not None:
            keypoints_2d = estabilizador_2d(keypoints_2d)

        # ==================================================================
        # FASE 2: Ensamblado de la pose 3D
        # ==================================================================
        keypoints_3d = None
        if keypoints_world is not None:
            keypoints_3d = elevador.elevar(keypoints_world)

        # ==================================================================
        # FASE 3: Evaluación RULA
        # ==================================================================
        resultado = None
        if keypoints_3d is not None:
            # Se pasa el frame (aún sin dibujos) para la detección de
            # encorvamiento por silueta en vista lateral.
            resultado = evaluador.evaluar(keypoints_3d, keypoints_2d, frame)

        # ==================================================================
        # DETECCIÓN ML DE ENCORVAMIENTO (modelo entrenado con tus datos)
        # ==================================================================
        # Devuelve {'prob', 'prob_cruda', 'encorvado'} o None si no hay modelo
        # o la pose no da features válidas. Ya viene suavizada e con histéresis.
        ml_postura = None
        if keypoints_3d is not None:
            ml_postura = clasificador.predecir(keypoints_3d)

        # ==================================================================
        # CONTROL DE ALARMA
        # ==================================================================
        en_riesgo = (resultado is not None
                     and resultado['score_final'] >= UMBRAL_RULA_ALARMA)
        if en_riesgo:
            frames_riesgo += 1
            frames_ok = 0
            if frames_riesgo >= UMBRAL_FRAMES_ALARMA and not alarma_activa:
                emitir_alarma_async()
                alarma_activa = True
                print(f"[!] ALARMA: Score RULA {resultado['score_final']} "
                      f"- {resultado['texto']}")
        else:
            # Histéresis: solo se libera tras varios frames buenos seguidos,
            # para que un único frame ruidoso no apague ni reinicie la alarma.
            frames_ok += 1
            if frames_ok >= UMBRAL_FRAMES_LIBERA:
                frames_riesgo = 0
                alarma_activa = False

        # ==================================================================
        # VISUALIZACIÓN
        # ==================================================================
        visualizador.dibujar(frame, keypoints_2d, resultado)

        # Banner del detector ML de encorvamiento (abajo-izquierda). Se dibuja
        # aparte del HUD RULA para no tocar el visualizador.
        if ml_postura is not None:
            if ml_postura['encorvado']:
                texto_ml = f"ML: ENCORVADO  ({ml_postura['prob']*100:.0f}%)"
                color_ml = (0, 80, 255)          # naranja-rojo
            else:
                texto_ml = f"ML: erguido  ({(1-ml_postura['prob'])*100:.0f}%)"
                color_ml = (0, 220, 0)           # verde
            h_f = frame.shape[0]
            cv2.putText(frame, texto_ml, (12, h_f - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ml, 2)

        # Indicador de cámara activa (abajo-derecha): recuerda que 'N' la cambia.
        texto_cam = f"CAM {indice_camara}  (N=cambiar)"
        (tw, _), _ = cv2.getTextSize(texto_cam, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, texto_cam, (frame.shape[1] - tw - 12, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostrar ventana
        cv2.imshow('Analisis de Postura Ergonomica - RULA', frame)

        # Teclas de control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            # Recalibrar la línea base de postura erguida (sentado derecho):
            # tanto el escorzo frontal como la silueta lateral del tronco.
            elevador.recalibrar()
            evaluador.recalibrar()
            clasificador.reset()
            if estabilizador_2d is not None:
                estabilizador_2d.reset()
            print("[Calibracion] Linea base de postura erguida reiniciada.")
        elif key == ord('n') or key == ord('N'):
            # Cambiar a la siguiente cámara. IMPORTANTE: no se toca ni se sondea
            # la cámara activa (eso interrumpiría su stream). Se prueban SOLO los
            # otros índices; si ninguno funciona, se conserva la cámara actual.
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
                cap.release()               # se suelta la vieja SOLO ahora
                cap = nuevo_cap
                indice_camara = nuevo_indice
                fallos_lectura = 0
                # Reiniciar suavizados/calibración porque cambió la fuente.
                elevador.recalibrar()
                evaluador.recalibrar()
                clasificador.reset()
                if estabilizador_2d is not None:
                    estabilizador_2d.reset()
                print(f"[Camara] Cambiada a indice {indice_camara}.")
            else:
                print("[Camara] No hay otra camara disponible; "
                      f"se mantiene la actual (indice {indice_camara}).")

    # ==========================================================================
    # LIMPIEZA
    # ==========================================================================
    cap.release()
    cv2.destroyAllWindows()
    detector.cerrar()
    print()
    print("[OK] Sistema cerrado correctamente.")


if __name__ == '__main__':
    main()