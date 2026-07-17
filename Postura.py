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
    print(f"[Cámara] Abriendo cámara (índice {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara.")
        print("  → Verifica que no esté siendo usada por otra aplicación.")
        return

    # Configurar resolución (opcional, mejora FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Cámara activada correctamente.")
    print()
    print("  Controles:")
    print("    Q  → Salir")
    print("    C  → Recalibrar postura erguida (presiona sentado derecho)")
    print()
    print("-" * 60)

    # ==========================================================================
    # VARIABLES DE CONTROL
    # ==========================================================================
    frames_riesgo = 0        # Frames consecutivos con RULA alto
    frames_ok = 0            # Frames consecutivos con buena postura (para histéresis)
    alarma_activa = False    # Evitar alarmas repetitivas

    # ==========================================================================
    # BUCLE PRINCIPAL
    # ==========================================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame de la cámara.")
            break

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
            if estabilizador_2d is not None:
                estabilizador_2d.reset()
            print("[Calibracion] Linea base de postura erguida reiniciada.")

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