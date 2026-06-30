"""
Postura.py — Sistema de Análisis de Postura Ergonómica en Tiempo Real.

Arquitectura Híbrida de 3 Fases:
    Fase 1: Extracción 2D ultrarrápida (YOLOv8-Pose)
    Fase 2: Elevación geométrica a 3D (proporciones antropométricas)
    Fase 3: Evaluación ergonómica RULA (Rapid Upper Limb Assessment)

Uso:
    python Postura.py
    
    - Se activa la cámara de la laptop
    - Presiona 'Q' para salir
    - El panel HUD muestra el score RULA en tiempo real
    - Alarma sonora si se detecta riesgo sostenido
"""

import cv2
import winsound
import threading
from detector_2d import Detector2D
from elevador_3d import Elevador3D
from evaluador_rula import EvaluadorRULA
from visualizador import Visualizador
from config import (
    CAMERA_INDEX, ALTURA_SUJETO_CM,
    ALARMA_FRECUENCIA_HZ, ALARMA_DURACION_MS,
    UMBRAL_FRAMES_ALARMA, UMBRAL_RULA_ALARMA,
)


def emitir_alarma_async():
    """Emite alarma sonora en un hilo separado (no bloquea el video)."""
    def _beep():
        winsound.Beep(ALARMA_FRECUENCIA_HZ, ALARMA_DURACION_MS)
    threading.Thread(target=_beep, daemon=True).start()


def main():
    # ==========================================================================
    # INICIALIZACIÓN DE MÓDULOS
    # ==========================================================================
    print("=" * 60)
    print("  SISTEMA DE ANÁLISIS DE POSTURA ERGONÓMICA")
    print("  Arquitectura Híbrida: YOLO-Pose → 3D Lifting → RULA")
    print("=" * 60)
    print()

    print("[1/4] Inicializando detector 2D (YOLOv8-Pose)...")
    detector = Detector2D()

    print("[2/4] Inicializando elevador 3D (geometría antropométrica)...")
    elevador = Elevador3D(altura_sujeto_cm=ALTURA_SUJETO_CM)

    print("[3/4] Inicializando evaluador RULA...")
    evaluador = EvaluadorRULA()

    print("[4/4] Inicializando visualizador...")
    visualizador = Visualizador()

    # ==========================================================================
    # ACTIVAR CÁMARA
    # ==========================================================================
    print()
    print(f"[Cámara] Abriendo cámara (índice {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

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
    print()
    print("-" * 60)

    # ==========================================================================
    # VARIABLES DE CONTROL
    # ==========================================================================
    frames_riesgo = 0        # Frames consecutivos con RULA alto
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
        # FASE 1: Extracción 2D (YOLOv8-Pose)
        # ==================================================================
        keypoints_2d = detector.detectar(frame)

        # ==================================================================
        # FASE 2: Elevación a 3D
        # ==================================================================
        keypoints_3d = None
        if keypoints_2d is not None:
            keypoints_3d = elevador.elevar(keypoints_2d, frame.shape)

        # ==================================================================
        # FASE 3: Evaluación RULA
        # ==================================================================
        resultado = None
        if keypoints_3d is not None:
            resultado = evaluador.evaluar(keypoints_3d)

        # ==================================================================
        # CONTROL DE ALARMA
        # ==================================================================
        if resultado is not None and resultado['score_final'] >= UMBRAL_RULA_ALARMA:
            frames_riesgo += 1
            if frames_riesgo >= UMBRAL_FRAMES_ALARMA and not alarma_activa:
                emitir_alarma_async()
                alarma_activa = True
                print(f"[!] ALARMA: Score RULA {resultado['score_final']} "
                      f"- {resultado['texto']}")
        else:
            frames_riesgo = 0
            alarma_activa = False

        # ==================================================================
        # VISUALIZACIÓN
        # ==================================================================
        visualizador.dibujar(frame, keypoints_2d, resultado)

        # Mostrar ventana
        cv2.imshow('Analisis de Postura Ergonomica - RULA', frame)

        # Tecla de salida
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # ==========================================================================
    # LIMPIEZA
    # ==========================================================================
    cap.release()
    cv2.destroyAllWindows()
    print()
    print("[OK] Sistema cerrado correctamente.")


if __name__ == '__main__':
    main()