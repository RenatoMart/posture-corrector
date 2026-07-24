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
import numpy as np
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
    ML_ACTIVO, ML_UMBRAL_ON, ML_UMBRAL_OFF, ML_ALPHA,
    ML_DISPARA_ALARMA, ML_FRAMES_ALARMA,
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

WINDOW = 'Analisis de Postura Ergonomica - RULA'   # título de la ventana


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


PANEL_ALTURA = 118   # alto del panel de control debajo del vídeo


def _dibujar_aviso_ml(frame, ml_postura, frames_ml):
    """
    Dibuja sobre el VÍDEO el aviso del detector ML de encorvamiento.

    El panel de abajo da el detalle numérico, pero mirando la cámara uno no lee
    texto pequeño: esto es la señal grande y de color que sí se percibe de reojo.
    Solo aparece cuando el modelo marca encorvado, para no ensuciar la imagen.

    Args:
        frame: imagen del vídeo (se modifica in-place).
        ml_postura: dict del clasificador o None.
        frames_ml: frames seguidos que el modelo lleva marcando encorvado.
    """
    if ml_postura is None or not ml_postura['encorvado']:
        return

    h, w = frame.shape[:2]
    # Rojo cuando el encorvamiento ya se sostuvo lo suficiente para ser riesgo;
    # naranja mientras solo es un aviso temprano.
    confirmado = ML_DISPARA_ALARMA and frames_ml >= ML_FRAMES_ALARMA
    color = (0, 0, 255) if confirmado else (0, 140, 255)
    texto = "ENCORVADO" if confirmado else "encorvandote..."

    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x0, y0 = (w - tw) // 2 - 14, 8
    x1, y1 = x0 + tw + 28, y0 + th + 18

    # Fondo semitransparente para que el texto se lea sobre cualquier escena.
    capa = frame.copy()
    cv2.rectangle(capa, (x0, y0), (x1, y1), (25, 25, 25), -1)
    cv2.addWeighted(capa, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(frame, texto, (x0 + 14, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)

    # Marco rojo alrededor del vídeo cuando ya es riesgo confirmado.
    if confirmado:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)


def _dibujar_panel_control(lienzo, y0, vista_manual, ml_postura, indice_camara,
                           frames_ml=0):
    """
    Dibuja el panel de control DEBAJO del vídeo (no encima), estilo recolectar.

    Así los botones, el estado ML y la leyenda de teclas siempre se ven completos
    y no tapan la imagen de la cámara.

    Args:
        lienzo: imagen compuesta (vídeo + panel) sobre la que dibujar (in-place).
        y0: coordenada Y donde empieza el panel (= alto del vídeo).
        vista_manual: vista fijada a mano o None (automático). Resalta el activo.
        ml_postura: dict del clasificador ML o None.
        indice_camara: índice de la cámara activa.
        frames_ml: frames seguidos encorvado según el ML (para la barra de progreso
            hacia la alarma).

    Returns:
        Lista de rectángulos [(x1, y1, x2, y2, vista), ...] EN COORDENADAS DEL
        LIENZO, para que el callback de ratón detecte los clicks en los botones.
    """
    w = lienzo.shape[1]
    cv2.line(lienzo, (0, y0), (w, y0), (70, 70, 70), 1)

    # --- Fila de botones de VISTA (clicables) ---
    cv2.putText(lienzo, "VISTA:", (12, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (220, 220, 220), 1, cv2.LINE_AA)
    opciones = [('FRENTE', 'frontal'), ('COSTADO', 'lateral'),
                ('ANGULO', 'angulo'), ('AUTO', None)]
    bx, by1, bw, bh, gap = 92, y0 + 12, 118, 30, 8
    rects = []
    for etiqueta, vista in opciones:
        bx2, by2 = bx + bw, by1 + bh
        activo = (vista_manual == vista)
        if activo:
            cv2.rectangle(lienzo, (bx, by1), (bx2, by2), (0, 170, 0), -1)
            cv2.rectangle(lienzo, (bx, by1), (bx2, by2), (0, 255, 0), 2)
            color_txt = (0, 0, 0)
        else:
            cv2.rectangle(lienzo, (bx, by1), (bx2, by2), (55, 55, 55), -1)
            cv2.rectangle(lienzo, (bx, by1), (bx2, by2), (130, 130, 130), 1)
            color_txt = (235, 235, 235)
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(lienzo, etiqueta, (bx + (bw - tw) // 2, by1 + (bh + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_txt, 1, cv2.LINE_AA)
        rects.append((bx, by1, bx2, by2, vista))
        bx = bx2 + gap

    # --- Estado del detector ML + cámara ---
    y_estado = y0 + 66
    if ml_postura is not None:
        if ml_postura['encorvado']:
            txt_ml, col_ml = f"ML: ENCORVADO ({ml_postura['prob']*100:.0f}%)", (0, 80, 255)
        else:
            txt_ml, col_ml = f"ML: erguido ({(1 - ml_postura['prob'])*100:.0f}%)", (0, 220, 0)
    else:
        txt_ml, col_ml = "ML: (sin modelo)", (150, 150, 150)
    cv2.putText(lienzo, txt_ml, (12, y_estado), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, col_ml, 2, cv2.LINE_AA)
    cv2.putText(lienzo, f"CAM {indice_camara}", (w - 95, y_estado),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Barra de "cuánto llevas encorvado" hacia la alarma: hace visible que el ML
    # exige sostener la postura y no dispara por un mal frame suelto.
    if ML_DISPARA_ALARMA and frames_ml > 0:
        frac = min(1.0, frames_ml / float(max(1, ML_FRAMES_ALARMA)))
        bx1, bar_y, bx2, bar_h = 300, y_estado - 14, w - 110, 16
        cv2.rectangle(lienzo, (bx1, bar_y), (bx2, bar_y + bar_h), (60, 60, 60), -1)
        ancho = int((bx2 - bx1) * frac)
        col_bar = (0, 0, 255) if frac >= 1.0 else (0, 140, 255)
        if ancho > 0:
            cv2.rectangle(lienzo, (bx1, bar_y), (bx1 + ancho, bar_y + bar_h),
                          col_bar, -1)
        cv2.rectangle(lienzo, (bx1, bar_y), (bx2, bar_y + bar_h), (140, 140, 140), 1)

    # --- Leyenda de teclas ---
    cv2.putText(lienzo,
                "Q=salir   C=recalibrar   N=cambiar camara   1/2/3/0=vista (o click)",
                (12, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1,
                cv2.LINE_AA)
    return rects


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
    clasificador = ClasificadorPostura(
        umbral_on=ML_UMBRAL_ON,
        umbral_off=ML_UMBRAL_OFF,
        alpha=ML_ALPHA,
        activo=ML_ACTIVO,
    )

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
    print("    Q        → Salir")
    print("    C        → Recalibrar postura erguida (presiona sentado derecho)")
    print("    N        → Cambiar a la siguiente cámara (integrada / DroidCam / etc.)")
    print("    1/2/3/0  → Vista: Frente / Costado / Angulo / Automatico")
    print("               (o haz click en los botones de la derecha)")
    print()
    print("-" * 60)

    # ==========================================================================
    # VENTANA + BOTONES DE VISTA (clicables con el ratón)
    # ==========================================================================
    # AUTOSIZE: la ventana toma el tamaño exacto del lienzo (vídeo + panel) y NO
    # se puede redimensionar, de modo que las coordenadas del ratón coinciden 1:1
    # con la imagen y los clicks en los botones caen siempre donde deben.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    # El callback lee estos rects (se rellenan cada frame). Se muta la MISMA lista
    # in-place para que el closure vea siempre la última posición de los botones.
    botones_rects = []

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (bx1, by1, bx2, by2, vista) in botones_rects:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                evaluador.set_vista_manual(vista)
                nombre = 'AUTOMATICA' if vista is None else vista.upper()
                print(f"[Vista] {nombre}")
                break

    cv2.setMouseCallback(WINDOW, _on_mouse)

    # ==========================================================================
    # VARIABLES DE CONTROL
    # ==========================================================================
    frames_riesgo = 0          # Frames consecutivos en riesgo (RULA y/o ML)
    frames_ok = 0              # Frames consecutivos con buena postura (para histéresis)
    frames_ml_encorvado = 0    # Frames seguidos que el modelo ML marca encorvado
    alarma_activa = False      # Evitar alarmas repetitivas
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
        # Dos fuentes de riesgo INDEPENDIENTES, cada una con su propio contador:
        #   - RULA: ángulos contra las tablas ergonómicas estándar.
        #   - ML:   el Random Forest entrenado con tus propias sesiones, que
        #           reconoce tu encorvamiento aunque el score RULA no llegue al
        #           umbral (típico al estar sentado, donde la cadera se ve mal).
        # El ML necesita sostenerse MÁS frames (ML_FRAMES_ALARMA) porque tiene
        # más falsos positivos que RULA; así un aviso suelto no hace sonar nada.
        riesgo_rula = (resultado is not None
                       and resultado['score_final'] >= UMBRAL_RULA_ALARMA)

        if ml_postura is not None and ml_postura['encorvado']:
            frames_ml_encorvado += 1
        else:
            frames_ml_encorvado = 0
        riesgo_ml = (ML_DISPARA_ALARMA
                     and frames_ml_encorvado >= ML_FRAMES_ALARMA)

        en_riesgo = riesgo_rula or riesgo_ml
        if en_riesgo:
            frames_riesgo += 1
            frames_ok = 0
            if frames_riesgo >= UMBRAL_FRAMES_ALARMA and not alarma_activa:
                emitir_alarma_async()
                alarma_activa = True
                if riesgo_rula:
                    print(f"[!] ALARMA: Score RULA {resultado['score_final']} "
                          f"- {resultado['texto']}")
                else:
                    print(f"[!] ALARMA: encorvamiento sostenido detectado por el "
                          f"modelo ({ml_postura['prob'] * 100:.0f}% de certeza).")
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

        # Aviso grande del detector ML sobre el propio vídeo (solo si encorvado).
        _dibujar_aviso_ml(frame, ml_postura, frames_ml_encorvado)

        # Panel de control DEBAJO del vídeo (no encima), para que nada tape la
        # imagen y todo se vea completo: botones de vista, estado ML y teclas.
        h_f, w_f = frame.shape[:2]
        panel = np.zeros((PANEL_ALTURA, w_f, 3), dtype=np.uint8)
        lienzo = cv2.vconcat([frame, panel])
        botones_rects[:] = _dibujar_panel_control(
            lienzo, h_f, evaluador.vista_manual, ml_postura, indice_camara,
            frames_ml_encorvado
        )

        # Mostrar ventana
        cv2.imshow(WINDOW, lienzo)

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
            frames_ml_encorvado = 0
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
                frames_ml_encorvado = 0
                if estabilizador_2d is not None:
                    estabilizador_2d.reset()
                print(f"[Camara] Cambiada a indice {indice_camara}.")
            else:
                print("[Camara] No hay otra camara disponible; "
                      f"se mantiene la actual (indice {indice_camara}).")
        elif key == ord('1'):
            evaluador.set_vista_manual('frontal')
            print("[Vista] FRENTE (manual)")
        elif key == ord('2'):
            evaluador.set_vista_manual('lateral')
            print("[Vista] COSTADO (manual)")
        elif key == ord('3'):
            evaluador.set_vista_manual('angulo')
            print("[Vista] ANGULO (manual)")
        elif key == ord('0'):
            evaluador.set_vista_manual(None)
            print("[Vista] AUTOMATICA")

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