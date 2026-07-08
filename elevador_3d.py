"""
elevador_3d.py — Fase 2: Elevación Geométrica 2D → 3D.

Convierte coordenadas 2D (píxeles) a un espacio 3D estimado (centímetros)
usando proporciones antropométricas y un modelo de cámara pinhole.

Principio:
    Z = (focal_length × longitud_real) / longitud_píxeles
    
Cuando un segmento corporal aparece más corto de lo esperado en la imagen,
inferimos que tiene un componente de profundidad (se aleja o acerca a la cámara).
"""

import numpy as np
from config import (
    PROPORCIONES_CORPORALES, CAMERA_FOV_GRADOS, ALTURA_SUJETO_CM,
    SEGMENTOS_CADENA, SUAVIZADO_ALPHA,
    ANCHO_HOMBROS_MIN_PX, ESCORZO_DEADZONE, ESCORZO_DZ_MAX_FRAC,
    CALIB_ALPHA_SUBIDA, CALIB_ALPHA_BAJADA,
)


class Elevador3D:
    """
    Eleva keypoints 2D a coordenadas 3D estimadas usando geometría
    proyectiva y proporciones antropométricas.
    """

    def __init__(self, altura_sujeto_cm=ALTURA_SUJETO_CM):
        """
        Args:
            altura_sujeto_cm: Altura del sujeto en cm (para escalar proporciones).
        """
        self.altura_cm = altura_sujeto_cm

        # Pre-calcular longitudes reales de segmentos en cm
        self.segmentos_reales = {
            nombre: fraccion * altura_sujeto_cm
            for nombre, fraccion in PROPORCIONES_CORPORALES.items()
        }

        # Buffer de suavizado temporal (media móvil exponencial)
        self.historial = {}
        self.alpha = SUAVIZADO_ALPHA

        # Líneas base de calibración relativa: ratio segmento/ancho_hombros de
        # la postura MÁS erguida observada, por tipo de referencia (ver
        # _elevar_sagital). Invariante a la distancia a la cámara.
        self._baselines = {}

    def recalibrar(self):
        """
        Resetea la línea base personal de postura erguida.

        Llamar con la persona sentada erguida (tecla 'C' en la app): la nueva
        referencia se adopta en ~1 segundo y el escorzo del cuello/torso pasa
        a medirse contra esa postura.
        """
        self._baselines = {}

    def _estimar_focal(self, ancho_frame):
        """
        Estima la longitud focal en píxeles basándose en el FOV de la cámara.
        
        focal = (ancho / 2) / tan(FOV / 2)
        """
        fov_rad = np.radians(CAMERA_FOV_GRADOS)
        return (ancho_frame / 2) / np.tan(fov_rad / 2)

    def _distancia_2d(self, p1, p2):
        """Distancia euclidiana entre dos puntos 2D."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _punto_medio(self, p1, p2):
        """Punto medio entre dos puntos (x, y, conf)."""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _es_punto_estimado(self, keypoints_2d, idx):
        """
        Detecta si un keypoint fue estimado (confianza baja, no detectado directamente).
        Los puntos estimados por el detector tienen confianza <= 0.26.
        """
        if idx not in keypoints_2d:
            return True
        return keypoints_2d[idx][2] <= 0.26

    def _caderas_estimadas(self, keypoints_2d):
        """Detecta si las caderas fueron estimadas."""
        return self._es_punto_estimado(keypoints_2d, 11) or self._es_punto_estimado(keypoints_2d, 12)

    def _hombro_estimado(self, keypoints_2d):
        """Detecta si algún hombro fue estimado (vista lateral)."""
        return self._es_punto_estimado(keypoints_2d, 5) or self._es_punto_estimado(keypoints_2d, 6)

    def _dz_por_escorzo(self, seg_proj_cm, seg_real_cm):
        """
        Calcula la profundidad (Z) implícita en el escorzo de un segmento.

        Si un segmento de longitud real conocida aparece más corto de lo
        esperado en la imagen, la diferencia se debe a que está inclinado en
        profundidad. Por Pitágoras: dz = sqrt(real² - proyectado²).

        Se aplica una zona muerta (ESCORZO_DEADZONE) para no reaccionar a
        acortamientos pequeños, que casi siempre son variación corporal entre
        personas y no postura. La estimación se calcula respecto al borde de la
        zona muerta para que dz crezca de forma continua desde 0 (sin saltos).

        Returns:
            float: profundidad estimada en cm (0 si no hay escorzo relevante).
        """
        if seg_real_cm <= 1e-6:
            return 0.0

        base = seg_real_cm * ESCORZO_DEADZONE
        if seg_proj_cm >= base:
            return 0.0

        dz = np.sqrt(base ** 2 - seg_proj_cm ** 2)
        # Limitar para estabilidad (evita saltos enormes por detecciones espurias)
        return min(dz, seg_real_cm * ESCORZO_DZ_MAX_FRAC)

    def _ref_cabeza_2d(self, keypoints_2d):
        """
        Punto de referencia 2D de la cabeza, su fracción antropométrica y el
        tipo de referencia usado.

        Usa la misma prioridad que el evaluador RULA (orejas > ojos > nariz)
        para que el cálculo de profundidad sea coherente con el del ángulo del
        cuello. El tipo permite mantener una línea base de calibración
        independiente por referencia (la distancia hombros→orejas no es la
        misma que hombros→nariz).

        Returns:
            (punto_xy, fraccion_altura, tipo) o (None, 0, None) si no hay referencia.
        """
        if 3 in keypoints_2d and 4 in keypoints_2d:
            return self._punto_medio(keypoints_2d[3], keypoints_2d[4]), 0.13, 'orejas'
        if 1 in keypoints_2d and 2 in keypoints_2d:
            return self._punto_medio(keypoints_2d[1], keypoints_2d[2]), 0.14, 'ojos'
        if 3 in keypoints_2d:
            return (keypoints_2d[3][0], keypoints_2d[3][1]), 0.13, 'oreja'
        if 4 in keypoints_2d:
            return (keypoints_2d[4][0], keypoints_2d[4][1]), 0.13, 'oreja'
        if 0 in keypoints_2d:
            return (keypoints_2d[0][0], keypoints_2d[0][1]), 0.15, 'nariz'
        return None, 0, None

    def _actualizar_baseline(self, clave, ratio):
        """
        Mantiene la línea base (postura más erguida) de un ratio escala-invariante.

        Sube rápido cuando aparece una postura más erguida (~1 s de evidencia
        sostenida, para no adoptar picos de ruido de golpe) y baja lentísimo,
        de modo que estar encorvado mucho rato NO erosiona la referencia.
        """
        base = self._baselines.get(clave)
        if base is None:
            self._baselines[clave] = ratio
            return ratio
        if ratio > base:
            base += (ratio - base) * CALIB_ALPHA_SUBIDA
        else:
            base += (ratio - base) * CALIB_ALPHA_BAJADA
        self._baselines[clave] = base
        return base

    def _dz_calibrado(self, clave, ratio, seg_real_cm):
        """
        Profundidad por escorzo RELATIVO a la línea base personal.

        En vez de comparar la longitud proyectada contra una proporción
        antropométrica absoluta (sensible a la altura configurada, al FOV y a
        la variación entre personas), compara el ratio segmento/ancho_hombros
        actual contra el mejor ratio observado de la propia persona. Si el
        segmento se ve un X% más corto que erguido, ese acortamiento se
        convierte a profundidad con la misma geometría de _dz_por_escorzo.
        """
        base = self._actualizar_baseline(clave, ratio)
        if base <= 1e-6:
            return 0.0
        seg_proj_eq = seg_real_cm * min(ratio / base, 1.0)
        return self._dz_por_escorzo(seg_proj_eq, seg_real_cm)

    def _reubicar_a_z(self, px, py, z, focal, cx, cy):
        """Re-proyecta un punto 2D a 3D a una profundidad Z dada."""
        return np.array([(px - cx) * z / focal, (py - cy) * z / focal, z])

    def _elevar_sagital(self, kp2d, kp3d, z_ref, focal, cx, cy,
                        hombro_est, caderas_est, ancho_hombros_px):
        """
        Reasigna la profundidad de caderas y cabeza según el escorzo, para
        revelar la inclinación sagital (encorvarse / cabeza adelantada) que en
        vista frontal no aparece en X-Y.

        El escorzo se mide de forma RELATIVA: cada ratio segmento/ancho_hombros
        se compara contra la línea base de la postura más erguida de la propia
        persona (ver _dz_calibrado). Así la detección de encorvamiento frontal
        funciona sin necesidad de ver las caderas (caso típico de escritorio,
        donde el encorvamiento se manifiesta como cabeza adelantada) y sin
        depender de acertar ALTURA_SUJETO_CM ni el FOV de la cámara.

        Solo se aplica en vista frontal/ángulo fiable (ambos hombros reales y
        bien separados). En vista lateral la inclinación YA se ve en X-Y, así
        que no se añade nada (evita contar dos veces).
        """
        # Vista lateral o hombros poco fiables: no tocar (ya se mide en X-Y)
        if hombro_est or ancho_hombros_px < ANCHO_HOMBROS_MIN_PX:
            return
        if 5 not in kp2d or 6 not in kp2d:
            return

        mid_hombro = self._punto_medio(kp2d[5], kp2d[6])

        # --- TRONCO: profundidad de las caderas por escorzo del torso ---
        # Solo con caderas reales (si fueron estimadas, el torso es inventado).
        if not caderas_est and 11 in kp3d and 12 in kp3d:
            mid_cadera = self._punto_medio(kp2d[11], kp2d[12])
            torso_px = self._distancia_2d(mid_hombro, mid_cadera)
            seg_real = self.segmentos_reales['hombro_cadera']
            ratio_torso = torso_px / ancho_hombros_px
            dz = self._dz_calibrado('torso', ratio_torso, seg_real)
            if dz > 0:
                # Caderas más profundas que los hombros → tronco inclinado.
                # (El signo no afecta la magnitud de flexión que mide RULA.)
                z_cadera = z_ref + dz
                for idx in (11, 12):
                    px, py, _ = kp2d[idx]
                    kp3d[idx] = self._reubicar_a_z(px, py, z_cadera, focal, cx, cy)

        # --- CUELLO: cabeza adelantada (forward-head) por escorzo del cuello ---
        # Al encorvarse frente al escritorio la cabeza baja y se adelanta: la
        # distancia hombros→cabeza se acorta respecto a la línea base erguida.
        cabeza_xy, frac, tipo_ref = self._ref_cabeza_2d(kp2d)
        if cabeza_xy is not None:
            cuello_px = self._distancia_2d(mid_hombro, cabeza_xy)
            seg_real = self.altura_cm * frac
            ratio_cuello = cuello_px / ancho_hombros_px
            dz = self._dz_calibrado(f'cuello_{tipo_ref}', ratio_cuello, seg_real)
            if dz > 0:
                # Cabeza adelantada hacia la cámara (más cerca = menor Z).
                z_cabeza = z_ref - dz
                for idx in (0, 1, 2, 3, 4):
                    if idx in kp3d and idx in kp2d:
                        px, py, _ = kp2d[idx]
                        kp3d[idx] = self._reubicar_a_z(px, py, z_cabeza, focal, cx, cy)

    def elevar(self, keypoints_2d, frame_shape):
        """
        Convierte keypoints 2D a coordenadas 3D estimadas.

        Maneja vistas frontales, laterales y en ángulo.

        Args:
            keypoints_2d: dict {idx: (x, y, conf)} de la Fase 1.
            frame_shape: tuple (alto, ancho, canales) del frame.

        Returns:
            dict {idx: np.array([X, Y, Z])} en centímetros,
            donde X = horizontal, Y = vertical (abajo +), Z = profundidad.
        """
        if keypoints_2d is None:
            return None

        alto, ancho = frame_shape[:2]
        focal = self._estimar_focal(ancho)
        cx, cy = ancho / 2, alto / 2  # Centro óptico

        # =====================================================================
        # Paso 1: Estimar profundidad de referencia
        # =====================================================================
        hombro_izq = keypoints_2d[5]
        hombro_der = keypoints_2d[6]
        cadera_izq = keypoints_2d[11]
        cadera_der = keypoints_2d[12]

        mid_hombro = self._punto_medio(hombro_izq, hombro_der)
        mid_cadera = self._punto_medio(cadera_izq, cadera_der)

        # Decidir la referencia de profundidad según lo que sea más confiable
        caderas_est = self._caderas_estimadas(keypoints_2d)
        hombro_est = self._hombro_estimado(keypoints_2d)

        ancho_hombros_px = self._distancia_2d(
            (hombro_izq[0], hombro_izq[1]),
            (hombro_der[0], hombro_der[1])
        )

        if not hombro_est and ancho_hombros_px > 20:
            # Hombros reales y separados (vista frontal o en ángulo):
            # usar el ANCHO DE HOMBROS como referencia de profundidad.
            #
            # Clave: a diferencia del torso, el ancho de hombros NO se acorta
            # cuando la persona se inclina hacia adelante/atrás. Por eso da una
            # profundidad de referencia estable e independiente de la postura,
            # y sirve de ancla para medir el escorzo del tronco (ver Paso 2.5).
            ref_pixels = ancho_hombros_px
            ref_real = self.segmentos_reales['ancho_hombros']
        else:
            # Vista lateral o todo estimado: usar distancia cabeza-hombro
            # como referencia (la más robusta en vistas laterales)
            if 0 in keypoints_2d and not self._es_punto_estimado(keypoints_2d, 0):
                # Distancia nariz a centro de hombros
                ref_pixels = self._distancia_2d(
                    (keypoints_2d[0][0], keypoints_2d[0][1]),
                    mid_hombro
                )
                # Cabeza-cuello ≈ 15% de la altura total
                ref_real = self.altura_cm * 0.15
            elif 3 in keypoints_2d or 4 in keypoints_2d:
                # Usar oreja visible
                oreja_idx = 3 if 3 in keypoints_2d else 4
                ref_pixels = self._distancia_2d(
                    (keypoints_2d[oreja_idx][0], keypoints_2d[oreja_idx][1]),
                    mid_hombro
                )
                ref_real = self.altura_cm * 0.13
            else:
                # Último recurso: usar torso estimado
                ref_pixels = self._distancia_2d(mid_hombro, mid_cadera)
                ref_real = self.segmentos_reales['hombro_cadera']

        if ref_pixels < 1:
            ref_pixels = 1  # Evitar división por cero

        # Z_ref = profundidad estimada del centro del torso
        z_ref = (focal * ref_real) / ref_pixels

        # =====================================================================
        # Paso 2: Asignar profundidad a keypoints del torso y cabeza
        # =====================================================================
        keypoints_3d = {}

        # Puntos del torso y cabeza: asumimos están a Z_ref (misma profundidad)
        indices_torso = [0, 1, 2, 3, 4, 5, 6, 11, 12]
        for idx in indices_torso:
            if idx in keypoints_2d:
                px, py, _ = keypoints_2d[idx]
                X = (px - cx) * z_ref / focal
                Y = (py - cy) * z_ref / focal
                keypoints_3d[idx] = np.array([X, Y, z_ref])

        # =====================================================================
        # Paso 3: Propagar profundidad por las extremidades
        # =====================================================================
        # Para cada segmento: si aparece más corto que lo esperado,
        # la diferencia se debe a un componente en Z.
        for padre_idx, hijo_idx, nombre_seg in SEGMENTOS_CADENA:
            # Saltar segmentos de piernas si las caderas son estimadas
            # (no tiene sentido propagar profundidad desde puntos estimados a piernas)
            if self._caderas_estimadas(keypoints_2d) and padre_idx in (11, 12, 13, 14):
                continue
            if padre_idx in keypoints_3d and hijo_idx in keypoints_2d:
                padre_3d = keypoints_3d[padre_idx]
                hijo_px, hijo_py, _ = keypoints_2d[hijo_idx]

                # Longitud real del segmento en cm
                seg_real = self.segmentos_reales[nombre_seg]

                # Longitud del segmento en píxeles
                padre_2d = keypoints_2d[padre_idx]
                seg_pixels = self._distancia_2d(
                    (padre_2d[0], padre_2d[1]),
                    (hijo_px, hijo_py)
                )

                # Convertir longitud en píxeles a cm a la profundidad del padre
                seg_proyectado = seg_pixels * padre_3d[2] / focal

                # Si el segmento proyectado es más corto que el real,
                # hay un componente de profundidad (teorema de Pitágoras)
                if seg_proyectado < seg_real:
                    dz = np.sqrt(seg_real ** 2 - seg_proyectado ** 2)
                else:
                    dz = 0  # El segmento es visible en toda su longitud

                # Estimar profundidad del hijo
                # (la dirección ±dz es ambigua; usamos valor absoluto)
                hijo_z = padre_3d[2] + dz

                X = (hijo_px - cx) * hijo_z / focal
                Y = (hijo_py - cy) * hijo_z / focal
                keypoints_3d[hijo_idx] = np.array([X, Y, hijo_z])

        # =====================================================================
        # Paso 2.5: Elevación sagital (inclinación frontal hacia adelante)
        # =====================================================================
        # En vista frontal, encorvarse o adelantar la cabeza son movimientos en
        # profundidad (Z) invisibles en X-Y. Aquí se recupera esa profundidad a
        # partir del escorzo del torso y del cuello, anclando en el plano de los
        # hombros (cuyo ancho no se acorta al inclinarse).
        self._elevar_sagital(
            keypoints_2d, keypoints_3d, z_ref, focal, cx, cy,
            hombro_est, caderas_est, ancho_hombros_px,
        )

        # =====================================================================
        # Paso 4: Agregar puntos virtuales (cuello, centro de torso)
        # =====================================================================
        # Punto virtual "cuello" = punto medio entre hombros (en 3D)
        if 5 in keypoints_3d and 6 in keypoints_3d:
            keypoints_3d[17] = (keypoints_3d[5] + keypoints_3d[6]) / 2  # Cuello

        # Punto virtual "centro_cadera" = punto medio entre caderas (en 3D)
        if 11 in keypoints_3d and 12 in keypoints_3d:
            keypoints_3d[18] = (keypoints_3d[11] + keypoints_3d[12]) / 2  # Centro cadera

        # =====================================================================
        # Paso 5: Suavizado temporal adaptativo (media móvil exponencial)
        # =====================================================================
        # Si un keypoint salta bruscamente (>40 cm en XY) entre frames,
        # casi siempre indica un cambio de persona rastreada o una detección
        # espuria. En ese caso se reduce el peso del historial para adaptarse
        # rápido al estado nuevo sin arrastrar valores de otra persona.
        for idx, pos in keypoints_3d.items():
            if idx in self.historial:
                dist_xy = np.linalg.norm(pos[:2] - self.historial[idx][:2])
                alpha_eff = min(self.alpha + 0.5, 0.95) if dist_xy > 40 else self.alpha
                keypoints_3d[idx] = alpha_eff * pos + (1 - alpha_eff) * self.historial[idx]
            self.historial[idx] = keypoints_3d[idx].copy()

        return keypoints_3d
