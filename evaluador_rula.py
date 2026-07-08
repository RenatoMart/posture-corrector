"""
evaluador_rula.py — Fase 3: Evaluación Ergonómica RULA completa.

Implementa el método RULA (Rapid Upper Limb Assessment) con las tablas
oficiales de McAtamney & Corlett (1993).

Pipeline:
    1. Calcular ángulos articulares 3D (producto escalar)
    2. Mapear ángulos → scores individuales
    3. Tabla A: brazo + antebrazo + muñeca → Score A
    4. Tabla B: cuello + tronco + piernas → Score B
    5. Tabla C: Score A × Score B → Score RULA final (1-7)
    6. Clasificar nivel de acción
"""

import numpy as np
from config import (
    CONF_FABRICADO,
    VISTA_ASIMETRIA_LATERAL, VISTA_SPREAD_FRONTAL, VISTA_SPREAD_LATERAL,
    VISTA_NARIZ_OFFSET_LATERAL, VISTA_HISTERESIS_FRAMES,
    ENCORVADO_ACTIVO,
)
from detector_encorvamiento import AnalizadorSiluetaLateral


class EvaluadorRULA:
    """
    Evaluador ergonómico RULA (Rapid Upper Limb Assessment).
    
    Calcula el score RULA a partir de keypoints 3D, evaluando la postura
    del cuello, tronco, brazos, antebrazos y muñecas.
    """

    # =========================================================================
    # TABLAS RULA OFICIALES
    # =========================================================================

    # Tabla A: [brazo_sup (1-6)][antebrazo (1-3)][muñeca (1-4)][giro_muñeca (1-2)]
    # Dimensiones: 6 × 3 × 4 × 2
    TABLA_A = np.array([
        # Brazo superior = 1
        [[[1, 2], [2, 2], [2, 3], [3, 3]],
         [[2, 2], [2, 2], [3, 3], [3, 3]],
         [[2, 3], [3, 3], [3, 3], [4, 4]]],
        # Brazo superior = 2
        [[[2, 3], [3, 3], [3, 4], [4, 4]],
         [[3, 3], [3, 3], [3, 4], [4, 4]],
         [[3, 4], [4, 4], [4, 4], [5, 5]]],
        # Brazo superior = 3
        [[[3, 3], [4, 4], [4, 4], [5, 5]],
         [[3, 4], [4, 4], [4, 4], [5, 5]],
         [[4, 4], [4, 4], [4, 5], [5, 5]]],
        # Brazo superior = 4
        [[[4, 4], [4, 4], [4, 5], [5, 5]],
         [[4, 4], [4, 4], [4, 5], [5, 5]],
         [[4, 4], [4, 5], [5, 5], [6, 6]]],
        # Brazo superior = 5
        [[[5, 5], [5, 5], [5, 6], [6, 7]],
         [[5, 6], [6, 6], [6, 7], [7, 7]],
         [[6, 6], [6, 7], [7, 7], [7, 8]]],
        # Brazo superior = 6
        [[[7, 7], [7, 7], [7, 8], [8, 9]],
         [[8, 8], [8, 8], [8, 9], [9, 9]],
         [[9, 9], [9, 9], [9, 9], [9, 9]]],
    ])

    # Tabla B: [cuello (1-6)][tronco (1-6)][piernas (1-2)]
    # Dimensiones: 6 × 6 × 2
    TABLA_B = np.array([
        # Cuello = 1
        [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
        # Cuello = 2
        [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
        # Cuello = 3
        [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
        # Cuello = 4
        [[5, 5], [4, 5], [5, 6], [6, 7], [7, 7], [7, 8]],
        # Cuello = 5
        [[7, 7], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
        # Cuello = 6
        [[8, 8], [6, 7], [7, 8], [7, 8], [7, 8], [8, 8]],
    ])

    # Tabla C: [score_A (1-8+)][score_B (1-7+)] → Score RULA final
    # Dimensiones: 8 × 7
    TABLA_C = np.array([
        [1, 2, 3, 3, 4, 5, 5],  # Score A = 1
        [2, 2, 3, 4, 4, 5, 5],  # Score A = 2
        [3, 3, 3, 4, 4, 5, 6],  # Score A = 3
        [3, 3, 3, 4, 5, 6, 6],  # Score A = 4
        [4, 4, 4, 5, 6, 7, 7],  # Score A = 5
        [4, 4, 5, 6, 6, 7, 7],  # Score A = 6
        [5, 5, 6, 6, 7, 7, 7],  # Score A = 7
        [5, 5, 6, 7, 7, 7, 7],  # Score A = 8
    ])

    # =========================================================================
    # NIVELES DE ACCIÓN
    # =========================================================================
    NIVELES = {
        1: {'nivel': 1, 'texto': 'Postura aceptable', 'color_key': 'verde'},
        2: {'nivel': 1, 'texto': 'Postura aceptable', 'color_key': 'verde'},
        3: {'nivel': 2, 'texto': 'Investigar, posible cambio', 'color_key': 'amarillo'},
        4: {'nivel': 2, 'texto': 'Investigar, posible cambio', 'color_key': 'amarillo'},
        5: {'nivel': 3, 'texto': 'Investigar, cambio pronto', 'color_key': 'naranja'},
        6: {'nivel': 3, 'texto': 'Investigar, cambio pronto', 'color_key': 'naranja'},
        7: {'nivel': 4, 'texto': 'Cambio INMEDIATO', 'color_key': 'rojo'},
    }

    def __init__(self):
        # Estado para la histéresis de la clasificación de vista (anti-parpadeo)
        self._vista_actual = None        # vista estable mostrada/usada
        self._lado_actual = None
        self._vista_pendiente = None     # vista candidata a reemplazar la actual
        self._vista_pendiente_n = 0      # frames consecutivos viendo la candidata

        # Analizador de encorvamiento por silueta (vista lateral con cadera oculta)
        self._analizador_silueta = AnalizadorSiluetaLateral() if ENCORVADO_ACTIVO else None

    def recalibrar(self):
        """Reinicia la línea base del detector de encorvamiento lateral (tecla 'C')."""
        if self._analizador_silueta is not None:
            self._analizador_silueta.recalibrar()

    # =========================================================================
    # UTILIDADES DE CONFIANZA (para vistas laterales)
    # =========================================================================

    @staticmethod
    def _confianza_lado(kp3d, lado):
        """
        Calcula la confianza promedio de los keypoints de un lado del cuerpo.
        
        Esto permite distinguir entre keypoints reales (detectados por YOLO)
        y estimados (inferidos por el detector). Los keypoints 3D no tienen
        confianza directa, así que usamos la presencia de puntos clave.
        
        Args:
            kp3d: dict de keypoints 3D.
            lado: 'izq' o 'der'
            
        Returns:
            float entre 0 y 1 indicando confianza del lado.
        """
        if lado == 'izq':
            indices = [5, 7, 9]   # hombro, codo, muñeca izq
        else:
            indices = [6, 8, 10]  # hombro, codo, muñeca der
        
        presentes = sum(1 for idx in indices if idx in kp3d)
        return presentes / len(indices)

    @staticmethod
    def _seleccionar_por_confianza(score_izq, conf_izq, score_der, conf_der):
        """
        Selecciona el score más apropiado basado en la confianza de cada lado.
        
        - Si ambos lados son confiables (>0.5): usa el peor (conservador)
        - Si solo un lado es confiable: usa ese lado
        - Si ninguno es confiable: usa el promedio
        
        Args:
            score_izq: score RULA del lado izquierdo
            conf_izq: confianza del lado izquierdo (0-1)
            score_der: score RULA del lado derecho
            conf_der: confianza del lado derecho (0-1)
            
        Returns:
            score seleccionado (int)
        """
        ambos_confiables = conf_izq > 0.5 and conf_der > 0.5
        
        if ambos_confiables:
            return max(score_izq, score_der)  # Conservador: peor caso
        elif conf_izq > conf_der:
            return score_izq  # Solo confiar en el izquierdo
        elif conf_der > conf_izq:
            return score_der  # Solo confiar en el derecho
        else:
            return min(score_izq, score_der)  # Ninguno domina: ser generoso

    # =========================================================================
    # DETECCIÓN DE VISTA DE CÁMARA (frontal / ángulo / lateral)
    # =========================================================================

    @staticmethod
    def _indices_lado(lado):
        """Mapea 'izq'/'der' a los índices COCO de ese lado del cuerpo."""
        if lado == 'izq':
            return {'hombro': 5, 'codo': 7, 'muneca': 9, 'cadera': 11, 'oreja': 3}
        return {'hombro': 6, 'codo': 8, 'muneca': 10, 'cadera': 12, 'oreja': 4}

    @staticmethod
    def _clasificar_vista_cruda(keypoints_2d):
        """
        Clasificación instantánea (sin histéresis) del ángulo de cámara.

        Combina tres señales robustas que NO dependen de las caderas (que suelen
        estar fabricadas al estar sentado, lo que antes rompía la detección):

          1) Asimetría de confianza entre lados. En perfil, todo un lado del
             cuerpo (ojo, oreja, hombro, cadera) queda ocluido y recibe baja
             confianza o se fabrica. Si el lado débil tiene mucha menos
             confianza que el fuerte → lateral.
          2) Separación de hombros normalizada por el alto de la cabeza
             (hombros→nariz/oreja). De frente los hombros se ven anchos respecto
             a la cabeza; de costado se juntan. Es escala-invariante e
             independiente de las caderas.
          3) Desplazamiento horizontal de la nariz respecto al centro de los
             hombros. De frente la nariz queda centrada entre los hombros
             (incluso girando la cabeza); de costado queda claramente hacia el
             hombro visible. Esta señal reconoce la vista lateral aunque la
             persona gire la cabeza hacia la cámara (antes eso confundía a las
             señales 1 y 2 y el sistema "exigía" mirar a la cámara de costado).

        Returns:
            (vista, lado_fiable)
        """
        if keypoints_2d is None:
            return 'frontal', None

        tiene_izq = 5 in keypoints_2d
        tiene_der = 6 in keypoints_2d
        if not tiene_izq and not tiene_der:
            return 'frontal', None
        # Un solo hombro presente: vista lateral inequívoca (defensivo; el
        # detector normalmente fabrica el faltante, pero por si acaso).
        if tiene_izq != tiene_der:
            return 'lateral', ('izq' if tiene_izq else 'der')

        # --- Señal 1: confianza real (no fabricada) por lado del cuerpo ---
        def _conf_lado(indices):
            return sum(
                keypoints_2d[i][2]
                for i in indices
                if i in keypoints_2d and keypoints_2d[i][2] > CONF_FABRICADO
            )

        conf_izq = _conf_lado([1, 3, 5, 11])   # ojo, oreja, hombro, cadera izq
        conf_der = _conf_lado([2, 4, 6, 12])   # ojo, oreja, hombro, cadera der

        lado_fuerte = 'izq' if conf_izq >= conf_der else 'der'
        mayor = max(conf_izq, conf_der)
        menor = min(conf_izq, conf_der)
        asimetria = (menor / mayor) if mayor > 1e-6 else 1.0

        # --- Señal 2: separación de hombros / alto de la cabeza ---
        hombro_izq = keypoints_2d[5]
        hombro_der = keypoints_2d[6]
        ancho_hombros = abs(hombro_izq[0] - hombro_der[0])
        centro_hombros_y = (hombro_izq[1] + hombro_der[1]) / 2

        cabeza_y = None
        for idx in (0, 3, 4, 1, 2):  # nariz, orejas, ojos (lo primero disponible y real)
            if idx in keypoints_2d and keypoints_2d[idx][2] > CONF_FABRICADO:
                cabeza_y = keypoints_2d[idx][1]
                break

        norm_spread = None
        alto_cabeza = None
        if cabeza_y is not None:
            alto_cabeza = abs(centro_hombros_y - cabeza_y)
            if alto_cabeza > 1e-3:
                norm_spread = ancho_hombros / alto_cabeza

        # --- Señal 3: nariz descentrada respecto al centro de los hombros ---
        nariz_offset = None
        if (0 in keypoints_2d and keypoints_2d[0][2] > CONF_FABRICADO
                and alto_cabeza is not None and alto_cabeza > 1e-3):
            centro_hombros_x = (hombro_izq[0] + hombro_der[0]) / 2
            nariz_offset = abs(keypoints_2d[0][0] - centro_hombros_x) / alto_cabeza

        # --- Decisión combinada ---
        muy_asimetrico = asimetria < VISTA_ASIMETRIA_LATERAL
        hombros_juntos = norm_spread is not None and norm_spread < VISTA_SPREAD_LATERAL
        hombros_separados = norm_spread is not None and norm_spread > VISTA_SPREAD_FRONTAL
        nariz_descentrada = (nariz_offset is not None
                             and nariz_offset > VISTA_NARIZ_OFFSET_LATERAL)

        # Lateral si un lado está ocluido, los hombros se ven casi superpuestos
        # o la nariz cae claramente fuera del centro de los hombros (cuerpo de
        # perfil aunque la cara mire a la cámara)
        if muy_asimetrico or hombros_juntos or nariz_descentrada:
            # Si la decisión vino por geometría (no por oclusión), el lado
            # fiable es hacia donde apunta la nariz: el hombro más cercano a
            # ella es el visible/delantero.
            if not muy_asimetrico and nariz_offset is not None:
                d_izq = abs(keypoints_2d[0][0] - hombro_izq[0])
                d_der = abs(keypoints_2d[0][0] - hombro_der[0])
                lado_geom = 'izq' if d_izq <= d_der else 'der'
                return 'lateral', lado_geom
            return 'lateral', lado_fuerte

        # Frontal solo si los hombros se ven claramente anchos y no hay oclusión
        if hombros_separados:
            return 'frontal', None

        # Sin referencia geométrica de cabeza: decidir solo por asimetría
        if norm_spread is None:
            return ('frontal', None) if asimetria > 0.7 else ('angulo', lado_fuerte)

        # Zona intermedia
        return 'angulo', lado_fuerte

    def _detectar_vista(self, keypoints_2d):
        """
        Clasifica el ángulo de cámara con histéresis temporal (anti-parpadeo).

        En vista lateral/ángulo, el hombro/cadera del lado oculto son fabricados
        (ver detector_2d); medir el tronco/cuello con ellos (vía los puntos
        medios 17/18) introduce un desplazamiento horizontal falso aunque la
        persona esté erguida. Por eso, fuera de la vista frontal, se mide con los
        puntos REALES de un solo lado (el fiable), nunca con los fabricados.

        La histéresis evita que la etiqueta y la ruta de medición salten de un
        frame a otro en posiciones límite: una vista nueva debe sostenerse varios
        frames antes de reemplazar a la actual.

        Returns:
            (vista, lado_fiable)
            vista: 'frontal' | 'lateral' | 'angulo'
            lado_fiable: 'izq' | 'der' | None (None solo en vista frontal)
        """
        vista_cruda, lado_crudo = self._clasificar_vista_cruda(keypoints_2d)

        # Primer frame: adoptar de inmediato (sin esperar histéresis)
        if self._vista_actual is None:
            self._vista_actual = vista_cruda
            self._lado_actual = lado_crudo
            self._vista_pendiente = None
            self._vista_pendiente_n = 0
            return self._vista_actual, self._lado_actual

        # Si coincide con la vista estable: actualizar el lado y limpiar pendiente
        if vista_cruda == self._vista_actual:
            self._lado_actual = lado_crudo
            self._vista_pendiente = None
            self._vista_pendiente_n = 0
            return self._vista_actual, self._lado_actual

        # Vista distinta: acumular evidencia antes de cambiar
        if vista_cruda == self._vista_pendiente:
            self._vista_pendiente_n += 1
        else:
            self._vista_pendiente = vista_cruda
            self._vista_pendiente_n = 1

        if self._vista_pendiente_n >= VISTA_HISTERESIS_FRAMES:
            self._vista_actual = vista_cruda
            self._lado_actual = lado_crudo
            self._vista_pendiente = None
            self._vista_pendiente_n = 0

        return self._vista_actual, self._lado_actual

    # =========================================================================
    # CÁLCULO DE ÁNGULOS 3D
    # =========================================================================

    @staticmethod
    def _angulo_entre_vectores(v1, v2):
        """
        Calcula el ángulo en grados entre dos vectores 3D.
        Usa producto escalar: θ = arccos(v1·v2 / |v1||v2|)
        """
        norma1 = np.linalg.norm(v1)
        norma2 = np.linalg.norm(v2)
        if norma1 < 1e-6 or norma2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (norma1 * norma2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    @staticmethod
    def _angulo_en_articulacion(punto_a, punto_b, punto_c):
        """
        Calcula el ángulo en el punto B formado por los vectores BA y BC.
        """
        v1 = punto_a - punto_b  # Vector B→A
        v2 = punto_c - punto_b  # Vector B→C
        return EvaluadorRULA._angulo_entre_vectores(v1, v2)

    @staticmethod
    def _angulo_brazo_sagital(kp3d, hombro_idx, codo_idx, cadera_idx):
        """
        Calcula el ángulo de flexión/extensión del brazo superior en el plano sagital.

        RULA mide la flexión del brazo en el plano sagital del cuerpo (el plano
        que contiene la columna y es perpendicular al eje lateral hombro-hombro).
        Este cálculo proyecta los vectores sobre ese plano antes de medir el ángulo,
        lo que da resultados correctos tanto en vista frontal como lateral o en ángulo.

        Sin la proyección, en vista frontal el ángulo 3D del brazo incluye el
        componente lateral (abducción) y sobreestima la flexión.
        En vista lateral, el componente de profundidad del brazo es muy incierto
        y contamina el resultado.

        Args:
            kp3d: dict {idx: np.array([X, Y, Z])}
            hombro_idx: 5 (izq) o 6 (der)
            codo_idx:   7 (izq) o 8 (der)
            cadera_idx: 11 (izq) o 12 (der)

        Returns:
            float: ángulo de flexión en grados.
        """
        if hombro_idx not in kp3d or codo_idx not in kp3d:
            return 0.0

        hombro = kp3d[hombro_idx]
        codo = kp3d[codo_idx]
        vec_brazo = codo - hombro

        # Eje del tronco hacia abajo (referencia RULA: brazo colgando = 0°)
        if cadera_idx in kp3d:
            eje_tronco = kp3d[cadera_idx] - hombro
        else:
            eje_tronco = np.array([0.0, 1.0, 0.0])

        # Definir el eje lateral del cuerpo desde los dos hombros
        otro_hombro_idx = 6 if hombro_idx == 5 else 5
        if otro_hombro_idx in kp3d:
            # La normal al plano sagital apunta lateralmente (hombro a hombro)
            vec_lateral = kp3d[otro_hombro_idx] - kp3d[hombro_idx]
            norma_lat = np.linalg.norm(vec_lateral)
            if norma_lat > 1e-6:
                n_sagital = vec_lateral / norma_lat

                # Proyectar ambos vectores sobre el plano sagital
                # (se elimina el componente lateral de cada vector)
                vec_brazo_p = vec_brazo - np.dot(vec_brazo, n_sagital) * n_sagital
                eje_tronco_p = eje_tronco - np.dot(eje_tronco, n_sagital) * n_sagital

                if np.linalg.norm(vec_brazo_p) > 1e-6 and np.linalg.norm(eje_tronco_p) > 1e-6:
                    return EvaluadorRULA._angulo_entre_vectores(vec_brazo_p, eje_tronco_p)

        # Fallback: ángulo 3D completo (cuando solo hay un hombro)
        return EvaluadorRULA._angulo_entre_vectores(vec_brazo, eje_tronco)

    @staticmethod
    def _detectar_abduccion(kp3d, hombro_idx, codo_idx):
        """
        Detecta abducción usando el eje lateral real del cuerpo en 3D.

        Calcula qué fracción del movimiento del brazo es lateral (fuera del
        plano sagital). Si supera el 35% del vector total, hay abducción.

        Args:
            kp3d: dict {idx: np.array([X, Y, Z])}
            hombro_idx: 5 (izq) o 6 (der)
            codo_idx:   7 (izq) o 8 (der)

        Returns:
            bool: True si el brazo está abducido.
        """
        if hombro_idx not in kp3d or codo_idx not in kp3d:
            return False

        otro_hombro_idx = 6 if hombro_idx == 5 else 5
        if otro_hombro_idx not in kp3d:
            return False

        vec_brazo = kp3d[codo_idx] - kp3d[hombro_idx]
        vec_lateral = kp3d[otro_hombro_idx] - kp3d[hombro_idx]
        norma_lat = np.linalg.norm(vec_lateral)
        norma_brazo = np.linalg.norm(vec_brazo)
        if norma_lat < 1e-6 or norma_brazo < 1e-6:
            return False

        n_sagital = vec_lateral / norma_lat
        comp_lateral = abs(np.dot(vec_brazo, n_sagital))
        return (comp_lateral / norma_brazo) > 0.35

    def _calcular_angulos(self, kp3d, vista='frontal', lado_fiable=None,
                          tronco_silueta=None, keypoints_2d=None):
        """
        Calcula todos los ángulos necesarios para RULA a partir de keypoints 3D.

        Despacha a una de dos rutas de medición:
        - Frontal (vista='frontal'): usa los puntos medios virtuales 17/18
          (cuello, centro de cadera), que combinan información de ambos lados.
        - Lateral/ángulo (lado_fiable definido): usa SOLO los puntos reales del
          lado visible, evitando los keypoints fabricados del lado oculto que
          contaminarían el tronco/cuello con un desplazamiento horizontal falso.

        Args:
            kp3d: dict {idx: np.array([X, Y, Z])}
                  Índices 17 = cuello virtual, 18 = centro cadera virtual
            vista: 'frontal' | 'lateral' | 'angulo'
            lado_fiable: 'izq' | 'der' | None
            tronco_silueta: flexión de tronco (grados) estimada por silueta en
                vista lateral cuando la cadera es fabricada, o None.
            keypoints_2d: dict 2D para saber si la cadera del lado visible es real.

        Returns:
            dict con ángulos en grados para cada segmento.
        """
        if vista in ('lateral', 'angulo') and lado_fiable is not None:
            return self._angulos_lateral(kp3d, lado_fiable, tronco_silueta, keypoints_2d)
        return self._angulos_frontal(kp3d)

    def _angulos_frontal(self, kp3d):
        """Calcula los ángulos usando ambos lados (puntos medios 17/18)."""
        angulos = {}

        # Vector vertical (gravedad apunta hacia abajo = Y positivo)
        vertical = np.array([0, 1, 0])
        vertical_arriba = np.array([0, -1, 0])

        # --- CUELLO ---
        # Ángulo de inclinación de la cabeza respecto a la vertical, medido
        # SOLO en el plano sagital (componentes Y y Z). En vista frontal, la
        # flexión real del cuello es un movimiento en profundidad: la
        # componente X solo aporta artefactos cuando la persona gira la cabeza
        # o cuando la referencia es asimétrica (una sola oreja / nariz).
        # Antes ese artefacto penalizaba a quien no miraba a la cámara.
        if 17 in kp3d:
            cuello = kp3d[17]

            # Referencia de la cabeza (prioridad: orejas > ojos > una oreja > nariz).
            # Usar más puntos de la cara hace la referencia estable ante giros.
            cabeza = None
            if 3 in kp3d and 4 in kp3d:
                cabeza = (kp3d[3] + kp3d[4]) / 2      # Punto medio entre orejas
            elif 1 in kp3d and 2 in kp3d:
                cabeza = (kp3d[1] + kp3d[2]) / 2      # Punto medio entre ojos
            elif 3 in kp3d:
                cabeza = kp3d[3]
            elif 4 in kp3d:
                cabeza = kp3d[4]
            elif 0 in kp3d:
                cabeza = kp3d[0]

            if cabeza is not None:
                vec_cuello = cabeza - cuello
                # Proyección sagital: descartar X (giros/inclinación lateral de
                # cabeza no son flexión). La profundidad Z proviene del escorzo
                # calibrado (ver elevador_3d._elevar_sagital): cabeza adelantada
                # => Z menor que los hombros => ángulo crece.
                vec_sagital = np.array([0.0, vec_cuello[1], vec_cuello[2]])
                angulo_cuello = self._angulo_entre_vectores(vec_sagital, vertical_arriba)
                angulos['cuello_flexion'] = angulo_cuello

                # Extensión (mirar hacia arriba): la nariz sube por encima de la
                # línea de los ojos. En reposo la nariz queda claramente por
                # debajo de los ojos; solo una extensión franca invierte eso.
                # (Y crece hacia abajo, por eso el signo.)
                en_extension = False
                if 0 in kp3d and 1 in kp3d and 2 in kp3d:
                    ojos_y = (kp3d[1][1] + kp3d[2][1]) / 2
                    escala_cabeza = max(np.linalg.norm(vec_cuello), 1e-6)
                    en_extension = (ojos_y - kp3d[0][1]) > 0.10 * escala_cabeza
                angulos['cuello_extension'] = en_extension
            else:
                angulos['cuello_flexion'] = 0
                angulos['cuello_extension'] = False
        else:
            angulos['cuello_flexion'] = 0
            angulos['cuello_extension'] = False

        # --- TRONCO ---
        # Ángulo entre vector (cadera_centro → cuello) y vertical hacia arriba
        if 18 in kp3d and 17 in kp3d:
            cadera_centro = kp3d[18]
            cuello = kp3d[17]
            vec_tronco = cuello - cadera_centro
            angulo_tronco = self._angulo_entre_vectores(vec_tronco, vertical_arriba)
            angulos['tronco_flexion'] = angulo_tronco
        else:
            angulos['tronco_flexion'] = 0

        # --- BRAZO SUPERIOR (izquierdo) ---
        # Ángulo de flexión en el plano sagital del cuerpo.
        # Al proyectar sobre el plano sagital se excluye la componente lateral
        # (abducción) y la estimación de profundidad del brazo en vistas laterales,
        # dando un resultado más estable en cualquier ángulo de cámara.
        if 5 in kp3d and 7 in kp3d:
            angulos['brazo_sup_izq'] = self._angulo_brazo_sagital(kp3d, 5, 7, 11)

            # Hombro elevado: comparar altura Y entre hombros
            if 6 in kp3d:
                diff_y = kp3d[5][1] - kp3d[6][1]
                angulos['hombro_elevado_izq'] = diff_y < -5

            # Abducción: fracción lateral del vector brazo respecto al eje hombro-hombro
            angulos['brazo_abducido_izq'] = self._detectar_abduccion(kp3d, 5, 7)
        else:
            angulos['brazo_sup_izq'] = 0
            angulos['hombro_elevado_izq'] = False
            angulos['brazo_abducido_izq'] = False

        # --- BRAZO SUPERIOR (derecho) ---
        if 6 in kp3d and 8 in kp3d:
            angulos['brazo_sup_der'] = self._angulo_brazo_sagital(kp3d, 6, 8, 12)

            if 5 in kp3d:
                diff_y = kp3d[6][1] - kp3d[5][1]
                angulos['hombro_elevado_der'] = diff_y < -5

            angulos['brazo_abducido_der'] = self._detectar_abduccion(kp3d, 6, 8)
        else:
            angulos['brazo_sup_der'] = 0
            angulos['hombro_elevado_der'] = False
            angulos['brazo_abducido_der'] = False

        # --- ANTEBRAZO (ángulo en el codo) ---
        # Izquierdo
        if 5 in kp3d and 7 in kp3d and 9 in kp3d:
            angulo_codo = self._angulo_en_articulacion(kp3d[5], kp3d[7], kp3d[9])
            angulos['antebrazo_izq'] = 180 - angulo_codo  # Flexión desde extendido
        else:
            angulos['antebrazo_izq'] = 90  # Asumir neutral

        # Derecho
        if 6 in kp3d and 8 in kp3d and 10 in kp3d:
            angulo_codo = self._angulo_en_articulacion(kp3d[6], kp3d[8], kp3d[10])
            angulos['antebrazo_der'] = 180 - angulo_codo
        else:
            angulos['antebrazo_der'] = 90

        # --- MUÑECA ---
        # Sin keypoints de mano en COCO, aproximamos como neutral
        angulos['muneca_izq'] = 0
        angulos['muneca_der'] = 0
        angulos['giro_muneca'] = 1  # 1 = rango medio (asumido)

        return angulos

    def _angulos_lateral(self, kp3d, lado, tronco_silueta=None, keypoints_2d=None):
        """
        Calcula los ángulos usando SOLO los puntos reales del lado visible.

        De costado, la imagen ya ES el plano sagital: no hace falta proyectar
        nada ni estimar profundidad. Se evita por completo el punto medio
        17/18 (que mezclaría el lado real con el hombro/cadera fabricados del
        lado oculto) y se mide directamente hombro→cadera y hombro→oreja del
        lado visible.

        Nota: el elevador 3D asigna la MISMA profundidad Z a todos los puntos
        del torso y la cabeza (ver elevador_3d.py Paso 1), así que el vector
        hombro→cadera de un mismo lado ya tiene componente Z = 0 — no hace
        falta descartarlo explícitamente, el resultado es equivalente a
        medir en el plano de la imagen (X-Y).

        Args:
            kp3d: dict {idx: np.array([X, Y, Z])}
            lado: 'izq' o 'der' — el lado fiable detectado por _detectar_vista
            tronco_silueta: flexión de tronco por silueta (grados) o None.
            keypoints_2d: dict 2D, para saber si la cadera del lado es real.

        Returns:
            dict con ángulos en grados (mismo formato que _angulos_frontal).
        """
        angulos = {}
        vertical_arriba = np.array([0, -1, 0])

        idx = self._indices_lado(lado)
        h, cd, m, c, o = idx['hombro'], idx['codo'], idx['muneca'], idx['cadera'], idx['oreja']

        # --- TRONCO --- hombro → cadera del lado visible, vs vertical.
        # Si la cadera es fabricada (típico sentado: pelvis oculta tras el
        # escritorio), este vector sale casi vertical y NO refleja el
        # encorvamiento. En ese caso se usa la estimación por silueta de la
        # espalda (bordes Canny), que sí captura la inclinación del tronco.
        cadera_fabricada = (
            keypoints_2d is None
            or c not in keypoints_2d
            or keypoints_2d[c][2] <= CONF_FABRICADO
        )
        if h in kp3d and c in kp3d:
            vec_tronco = kp3d[h] - kp3d[c]
            tronco_kp = self._angulo_entre_vectores(vec_tronco, vertical_arriba)
        else:
            tronco_kp = 0

        if cadera_fabricada and tronco_silueta is not None:
            # Cadera oculta: la silueta es la única fuente fiable del tronco.
            angulos['tronco_flexion'] = tronco_silueta
        elif tronco_silueta is not None:
            # Cadera real: usar keypoints, pero no reportar menos que la silueta
            # (conservador, coherente con el "peor caso" del resto del RULA).
            angulos['tronco_flexion'] = max(tronco_kp, tronco_silueta)
        else:
            angulos['tronco_flexion'] = tronco_kp

        # --- CUELLO --- oreja → hombro del lado visible, vs vertical.
        # Las orejas nunca son fabricadas por el detector, así que si la del
        # lado fiable falta (p.ej. cabeza girada hacia la cámara) se usa la
        # otra oreja real. La medición NO depende de hacia dónde mire la cara.
        oreja_ref = None
        for idx_oreja in (o, 7 - o):  # oreja del lado fiable, luego la otra (3+4=7)
            if idx_oreja in kp3d:
                oreja_ref = kp3d[idx_oreja]
                break

        if oreja_ref is not None and h in kp3d:
            vec_cuello = oreja_ref - kp3d[h]
            angulo_cuello = self._angulo_entre_vectores(vec_cuello, vertical_arriba)
            angulos['cuello_flexion'] = angulo_cuello

            # En perfil la nariz queda DELANTE de la oreja: eso indica hacia
            # dónde mira la persona. Si la oreja queda por DETRÁS del hombro
            # (en contra de la mirada) la cabeza va hacia atrás => extensión.
            en_extension = False
            if 0 in kp3d:
                direccion_frente = np.sign(kp3d[0][0] - oreja_ref[0])
                lean = (oreja_ref[0] - kp3d[h][0]) * direccion_frente
                en_extension = lean < 0 and angulo_cuello > 8
            angulos['cuello_extension'] = en_extension
        else:
            angulos['cuello_flexion'] = 0
            angulos['cuello_extension'] = False

        # --- BRAZO SUPERIOR --- flexión relativa al eje del tronco (no al eje
        # hombro-hombro, que en lateral incluye un hombro fabricado)
        if h in kp3d and cd in kp3d and c in kp3d:
            vec_brazo = kp3d[cd] - kp3d[h]
            vec_tronco_abajo = kp3d[c] - kp3d[h]  # dirección "brazo colgando" = 0°
            angulo_brazo = self._angulo_entre_vectores(vec_brazo, vec_tronco_abajo)
        else:
            angulo_brazo = 0

        # Abducción y elevación de hombro requieren el hombro opuesto real:
        # en lateral ese punto es fabricado, así que no se evalúan (evita
        # falsos +1 en el score de brazo; ver PLAN_LATERAL.md causa B).
        angulos[f'brazo_sup_{lado}'] = angulo_brazo
        angulos[f'hombro_elevado_{lado}'] = False
        angulos[f'brazo_abducido_{lado}'] = False

        # --- ANTEBRAZO --- ángulo en el codo del lado visible (ya usaba solo
        # puntos reales, no cambia respecto a la versión frontal)
        if h in kp3d and cd in kp3d and m in kp3d:
            angulo_codo = self._angulo_en_articulacion(kp3d[h], kp3d[cd], kp3d[m])
            angulos[f'antebrazo_{lado}'] = 180 - angulo_codo
        else:
            angulos[f'antebrazo_{lado}'] = 90

        # Lado no fiable: se deja en neutral por si algo lo lee (HUD, etc.);
        # evaluar() nunca usa estas claves en vista lateral/ángulo.
        otro = 'der' if lado == 'izq' else 'izq'
        angulos[f'brazo_sup_{otro}'] = 0
        angulos[f'hombro_elevado_{otro}'] = False
        angulos[f'brazo_abducido_{otro}'] = False
        angulos[f'antebrazo_{otro}'] = 90

        angulos['muneca_izq'] = 0
        angulos['muneca_der'] = 0
        angulos['giro_muneca'] = 1

        return angulos

    # =========================================================================
    # MAPEO DE ÁNGULOS A SCORES RULA
    # =========================================================================

    @staticmethod
    def _score_brazo_superior(angulo, hombro_elevado=False, abducido=False):
        """Mapea ángulo del brazo superior a score RULA (1-6).
        
        Rangos adaptados para entorno de oficina donde 0-40° es la
        posición natural de trabajo con teclado/mouse.
        """
        if angulo <= 40:
            score = 1  # Rango natural de oficina: 0° - 40°
        elif angulo <= 60:
            score = 2  # Aceptable: 40° - 60°
        elif angulo <= 90:
            score = 3  # Investigar: 60° - 90°
        else:
            score = 4  # Cambio necesario: > 90°

        # Ajustes
        if hombro_elevado:
            score += 1
        if abducido:
            score += 1

        return min(score, 6)

    @staticmethod
    def _score_antebrazo(angulo_flexion):
        """Mapea ángulo del antebrazo (flexión en codo) a score RULA (1-3).
        
        Rango ampliado para oficina: personas usan teclado, mouse,
        teléfono con variedad de ángulos de codo.
        """
        if 45 <= angulo_flexion <= 120:
            return 1  # Rango amplio de oficina
        else:
            return 2  # Fuera de rango

    @staticmethod
    def _score_muneca(angulo):
        """Mapea ángulo de muñeca a score RULA (1-4)."""
        if angulo <= 5:
            return 1   # Neutral
        elif angulo <= 15:
            return 2   # Ligera flexión/extensión
        else:
            return 3   # Mayor flexión/extensión

    @staticmethod
    def _score_cuello(angulo_flexion, en_extension=False):
        """Mapea ángulo del cuello a score RULA (1-6).

        El límite del score 1 (12°) deja una pequeña tolerancia para el ruido
        de la estimación de profundidad (cabeza adelantada) en vista frontal.
        """
        if en_extension:
            return 4  # Cuello en extensión
        elif angulo_flexion <= 12:
            return 1  # 0° - 12° (neutral)
        elif angulo_flexion <= 22:
            return 2  # 12° - 22°
        else:
            return 3  # > 22° flexión (cabeza caída o muy adelantada)

    @staticmethod
    def _score_tronco(angulo_flexion):
        """Mapea ángulo del tronco a score RULA (1-6).

        Umbrales ajustados para postura sentada de oficina, donde el ángulo del
        tronco ahora incorpora la inclinación sagital estimada por escorzo. El
        límite del score 1 (10°) absorbe el ruido propio de esa estimación.
        """
        if angulo_flexion <= 10:
            return 1   # Sentado erguido (con tolerancia a ruido)
        elif angulo_flexion <= 20:
            return 2   # Ligera inclinación
        elif angulo_flexion <= 45:
            return 3   # Encorvado moderado
        else:
            return 4   # Encorvado severo

    # =========================================================================
    # EVALUACIÓN COMPLETA
    # =========================================================================

    def evaluar(self, keypoints_3d, keypoints_2d=None, frame=None):
        """
        Evalúa la postura completa según el método RULA.

        Args:
            keypoints_3d: dict {idx: np.array([X, Y, Z])} de la Fase 2.
            keypoints_2d: dict {idx: (x, y, conf)} de la Fase 1. Se usa para
                detectar el ángulo de cámara (frontal/lateral/ángulo) y elegir
                la ruta de medición correcta. Si se omite, se asume frontal.
            frame: imagen BGR original (sin dibujos). Opcional. Si se provee y
                la vista es lateral, se usa para estimar el encorvamiento del
                tronco por silueta (cuando la cadera está oculta/fabricada).

        Returns:
            dict con:
                'angulos': ángulos calculados (grados)
                'scores': scores individuales por segmento
                'score_a': score del Grupo A (brazo/muñeca)
                'score_b': score del Grupo B (cuello/tronco/piernas)
                'score_final': score RULA final (1-7)
                'nivel': nivel de acción RULA
                'texto': descripción del nivel
                'color_key': clave de color para visualización
                'vista': 'frontal' | 'lateral' | 'angulo'
                'lado_fiable': 'izq' | 'der' | None
        """
        if keypoints_3d is None:
            return None

        # Detectar ángulo de cámara
        vista, lado_fiable = self._detectar_vista(keypoints_2d)

        # En vista lateral, estimar el encorvamiento del tronco por silueta
        # (bordes) cuando la cadera no da información fiable.
        tronco_silueta = None
        if (self._analizador_silueta is not None and frame is not None
                and vista in ('lateral', 'angulo') and lado_fiable is not None):
            tronco_silueta = self._analizador_silueta.analizar(
                frame, keypoints_2d, lado_fiable
            )

        # Calcular ángulos 3D por la ruta correcta
        angulos = self._calcular_angulos(
            keypoints_3d, vista, lado_fiable, tronco_silueta, keypoints_2d
        )

        # =====================================================================
        # GRUPO A: Brazo, Antebrazo, Muñeca
        # =====================================================================
        if vista in ('lateral', 'angulo') and lado_fiable is not None:
            # Vista lateral/ángulo: usar directamente el lado fiable medido
            # con puntos reales (ver _angulos_lateral). No mezclar con el
            # lado oculto, cuyos keypoints son fabricados.
            sufijo = lado_fiable
            score_brazo = self._score_brazo_superior(
                angulos[f'brazo_sup_{sufijo}'],
                angulos.get(f'hombro_elevado_{sufijo}', False),
                angulos.get(f'brazo_abducido_{sufijo}', False)
            )
            score_antebrazo = self._score_antebrazo(angulos[f'antebrazo_{sufijo}'])
        else:
            # Vista frontal: evaluar ambos lados y elegir según confianza
            # (ambos confiables → el peor caso, más conservador)
            score_brazo_izq = self._score_brazo_superior(
                angulos['brazo_sup_izq'],
                angulos.get('hombro_elevado_izq', False),
                angulos.get('brazo_abducido_izq', False)
            )
            score_brazo_der = self._score_brazo_superior(
                angulos['brazo_sup_der'],
                angulos.get('hombro_elevado_der', False),
                angulos.get('brazo_abducido_der', False)
            )

            conf_izq = self._confianza_lado(keypoints_3d, 'izq')
            conf_der = self._confianza_lado(keypoints_3d, 'der')

            score_brazo = self._seleccionar_por_confianza(
                score_brazo_izq, conf_izq, score_brazo_der, conf_der
            )

            score_antebrazo_izq = self._score_antebrazo(angulos['antebrazo_izq'])
            score_antebrazo_der = self._score_antebrazo(angulos['antebrazo_der'])
            score_antebrazo = self._seleccionar_por_confianza(
                score_antebrazo_izq, conf_izq, score_antebrazo_der, conf_der
            )

        score_muneca = self._score_muneca(angulos['muneca_izq'])
        giro_muneca = angulos['giro_muneca']

        # Lookup Tabla A (índices 0-based)
        idx_brazo = min(score_brazo, 6) - 1
        idx_antebrazo = min(score_antebrazo, 3) - 1
        idx_muneca = min(score_muneca, 4) - 1
        idx_giro = min(giro_muneca, 2) - 1

        postura_a = int(self.TABLA_A[idx_brazo][idx_antebrazo][idx_muneca][idx_giro])

        # Agregar uso muscular (+1 si postura estática mantenida)
        # Para análisis en tiempo real, asumimos postura estática
        uso_muscular = 1
        score_a = postura_a + uso_muscular

        # =====================================================================
        # GRUPO B: Cuello, Tronco, Piernas
        # =====================================================================
        score_cuello = self._score_cuello(
            angulos['cuello_flexion'],
            angulos.get('cuello_extension', False)
        )
        score_tronco = self._score_tronco(angulos['tronco_flexion'])
        score_piernas = 1  # Asumimos piernas apoyadas (sentado)

        # Lookup Tabla B (índices 0-based)
        idx_cuello = min(score_cuello, 6) - 1
        idx_tronco = min(score_tronco, 6) - 1
        idx_piernas = min(score_piernas, 2) - 1

        postura_b = int(self.TABLA_B[idx_cuello][idx_tronco][idx_piernas])

        # Agregar uso muscular
        score_b = postura_b + uso_muscular

        # =====================================================================
        # TABLA C: Score Final
        # =====================================================================
        idx_sa = min(score_a, 8) - 1
        idx_sb = min(score_b, 7) - 1

        score_final = int(self.TABLA_C[idx_sa][idx_sb])

        # Determinar nivel de acción
        nivel_key = min(score_final, 7)
        nivel_info = self.NIVELES[nivel_key]

        return {
            'angulos': angulos,
            'scores': {
                'brazo_superior': score_brazo,
                'antebrazo': score_antebrazo,
                'muneca': score_muneca,
                'cuello': score_cuello,
                'tronco': score_tronco,
                'piernas': score_piernas,
            },
            'score_a': score_a,
            'score_b': score_b,
            'score_final': score_final,
            'nivel': nivel_info['nivel'],
            'texto': nivel_info['texto'],
            'color_key': nivel_info['color_key'],
            'vista': vista,
            'lado_fiable': lado_fiable,
        }
