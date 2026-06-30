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
        pass

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

    def _calcular_angulos(self, kp3d):
        """
        Calcula todos los ángulos necesarios para RULA a partir de keypoints 3D.
        
        Args:
            kp3d: dict {idx: np.array([X, Y, Z])}
                  Índices 17 = cuello virtual, 18 = centro cadera virtual
        
        Returns:
            dict con ángulos en grados para cada segmento.
        """
        angulos = {}

        # Vector vertical (gravedad apunta hacia abajo = Y positivo)
        vertical = np.array([0, 1, 0])
        vertical_arriba = np.array([0, -1, 0])

        # --- CUELLO ---
        # Ángulo de inclinación de la cabeza respecto a la vertical.
        # Usa orejas como referencia primaria (independiente de dirección de mirada).
        # La nariz solo se usa como fallback si no hay orejas disponibles.
        if 17 in kp3d:
            cuello = kp3d[17]
            
            # Determinar punto de referencia de la cabeza (prioridad: orejas > nariz)
            cabeza = None
            if 3 in kp3d and 4 in kp3d:
                # Mejor caso: punto medio entre orejas
                cabeza = (kp3d[3] + kp3d[4]) / 2
            elif 3 in kp3d:
                cabeza = kp3d[3]  # Solo oreja izquierda
            elif 4 in kp3d:
                cabeza = kp3d[4]  # Solo oreja derecha
            elif 0 in kp3d:
                cabeza = kp3d[0]  # Fallback: nariz
            
            if cabeza is not None:
                vec_cuello = cabeza - cuello
                angulo_cuello = self._angulo_entre_vectores(vec_cuello, vertical_arriba)
                angulos['cuello_flexion'] = angulo_cuello
                
                # Extensión del cuello: cabeza por detrás/encima de la línea del cuello
                # Usamos Y (vertical) en lugar de Z, ya que Y es independiente del
                # ángulo de cámara. Si la cabeza está más arriba de lo normal = extensión
                vec_tronco_ref = kp3d[18] - cuello if 18 in kp3d else vertical
                # Proyección del vector cuello en la dirección opuesta al tronco
                angulos['cuello_extension'] = vec_cuello[1] < cuello[1] * 0.1 and angulo_cuello > 15
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
        """Mapea ángulo del cuello a score RULA (1-6)."""
        if en_extension:
            return 4  # Cuello en extensión
        elif angulo_flexion <= 10:
            return 1  # 0° - 10°
        elif angulo_flexion <= 20:
            return 2  # 10° - 20°
        else:
            return 3  # > 20° flexión

    @staticmethod
    def _score_tronco(angulo_flexion):
        """Mapea ángulo del tronco a score RULA (1-6)."""
        if angulo_flexion <= 5:
            return 1   # Neutral (sentado derecho)
        elif angulo_flexion <= 20:
            return 2   # 0° - 20° flexión
        elif angulo_flexion <= 60:
            return 3   # 20° - 60° flexión
        else:
            return 4   # > 60° flexión

    # =========================================================================
    # EVALUACIÓN COMPLETA
    # =========================================================================

    def evaluar(self, keypoints_3d):
        """
        Evalúa la postura completa según el método RULA.

        Args:
            keypoints_3d: dict {idx: np.array([X, Y, Z])} de la Fase 2.

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
        """
        if keypoints_3d is None:
            return None

        # Calcular ángulos 3D
        angulos = self._calcular_angulos(keypoints_3d)

        # =====================================================================
        # GRUPO A: Brazo, Antebrazo, Muñeca
        # =====================================================================
        # Evaluar ambos lados. En vistas laterales, un lado puede tener
        # keypoints estimados (baja confianza) — priorizar el lado confiable.
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
        
        # Determinar confianza de cada lado
        conf_izq = self._confianza_lado(keypoints_3d, 'izq')
        conf_der = self._confianza_lado(keypoints_3d, 'der')
        
        # Seleccionar score del brazo: preferir el lado con mayor confianza
        # Si ambos son confiables, usar el peor (más conservador)
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
        }
