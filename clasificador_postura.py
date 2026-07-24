"""
clasificador_postura.py — Inferencia en vivo del detector de encorvamiento.

Carga el modelo entrenado (`modelo_postura.joblib`) y, dado un frame de pose,
devuelve la probabilidad de "encorvado" ya suavizada en el tiempo y con
histéresis (porque el encorvamiento es una postura sostenida, no un parpadeo).

Usa `features_postura.vector_features`, la MISMA función con la que se recolectó
y se entrenó: así el modelo recibe exactamente los números que aprendió.

Uso típico (dentro del bucle de Postura.py):

    clf = ClasificadorPostura()          # carga el modelo si existe
    ...
    res = clf.predecir(kp3d)             # {'prob':.., 'encorvado':bool} o None

Si no hay modelo entrenado todavía, `disponible` es False y `predecir` devuelve
None, para que el sistema siga funcionando con la lógica RULA de siempre.
"""

import os

import numpy as np

from features_postura import FEATURE_NAMES, vector_features

try:
    import joblib
    _JOBLIB_OK = True
except Exception:
    _JOBLIB_OK = False

_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(_DIR, 'modelo_postura.joblib')


class ClasificadorPostura:
    """
    Envuelve el Random Forest entrenado y añade suavizado temporal e histéresis.

    Args:
        umbral_on: probabilidad a partir de la cual se ENCIENDE la señal de
            encorvado.
        umbral_off: probabilidad por debajo de la cual se APAGA (más baja que
            umbral_on para dar histéresis y evitar parpadeos).
        alpha: factor del suavizado exponencial (EMA) de la probabilidad.
        activo: False desactiva el detector aunque el .joblib exista (config.ML_ACTIVO).
    """

    def __init__(self, ruta=MODELO_PATH, umbral_on=0.6, umbral_off=0.4, alpha=0.3,
                 activo=True):
        self.umbral_on = umbral_on
        self.umbral_off = umbral_off
        self.alpha = alpha

        self._prob_suave = None
        self._encorvado = False

        self.modelo = None
        self.features = None
        self.n_muestras = None
        self.disponible = False

        if not activo:
            print("[ClasificadorPostura] Desactivado por config (ML_ACTIVO=False). "
                  "El sistema usará solo RULA.")
        elif _JOBLIB_OK and os.path.exists(ruta):
            try:
                data = joblib.load(ruta)
                self.modelo = data['modelo']
                self.features = data.get('features')
                self.n_muestras = data.get('n_muestras')
                self.disponible = True
                print(f"[ClasificadorPostura] Modelo cargado "
                      f"({data.get('n_muestras', '?')} muestras de entrenamiento).")
                # Guarda contra el fallo SILENCIOSO más peligroso de este flujo:
                # que el modelo se haya entrenado con otras features (u otro
                # orden) que las que calcula features_postura.py ahora mismo. En
                # ese caso el RF recibiría números que no corresponden a las
                # columnas que aprendió y acertaría por casualidad.
                if self.features is not None and list(self.features) != list(FEATURE_NAMES):
                    print("[ClasificadorPostura] AVISO: las features del modelo NO "
                          "coinciden con features_postura.py.")
                    print(f"    modelo : {list(self.features)}")
                    print(f"    actual : {list(FEATURE_NAMES)}")
                    print("    → Vuelve a correr entrenar_modelo.py. Se desactiva el ML.")
                    self.disponible = False
            except Exception as e:
                print(f"[ClasificadorPostura] No se pudo cargar el modelo: {e}")
        else:
            print("[ClasificadorPostura] Sin modelo entrenado "
                  "(corre entrenar_modelo.py). El sistema usará solo RULA.")

    def reset(self):
        """Reinicia el suavizado/histéresis (p.ej. al recalibrar con 'C')."""
        self._prob_suave = None
        self._encorvado = False

    def predecir(self, kp3d):
        """
        Devuelve el veredicto de encorvamiento para la pose actual.

        Returns:
            dict {'prob': float 0-1, 'prob_cruda': float, 'encorvado': bool} o
            None si no hay modelo o la pose no da features válidas.
        """
        if not self.disponible:
            return None

        vec = vector_features(kp3d)
        if vec is None:
            return None

        # Probabilidad de la clase 1 (encorvado)
        prob = float(self.modelo.predict_proba(vec.reshape(1, -1))[0][1])

        # Suavizado temporal (EMA)
        if self._prob_suave is None:
            self._prob_suave = prob
        else:
            self._prob_suave = self.alpha * prob + (1 - self.alpha) * self._prob_suave

        # Histéresis: encender/apagar con umbrales distintos
        if self._encorvado:
            if self._prob_suave < self.umbral_off:
                self._encorvado = False
        else:
            if self._prob_suave > self.umbral_on:
                self._encorvado = True

        return {
            'prob': self._prob_suave,
            'prob_cruda': prob,
            'encorvado': self._encorvado,
        }
