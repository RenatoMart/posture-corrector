"""
filtro_oneeuro.py — Filtro One-Euro para estabilizar landmarks (anti-jitter).

El filtro One-Euro (Casiez, Roussel & Vogel, 2012) resuelve el compromiso
clásico entre TEMBLOR y RETRASO que tiene una media móvil de alpha fijo (EMA):

    - Con un punto casi quieto usa un cutoff bajo → suaviza fuerte y elimina el
      jitter (el temblor de los landmarks cuando estás inmóvil).
    - Cuando el punto se mueve rápido sube el cutoff → deja pasar el movimiento
      sin añadir "lag" ni arrastre.

Es el estándar de facto para señales de tracking en tiempo real. A diferencia de
la EMA, no hay que elegir entre "suave pero con retraso" o "reactivo pero
tembloroso": se adapta solo a la velocidad de cada punto.
"""

import time
import numpy as np


def _alpha(cutoff, dt):
    """Coeficiente de un pasa-bajos de primer orden para el cutoff (Hz) y dt (s)."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class FiltroOneEuro:
    """
    Filtro One-Euro para una señal vectorial (cada componente se filtra aparte).

    Args:
        min_cutoff: frecuencia de corte mínima (Hz). Menor = más suave en reposo.
        beta: cuánto sube el cutoff con la velocidad. Mayor = reacciona antes.
        d_cutoff: cutoff del filtro de la derivada (Hz). Se suele dejar en 1.0.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def reset(self):
        """Olvida el estado (usar cuando un punto reaparece tras ausentarse)."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t=None):
        x = np.asarray(x, dtype=float)
        if t is None:
            t = time.perf_counter()

        # Primer valor: no hay con qué comparar; se adopta tal cual.
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3  # protección ante timestamps no crecientes

        # Derivada (velocidad) suavizada de la señal
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Cutoff adaptativo POR COMPONENTE según la velocidad medida
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat.copy()


class EstabilizadorPose:
    """
    Aplica un filtro One-Euro independiente a cada keypoint de una pose.

    Sirve tanto para poses 2D {idx: (x, y, conf)} como 3D {idx: np.array([X,Y,Z])}.
    Solo suaviza las coordenadas; cualquier valor extra por punto (confianza o
    visibilidad) se conserva sin tocar. Un keypoint que desaparece y reaparece
    reinicia su filtro para no arrastrar una posición vieja (evita el "salto").

    Args:
        min_cutoff, beta, d_cutoff: parámetros del One-Euro (ver FiltroOneEuro).
        n_coords: nº de coordenadas iniciales a filtrar (2 para 2D en píxeles,
            para no tocar la confianza; None = filtrar todas, típico en 3D).
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0, n_coords=None):
        self._cfg = (min_cutoff, beta, d_cutoff)
        self._n_coords = n_coords
        self._filtros = {}
        self._vistos = set()

    def reset(self):
        """Reinicia todos los filtros (tecla 'C' / recalibración)."""
        for f in self._filtros.values():
            f.reset()
        self._vistos = set()

    def __call__(self, pose):
        if not pose:
            self._vistos = set()
            return pose

        t = time.perf_counter()
        presentes = set(pose.keys())

        # Reiniciar el filtro de los puntos que reaparecen tras haber faltado
        for idx in presentes - self._vistos:
            if idx in self._filtros:
                self._filtros[idx].reset()

        salida = {}
        for idx, valor in pose.items():
            es_secuencia = isinstance(valor, (tuple, list))
            arr = np.asarray(valor, dtype=float)
            n = arr.size if self._n_coords is None else min(self._n_coords, arr.size)

            if idx not in self._filtros:
                self._filtros[idx] = FiltroOneEuro(*self._cfg)

            suave = arr.copy()
            suave[:n] = self._filtros[idx](arr[:n], t)

            salida[idx] = tuple(suave.tolist()) if es_secuencia else suave

        self._vistos = presentes
        return salida
