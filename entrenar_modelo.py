"""
entrenar_modelo.py — Entrena el Random Forest de detección de encorvamiento.

Lee el CSV que generó `recolectar_datos.py`, entrena un RandomForestClassifier
de scikit-learn y guarda el modelo en `modelo_postura.joblib`. Ese archivo es el
que carga el sistema en vivo (`clasificador_postura.py`).

Uso:
    python entrenar_modelo.py

Qué reporta (léelo, es donde aprendes si vas bien):
    - Exactitud por validación cruzada POR SESIÓN (sin fugas de datos).
    - Matriz de confusión y precisión/recall de la clase "encorvado".
    - Importancia de cada feature: qué ángulos distinguen de verdad la postura.

Validación honesta
-------------------
Si hay varias sesiones (recomendado), se valida con GroupKFold agrupando por
sesión: el modelo se prueba SIEMPRE en sesiones que no vio al entrenar. Así la
exactitud reportada refleja cómo generalizará a un momento/persona nuevos, y no
el sobreajuste a tu fondo o tu ropa de una sola grabación.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from features_postura import FEATURE_NAMES

try:
    from config import SESIONES_EXCLUIDAS
except ImportError:      # config sin la constante (versiones antiguas)
    SESIONES_EXCLUIDAS = []

_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, 'datos_postura.csv')
MODELO_PATH = os.path.join(_DIR, 'modelo_postura.joblib')


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] No existe {CSV_PATH}. Corre primero recolectar_datos.py.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"[Datos] {len(df)} filas leídas de {os.path.basename(CSV_PATH)}")

    # Sesiones excluidas a propósito (config.SESIONES_EXCLUIDAS). Las filas siguen
    # en el CSV; solo se dejan fuera del entrenamiento. Se avisa SIEMPRE, para que
    # nadie interprete mal las métricas creyendo que se entrenó con todo.
    if SESIONES_EXCLUIDAS and 'sesion' in df.columns:
        presentes = [s for s in SESIONES_EXCLUIDAS if (df['sesion'] == s).any()]
        ausentes = [s for s in SESIONES_EXCLUIDAS if s not in presentes]
        if presentes:
            n_antes = len(df)
            df = df[~df['sesion'].isin(presentes)]
            print(f"[Datos] EXCLUIDAS {n_antes - len(df)} filas de "
                  f"{len(presentes)} sesión(es) por config.SESIONES_EXCLUIDAS: "
                  f"{', '.join(presentes)}")
            print("        (las filas siguen en el CSV; quita el id de la lista "
                  "para volver a incluirlas)")
        if ausentes:
            print(f"[AVISO] Estas sesiones de SESIONES_EXCLUIDAS no existen en el "
                  f"CSV: {', '.join(ausentes)}")
        if df.empty:
            print("[ERROR] No quedan filas tras excluir sesiones.")
            sys.exit(1)

    # Validar columnas
    faltan = [c for c in FEATURE_NAMES + ['label'] if c not in df.columns]
    if faltan:
        print(f"[ERROR] Faltan columnas en el CSV: {faltan}")
        sys.exit(1)

    df = df.dropna(subset=FEATURE_NAMES + ['label'])
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df['label'].to_numpy(dtype=int)
    grupos = df['sesion'].to_numpy() if 'sesion' in df.columns else None

    # Reparto de clases
    n_erg = int((y == 0).sum())
    n_enc = int((y == 1).sum())
    print(f"[Datos] erguido(0): {n_erg}   encorvado(1): {n_enc}")
    if n_erg < 30 or n_enc < 30:
        print("[AVISO] Muy pocos datos de alguna clase. Junta más antes de "
              "confiar en el modelo (idealmente 300+ por clase).")

    n_sesiones = len(np.unique(grupos)) if grupos is not None else 1
    print(f"[Datos] sesiones distintas: {n_sesiones}")

    # --- Validación cruzada honesta ---
    modelo_cv = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    if n_sesiones >= 2:
        n_splits = min(5, n_sesiones)
        cv = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(modelo_cv, X, y, groups=grupos, cv=cv)
        print(f"\n[Validación] GroupKFold por sesión ({n_splits} folds)")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(modelo_cv, X, y, cv=cv)
        print("\n[Validación] StratifiedKFold (5 folds)")
        print("  [AVISO] Solo hay 1 sesión: esta exactitud puede ser optimista "
              "(el modelo ve tu mismo fondo/ropa). Graba en varias sesiones.")

    print("\n--- Matriz de confusión (filas=real, columnas=predicho) ---")
    print("            pred_erguido  pred_encorvado")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print(f"real_erguido      {cm[0][0]:6d}        {cm[0][1]:6d}")
    print(f"real_encorvado    {cm[1][0]:6d}        {cm[1][1]:6d}")

    print("\n--- Reporte de clasificación ---")
    print(classification_report(y, y_pred, labels=[0, 1],
                                target_names=['erguido', 'encorvado'],
                                zero_division=0))

    # --- Modelo FINAL entrenado con todos los datos ---
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    modelo.fit(X, y)

    print("--- Importancia de features (qué distingue la postura) ---")
    orden = np.argsort(modelo.feature_importances_)[::-1]
    for i in orden:
        print(f"  {FEATURE_NAMES[i]:26s} {modelo.feature_importances_[i]:.3f}")

    # Guardar modelo + metadatos (el orden de features viaja con el modelo)
    joblib.dump({
        'modelo': modelo,
        'features': FEATURE_NAMES,
        'clases': {0: 'erguido', 1: 'encorvado'},
        'n_muestras': len(df),
    }, MODELO_PATH)
    print(f"\n[OK] Modelo guardado en {MODELO_PATH}")
    print("     Ya puedes usarlo en vivo (clasificador_postura.py / Postura.py).")


if __name__ == '__main__':
    main()
