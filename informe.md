# Informe Técnico — Sistema de Análisis de Postura Ergonómica (RULA) en Tiempo Real

> Documento técnico exhaustivo del proyecto **Postura**: herramientas, librerías,
> arquitectura, flujo de datos, funciones y las técnicas de visión por computador,
> biomecánica y aprendizaje automático que sustentan cada fase.

---

## 1. Resumen del sistema

El proyecto captura vídeo de una webcam estándar, detecta la postura de la
persona frame a frame y calcula un **score ergonómico RULA** (Rapid Upper Limb
Assessment) en tiempo real, mostrándolo sobre la imagen y emitiendo una alarma
sonora cuando se sostiene una mala postura. En paralelo, un **detector de
encorvamiento por Machine Learning** (Random Forest entrenado con datos propios)
clasifica la postura como *erguida* o *encorvada*.

La arquitectura es una **tubería (pipeline) de 3 fases** que parte de la pose 3D
real de MediaPipe, con módulos auxiliares de visualización, detección de
encorvamiento por silueta y clasificación por aprendizaje automático:

```
        ┌───────────────┐  keypoints 2D  ┌──────────────┐  keypoints 3D   ┌────────────────┐
 frame  │  Fase 1       │  {idx:(x,y,v)} │  Fase 2      │  {idx:[X,Y,Z]}  │  Fase 3        │  resultado
 ─────▶ │ Detector2D    │ ──────────────▶│ Elevador3D   │ ──────────────▶ │ EvaluadorRULA  │ ─────────▶
 (BGR)  │ (MediaPipe)   │  + world 3D    │ (ensamblado) │        │        │ (RULA + vista) │  (dict)
        └───────┬───────┘                └──────────────┘        │        └───────┬────────┘
                │                                                 │                │ frame (lateral)
                │                                       ┌─────────▼────────┐  ┌────▼───────────────┐
                │                                       │ ClasificadorPost.│  │ AnalizadorSilueta   │
                │                                       │ (Random Forest)  │  │ Lateral (Canny)     │
                │                                       └──────────────────┘  └─────────────────────┘
                ▼
        ┌───────────────┐
        │ Visualizador  │  ← dibuja esqueleto, ángulos, HUD, FPS sobre el frame
        └───────────────┘
```

Cada fase transforma la representación de los datos: **píxeles → puntos 2D +
pose 3D real → ángulos articulares → scores RULA → nivel de riesgo**; y en la
rama ML: **pose 3D → ~13 features geométricas → probabilidad de encorvamiento**.

---

## 2. Herramientas, librerías y entorno

### 2.1 Lenguaje y entorno de ejecución

| Elemento | Versión / valor | Notas |
|---|---|---|
| Lenguaje | Python 3.12 | El intérprete probado es CPython 3.12.3 |
| SO objetivo | Windows 11 | Depende de `winsound` (nativa de Windows) para la alarma; la cámara usa el backend **DirectShow** |
| Entorno virtual | `.venv` (Python 3.12) | Aislamiento de dependencias; ejecutable `.venv/Scripts/python.exe` |

### 2.2 Librerías de terceros

| Librería | Versión instalada | Rol en el proyecto |
|---|---|---|
| **mediapipe** | 0.10.35 | Estimación de pose (**Pose Landmarker**, Tasks API): 33 landmarks 2D + **world landmarks 3D reales** con profundidad |
| **opencv-python** (`cv2`) | 5.0.0 | Captura de vídeo, procesamiento de imagen (CLAHE, espacios de color, Canny, blur) y renderizado (líneas, círculos, texto, transparencias) |
| **numpy** | 2.5.1 | Álgebra vectorial (producto punto, normas), operaciones matriciales y manejo eficiente de los tensores de keypoints y features |
| **scikit-learn** | 1.9.0 | Detector ML: `RandomForestClassifier`, validación cruzada `GroupKFold`, métricas |
| **pandas** | 3.0.3 | Lectura y manejo del dataset `datos_postura.csv` durante el entrenamiento |
| **joblib** | 1.5.3 | Serialización del modelo entrenado (`modelo_postura.joblib`) |

> **Nota sobre YOLO (histórico):** el sistema **migró de YOLOv8-Pose a MediaPipe**.
> Ya **no** se usan `ultralytics` ni `torch`. El fichero `yolov8n-pose.pt` quedó
> obsoleto y puede borrarse.

### 2.3 Librerías estándar de Python

| Módulo | Uso |
|---|---|
| `winsound` | Emite el tono de alarma (`Beep`) en Windows |
| `threading` | Ejecuta la alarma en un hilo aparte para no bloquear el vídeo |
| `os`, `sys`, `contextlib` | Silenciado del stderr del motor C++ de MediaPipe durante la carga |
| `urllib.request` | Descarga automática del modelo `.task` la primera vez |
| `csv`, `datetime` | Recolección etiquetada de datos (CSV con marca de sesión) |

### 2.4 Modelo de estimación de pose

- **MediaPipe Pose Landmarker** (archivo `.task`, hasta ~30 MB): red que devuelve
  **33 landmarks** anatómicos por persona, cada uno con:
  - Coordenadas 2D normalizadas `(x, y)` en la imagen, más `visibility` y
    `presence` por punto.
  - **World landmarks**: coordenadas **3D reales en metros** (`X, Y, Z`), con el
    origen en el centro de las caderas. Esta es la diferencia clave con YOLO: la
    profundidad **viene medida**, no estimada.
- Se usa la **Tasks API moderna** (`mediapipe.tasks.python.vision.PoseLandmarker`,
  modo `VIDEO`), ya que la antigua `mp.solutions.pose` fue **retirada** en
  mediapipe 0.10.35. El modelo rastrea a **una sola persona** de forma nativa.
- Se descarga automáticamente la primera vez según `MP_MODEL_COMPLEXITY`
  (`0` lite / `1` full / `2` heavy) y se guarda junto al código.

### 2.5 Conjunto de keypoints COCO (17 puntos)

Los 33 landmarks de MediaPipe se mapean a los **17 índices estilo COCO** que ya
usaba el resto del sistema (ver `_MP_A_COCO` en `detector_2d.py`), por lo que el
evaluador y el visualizador siguen operando sin cambios:

```
0 nariz     1 ojo_izq    2 ojo_der    3 oreja_izq   4 oreja_der
5 hombro_izq 6 hombro_der 7 codo_izq   8 codo_der    9 muñeca_izq
10 muñeca_der 11 cadera_izq 12 cadera_der 13 rodilla_izq 14 rodilla_der
15 tobillo_izq 16 tobillo_der
```

Además, el sistema crea **dos keypoints virtuales** en 3D:
- `17` = **cuello** (punto medio entre hombros).
- `18` = **centro de cadera** (punto medio entre caderas).

---

## 3. Estructura de archivos y responsabilidades

| Archivo | Clase / función principal | Responsabilidad |
|---|---|---|
| `Postura.py` | `main()` | Orquestador: cámara robusta, bucle principal, control de alarma, integración ML, teclas |
| `config.py` | (constantes) | Configuración central: keypoints, umbrales MediaPipe, cámara, alarma, colores, parámetros |
| `detector_2d.py` | `Detector2D` | **Fase 1**: inferencia MediaPipe, preprocesamiento CLAHE, salida 2D + 3D real |
| `elevador_3d.py` | `Elevador3D` | **Fase 2**: ensamblado de la pose 3D (puntos virtuales + suavizado) |
| `evaluador_rula.py` | `EvaluadorRULA` | **Fase 3**: detección de vista, cálculo de ángulos, tablas RULA, score final |
| `detector_encorvamiento.py` | `AnalizadorSiluetaLateral` | Encorvamiento lateral por bordes (Canny) |
| `filtro_oneeuro.py` | `EstabilizadorPose` | Filtro One-Euro para suavizar pose 2D y 3D |
| `visualizador.py` | `Visualizador` | Renderizado del esqueleto, ángulos, HUD, FPS |
| **`features_postura.py`** | `vector_features`, `extraer_features` | **Detector ML**: extracción de las ~13 features geométricas (fuente única) |
| **`recolectar_datos.py`** | `main()` | **Detector ML**: captura etiquetada de datos en vivo → `datos_postura.csv` |
| **`entrenar_modelo.py`** | `main()` | **Detector ML**: entrena el Random Forest → `modelo_postura.joblib` |
| **`clasificador_postura.py`** | `ClasificadorPostura` | **Detector ML**: inferencia en vivo (probabilidad de encorvamiento) |

---

## 4. Fase 1 — Detección de pose (`detector_2d.py`)

Clase **`Detector2D`**. Convierte un frame BGR en dos representaciones alineadas
por índice COCO: la pose **2D en píxeles** (para dibujar y clasificar la vista) y
la pose **3D real en cm** (world landmarks, para el evaluador y el clasificador).

### 4.1 Preprocesamiento CLAHE — `_preprocesar_frame()`

**Técnica: CLAHE (Contrast Limited Adaptive Histogram Equalization).**

- La imagen BGR se convierte al espacio de color **LAB** (`cv2.cvtColor`), que
  separa la luminosidad (canal **L**) del color (canales **a**, **b**).
- CLAHE (`cv2.createCLAHE`) se aplica **solo al canal L**, mejorando el contraste
  local en zonas oscuras/contraluz **sin distorsionar los colores**.
- El parámetro `clipLimit` limita la amplificación del ruido; `tileGridSize`
  define la grilla de ecualización adaptativa.
- El objeto CLAHE se crea **una sola vez** en el constructor (reutilizable). Se
  activa/desactiva con `PREPROCESAMIENTO_ACTIVO`. Coste ~1-2 ms/frame.

### 4.2 Inferencia MediaPipe — `detectar()`

- El frame RGB se envuelve en un `mp.Image` y se procesa con
  `landmarker.detect_for_video(mp_image, ts_ms)`. El modo VIDEO exige
  **timestamps estrictamente crecientes** (se incrementan 33 ms por frame).
- MediaPipe rastrea **una sola persona** de forma nativa entre frames, por lo que
  **ya no hace falta el tracker por centroide** del detector YOLO anterior, ni la
  **fabricación anatómica de puntos ocluidos** (la pose 3D los infiere sola).
- **Delegado GPU/CPU con caída automática** (`_crear_landmarker`): si se pide GPU
  y el build de MediaPipe no la soporta (caso de Windows, *GPU processing is
  disabled in build flags*), se captura la excepción y se reintenta con CPU
  avisando por consola. Nunca deja el sistema sin detector.
- El motor C++ escupe avisos informativos (XNNPACK, feedback manager, absl) que
  **no son errores**; se silencian con variables de entorno (`GLOG_minloglevel`,
  `TF_CPP_MIN_LOG_LEVEL`) fijadas antes de importar mediapipe, más un redirector
  temporal del descriptor de stderr durante la carga del modelo.

### 4.3 Filtrado desacoplado 2D / 3D (clave para el primer plano)

Cada uno de los 33→17 landmarks trae una **presencia** `presence ∈ [0,1]`
(probabilidad de que el punto esté **dentro del encuadre**). El detector la usa de
forma **distinta para 2D y para 3D**, y esta separación es lo que permite medir la
postura cuando la persona está **muy cerca** de la cámara:

- **Puntos 2D** (para dibujar el esqueleto y clasificar la vista): se conservan
  solo si `presence ≥ MP_PRESENCE_MIN` (=0.5). No tiene sentido dibujar un punto
  que está fuera de cuadro.
- **Puntos 3D (world landmarks):** se conservan con un umbral **mucho más bajo**,
  `presence ≥ MP_PRESENCE_MIN_3D` (=0.0, es decir, **todos**). MediaPipe estima el
  **esqueleto 3D completo** aunque un punto esté fuera de cuadro (lo *infiere* por
  proporciones humanas). Conservar esos puntos inferidos permite reconstruir el
  tronco y las caderas cuando la persona está tan cerca que solo se le ve cabeza,
  hombros y parte del torso → así la Fase 3 (RULA) y el detector ML **siguen
  funcionando** en primer plano en lugar de reportar "sin pose".

> `visibility` se guarda igual en el slot de confianza de los puntos 2D, porque el
> evaluador la usa para **clasificar la vista** (frente / lado).

### 4.4 Salida

`detectar()` exige como mínimo **un hombro en 2D** (índice 5 o 6); si no, devuelve
`(None, None)`. En caso normal devuelve `(keypoints_2d, keypoints_world)`, con
las coordenadas 3D ya convertidas de metros a **centímetros** (`× 100`).

---

## 5. Fase 2 — Ensamblado de la pose 3D (`elevador_3d.py`)

Clase **`Elevador3D`**. Con YOLO esta fase era compleja: había que **estimar** la
profundidad (Z) por escorzo, con modelo de cámara *pinhole*, proporciones
antropométricas (Drillis & Contini) y calibración relativa — todo un rodeo para
suplir que YOLO no daba profundidad, y justo lo que fallaba al encorvarse.

**Con MediaPipe la profundidad viene medida**, así que esta fase se reduce a tres
pasos:

### 5.1 Pose 3D real
Se toman directamente los world landmarks en cm (X derecha, Y **abajo** positiva,
Z profundidad). Encorvarse o adelantar la cabeza son movimientos en **Z** que
ahora se miden de forma directa en **cualquier vista** (frontal incluida), sin
trucos geométricos.

### 5.2 Puntos virtuales
Se añaden los dos keypoints que consume el evaluador RULA y el detector ML:
- `17` = **cuello** = punto medio entre hombros (índices 5 y 6).
- `18` = **centro de cadera** = punto medio entre caderas (índices 11 y 12).

### 5.3 Suavizado temporal
MediaPipe ya suaviza, pero un filtro extra estabiliza los ángulos y el HUD:
- **Filtro One-Euro** (`filtro_oneeuro.EstabilizadorPose`, por defecto con
  `SUAVIZADO_ONEEURO`): se adapta a la velocidad del movimiento — quita el temblor
  en reposo **sin añadir retraso** al moverse. Parámetros `ONEEURO_3D_*`.
- **Fallback EMA**: si el One-Euro está desactivado, se usa una Media Móvil
  Exponencial de `alpha` fijo (=0.4), que sube a 0.95 cuando un punto salta
  >40 cm entre frames (para no arrastrar un valor viejo).

### 5.4 Recalibración
`recalibrar()` (tecla **`C`**) ahora **solo limpia el historial de suavizado**.
Con la pose 3D real **ya no hay una "línea base de postura erguida"** que calibrar
(eso era propio de la estimación por escorzo de la época YOLO).

---

## 6. Fase 3 — Evaluación RULA (`evaluador_rula.py`)

Clase **`EvaluadorRULA`**. Implementa el método RULA de McAtamney & Corlett
(1993) con sus tablas oficiales.

### 6.1 Detección del ángulo de cámara — `_detectar_vista()`

Antes de medir, se decide si la vista es **frontal**, **lateral** o **ángulo**,
porque la ruta de medición cambia. Se combinan **tres señales robustas** que **no
dependen de las caderas**:

1. **Asimetría de confianza entre lados.** En perfil, un lado completo (ojo,
   oreja, hombro) queda ocluido y recibe baja confianza. Si
   `conf_débil / conf_fuerte < VISTA_ASIMETRIA_LATERAL` → lateral.
2. **Separación de hombros / alto de la cabeza.** De frente los hombros se ven
   anchos; de costado se juntan. Umbrales `VISTA_SPREAD_FRONTAL` y
   `VISTA_SPREAD_LATERAL`. Escala-invariante.
3. **Desplazamiento horizontal de la nariz** respecto al centro de los hombros.
   De frente la nariz queda centrada **aunque gires la cabeza**; de costado queda
   desplazada hacia el hombro visible **aunque mires a la cámara**.

- **Histéresis temporal**: una vista nueva debe sostenerse varios frames antes de
  reemplazar a la actual, evitando parpadeo en posiciones límite.
- Devuelve `(vista, lado_fiable)`. Esta función la reutiliza también
  `recolectar_datos.py` para etiquetar la vista de cada muestra.

### 6.2 Cálculo de ángulos — dos rutas de medición

**Técnica base — ángulo entre vectores** (`_angulo_entre_vectores`): producto
punto y arcocoseno:

```
θ = arccos( (u·v) / (|u|·|v|) )
```

- **Ruta frontal**: usa los puntos medios virtuales 17/18 (combinan ambos lados).
  - **Brazo superior**: flexión **proyectada sobre el plano sagital**, eliminando
    la componente lateral (abducción) mediante la normal del eje hombro-hombro.
  - **Cuello**: se mide **solo en el plano sagital (Y, Z)**, descartando X para
    que girar la cabeza no lo penalice. Referencia de cabeza por prioridad:
    orejas → ojos → una oreja → nariz.
- **Ruta lateral/ángulo**: usa **solo los puntos reales del lado visible**. De
  costado la imagen ya ES el plano sagital, así que se mide directamente
  hombro→cadera (tronco) y oreja→hombro (cuello).

### 6.3 Mapeo de ángulos a scores

Funciones `_score_brazo_superior`, `_score_antebrazo`, `_score_muneca`,
`_score_cuello`, `_score_tronco`. Los **rangos están adaptados a oficina** (p.ej.
brazo 0-40° = natural con teclado/mouse; antebrazo 45-120° = aceptable),
reduciendo falsas alarmas frente a algoritmos genéricos estrictos.

### 6.4 Tablas RULA y score final

- **Tabla A**: brazo superior × antebrazo × muñeca × giro de muñeca.
- **Tabla B**: cuello × tronco × piernas.
- **Tabla C**: combina Score A y Score B → **score RULA final (1-7)**.
- Se añade `+1` por uso muscular (postura estática asumida en tiempo real).
- **Filtro de confianza**: en vista frontal se evalúan ambos lados y se reporta el
  peor caso si ambos son confiables, o el lado más confiable si hay oclusión.
- El score final se mapea a un **nivel de acción** con texto y color (verde /
  amarillo / naranja / rojo).

---

## 7. Módulo auxiliar — Encorvamiento lateral por silueta (`detector_encorvamiento.py`)

Clase **`AnalizadorSiluetaLateral`**. Resuelve un caso que la pose por sí sola no
cubre.

### 7.1 El problema
De costado y sentado, la cadera queda oculta tras el escritorio y su estimación
3D puede quedar casi bajo el hombro → el vector hombro→cadera sale **casi
vertical** → el tronco parece erguido aunque la persona esté encorvada. La única
información del tronco que queda en la imagen es el **contorno de la espalda**.

### 7.2 La técnica — análisis de silueta por bordes (Canny)

1. **ROI acotado por keypoints**: recorta una región del torso alrededor del
   hombro visible, dimensionada por la escala hombro-oreja (invariante a la
   distancia). Elimina casi todo el fondo.
2. **Preprocesamiento**: escala de grises + **desenfoque gaussiano**
   (`cv2.GaussianBlur`) para reducir bordes espurios.
3. **Detección de bordes**: `cv2.Canny`.
4. **Extracción del contorno**: fila por fila, toma el borde **más cercano a la
   posición esperada** de la espalda. Ignora bordes internos de la ropa y el
   respaldo de la silla.
5. **Ajuste robusto de recta**: `np.polyfit` grado 1 con **rechazo de outliers**.
   Si el residuo supera `ENCORVADO_RESIDUO_MAX`, la señal se descarta y el
   evaluador cae al método por keypoints — sin falsas alarmas.
6. **Ángulo con signo** respecto a la vertical, **suavizado temporal (EMA)** y
   **calibración personal**: línea base = postura más erguida observada; se
   reporta solo el exceso sobre esa base (menos una zona muerta), escalado por una
   **ganancia** que convierte la inclinación del contorno en flexión efectiva.

### 7.3 Integración con la Fase 3
Solo se activa en **vista lateral/ángulo**. `recalibrar()` reinicia su línea base;
se invoca junto con la del elevador al pulsar **`C`**.

---

## 8. Detector ML de encorvamiento (Random Forest)

Un clasificador supervisado, entrenado con **datos propios del usuario**,
distingue postura *erguida* (0) de *encorvada* (1). Vive en cuatro archivos que
comparten una **única** función de extracción de features, para que el modelo
reciba en producción exactamente los mismos números con los que se entrenó.

### 8.1 Ingeniería de características — `features_postura.py`

No se entrena sobre píxeles, sino sobre **~13 ángulos y proporciones** derivados de
la pose 3D real. Todas las features se **normalizan por la escala del torso**
(ancho de hombros) para dar igual a cualquier distancia de la cámara, y se apoyan
en la **proyección al plano sagital** para tolerar frente/costado.

`FEATURE_NAMES` (orden canónico, no reordenar sin reentrenar): flexión del tronco
sagital / 3D / lateral, flexión del cuello (vs. tronco y vs. vertical), ángulo
craneovertebral, cabeza adelantada en Z, protracción de hombros, curvatura
torácica, caída vertical de la cabeza, nariz vs. orejas, altura relativa del
torso y asimetría de profundidad de hombros.

`features_validos(kp3d)` decide si hay pose suficiente: exige los virtuales
**17 y 18**, al menos un hombro real y una referencia de cabeza. Gracias al
filtrado 3D permisivo (§4.3), esto se cumple **incluso en primer plano**.

### 8.2 Recolección etiquetada — `recolectar_datos.py`

Abre la webcam, corre la misma pose de MediaPipe que el sistema real y permite
**etiquetar en vivo** con el teclado. Cada frame válido se guarda como una fila de
números (no imágenes) en `datos_postura.csv`, que se **acumula** entre sesiones.

- Teclas: `G` graba **erguido** (0), `B` graba **encorvado** (1), `ESPACIO` pausa,
  `1/2/3/0` marcan la vista (frente/costado/ángulo/auto), `Z` deshace la última
  fila, `C` recalibra, **`N` cambia de cámara**, `Q` sale.
- Cada corrida marca sus filas con un identificador de **sesión** (fecha-hora),
  que el entrenador usa para validar **sin fugas de datos**.
- Solo graba cuando `features_validos` es `True` (indicador **`pose OK`** verde),
  para no meter ruido al dataset.

### 8.3 Entrenamiento — `entrenar_modelo.py`

- Lee el CSV, entrena un **`RandomForestClassifier`** (300 árboles,
  `min_samples_leaf=3`, `class_weight='balanced'`) y guarda
  `modelo_postura.joblib` con el **orden de features** y metadatos.
- **Validación honesta con `GroupKFold` por sesión**: el modelo se prueba siempre
  en sesiones que **no vio** al entrenar, de modo que la exactitud reportada
  refleja cómo generalizará a un momento/persona nuevos y no el sobreajuste al
  fondo o la ropa de una sola grabación. (Con una sola sesión cae a
  `StratifiedKFold` y avisa de que la cifra puede ser optimista.)
- Reporta **matriz de confusión**, precisión/recall por clase e **importancia de
  cada feature** (qué ángulos distinguen de verdad la postura).

### 8.4 Inferencia en vivo — `clasificador_postura.py`

Clase **`ClasificadorPostura`**. Dado `kp3d`, `predecir()` devuelve
`{'prob', 'prob_cruda', 'encorvado'}` o `None`:

- **Probabilidad** de la clase "encorvado" (`predict_proba`).
- **Suavizado temporal (EMA, `alpha=0.3`)** para no reaccionar al ruido de un solo
  frame.
- **Histéresis** con dos umbrales (`umbral_on=0.6`, `umbral_off=0.4`): la señal se
  **enciende** al superar 0.6 y solo se **apaga** al bajar de 0.4, evitando el
  parpadeo cerca del punto de corte. Subir `umbral_on` lo hace más conservador
  (menos falsas alarmas).
- **Degradación elegante**: si no existe `modelo_postura.joblib`, `disponible` es
  `False` y `predecir` devuelve `None`; el sistema sigue funcionando solo con RULA.

---

## 9. Orquestación y bucle principal (`Postura.py`)

`main()` conecta todo:

1. **Inicialización** de los cinco módulos: `Detector2D`, `Elevador3D`,
   `EvaluadorRULA`, `Visualizador` y `ClasificadorPostura`.
2. **Apertura robusta de cámara**:
   - `_abrir_camara(indice)` abre con el backend **DirectShow** (`cv2.CAP_DSHOW`),
     mucho más fiable para la webcam integrada en Windows, y **verifica que
     entregue un frame** (no basta con que "abra").
   - `_listar_camaras()` sondea los índices disponibles al arrancar.
   - Se intenta primero `CAMERA_INDEX`; si no responde, cae a la **primera cámara
     disponible** automáticamente.
3. **Bucle principal**:
   - `cap.read()` → `cv2.flip(frame, 1)` (espejo, más natural).
   - **Tolerancia a fallos de lectura**: un frame fallido **no cierra el
     programa**; solo se rinde tras `MAX_FALLOS` (~2 s) seguidos, para sobrevivir
     a hipos temporales (cambio de cámara, USB).
   - **Fase 1**: `detector.detectar(frame)` → estabilizador One-Euro de los
     puntos 2D (anti-jitter del dibujo).
   - **Fase 2**: `elevador.elevar(keypoints_world)`.
   - **Fase 3**: `evaluador.evaluar(keypoints_3d, keypoints_2d, frame)`.
   - **Detector ML**: `clasificador.predecir(keypoints_3d)`.
   - **Control de alarma con histéresis**: en riesgo (`score >= UMBRAL_RULA_ALARMA`)
     durante `UMBRAL_FRAMES_ALARMA` frames → dispara `emitir_alarma_async()`; se
     libera solo tras `UMBRAL_FRAMES_LIBERA` frames buenos seguidos.
   - **Visualización**: `visualizador.dibujar(...)`, más el **banner ML**
     (`ML: ENCORVADO / erguido` con su probabilidad) y el **indicador de cámara**
     activa (`CAM n`).
   - **Teclas**: `Q` sale; `C` recalibra (elevador + evaluador + clasificador +
     estabilizador); **`N` cambia a la siguiente cámara** disponible **sin tocar la
     activa** (prueba solo otros índices y, si ninguno funciona, conserva la
     actual — nunca mata el programa).
4. **Limpieza**: `cap.release()` + `cv2.destroyAllWindows()` + `detector.cerrar()`.

**Alarma no bloqueante** (`emitir_alarma_async`): `winsound.Beep` se lanza en un
`threading.Thread(daemon=True)` para no congelar el vídeo.

---

## 10. Visualización (`visualizador.py`)

Clase **`Visualizador`**. Dibuja sobre el frame (in-place) en capas:

- **Esqueleto** (`_dibujar_esqueleto`): líneas (`cv2.line`) entre keypoints según
  las conexiones, coloreadas por nivel de riesgo; nodos como círculos.
- **Ángulos** (`_dibujar_angulos`): valores numéricos junto a cuello y codos
  (doble `cv2.putText`: contorno + relleno para legibilidad).
- **HUD** (`_dibujar_hud`): panel semitransparente (`cv2.addWeighted`) con el
  score RULA, nivel de acción, scores individuales por segmento con mini-barras y
  barra de riesgo global.
- **Indicador de vista** (`_dibujar_vista`): muestra FRENTE / LADO / ANGULO.
- **FPS** (`_dibujar_fps`): media móvil sobre varias muestras.

> El **banner ML** y el **indicador de cámara** se dibujan aparte, directamente en
> `Postura.py`, para no acoplar el visualizador RULA con el detector ML.

---

## 11. Configuración central (`config.py`)

Agrupa todos los parámetros ajustables. Bloques principales:

| Bloque | Contenido |
|---|---|
| Keypoints COCO | Nombres, índices, conexiones del esqueleto |
| Cámara / sujeto | `CAMERA_INDEX`, `CAMERA_FOV_GRADOS`, `ALTURA_SUJETO_CM` |
| MediaPipe | `MP_MODEL_COMPLEXITY`, `MP_MODELOS` (URLs), `MP_USAR_GPU`, confianzas de detección/seguimiento/presencia |
| Filtrado de landmarks | `MP_PRESENCE_MIN` (2D, =0.5) y **`MP_PRESENCE_MIN_3D`** (3D, =0.0 → conservar puntos inferidos para el primer plano) |
| Preprocesamiento | `PREPROCESAMIENTO_ACTIVO`, parámetros CLAHE |
| Clasificación de vista | Umbrales de asimetría, spread, offset de nariz, histéresis |
| Encorvamiento lateral | Parámetros Canny, ROI, calibración, ganancia (§7) |
| Suavizado | `SUAVIZADO_ALPHA` (EMA) y `SUAVIZADO_ONEEURO` + `ONEEURO_2D_*` / `ONEEURO_3D_*` |
| Alarma | Frecuencia, duración, umbrales de frames y de RULA |
| Colores | Paleta BGR por nivel de riesgo |

---

## 12. Glosario de técnicas

| Técnica | Dónde se usa | Para qué |
|---|---|---|
| **Estimación de pose (MediaPipe Pose Landmarker)** | Fase 1 | Detectar 33 landmarks 2D + **world landmarks 3D reales** |
| **World landmarks 3D** | Fases 1-3 | Profundidad medida (encorvamiento en Z) en cualquier vista |
| **Filtrado por presencia (2D vs 3D)** | Fase 1 | Dibujar solo lo que está en cuadro, pero conservar el 3D inferido en primer plano |
| **CLAHE + espacio LAB** | Fase 1 | Robustez ante baja luz / contraluz sin tocar el color |
| **Delegado GPU→CPU con caída automática** | Fase 1 | Nunca quedarse sin detector si la GPU no está disponible |
| **Puntos virtuales (cuello / cadera)** | Fase 2 | Ejes de tronco y cuello para RULA y features |
| **Filtro One-Euro / EMA** | Fase 2, §7, §8 | Suavizado temporal adaptativo / anti-jitter |
| **Producto punto / arcocoseno** | Fase 3, §8 | Ángulos entre vectores 3D |
| **Proyección sobre plano sagital** | Fase 3, §8 | Aislar la flexión real de la abducción / giro |
| **Histéresis temporal** | Fase 3, alarma, §8 | Evitar parpadeo de estado |
| **Método RULA (tablas A/B/C)** | Fase 3 | Score ergonómico estándar |
| **Canny + desenfoque gaussiano + `polyfit`** | §7 | Contorno de la espalda cuando la pelvis está oculta |
| **Random Forest (scikit-learn)** | §8 | Clasificar erguido vs. encorvado |
| **Ingeniería de features normalizadas** | §8 | Ángulos/proporciones invariantes a distancia y vista |
| **Validación `GroupKFold` por sesión** | §8 | Exactitud honesta sin fugas de datos entre grabaciones |

---

## 13. Ejecución

```bash
# Dependencias (entorno Python 3.12)
pip install -r requirements.txt

# 1) (opcional) Recolectar datos etiquetados para el detector ML
python recolectar_datos.py     # G=erguido  B=encorvado  N=cambiar cámara  Q=salir

# 2) (opcional) Entrenar / reentrenar el modelo con todo el CSV acumulado
python entrenar_modelo.py      # genera modelo_postura.joblib

# 3) Ejecutar el sistema en tiempo real
python Postura.py
python Postura.py --gpu        # intenta GPU (cae a CPU si el build no la soporta)
python Postura.py --cpu        # fuerza CPU
```

**Controles en vivo**: `Q` sale · `C` recalibra la postura erguida (siéntate
derecho y púlsala) · **`N` cambia de cámara**. El modelo `.task` de MediaPipe se
descarga solo la primera vez.

El ciclo del detector ML es iterativo y acumulativo: `recolectar_datos.py` **añade**
sesiones al mismo CSV; `entrenar_modelo.py` **reentrena con todo** y sobrescribe el
`.joblib`; `Postura.py` carga siempre el último modelo.

---

*Informe generado a partir del código fuente del proyecto. Cada afirmación técnica
corresponde a una función o parámetro concreto de los módulos descritos.*
