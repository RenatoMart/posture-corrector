# Informe Técnico — Sistema de Análisis de Postura Ergonómica (RULA) en Tiempo Real

> Documento técnico exhaustivo del proyecto **Postura**: herramientas, librerías,
> arquitectura, flujo de datos, funciones y las técnicas de visión por computador
> y biomecánica que sustentan cada fase.

---

## 1. Resumen del sistema

El proyecto captura vídeo de una webcam estándar, detecta la postura de la
persona frame a frame y calcula un **score ergonómico RULA** (Rapid Upper Limb
Assessment) en tiempo real, mostrándolo sobre la imagen y emitiendo una alarma
sonora cuando se sostiene una mala postura.

La arquitectura es una **tubería (pipeline) híbrida de 3 fases**, con un cuarto
módulo de visualización y un módulo auxiliar de detección de encorvamiento por
silueta:

```
        ┌─────────────┐   keypoints 2D    ┌──────────────┐   keypoints 3D   ┌────────────────┐
 frame  │  Fase 1     │   {idx:(x,y,c)}   │  Fase 2      │  {idx:[X,Y,Z]}   │  Fase 3        │  resultado
 ─────▶ │ Detector2D  │ ────────────────▶ │ Elevador3D   │ ───────────────▶ │ EvaluadorRULA  │ ─────────▶
 (BGR)  │ (YOLOv8)    │                   │ (geometría)  │                  │ (RULA + vista) │  (dict)
        └─────────────┘                   └──────────────┘                  └───────┬────────┘
              │                                                                     │ frame (lateral)
              │                                                            ┌────────▼───────────┐
              │                                                            │ AnalizadorSilueta   │
              │                                                            │ Lateral (Canny)     │
              │                                                            └─────────────────────┘
              ▼
        ┌───────────────┐
        │ Visualizador  │  ← dibuja esqueleto, ángulos, HUD, FPS sobre el frame
        └───────────────┘
```

Cada fase transforma la representación de los datos: **píxeles → puntos 2D →
puntos 3D estimados → ángulos articulares → scores RULA → nivel de riesgo**.

---

## 2. Herramientas, librerías y entorno

### 2.1 Lenguaje y entorno de ejecución

| Elemento | Versión / valor | Notas |
|---|---|---|
| Lenguaje | Python 3.11 / 3.12 | El intérprete de ejecución probado es CPython 3.12 |
| SO objetivo | Windows 11 | Depende de `winsound` (nativa de Windows) para la alarma |
| Entorno virtual | `.venv` | Aislamiento de dependencias |

### 2.2 Librerías de terceros

| Librería | Versión instalada | Rol en el proyecto |
|---|---|---|
| **ultralytics** | 8.4.66 | Framework que provee el modelo **YOLOv8-Pose**; realiza la inferencia de los 17 keypoints |
| **opencv-python** (`cv2`) | 4.13.0 | Captura de vídeo, procesamiento de imagen (CLAHE, conversión de espacios de color, Canny, blur), y renderizado (líneas, círculos, texto, transparencias) |
| **numpy** | 2.4.5 | Álgebra vectorial (producto punto, normas), operaciones matriciales y manejo eficiente de tensores de keypoints |
| **torch** (PyTorch) | 2.12.0+cpu | Backend de inferencia de YOLO. La build es **CPU-only** (no requiere GPU) |

### 2.3 Librerías estándar de Python

| Módulo | Uso |
|---|---|
| `winsound` | Emite el tono de alarma (`Beep`) en Windows |
| `threading` | Ejecuta la alarma en un hilo aparte para no bloquear el vídeo |
| `time` | Cálculo de FPS con media móvil en el visualizador |

### 2.4 Modelo de deep learning

- **YOLOv8-Nano-Pose** (`yolov8n-pose.pt`, ~6.4 MB): red de estimación de pose
  que devuelve, por persona, los **17 keypoints del estándar COCO** con
  coordenadas `(x, y)` y una confianza `conf ∈ [0,1]` por punto.
- Se descarga automáticamente la primera vez que se instancia el modelo.
- Alternativa documentada en `config.py`: `yolov8s-pose.pt` (~23 MB), más preciso
  pero 2-3× más lento en CPU.

### 2.5 Conjunto de keypoints COCO (17 puntos)

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
| `Postura.py` | `main()` | Orquestador: captura de cámara, bucle principal, control de alarma, teclas |
| `config.py` | (constantes) | Configuración central: keypoints, proporciones, umbrales, colores, parámetros |
| `detector_2d.py` | `Detector2D` | **Fase 1**: inferencia YOLO, preprocesamiento CLAHE, tracking, estimación de puntos faltantes |
| `elevador_3d.py` | `Elevador3D` | **Fase 2**: elevación geométrica 2D→3D, escorzo, calibración relativa, suavizado |
| `evaluador_rula.py` | `EvaluadorRULA` | **Fase 3**: detección de vista, cálculo de ángulos, tablas RULA, score final |
| `detector_encorvamiento.py` | `AnalizadorSiluetaLateral` | Detección de encorvamiento lateral por bordes (Canny) |
| `visualizador.py` | `Visualizador` | Renderizado del esqueleto, ángulos, HUD, FPS |

---

## 4. Fase 1 — Detección 2D (`detector_2d.py`)

Clase **`Detector2D`**. Convierte un frame BGR en un diccionario
`{idx: (x, y, conf)}` de keypoints de la persona principal.

### 4.1 Preprocesamiento CLAHE — `_preprocesar_frame()`

**Técnica: CLAHE (Contrast Limited Adaptive Histogram Equalization).**

- La imagen BGR se convierte al espacio de color **LAB** (`cv2.cvtColor`), que
  separa la luminosidad (canal **L**) del color (canales **a**, **b**).
- CLAHE (`cv2.createCLAHE`) se aplica **solo al canal L**, mejorando el contraste
  local en zonas oscuras/contraluz **sin distorsionar los colores**.
- El parámetro `clipLimit` (=2.0) limita la amplificación del ruido; `tileGridSize`
  (=8×8) define la grilla de ecualización adaptativa.
- El objeto CLAHE se crea **una sola vez** en el constructor (reutilizable, más
  eficiente que recrearlo por frame). Coste ~1-2 ms/frame.

### 4.2 Inferencia y selección de persona

- `self.modelo(frame, conf, imgsz=YOLO_IMGSZ, verbose=False)` ejecuta YOLOv8-Pose.
  `imgsz` (=640) controla la resolución de inferencia; subirla mejora keypoints
  lejanos a costa de FPS.
- **`_seleccionar_persona()`** implementa un **tracker por centroide** para seguir
  a la misma persona entre frames cuando hay varias:
  - Sin historial → elige la persona con **mayor bounding box** (más cercana).
  - Con historial y varias personas → sigue la **más cercana al centroide** del
    frame anterior (distancia euclidiana con `np.linalg.norm`), dentro de un
    umbral (`_MAX_DIST_TRACKING = 250 px`).
  - Tras `_MAX_FRAMES_PERDIDOS = 15` frames sin detección, reinicia el tracker.

### 4.3 Estimación de puntos ocluidos (inferencia anatómica)

Para funcionar con oclusiones (medio cuerpo, vista lateral), el detector
**fabrica** keypoints faltantes con confianza baja (`CONF_ESTIMADA = 0.25`) para
que las fases posteriores sepan que no son medidas reales:

- **`_estimar_hombro_faltante()`** (vista lateral, 1 hombro visible): estima el
  ancho biacromial a partir de la distancia entre orejas (×2.2), nariz-oreja
  (×3.0) o del torso visible (×0.6), y coloca el hombro oculto en dirección
  **perpendicular al eje del torso** (para que el esqueleto acompañe la
  inclinación real).
- **`_estimar_caderas()`**: cuando las caderas no son visibles (típico sentado),
  las coloca por debajo de los hombros usando la proporción antropométrica
  torso/hombros (`RATIO_TORSO_HOMBROS = 1.4`).

> ⚠️ **Consecuencia clave para la Fase 3:** las caderas fabricadas quedan justo
> debajo de los hombros → en vista lateral el tronco parece siempre vertical.
> Esto motivó el módulo de detección por silueta (§8).

### 4.4 Salida

`detectar()` requiere **al menos 1 hombro visible**; devuelve el dict de keypoints
(reales + fabricados) o `None`. La confianza de cada punto permite distinguir
medidos vs fabricados (`conf <= CONF_FABRICADO = 0.26`).

---

## 5. Fase 2 — Elevación geométrica a 3D (`elevador_3d.py`)

Clase **`Elevador3D`**. Reconstruye una coordenada Z (profundidad) a partir de
una sola cámara, usando óptica y antropometría.

### 5.1 Modelo de cámara Pinhole — `_estimar_focal()`

La distancia focal en píxeles se deriva del campo de visión (FOV) de la cámara:

```
focal = (ancho_frame / 2) / tan(FOV / 2)
```

Con la focal, la relación **Z = (focal × longitud_real) / longitud_píxeles**
permite estimar la profundidad de un segmento a partir de su tamaño aparente.

### 5.2 Proporciones antropométricas (Drillis & Contini, 1966)

`config.PROPORCIONES_CORPORALES` define la longitud de cada segmento como
fracción de la altura total del sujeto (`ALTURA_SUJETO_CM`). Ejemplo:
tronco = altura × 0.288, ancho de hombros = altura × 0.259. En el constructor se
precalculan las longitudes reales en cm.

### 5.3 Profundidad de referencia y propagación por escorzo

- **Paso 1 — Z de referencia:** se prefiere el **ancho de hombros** como ancla
  porque, a diferencia del torso, **no se acorta al inclinarse hacia adelante**,
  dando una profundidad estable e independiente de la postura. En vista lateral
  (hombros juntos), se usa la distancia cabeza-hombro.
- **Paso 2:** los puntos del torso y la cabeza se colocan a esa misma Z.
- **Paso 3 — propagación por cadenas** (`SEGMENTOS_CADENA`): para cada segmento
  de las extremidades, si su proyección en píxeles es más corta que la longitud
  real esperada, la diferencia se atribuye a profundidad mediante el **teorema de
  Pitágoras**: `dz = sqrt(real² - proyectado²)`.

### 5.4 Elevación sagital y calibración relativa — `_elevar_sagital()` + `_dz_calibrado()`

En vista frontal, **encorvarse o adelantar la cabeza es un movimiento en
profundidad (Z) invisible en X-Y**. Para recuperarlo:

- Se mide el **escorzo** (acortamiento) del torso y del cuello.
- **Técnica de calibración personal (línea base):** en lugar de comparar contra
  proporciones absolutas (sensibles a la altura configurada y al FOV), se compara
  el **ratio `segmento / ancho_hombros`** (invariante a la distancia a la cámara)
  contra el **mejor ratio observado de la propia persona** en la sesión.
  - `_actualizar_baseline()`: la línea base **sube rápido** ante una postura más
    erguida (`CALIB_ALPHA_SUBIDA = 0.05`) y **baja lentísimo** (`CALIB_ALPHA_BAJADA
    = 0.0005`), de modo que estar encorvado mucho rato no erosiona la referencia.
  - La tecla **`C`** (`recalibrar()`) reinicia esta línea base.
- La zona muerta `ESCORZO_DEADZONE` evita reaccionar a acortamientos pequeños
  (variación corporal, no postura).

### 5.5 Puntos virtuales y suavizado temporal (EMA)

- **Paso 4:** crea los keypoints virtuales `17` (cuello) y `18` (centro de cadera).
- **Paso 5 — suavizado adaptativo:** aplica una **Media Móvil Exponencial (EMA)**
  a cada keypoint 3D para eliminar el *jitter*. El factor `alpha` (=0.4) sube a
  0.95 cuando un punto salta >40 cm entre frames (cambio de persona o detección
  espuria), para adaptarse rápido sin arrastrar valores ajenos.

---

## 6. Fase 3 — Evaluación RULA (`evaluador_rula.py`)

Clase **`EvaluadorRULA`**. Implementa el método RULA de McAtamney & Corlett
(1993) con sus tablas oficiales.

### 6.1 Detección del ángulo de cámara — `_clasificar_vista_cruda()` / `_detectar_vista()`

Antes de medir, se decide si la vista es **frontal**, **lateral** o **ángulo**,
porque la ruta de medición cambia. Se combinan **tres señales robustas** que **no
dependen de las caderas** (a menudo fabricadas al estar sentado):

1. **Asimetría de confianza entre lados.** En perfil, un lado completo (ojo,
   oreja, hombro, cadera) queda ocluido y recibe baja confianza. Si
   `conf_débil / conf_fuerte < VISTA_ASIMETRIA_LATERAL (0.45)` → lateral.
2. **Separación de hombros / alto de la cabeza.** De frente los hombros se ven
   anchos; de costado se juntan. Umbrales `VISTA_SPREAD_FRONTAL (1.40)` y
   `VISTA_SPREAD_LATERAL (0.70)`. Escala-invariante.
3. **Desplazamiento horizontal de la nariz** respecto al centro de los hombros
   (`VISTA_NARIZ_OFFSET_LATERAL = 0.60`). De frente la nariz queda centrada
   **aunque gires la cabeza**; de costado queda desplazada hacia el hombro visible
   **aunque mires a la cámara**. Esta señal reconoce el perfil sin exigir que la
   persona mire a la cámara (corrige un bug previo).

- **Histéresis temporal** (`_detectar_vista`): una vista nueva debe sostenerse
  `VISTA_HISTERESIS_FRAMES (4)` frames antes de reemplazar a la actual, evitando
  parpadeo en posiciones límite.
- Devuelve `(vista, lado_fiable)`. En lateral/ángulo, `lado_fiable` indica qué
  lado real usar; en frontal es `None`.

### 6.2 Cálculo de ángulos — dos rutas de medición

**Técnica base — ángulo entre vectores** (`_angulo_entre_vectores`): producto
punto y arcocoseno:

```
θ = arccos( (u·v) / (|u|·|v|) )
```

- **Ruta frontal** (`_angulos_frontal`): usa los puntos medios virtuales 17/18
  (combinan ambos lados).
  - **Brazo superior**: se mide la flexión **proyectada sobre el plano sagital**
    (`_angulo_brazo_sagital`), eliminando la componente lateral (abducción) usando
    la normal definida por el eje hombro-hombro. Esto da resultados estables en
    cualquier ángulo de cámara.
  - **Cuello**: se mide **solo en el plano sagital (Y, Z)**, descartando X para
    que girar la cabeza no lo penalice. Referencia de cabeza por prioridad:
    orejas → ojos → una oreja → nariz.
  - **Abducción** (`_detectar_abduccion`): fracción del vector del brazo que cae
    fuera del plano sagital; si supera el 35% → abducido.

- **Ruta lateral/ángulo** (`_angulos_lateral`): usa **solo los puntos reales del
  lado visible**, evitando los keypoints fabricados del lado oculto. De costado
  la imagen ya ES el plano sagital, así que se mide directamente hombro→cadera
  (tronco) y oreja→hombro (cuello). Si falta la oreja del lado fiable, usa la otra
  oreja real (nunca puntos fabricados).

### 6.3 Mapeo de ángulos a scores

Funciones `_score_brazo_superior`, `_score_antebrazo`, `_score_muneca`,
`_score_cuello`, `_score_tronco`. Los **rangos están adaptados a oficina** (p.ej.
brazo 0-40° = natural con teclado/mouse; antebrazo 45-120° = aceptable),
reduciendo falsas alarmas frente a algoritmos genéricos estrictos.

### 6.4 Tablas RULA y score final

- **Tabla A** (6×3×4×2): brazo superior × antebrazo × muñeca × giro de muñeca.
- **Tabla B** (6×6×2): cuello × tronco × piernas.
- **Tabla C** (8×7): combina Score A y Score B → **score RULA final (1-7)**.
- Se añade `+1` por uso muscular (postura estática asumida en tiempo real).
- **Filtro de confianza** (`_seleccionar_por_confianza`): en vista frontal se
  evalúan ambos lados y se reporta el **peor caso** si ambos son confiables, o el
  lado más confiable si hay oclusión.
- El score final se mapea a un **nivel de acción** con texto y color (verde /
  amarillo / naranja / rojo).

---

## 7. Módulo auxiliar — Detección de encorvamiento lateral (`detector_encorvamiento.py`)

Clase **`AnalizadorSiluetaLateral`**. Resuelve un caso que ninguna fase anterior
cubre.

### 7.1 El problema

De costado y sentado, la cadera queda oculta tras el escritorio → el detector la
fabrica justo debajo del hombro → el vector hombro→cadera sale **siempre
vertical** → el tronco parece erguido aunque la persona esté encorvada. No hay
keypoint que recupere la pelvis oculta.

### 7.2 La técnica — análisis de silueta por bordes (Canny)

Cuando la pelvis no es visible, la única información del tronco que queda en la
imagen es el **contorno de la espalda alta**, que se inclina hacia adelante al
encorvarse. El algoritmo:

1. **ROI acotado por keypoints** (`analizar`): recorta una región del torso
   alrededor del hombro visible, dimensionada por la escala hombro-oreja
   (invariante a la distancia). Esto elimina casi todo el fondo.
2. **Preprocesamiento**: escala de grises + **desenfoque gaussiano**
   (`cv2.GaussianBlur`) para reducir bordes espurios.
3. **Detección de bordes**: `cv2.Canny(gris, LOW=40, HIGH=120)`.
4. **Extracción del contorno** (`_contorno_espalda`): fila por fila, toma el
   borde **más cercano a la posición esperada** de la espalda
   (`ENCORVADO_OFFSET_ESPALDA = 0.6 × escala`). Esta heurística lo hace robusto:
   ignora bordes internos de la ropa (más al centro) y el respaldo de la silla
   (más lejos).
5. **Dirección de la espalda** (`_direccion_frente`): la nariz indica hacia dónde
   mira la persona; la espalda está en el lado opuesto.
6. **Ajuste robusto de recta**: `np.polyfit` grado 1, con **rechazo de outliers**
   por residuo. Si el RMS del residuo supera `ENCORVADO_RESIDUO_MAX (14 px)`, la
   señal se descarta (`return None`) y el evaluador **cae al método anterior** —
   sin falsas alarmas.
7. **Ángulo con signo** respecto a la vertical, en la dirección "hacia adelante".
8. **Suavizado temporal (EMA)** del ángulo (`ENCORVADO_SUAVIZADO = 0.35`).
9. **Calibración personal** (igual filosofía que la Fase 2): línea base = postura
   más erguida observada; se reporta solo el exceso sobre esa base, menos una
   **zona muerta** (`ENCORVADO_MARGEN_GRADOS = 4°`), escalado por una **ganancia**
   (`ENCORVADO_GANANCIA = 1.8`) que convierte la inclinación del contorno (que
   subestima la flexión de tronco completa) en flexión efectiva.

### 7.3 Integración con la Fase 3

- `EvaluadorRULA.evaluar(kp3d, kp2d, frame)` recibe ahora el **frame** (sin
  dibujos).
- Solo se activa en **vista lateral/ángulo**.
- En `_angulos_lateral`, si la **cadera del lado visible es fabricada**
  (`conf <= CONF_FABRICADO`), se usa la flexión por silueta en lugar del vector
  hombro→cadera (que sería ~0°). Si la cadera es real, se mantiene el método por
  keypoints (y se toma el **máximo** con la silueta, criterio conservador).
- `recalibrar()` reinicia la línea base; se invoca junto con la del elevador al
  pulsar **`C`**.

---

## 8. Orquestación y bucle principal (`Postura.py`)

`main()` conecta todo:

1. **Inicialización** de los 4 módulos (`Detector2D`, `Elevador3D`,
   `EvaluadorRULA`, `Visualizador`).
2. **Apertura de cámara** con `cv2.VideoCapture(CAMERA_INDEX)`, resolución
   640×480.
3. **Bucle principal** (`while cap.isOpened()`):
   - `cap.read()` → `cv2.flip(frame, 1)` (espejo, más natural).
   - **Fase 1**: `detector.detectar(frame)`.
   - **Fase 2**: `elevador.elevar(keypoints_2d, frame.shape)`.
   - **Fase 3**: `evaluador.evaluar(keypoints_3d, keypoints_2d, frame)`.
   - **Control de alarma con histéresis**:
     - En riesgo (`score >= UMBRAL_RULA_ALARMA = 5`) durante
       `UMBRAL_FRAMES_ALARMA = 45` frames → dispara `emitir_alarma_async()`.
     - Se libera solo tras `UMBRAL_FRAMES_LIBERA = 12` frames buenos seguidos
       (un frame ruidoso no apaga la alarma).
   - **Visualización**: `visualizador.dibujar(...)` + `cv2.imshow`.
   - **Teclas**: `Q` sale; `C` recalibra la línea base (elevador + evaluador).
4. **Limpieza**: `cap.release()` + `cv2.destroyAllWindows()`.

**Alarma no bloqueante** (`emitir_alarma_async`): `winsound.Beep` se lanza en un
`threading.Thread(daemon=True)` para no congelar el vídeo.

---

## 9. Visualización (`visualizador.py`)

Clase **`Visualizador`**. Dibuja sobre el frame (in-place) en capas:

- **Esqueleto** (`_dibujar_esqueleto`): líneas (`cv2.line`) entre keypoints según
  `SKELETON_CONEXIONES`, coloreadas por nivel de riesgo; nodos como círculos
  (`cv2.circle`), más grandes en articulaciones principales.
- **Ángulos** (`_dibujar_angulos`): valores numéricos junto a cuello y codos
  (doble `cv2.putText`: contorno + relleno para legibilidad).
- **HUD** (`_dibujar_hud`): panel semitransparente (`cv2.addWeighted`) con el
  score RULA, nivel de acción, scores individuales por segmento con mini-barras y
  barra de riesgo global.
- **Indicador de vista** (`_dibujar_vista`): muestra FRENTE / LADO / ANGULO;
  marca "postura sagital incierta" en la vista ANGULO (zona ambigua).
- **FPS** (`_dibujar_fps`): media móvil sobre 30 muestras (`time.time()`).
- **Instrucciones**: recordatorio de teclas `C` / `Q`.

---

## 10. Configuración central (`config.py`)

Agrupa todos los parámetros ajustables. Bloques principales:

| Bloque | Contenido |
|---|---|
| Keypoints COCO | Nombres, índices, conexiones del esqueleto |
| Proporciones | Antropometría (Drillis & Contini) y cadenas de segmentos |
| Cámara / sujeto | `CAMERA_INDEX`, `CAMERA_FOV_GRADOS`, `ALTURA_SUJETO_CM` |
| Detección | `YOLO_MODELO`, `YOLO_IMGSZ`, `YOLO_CONFIANZA`, umbrales de confianza |
| Clasificación de vista | Umbrales de asimetría, spread, offset de nariz, histéresis |
| Preprocesamiento | Parámetros CLAHE |
| Elevación sagital | Zona muerta de escorzo, límites, calibración relativa |
| Encorvamiento lateral | Parámetros Canny, ROI, calibración, ganancia (§7) |
| Alarma | Frecuencia, duración, umbrales de frames y RULA |
| Suavizado | `SUAVIZADO_ALPHA` (EMA de la Fase 2) |
| Colores | Paleta BGR por nivel de riesgo |

---

## 11. Glosario de técnicas de visión por computador y matemáticas

| Técnica | Dónde se usa | Para qué |
|---|---|---|
| **Estimación de pose (YOLOv8-Pose)** | Fase 1 | Detectar los 17 keypoints |
| **CLAHE** (ecualización adaptativa de contraste limitado) | Fase 1 | Robustez ante baja luz / contraluz |
| **Espacio de color LAB** | Fase 1 | Ecualizar luminosidad sin tocar el color |
| **Tracking por centroide** | Fase 1 | Seguir a la misma persona entre frames |
| **Modelo de cámara Pinhole** | Fase 2 | Estimar la focal y la profundidad Z |
| **Antropometría (Drillis & Contini)** | Fase 2 | Longitudes reales de segmentos |
| **Escorzo + Teorema de Pitágoras** | Fase 2 | Recuperar profundidad de segmentos acortados |
| **Calibración relativa (línea base personal)** | Fase 2 y §7 | Medir postura contra la mejor propia, no contra tablas absolutas |
| **Media Móvil Exponencial (EMA)** | Fase 2 y §7 | Suavizado temporal / anti-jitter |
| **Producto punto / arcocoseno** | Fase 3 | Ángulos entre vectores 3D |
| **Proyección sobre plano sagital** | Fase 3 | Aislar la flexión real de la abducción/giro |
| **Histéresis temporal** | Fase 3 y alarma | Evitar parpadeo de estado |
| **Detección de bordes Canny** | §7 | Contorno de la espalda cuando la pelvis está oculta |
| **Desenfoque gaussiano** | §7 | Reducir ruido antes de Canny |
| **Ajuste de recta (`polyfit`) con rechazo de outliers** | §7 | Medir la inclinación robusta del contorno |
| **Método RULA (tablas A/B/C)** | Fase 3 | Score ergonómico estándar |

---

## 12. Ejecución

```bash
# Dependencias
pip install ultralytics opencv-python numpy

# Ejecutar
python Postura.py
```

**Controles**: `Q` sale · `C` recalibra la postura erguida (siéntate derecho y
púlsala). El modelo YOLO se descarga solo la primera vez.

---

*Informe generado a partir del código fuente del proyecto. Cada afirmación técnica
corresponde a una función o parámetro concreto de los módulos descritos.*
