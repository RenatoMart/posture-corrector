# Sistema de Análisis de Postura Ergonómica (RULA) en Tiempo Real

Este proyecto es un sistema avanzado de visión artificial diseñado para monitorear y evaluar la postura ergonómica de personas en entornos de oficina en tiempo real. Utiliza una cámara web estándar y aplica el **Método RULA** (Rapid Upper Limb Assessment) para determinar el nivel de riesgo ergonómico y emitir alertas si se detectan posturas perjudiciales sostenidas.

## 🌟 Características Principales

El sistema ha sido adaptado específicamente para **entornos reales de oficina**, superando las limitaciones de los sistemas de detección tradicionales:

- **Independencia del Ángulo de Cámara (Vistas Laterales):** Funciona perfectamente incluso si la cámara está ubicada a un costado. El sistema es capaz de detectar la postura viendo un solo hombro y estimando anatómicamente el lado oculto.
- **Clasificación de Vista Robusta (Frente / Lado / Ángulo):** Combina tres señales independientes de las caderas (asimetría de confianza, separación de hombros y desplazamiento de la nariz) con histéresis temporal. Reconoce el perfil **aunque gires la cara hacia la cámara**, sin exigir que mires a la lente.
- **Detección de Encorvamiento por Silueta (Vista Lateral):** Cuando estás de costado y las caderas quedan ocultas tras el escritorio, el sistema analiza el **contorno de la espalda con detección de bordes (Canny)** para medir el encorvamiento del tronco, algo que la sola posición de los keypoints no puede capturar.
- **Detección de Medio Cuerpo (Oclusión Inferior):** No requiere que la persona esté de cuerpo entero en la cámara. Si las caderas o piernas no son visibles, el sistema calcula las proporciones antropométricas usando los hombros y el torso superior.
- **Calibración Personal (Tecla `C`):** Aprende tu postura erguida ideal como línea base. En lugar de comparar contra tablas absolutas (sensibles a tu altura o al FOV), mide cuánto te desvías de tu propia mejor postura, adaptándose a cada persona.
- **Independencia de la Dirección de la Mirada:** Utiliza las orejas como referencia principal para la inclinación de la cabeza y mide el cuello solo en el plano sagital. La persona puede mirar monitores laterales o el teclado sin que el sistema asuma posturas incorrectas del cuello.
- **Filtro de Iluminación Inteligente (CLAHE):** Integra ecualización adaptativa de contraste. Funciona excelente en condiciones de baja luz o contraluz sin afectar el rendimiento (FPS).
- **Rangos RULA Adaptados a Oficina:** Reconoce que la flexión natural del brazo (0-40°) y antebrazo (45-120°) al usar teclado y mouse son posturas aceptables y cómodas, evitando falsas alarmas (rojos) comunes en algoritmos estrictos genéricos.
- **Alerta Sonora Integrada:** Emite un pitido ("beep") automático si el usuario mantiene un Score RULA alto (riesgo) durante un tiempo prolongado.

---

## 🏗️ Arquitectura y Fundamentos Teóricos (3 Fases)

El flujo de procesamiento se divide en 3 módulos principales, sustentados en principios matemáticos y de visión computacional:

### 1. Fase de Detección 2D y Preprocesamiento (`detector_2d.py`)
Antes de pasar la imagen a la red neuronal, se aplica un filtro para estabilizar las condiciones de luz:
- **Preprocesamiento CLAHE (Contrast Limited Adaptive Histogram Equalization):** En lugar de ecualizar la imagen BGR entera (lo que distorsiona los colores), la imagen se convierte al espacio de color **LAB**. El filtro CLAHE se aplica *únicamente* al canal **L** (Luminosidad). Esto mejora drásticamente el contraste local en zonas oscuras o con contraluz, previniendo la amplificación del ruido mediante el parámetro `clipLimit`.
- **Inferencia (MediaPipe Pose Landmarker):** Se extraen **33 landmarks** anatómicos, de los que se usan los 17 índices estilo COCO. MediaPipe entrega tanto los puntos 2D en la imagen (con una métrica de *visibilidad* por punto) como los **world landmarks: coordenadas 3D reales en metros** con el origen en el centro de las caderas. Rastrea automáticamente a una sola persona entre frames.

### 2. Fase de Ensamblado de la Pose 3D (`elevador_3d.py`)
A diferencia de YOLO (que solo daba 2D y obligaba a *estimar* la profundidad por escorzo, algo frágil que fallaba justo al encorvarse), **MediaPipe ya mide la profundidad**. Por eso esta fase se simplifica a:
- **Pose 3D real:** Se toman directamente los world landmarks (X derecha, Y abajo, Z profundidad, en cm). Encorvarse o adelantar la cabeza son movimientos en Z que ahora se ven de forma directa en **cualquier vista** (frontal incluida), sin trucos geométricos.
- **Puntos virtuales:** Se añaden el *cuello* (punto medio entre hombros) y el *centro de cadera* (punto medio entre caderas) que consume el evaluador RULA.
- **Suavizado Temporal:** Se aplica una **Media Móvil Exponencial (EMA)** a los vectores 3D para eliminar el *jitter* (temblor) residual de los keypoints en tiempo real.

### 3. Fase de Evaluación Biomecánica RULA (`evaluador_rula.py`)
El Método RULA divide el cuerpo en el Grupo A (brazos, muñecas) y Grupo B (cuello, tronco, piernas). Las matemáticas detrás de esta fase incluyen:
- **Álgebra Vectorial:** Todos los ángulos articulares se calculan mediante el producto punto entre vectores 3D. 
  $\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{|\vec{u}| |\vec{v}|}\right)$
- **Marcos de Referencia Relativos:** A diferencia de sistemas ingenuos que miden los brazos contra la "gravedad absoluta", este sistema proyecta un eje local a lo largo del tronco. La flexión del brazo superior se mide *relativa al ángulo de inclinación del torso*, lo cual es el estándar biomecánico correcto.
- **Filtro de Confianza (Confidence Scoring):** En vistas donde la cámara solo capta un lado del cuerpo de manera fiable, el sistema evalúa ambos lados pero confía y reporta el score del hemisferio corporal cuya sumatoria de confianza de keypoints ($\sum P_c$) sea mayor, evitando falsas alarmas generadas por oclusiones visuales.

### 4. Módulo Auxiliar: Detección de Encorvamiento Lateral (`detector_encorvamiento.py`)
De costado y sentado, la cadera queda oculta tras el escritorio y el detector la fabrica justo debajo del hombro, por lo que el tronco parece **siempre vertical** aunque la persona se encorve. La única señal que queda es el contorno de la espalda:
- **Detección de Bordes (Canny):** Sobre un ROI del torso acotado por los keypoints (hombro/oreja), se aplica desenfoque gaussiano y el operador **Canny** para extraer los bordes.
- **Trazado Robusto del Contorno:** Fila por fila se toma el borde más cercano a la posición esperada de la espalda, descartando bordes internos de la ropa y el respaldo de la silla. Se ajusta una recta (`polyfit`) con rechazo de *outliers*; si el contorno es demasiado ruidoso, la señal se descarta y el sistema no genera falsas alarmas.
- **Ángulo Calibrado:** La inclinación del contorno se mide respecto a la vertical, se calibra contra tu postura más erguida y se escala a flexión de tronco efectiva para alimentar el score RULA cuando la cadera es fabricada.

---

## 🛠️ Estructura de Archivos

- `Postura.py`: Archivo principal de ejecución. Maneja el ciclo de la cámara, hilos de alarma y orquesta los demás módulos.
- `config.py`: Configuraciones globales. Aquí puedes ajustar los límites RULA, variables de la alarma, uso de GPU/CPU, umbrales de detección, calibración y activación del filtro CLAHE.
- `detector_2d.py`: Clase `Detector2D` (MediaPipe Pose Landmarker y preprocesamiento). Devuelve la pose 2D en píxeles y la pose 3D real (world landmarks). El modelo `.task` se descarga solo la primera vez.
- `elevador_3d.py`: Clase `Elevador3D` (ensamblado de la pose 3D: puntos virtuales de cuello/cadera y suavizado).
- `evaluador_rula.py`: Clase `EvaluadorRULA` (Detección de vista y lógica ergonómica).
- `detector_encorvamiento.py`: Clase `AnalizadorSiluetaLateral` (Encorvamiento lateral por bordes Canny).
- `visualizador.py`: Encargado de renderizar el HUD (Panel de control semitransparente), el esqueleto de colores, barras de riesgo y los ángulos sobre la imagen de la cámara.
- `informe.md`: Informe técnico exhaustivo con todas las técnicas, librerías y la conexión detallada del código.

---

## 🚀 Requisitos e Instalación

Necesitarás Python 3.9 o superior (probado en 3.12). Instala las dependencias requeridas ejecutando:

```bash
pip install -r requirements.txt
```

O bien, manualmente:

```bash
pip install mediapipe opencv-python numpy
```

*(Nota: En Windows, el sonido utiliza la librería nativa `winsound`, la cual ya viene incluida en Python).*

> **📦 Modelos de pose:** Los archivos de modelo de MediaPipe (`pose_landmarker_*.task`, hasta 30 MB) **no** están incluidos en el repositorio. El programa los **descarga automáticamente** la primera vez que se ejecuta, según el `MP_MODEL_COMPLEXITY` configurado, y los guarda junto al código. No necesitas hacer nada manual, solo tener conexión a internet en el primer arranque.

---

## 💻 Uso

Para iniciar el sistema, simplemente ejecuta:

```bash
python Postura.py
```

Opciones de línea de comandos:

```bash
python Postura.py --gpu   # intenta acelerar por GPU
python Postura.py --cpu   # fuerza CPU
```

> **⚠️ Sobre la GPU:** la rueda oficial de **MediaPipe para Windows en Python está compilada solo para CPU** (*GPU processing is disabled in build flags*). El flag `--gpu` intenta usar la GPU y, si el build no lo permite (tu caso en Windows), **cae automáticamente a CPU** avisándote por consola — nunca se queda sin funcionar. Habilitar GPU de verdad requeriría recompilar MediaPipe desde el código fuente (o correr en Linux). En Windows, la forma real de ganar velocidad es bajar `MP_MODEL_COMPLEXITY` (2→1→0). Nota: el delegado `XNNPACK` que ves al arrancar **ya es** una optimización de CPU (multinúcleo/SIMD).

### Controles durante la ejecución:
- **`Q`**: Salir del programa y cerrar la cámara.
- **`C`**: Recalibrar tu línea base de postura. Siéntate erguido y púlsala: el sistema aprende esa postura como referencia y mide cuánto te desvías de ella (afecta tanto al encorvamiento frontal como al lateral).

### Interpretación del HUD (Panel en pantalla):
El sistema dibuja un esqueleto sobre la persona y un panel de información. El color indica el nivel de acción RULA:
- 🟢 **Verde (Score 1-2):** Postura aceptable.
- 🟡 **Amarillo (Score 3-4):** Investigar, posible cambio necesario.
- 🟠 **Naranja (Score 5-6):** Investigar y realizar cambios pronto.
- 🔴 **Rojo (Score 7+):** Riesgo alto. Cambio INMEDIATO.

---

## ⚙️ Configuración Personalizada

Puedes abrir el archivo `config.py` para adaptar el programa a tus necesidades. Algunos valores útiles:

- `CAMERA_INDEX`: Cambia a `1` si usas una cámara web externa.
- `UMBRAL_FRAMES_ALARMA`: Número de frames (tiempo) que debe mantenerse una mala postura antes de que suene la alarma.
- `PREPROCESAMIENTO_ACTIVO`: Cambia a `False` si tienes iluminación de estudio perfecta y deseas ganar 1-2 FPS.
- `MP_MODEL_COMPLEXITY`: `0` (lite, más rápido), `1` (full, equilibrado, por defecto) o `2` (heavy, más preciso pero más lento en CPU). El modelo correspondiente se descarga solo.
- `MP_VIS_MIN`: Visibilidad mínima (0-1) para usar un landmark. Bájalo si quieres que puntos parcialmente ocluidos (ej. caderas tras el escritorio) se usen igualmente.
- `ENCORVADO_ACTIVO`: Activa/desactiva la detección de encorvamiento lateral por silueta. Los parámetros `ENCORVADO_*` ajustan su sensibilidad: sube `ENCORVADO_RESIDUO_MAX` para ser más permisivo con fondos complejos, o baja `ENCORVADO_GANANCIA` si detecta encorvamiento de más.
