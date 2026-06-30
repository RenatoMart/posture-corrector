# Sistema de Análisis de Postura Ergonómica (RULA) en Tiempo Real

Este proyecto es un sistema avanzado de visión artificial diseñado para monitorear y evaluar la postura ergonómica de personas en entornos de oficina en tiempo real. Utiliza una cámara web estándar y aplica el **Método RULA** (Rapid Upper Limb Assessment) para determinar el nivel de riesgo ergonómico y emitir alertas si se detectan posturas perjudiciales sostenidas.

## 🌟 Características Principales

El sistema ha sido adaptado específicamente para **entornos reales de oficina**, superando las limitaciones de los sistemas de detección tradicionales:

- **Independencia del Ángulo de Cámara (Vistas Laterales):** Funciona perfectamente incluso si la cámara está ubicada a un costado. El sistema es capaz de detectar la postura viendo un solo hombro y estimando anatómicamente el lado oculto.
- **Detección de Medio Cuerpo (Oclusión Inferior):** No requiere que la persona esté de cuerpo entero en la cámara. Si las caderas o piernas no son visibles, el sistema calcula las proporciones antropométricas usando los hombros y el torso superior.
- **Independencia de la Dirección de la Mirada:** Utiliza las orejas como referencia principal para la inclinación de la cabeza. La persona puede mirar monitores laterales o el teclado sin que el sistema asuma posturas incorrectas del cuello.
- **Filtro de Iluminación Inteligente (CLAHE):** Integra ecualización adaptativa de contraste. Funciona excelente en condiciones de baja luz o contraluz sin afectar el rendimiento (FPS).
- **Rangos RULA Adaptados a Oficina:** Reconoce que la flexión natural del brazo (0-40°) y antebrazo (45-120°) al usar teclado y mouse son posturas aceptables y cómodas, evitando falsas alarmas (rojos) comunes en algoritmos estrictos genéricos.
- **Alerta Sonora Integrada:** Emite un pitido ("beep") automático si el usuario mantiene un Score RULA alto (riesgo) durante un tiempo prolongado.

---

## 🏗️ Arquitectura y Fundamentos Teóricos (Híbrida de 3 Fases)

El flujo de procesamiento se divide en 3 módulos principales, sustentados en principios matemáticos y de visión computacional:

### 1. Fase de Detección 2D y Preprocesamiento (`detector_2d.py`)
Antes de pasar la imagen a la red neuronal, se aplica un filtro para estabilizar las condiciones de luz:
- **Preprocesamiento CLAHE (Contrast Limited Adaptive Histogram Equalization):** En lugar de ecualizar la imagen BGR entera (lo que distorsiona los colores), la imagen se convierte al espacio de color **LAB**. El filtro CLAHE se aplica *únicamente* al canal **L** (Luminosidad). Esto mejora drásticamente el contraste local en zonas oscuras o con contraluz, previniendo la amplificación del ruido mediante el parámetro `clipLimit`.
- **Inferencia (YOLOv8-Nano-Pose):** Se extraen las coordenadas $(x, y)$ de los 17 puntos anatómicos clave. La red proporciona una métrica de "confianza" ($P_c$) para cada punto.
- **Inferencia Anatómica:** Si la oclusión de la cámara oculta puntos clave (ej. vista lateral), el sistema emplea heurísticas geométricas. Por ejemplo, estima el hombro oculto calculando un vector ortogonal al eje del tronco, limitando su separación al 30% del ancho biacromial visible.

### 2. Fase de Elevación Geométrica a 3D (`elevador_3d.py`)
Dado que una sola cámara carece de información de profundidad, el sistema reconstruye el eje Z utilizando antropometría y óptica básica:
- **Modelo de Cámara Pinhole:** Se estima la distancia focal ($f$) en píxeles usando el FOV (Field of View) de la cámara:  
  $f = \frac{ancho\_frame / 2}{\tan(FOV / 2)}$
- **Proporciones de Drillis & Contini:** Se usan tablas estadísticas poblacionales para conocer la longitud real de los segmentos corporales en función de la altura total del sujeto (ej. *Torso = Altura × 0.288*).
- **Cálculo de Profundidad (Z-Lifting):** Si la proyección 2D de un segmento corporal (en píxeles) aparece más corta que su longitud teórica proyectada, la diferencia se atribuye matemáticamente al escorzo (el segmento se aleja o acerca a la lente). Usando el teorema de Pitágoras en 3D, el sistema despeja el vector de profundidad ($Z$) para cada articulación hija partiendo del centro del torso.
- **Suavizado Temporal:** Se aplica una **Media Móvil Exponencial (EMA)** a los vectores 3D resultantes para eliminar el *jitter* (temblor) de los keypoints en tiempo real.

### 3. Fase de Evaluación Biomecánica RULA (`evaluador_rula.py`)
El Método RULA divide el cuerpo en el Grupo A (brazos, muñecas) y Grupo B (cuello, tronco, piernas). Las matemáticas detrás de esta fase incluyen:
- **Álgebra Vectorial:** Todos los ángulos articulares se calculan mediante el producto punto entre vectores 3D. 
  $\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{|\vec{u}| |\vec{v}|}\right)$
- **Marcos de Referencia Relativos:** A diferencia de sistemas ingenuos que miden los brazos contra la "gravedad absoluta", este sistema proyecta un eje local a lo largo del tronco. La flexión del brazo superior se mide *relativa al ángulo de inclinación del torso*, lo cual es el estándar biomecánico correcto.
- **Filtro de Confianza (Confidence Scoring):** En vistas donde la cámara solo capta un lado del cuerpo de manera fiable, el sistema evalúa ambos lados pero confía y reporta el score del hemisferio corporal cuya sumatoria de confianza de keypoints ($\sum P_c$) sea mayor, evitando falsas alarmas generadas por oclusiones visuales.

---

## 🛠️ Estructura de Archivos

- `Postura.py`: Archivo principal de ejecución. Maneja el ciclo de la cámara, hilos de alarma y orquesta los demás módulos.
- `config.py`: Configuraciones globales. Aquí puedes ajustar los límites RULA, variables de la alarma, uso de GPU/CPU, umbrales de detección y activación del filtro CLAHE.
- `detector_2d.py`: Clase `Detector2D` (YOLO y preprocesamiento).
- `elevador_3d.py`: Clase `Elevador3D` (Geometría).
- `evaluador_rula.py`: Clase `EvaluadorRULA` (Lógica ergonómica).
- `visualizador.py`: Encargado de renderizar el HUD (Panel de control semitransparente), el esqueleto de colores, barras de riesgo y los ángulos sobre la imagen de la cámara.

---

## 🚀 Requisitos e Instalación

Necesitarás Python 3.8 o superior. Instala las dependencias requeridas ejecutando:

```bash
pip install ultralytics opencv-python numpy
```

*(Nota: En Windows, el sonido utiliza la librería nativa `winsound`, la cual ya viene incluida en Python).*

---

## 💻 Uso

Para iniciar el sistema, simplemente ejecuta:

```bash
python Postura.py
```

### Controles durante la ejecución:
- **`Q`**: Salir del programa y cerrar la cámara.

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
- `ALTURA_SUJETO_CM`: Altura en cm usada para mejorar la precisión del modelo 3D.
