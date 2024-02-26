## Cv2Detector.py

Este modulo contiene la función get_panel.

#### Parámetros

- `image`: Imagen de entrada en formato BGR (Blue-Green-Red) de tipo `numpy.ndarray`.
- `debug`: Un booleano que indica si se debe mostrar la imagen con los contornos detectados para propósitos de depuración. Por defecto es `False`.

#### Retorna

Una tupla que contiene los siguientes elementos:

- Una lista de contornos finales detectados que representan paneles solares.
- La imagen filtrada después de la segmentación.
- La imagen binarizada después de la segmentación.
- La imagen original con los contornos dibujados.
- La concatenación de las tres imágenes mencionadas anteriormente para propósitos de visualización.

La función `get_panel` segmenta una imagen en base a colores específicos para detectar paneles solares. Utiliza un proceso que incluye la conversión de la imagen a escala de grises, aplicación de un filtro gaussiano para reducir el ruido, binarización de la imagen, y detección de contornos. Los contornos detectados que cumplen con ciertos criterios (como tener cuatro lados y un área mínima) son considerados como posibles paneles solares y se devuelven como resultado.

## YoloModel.py

La clase Yolo5 proporciona una abstracción del modelo YOLOv5 para la detección de objetos en imágenes.

### Constructor

El constructor genera un objeto en python el cual contiene el comportamiento de la red Yolov5 para realizar predicciones de imagenes

#### Parámetros del constructor

- `model_path`: Ruta al archivo de pesos pre-entrenados del modelo YOLOv5.
- `repo_or_dir`: Directorio o repositorio de donde cargar el modelo YOLOv5. Por defecto es 'ultralytics/yolov5'.
- `eval_mode`: Indica si se debe poner el modelo en modo de evaluación. Por defecto es True.

### Método `get_image_markers`

 Detecta objetos en una imagen y devuelve sus coordenadas y confianza.

#### Parámetros

- `image`: La imagen de entrada en formato compatible con PyTorch.

#### Retorna

Una lista de diccionarios donde cada diccionario contiene las siguientes claves:

- `x_min`: Coordenada x mínima de la caja del objeto.
- `y_min`: Coordenada y mínima de la caja del objeto.
- `x_max`: Coordenada x máxima de la caja del objeto.
- `y_max`: Coordenada y máxima de la caja del objeto.
- `confidence`: Confianza de la detección.
- `class_id`: Identificador de clase del objeto detectado.
- `class_name`: Nombre de la clase del objeto detectado.

### Método `get_image_markers_over_confidence`

Detecta objetos en una imagen y devuelve solo aquellos con una confianza mayor que un umbral dado.

#### Parámetros

- `image`: La imagen de entrada en formato compatible con PyTorch.
- `min_confidence`: Umbral mínimo de confianza para considerar una detección válida.

#### Retorna

Una lista de diccionarios donde cada diccionario contiene las siguientes claves:

- `x_min`: Coordenada x mínima de la caja del objeto.
- `y_min`: Coordenada y mínima de la caja del objeto.
- `x_max`: Coordenada x máxima de la caja del objeto.
- `y_max`: Coordenada y máxima de la caja del objeto.
- `confidence`: Confianza de la detección.
- `class_id`: Identificador de clase del objeto detectado.
- `class_name`: Nombre de la clase del objeto detectado.

## Cv2Drawer.py

Este modulo contiene el método box_drawer. 
Aquí está la documentación para la función `box_drawer` en el módulo `Cv2Drawer.py`:


### Función `box_drawer`

La función `box_drawer` Se utiliza para dibujar cajas delimitadoras (bounding boxes) en una imagen utilizando la biblioteca OpenCV (cv2).

### Parámetros

- `image`: La imagen en la cual se dibujará la caja delimitadora. Debe ser en formato BGR (compatible con CV2).
- `x_min`: La coordenada x mínima de la esquina superior izquierda de la caja delimitadora.
- `y_min`: La coordenada y mínima de la esquina superior izquierda de la caja delimitadora.
- `x_max`: La coordenada x máxima de la esquina inferior derecha de la caja delimitadora.
- `y_max`: La coordenada y máxima de la esquina inferior derecha de la caja delimitadora.
- `class_name` (opcional): El nombre de la clase asociada a la caja delimitadora. Por defecto es una cadena vacía.
- `confidence` (opcional): El nivel de confianza asociado a la predicción de la clase. Por defecto es `None`.
- `color` (opcional): El color de la caja delimitadora y el texto. Debe ser una tupla en formato BGR. Por defecto es `(255, 0, 228)`, que representa un tono de rosa.
- `**Kwargs` no utilizados

## PolyDustNet.py

El módulo `PolyDustNet` implementa una red neuronal para segmentar una imagen de un panel solar en secciones con polvo y secciones limpias.

### Clase `Unet_Model`

La clase `Unet_Model` encapsula la funcionalidad para realizar la segmentación de la imagen utilizando un modelo de red neuronal UNet.

### Parametros del constructor

- model_path : Ruta al archivo del modelo pre-entrenado de la red neuronal UNet.

### Método unet

Realiza la segmentación de una imagen de un panel solar en secciones con polvo y secciones limpias.

#### Parámetros:

​    - image (numpy.ndarray): Imagen de entrada en formato RGB.

#### Retorna:

   - numpy.ndarray: Una imagen segmentada donde:
     - 0 representa el panel solar.
     - 1 representa el fondo.
     - 2 representa el polvo.

### Método `show_images`

Muestra las imágenes original y segmentada.     

#### Parámetros:        

- original_image (numpy.ndarray): Imagen original en formato RGB.        

- dust_image (numpy.ndarray): Imagen segmentada con las regiones de polvo.        

- show (bool): Indica si se debe mostrar la imagen segmentada. Por defecto es False.     

#### Retorna:        

- numpy.ndarray: Una imagen que combina la imagen original y la imagen segmentada.

## Clase `WebCamRead`

La clase `WebCamRead` se utiliza para leer un flujo de video de una cámara web y aplicar diferentes modelos de detección de objetos y segmentación de imágenes según la configuración.

### Parámetros de Configuración

- `FRAME_STEP`: El paso entre cada análisis de fotogramas para la detección de paneles solares utilizando YOLO o detección de polvo utilizando UNet.
- `DUST_FRAME_STEP`: El paso entre cada análisis de fotogramas para la segmentación de polvo utilizando UNet.
- `VIDEO_URL`: La URL del flujo de video de la cámara web.
- `UNET_MODEL_PATH`: La ruta al archivo del modelo pre-entrenado de UNet.
- `USE_YOLO`: Un booleano que indica si se debe utilizar el modelo YOLO para la detección de paneles solares. Por defecto es `False`.
- `USE_UNET`: Un booleano que indica si se debe utilizar el modelo UNet para la segmentación de polvo. Por defecto es `False`.

