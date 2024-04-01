import os
import cv2
import Cv2Detector
import YoloModel
import Cv2Drawer


def mostrar_imagenes_en_directorio(directorio,
                                   label_dir):
    # Obtener la lista de archivos en el directorio
    yolo = YoloModel.Yolo5('Weights\yoloSolar.pt')
    lista_archivos = os.listdir(directorio)
   
    # Filtrar solo los archivos de imagen
    imagenes = [archivo for archivo in lista_archivos if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not imagenes:
        print("No se encontraron imágenes en el directorio.")
        return

    for imagen in imagenes:
        # Construir la ruta completa de la imagen
        ruta_imagen = os.path.join(directorio, imagen)
        
        # Cargar la imagen con OpenCV
        img = cv2.imread(ruta_imagen)
        alto, ancho, canales = img.shape

        boxes = yolo.get_image_markers_over_confidence(img,0.75)
        for box in boxes:
            Cv2Drawer.box_drawer(img,**box)
            name = ruta_imagen.split('.')[0].split('\\')[-1]
            _width = box['x_max'] - box['x_min']  
            _height = box['y_max'] - box['y_min'] 

            x_center = box['x_min'] + (_width / 2)
            y_center = box['y_min'] + (_height / 2)

            x_center /= ancho
            y_center /= alto

            _width /= ancho
            _height/= alto
            # print(f'ancho: {ancho} alto: {alto}')
            # print(box)
            # print(f"solar-panel {x_center:.2f} {y_center:.2f} {_width:.2f} {_height:.2f}")
            with open(f'{label_dir}\{name}.txt', 'a') as archivo:
                archivo.write(f"0 {x_center} {y_center} {_width} {_height}\n")

        cv2.imshow("PREDICCION", img)
        cv2.waitKey(5)
    cv2.destroyAllWindows()


# Ruta del directorio que contiene las imágenes
directorio_imagenes = f"decode"
dir_label = f"labels"
# Llamar a la función para mostrar las imágenes
mostrar_imagenes_en_directorio(directorio_imagenes,
                               dir_label)
