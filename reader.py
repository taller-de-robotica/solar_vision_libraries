import os
import cv2
import Cv2Detector


def mostrar_imagenes_en_directorio(directorio):
    # Obtener la lista de archivos en el directorio

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

        # Mostrar la imagen
        panel = Cv2Detector.get_panel(img)[-1]
        alto_original, ancho_original = panel.shape[:2]
        nuevo_ancho = int(ancho_original / 1.1)
        nuevo_alto = int(alto_original / 1.1)
        panel = cv2.resize(panel, (nuevo_ancho, nuevo_alto))

        cv2.imshow("PREDICCION", panel)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


# Ruta del directorio que contiene las imágenes
directorio_imagenes = "C:/Users/fcoem_7l4lie2/Downloads/Panel Solar.v3i.yolov5pytorch/train/images"

# Llamar a la función para mostrar las imágenes
mostrar_imagenes_en_directorio(directorio_imagenes)
