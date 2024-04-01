import os
import cv2
import Cv2Detector
import YoloModel
import Cv2Drawer


def mostrar_imagenes_en_directorio(directorio,
                                   label_dir):
    # Obtener la lista de archivos en el directorio
    old_yolo = YoloModel.Yolo5('Weights\\yoloSolar.pt')
    yolo = YoloModel.Yolo5('Weights\\newTrain2.pt')
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
        img2 = img.copy()
        
        boxes = yolo.get_image_markers_over_confidence(img, 0.40)
        for box in boxes:
            Cv2Drawer.box_drawer(img,**box)

        boxes2 = old_yolo.get_image_markers_over_confidence(img2, 0.40)
        for box in boxes2:
            Cv2Drawer.box_drawer(img2,**box)

        cv2.putText(img, 'Pesos Nuevos', (30,  30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0 ), 1)
        cv2.putText(img2, 'Pesos Originales', (30,  30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0 ), 1)
        alto, ancho, canales = img.shape
        img = cv2.resize(img, (int(ancho*0.8), int(alto*0.8)))
        img2 = cv2.resize(img2, (int(ancho*0.8), int(alto*0.8)))
        cv2.imshow("PREDICCION", cv2.hconcat([img2,img]))
        cv2.waitKey(50)
    cv2.destroyAllWindows()


# Ruta del directorio que contiene las imágenes
directorio_imagenes = f"decode"
dir_label = f"labels"
# Llamar a la función para mostrar las imágenes
mostrar_imagenes_en_directorio(directorio_imagenes,
                               dir_label)
