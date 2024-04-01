import cv2
import argparse

parser = argparse.ArgumentParser(description='Descripción de mi programa')

# Agregar argumento '--archivo'
parser.add_argument('--v', help='Nombre del video')
# Parsear los argumentos de la línea de comandos
args = parser.parse_args()

# Ruta del video
id  = args.v #"prueba1"
video_path = f'videos\{id}.mp4'

# Abre el video
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()
# cv2.namedWindow('Frame')
fps = cap.get(cv2.CAP_PROP_FPS)
jump = fps 
print("Frame Rate del video:", fps)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total de fotogramas del video:", total_frames)
# Lee y muestra cada frame del video
i = 0
f = 0
while True:
    
    ret, frame = cap.read() 
    i+=1
    print(f'Progreso: {i*100/total_frames:.2f}%',end='\r')

    if i % jump != 0:
        continue
    if ret == True: 
        cv2.imwrite(f'decode/{id}_frame_{f}.jpg', frame)
        f+=1
    else: 
        break
    
# Libera el objeto de captura y cierra todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
