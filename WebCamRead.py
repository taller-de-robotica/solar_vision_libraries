# pip3 install pandas
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# ó si hay GPUs elegir comando en:
# https://pytorch.org/get-started/locally/

import cv2
import urllib.request
import numpy as np

from PolyDustNet import Unet_Model


FRAME_STEP = 100
DUST_FRAME_STEP = 500
VIDEO_URL = "http://localhost:8000/stream.mjpg"
UNET_MODEL_PATH = "Weights/sm_unet4_03.hdf5"
USE_YOLO = False
USE_UNET = False


if __name__ == '__main__':
    if USE_YOLO:
        from YoloModel import Yolo5
        YOLO_MODEL_PATH = "Weights/yoloSolar.pt"
        yolo5 = Yolo5(YOLO_MODEL_PATH)
    else:
        from Cv2Detector import get_panel

    if USE_UNET:
        unet = Unet_Model(UNET_MODEL_PATH)


    with urllib.request.urlopen(VIDEO_URL) as stream:
        # https://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera
        bstring = b''
        num_frame = 1
        num_dust_frame = 1
        while True:
            b_read = stream.read(1024 * 8)
            bstring += b_read

            a = bstring.find(b'\xff\xd8')
            b = bstring.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bstring[a:b+2]
                bstring = bstring[b+2:]
                image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('i', image)
                num_frame += 1
                if cv2.waitKey(1) == 27:
                    exit(0)
                if num_frame == FRAME_STEP:
                    # Indaga si hay panel
                    if USE_YOLO:
                        results = yolo5.get_image_markers_over_confidence(image, 0.7)
                        print(results)
                    else:
                        results = get_panel(image, True)
                    num_frame = 1
                if USE_UNET:
                    num_dust_frame += 1
                    if num_dust_frame == DUST_FRAME_STEP:
                        dust_img = unet.unet_prediction(image)
                        unet.show_images(image, dust_img, True)
                        num_dust_frame = 1
