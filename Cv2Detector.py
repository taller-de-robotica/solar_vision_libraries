import cv2
import numpy as np

_Parameter_min_area = 1000

def get_panel(image, debug = False):
        final_cotours = []

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        # filtro gaussiano para reducir el ruido
        blurred_image = cv2.GaussianBlur(gray,(21,21),cv2.BORDER_DEFAULT)
        blurred_image = cv2.equalizeHist(blurred_image)
        # Binarizar la imagen 
        _, binary_image = cv2.threshold(blurred_image, 150, 85, cv2.THRESH_BINARY)

        binary_image = cv2.GaussianBlur(binary_image ,(21,21),cv2.BORDER_DEFAULT)

        _, binary_image = cv2.threshold(blurred_image, 82, 85, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i,contour in enumerate(contours):
            epsilon = 0.02 * cv2.arcLength(contour, True)  # tolerancia
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified_contour) == 4 and cv2.contourArea(simplified_contour) >= _Parameter_min_area:
                final_cotours.append(simplified_contour)
                cv2.drawContours(image, [simplified_contour], 0, (0, 0, 255), 3)

        # Mostrar la imagen con contornos
        blurred_image3 = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
        binary_image3 = cv2.cvtColor(binary_image , cv2.COLOR_GRAY2BGR)
        resultado = np.hstack((blurred_image3,binary_image3, image))

        if debug:
            cv2.imshow('Contours', resultado)
            cv2.waitKey(1)

        return (final_cotours, blurred_image3, binary_image3, image, resultado)