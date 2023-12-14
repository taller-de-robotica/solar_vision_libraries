import cv2

def box_drawer(image,  
               x_min: int, 
               y_min: int, 
               x_max: int, 
               y_max: int,
               class_name : str = '',
               confidence : float = None,
               color = ( 255, 0, 228 ),
               **Kwargs):
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    if class_name:
        text = class_name if confidence is None else f'{class_name} : {confidence:.2f}'
        cv2.putText(image, text, (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)