import torch
from typing import List


class Yolo5:

    def __init__(self, 
                 model_path : str, 
                 repo_or_dir : str = 'ultralytics/yolov5',
                 eval_mode = True
                 ) -> None:
        self.__model_path = model_path
        self.__model = torch.hub.load(repo_or_dir, 'custom', path=model_path)
        # Cambiar a modo de evaluacion
        if eval_mode:
            self.__model.eval()


    def get_image_markers(self, image) -> List[dict]:
        results = self.__model(image)
        boxes = results.xyxy[0].cpu().numpy()
        boxes_list = []
        labels = results.names
        for box in boxes:
            class_id = int(box[5])
            boxes_list.append({
                'x_min' : int(box[0]),
                'y_min' : int(box[1]),
                'x_max' : int(box[2]),
                'y_max' : int(box[3]),
                'confidence' : box[4],
                'class_id' : class_id ,
                'class_name' : labels[class_id]
            })
        return boxes_list
    
    def get_image_markers_over_confidence(self, image, min_confidence : float):
        boxes = self.get_image_markers(image)
        return [box for box in boxes if box['confidence'] > min_confidence]