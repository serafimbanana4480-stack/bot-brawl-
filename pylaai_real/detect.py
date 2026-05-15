"""
detect.py

Transforma output YOLOv8 num formato mais fácil de usar.
Baseado no PylaAI de ivanyordanovgt.
"""

from typing import Dict, List, Optional


class Detect:
    """
    Classe para detetar objetos usando YOLOv8.
    Requer classes porque modelos exportados perdem o atributo 'names'.
    """

    def __init__(self, model, ignore_classes=None, classes=None, conf=0.5, model_config=None):
        self.model = model
        self.classes = classes or {}
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.conf = conf
        self.model_config = model_config or {
            'conf': 0.65,
            'device': 'cpu',
            'verbose': False
        }

    def detect_objects(self, img) -> Dict[str, List[List[int]]]:
        """
        Processa output do modelo YOLOv8.
        Retorna: {class_name: [[x1, y1, x2, y2], ...]}
        """
        detected = {}
        results = self.model(img, conf=self.conf)
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.ignore_classes:
                    continue
                
                class_name = self.classes.get(cls_id, f"class_{cls_id}")
                
                bbox = box.xyxy[0].cpu().numpy()
                bbox = bbox.astype(int).tolist()
                
                if class_name not in detected:
                    detected[class_name] = []
                detected[class_name].append(bbox)
        
        return detected

    def get_center(self, bbox: List[int]) -> tuple:
        """Calcula centro de uma bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcula distância entre centros de duas bounding boxes"""
        c1 = self.get_center(bbox1)
        c2 = self.get_center(bbox2)
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
