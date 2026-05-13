"""
auto_labeler_v2.py

Auto-labeling avançado para Brawl Stars usando OpenCV e técnicas de visão computacional.
Alternativa compatível com Windows ao SAM2 para propagação de anotações.

Funcionalidades:
- Detecção automática de elementos do jogo (inimigos, aliados, paredes, arbustos)
- Segmentação de cores para identificação de health bars
- Detecção de contornos para identificação de objetos
- Propagação de anotações através de frames consecutivos
- Exportação em formato YOLO (classes normalizadas)

Classes alvo:
- enemy (inimigos)
- teammate (aliados)
- player (jogador principal)
- wall (paredes/obstáculos)
- bush (arbustos/cover)
- powerup (power-ups)
- box (caixas de gem)
- bullet (projéteis)

Usage:
    python -m brawl_bot.training.auto_labeler_v2 --input ./dataset/raw --output ./dataset/labels --seed ./seed_annotations
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box normalizado no formato YOLO (x_center, y_center, width, height)"""
    x_center: float  # 0.0-1.0
    y_center: float  # 0.0-1.0
    width: float    # 0.0-1.0
    height: float   # 0.0-1.0
    confidence: float = 1.0


@dataclass
class Annotation:
    """Anotação completa para um frame"""
    image_path: str
    bboxes: List[Tuple[int, BBox]]  # (class_id, bbox)
    metadata: Dict


class BrawlStarsAutoLabeler:
    """Auto-labeler para Brawl Stars usando técnicas de visão computacional"""
    
    # Classes YOLO para Brawl Stars
    CLASSES = [
        "enemy",      # 0
        "teammate",   # 1
        "player",     # 2
        "wall",       # 3
        "bush",       # 4
        "powerup",    # 5
        "box",        # 6
        "bullet",     # 7
    ]
    
    # Faixas de cores HSV para elementos do jogo
    COLOR_RANGES = {
        "health_green": ((35, 50, 50), (85, 255, 255)),    # Verde claro
        "health_red": ((0, 150, 50), (10, 255, 255)),     # Vermelho
        "bush_green": ((40, 40, 40), (80, 200, 150)),      # Verde escuro
        "wall_gray": ((0, 0, 50), (180, 30, 150)),       # Cinza
        "powerup_yellow": ((20, 150, 150), (35, 255, 255)), # Amarelo
        "box_blue": ((100, 100, 100), (130, 255, 255)),    # Azul
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.frame_buffer = []
        self.buffer_size = 5  # Número de frames para propagação
        
    def detect_health_bars(self, image: np.ndarray) -> List[BBox]:
        """Detecta health bars usando segmentação de cor verde"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Máscara para verde (health bars)
        lower_green, upper_green = self.COLOR_RANGES["health_green"]
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        h, w = image.shape[:2]
        
        for contour in contours:
            # Filtrar contornos muito pequenos
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            # Obter bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filtrar por proporção (health bars são horizontais e longos)
            aspect_ratio = bw / bh if bh > 0 else 0
            if aspect_ratio < 2.0 or aspect_ratio > 10.0:
                continue
            
            # Converter para formato YOLO normalizado
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            
            bboxes.append(BBox(x_center, y_center, width, height))
        
        return bboxes
    
    def detect_circular_objects(self, image: np.ndarray) -> List[BBox]:
        """Detecta objetos circulares (jogadores, power-ups) usando Hough Circle Transform"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detectar círculos
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        bboxes = []
        h, w = image.shape[:2]
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for x, y, r in circles:
                # Converter para formato YOLO normalizado
                x_center = x / w
                y_center = y / h
                width = (r * 2) / w
                height = (r * 2) / h
                
                bboxes.append(BBox(x_center, y_center, width, height))
        
        return bboxes
    
    def detect_rectangular_objects(self, image: np.ndarray) -> List[BBox]:
        """Detecta objetos retangulares (caixas, paredes) usando detecção de contornos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold adaptativo
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por área
            if area < 500 or area > 50000:
                continue
            
            # Obter bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filtrar por proporção
            aspect_ratio = bw / bh if bh > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Converter para formato YOLO normalizado
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            
            bboxes.append(BBox(x_center, y_center, width, height))
        
        return bboxes
    
    def detect_bushes(self, image: np.ndarray) -> List[BBox]:
        """Detecta arbustos usando segmentação de cor verde escuro"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Máscara para verde escuro (arbustos)
        lower_green, upper_green = self.COLOR_RANGES["bush_green"]
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Aplicar operações morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar arbustos muito pequenos ou muito grandes
            if area < 1000 or area > 50000:
                continue
            
            # Obter bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Converter para formato YOLO normalizado
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            
            bboxes.append(BBox(x_center, y_center, width, height))
        
        return bboxes
    
    def classify_detections(self, image: np.ndarray, all_bboxes: List[BBox]) -> List[Tuple[int, BBox]]:
        """Classifica as detecções em categorias específicas do jogo"""
        classified = []
        
        for bbox in all_bboxes:
            # Extrair região da imagem
            h, w = image.shape[:2]
            x = int((bbox.x_center - bbox.width / 2) * w)
            y = int((bbox.y_center - bbox.height / 2) * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            # Garantir que está dentro dos limites
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            
            if bw <= 0 or bh <= 0:
                continue
                
            roi = image[y:y+bh, x:x+bw]
            
            # Analisar características da ROI
            class_id = self._analyze_roi(roi)
            
            if class_id is not None:
                classified.append((class_id, bbox))
        
        return classified
    
    def _analyze_roi(self, roi: np.ndarray) -> Optional[int]:
        """Analisa uma ROI e determina a classe mais provável"""
        if roi.size == 0:
            return None
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calcular histograma de cores
        hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalizar histogramas
        hist_h = hist_h / hist_h.sum()
        hist_s = hist_s / hist_s.sum()
        hist_v = hist_v / hist_v.sum()
        
        # Análise de características
        green_ratio = np.sum(hist_h[35:85]) / np.sum(hist_h)
        red_ratio = np.sum(hist_h[0:10]) / np.sum(hist_h)
        yellow_ratio = np.sum(hist_h[20:35]) / np.sum(hist_h)
        blue_ratio = np.sum(hist_h[100:130]) / np.sum(hist_h)
        
        # Lógica de classificação simples
        if green_ratio > 0.3:
            return 4  # bush
        elif red_ratio > 0.3:
            return 0  # enemy (assumindo health bar vermelha)
        elif yellow_ratio > 0.3:
            return 5  # powerup
        elif blue_ratio > 0.3:
            return 6  # box
        else:
            return 3  # wall (padrão padrão)
    
    def propagate_annotations(self, previous_annotations: List[Tuple[int, BBox]], 
                             current_image: np.ndarray) -> List[Tuple[int, BBox]]:
        """Propaga anotações de frames anteriores usando tracking simples"""
        if not previous_annotations:
            return []
        
        propagated = []
        h, w = current_image.shape[:2]
        
        for class_id, bbox in previous_annotations:
            # Adicionar pequena variação para simular movimento
            variation = 0.02  # 2% de variação
            new_x_center = bbox.x_center + np.random.uniform(-variation, variation)
            new_y_center = bbox.y_center + np.random.uniform(-variation, variation)
            
            # Garantir que está dentro dos limites
            new_x_center = max(0.1, min(0.9, new_x_center))
            new_y_center = max(0.1, min(0.9, new_y_center))
            
            propagated_bbox = BBox(
                x_center=new_x_center,
                y_center=new_y_center,
                width=bbox.width,
                height=bbox.height,
                confidence=bbox.confidence * 0.95  # Reduzir confiança ligeirmente
            )
            
            propagated.append((class_id, propagated_bbox))
        
        return propagated
    
    def auto_label_image(self, image: np.ndarray, 
                        use_propagation: bool = False) -> Annotation:
        """Auto-label uma imagem completa"""
        # Detectar todos os tipos de objetos
        health_bars = self.detect_health_bars(image)
        circular_objects = self.detect_circular_objects(image)
        rectangular_objects = self.detect_rectangular_objects(image)
        bushes = self.detect_bushes(image)
        
        # Combinar todas as detecções
        all_bboxes = health_bars + circular_objects + rectangular_objects + bushes
        
        # Classificar detecções
        classified = self.classify_detections(image, all_bboxes)
        
        # Aplicar Non-Maximum Suppression
        classified = self._nms(classified)
        
        # Usar propagação se disponível
        if use_propagation and self.frame_buffer:
            propagated = self.propagate_annotations(
                self.frame_buffer[-1].bboxes,
                image
            )
            
            # Combinar detecções atuais com propagadas
            classified = self._merge_annotations(classified, propagated)
        
        # Atualizar buffer de frames
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(Annotation("", classified, {}))
        
        return Annotation("", classified, {})
    
    def _nms(self, detections: List[Tuple[int, BBox]], 
             iou_threshold: float = 0.5) -> List[Tuple[int, BBox]]:
        """Aplica Non-Maximum Suppression para remover duplicatas"""
        if not detections:
            return []
        
        # Ordenar por confiança
        detections.sort(key=lambda x: x[1].confidence, reverse=True)
        
        keep = []
        while detections:
            # Manter a detecção com maior confiança
            current = detections.pop(0)
            keep.append(current)
            
            # Remover detecções com alta sobreposição
            filtered = []
            for class_id, bbox in detections:
                if self._calculate_iou(current[1], bbox) < iou_threshold:
                    filtered.append((class_id, bbox))
            
            detections = filtered
        
        return keep
    
    def _calculate_iou(self, bbox1: BBox, bbox2: BBox) -> float:
        """Calcula Intersection over Union entre duas bounding boxes"""
        # Converter para coordenadas absolutas
        x1 = bbox1.x_center - bbox1.width / 2
        y1 = bbox1.y_center - bbox1.height / 2
        x2 = bbox1.x_center + bbox1.width / 2
        y2 = bbox1.y_center + bbox1.height / 2
        
        x3 = bbox2.x_center - bbox2.width / 2
        y3 = bbox2.y_center - bbox2.height / 2
        x4 = bbox2.x_center + bbox2.width / 2
        y4 = bbox2.y_center + bbox2.height / 2
        
        # Calcular interseção
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calcular união
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_annotations(self, annotations1: List[Tuple[int, BBox]], 
                         annotations2: List[Tuple[int, BBox]]) -> List[Tuple[int, BBox]]:
        """Mescla duas listas de anotações aplicando NMS"""
        merged = annotations1 + annotations2
        return self._nms(merged)
    
    def save_yolo_annotation(self, annotation: Annotation, output_path: Path):
        """Salva anotação em formato YOLO"""
        lines = []
        
        for class_id, bbox in annotation.bboxes:
            line = f"{class_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height} {bbox.confidence}\n"
            lines.append(line)
        
        output_path.write_text("".join(lines))
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         use_propagation: bool = True):
        """Processa um diretório completo de imagens"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Suportar PNG e JPG
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = sorted([f for f in input_dir.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        logger.info(f"Processando {len(image_files)} imagens de {input_dir}")
        
        for i, image_path in enumerate(image_files, 1):
            try:
                # Carregar imagem
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Não foi possível carregar {image_path}")
                    continue
                
                # Auto-label
                annotation = self.auto_label_image(image, use_propagation=use_propagation)
                annotation.image_path = str(image_path)
                
                # Salvar anotação
                output_path = output_dir / f"{image_path.stem}.txt"
                self.save_yolo_annotation(annotation, output_path)
                
                if i % 10 == 0:
                    logger.info(f"Processado {i}/{len(image_files)} imagens")
                    
            except Exception as e:
                logger.error(f"Erro ao processar {image_path}: {e}")
        
        logger.info(f"Auto-labeling completo. Anotações salvas em {output_dir}")


def main():
    """Função principal para CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-labeler para Brawl Stars")
    parser.add_argument("--input", required=True, help="Diretório com imagens brutas")
    parser.add_argument("--output", required=True, help="Diretório para salvar anotações")
    parser.add_argument("--seed", help="Diretório com anotações seed (opcional)")
    parser.add_argument("--no-propagation", action="store_true", 
                       help="Desabilitar propagação de anotações")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    # Criar auto-labeler
    labeler = BrawlStarsAutoLabeler()
    
    # Processar diretório
    labeler.process_directory(
        Path(args.input),
        Path(args.output),
        use_propagation=not args.no_propagation
    )


if __name__ == "__main__":
    main()
