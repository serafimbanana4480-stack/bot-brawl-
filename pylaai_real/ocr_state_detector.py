"""
ocr_state_detector.py

Sistema de detecção de estado usando OCR (Reconhecimento Ótico de Caracteres).
Usa EasyOCR para ler texto na tela e determinar o estado do jogo.

Funcionalidades:
- Detecção de estado baseada em texto (PLAY, VICTORY, DEFEAT, etc.)
- Detecção de botões por texto
- Detecção de mensagens de erro
- Complemento aos métodos existentes (pixel + template)
- Cache de resultados para performance
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class OCRElement(Enum):
    """Elementos detectáveis via OCR."""
    PLAY_BUTTON = "play_button"
    VICTORY = "victory"
    DEFEAT = "defeat"
    DRAW = "draw"
    BRAWL_STARS = "brawl_stars"
    TROPHY = "trophy"
    SHOP = "shop"
    SETTINGS = "settings"
    CONNECTION_LOST = "connection_lost"
    LOADING = "loading"
    PROCEED = "proceed"
    PLAY_AGAIN = "play_again"
    BACK = "back"


@dataclass
class OCRDetection:
    """Resultado de uma detecção OCR."""
    element: OCRElement
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    timestamp: float = field(default_factory=time.time)


class OCRStateDetector:
    """
    Detector de estado usando OCR.
    
    Vantagens:
    - Robusto a mudanças de cor/brilho
    - Funciona em qualquer resolução
    - Pode detectar texto dinâmico (nomes de mapas, etc.)
    
    Desvantagens:
    - Mais lento que pixel/template matching
    - Requer EasyOCR instalado
    - Pode ter falsos positivos com texto similar
    """
    
    def __init__(
        self,
        languages: List[str] = None,
        enable_gpu: bool = False,
        confidence_threshold: float = 0.6,
        cache_ttl: float = 5.0
    ):
        self.languages = languages or ['en']
        self.enable_gpu = enable_gpu
        self.confidence_threshold = confidence_threshold
        self.cache_ttl = cache_ttl
        
        # Reader OCR (lazy loading)
        self._reader = None
        
        # Cache de detecções
        self.detection_cache: Dict[OCRElement, OCRDetection] = {}
        
        # Padrões de texto para cada estado
        self.text_patterns = {
            OCRElement.PLAY_BUTTON: [r"PLAY", r"Play", r"play"],
            OCRElement.VICTORY: [r"VICTORY", r"Victory", r"victory", r"WIN", r"Win"],
            OCRElement.DEFEAT: [r"DEFEAT", r"Defeat", r"defeat", r"LOSE", r"Lose"],
            OCRElement.DRAW: [r"DRAW", r"Draw", r"draw"],
            OCRElement.BRAWL_STARS: [r"BRAWL STARS", r"Brawl Stars", r"brawl stars"],
            OCRElement.TROPHY: [r"TROPHY", r"Trophy", r"trophy"],
            OCRElement.SHOP: [r"SHOP", r"Shop", r"shop"],
            OCRElement.SETTINGS: [r"SETTINGS", r"Settings", r"settings"],
            OCRElement.CONNECTION_LOST: [r"CONNECTION", r"LOST", r"Connection Lost", r"connection lost"],
            OCRElement.LOADING: [r"LOADING", r"Loading", r"loading"],
            OCRElement.PROCEED: [r"PROCEED", r"Proceed", r"proceed"],
            OCRElement.PLAY_AGAIN: [r"PLAY AGAIN", r"Play Again", r"play again"],
            OCRElement.BACK: [r"BACK", r"Back", r"back"]
        }
        
        # Regiões de interesse para cada elemento (normalizadas 0-1)
        self.rois = {
            OCRElement.PLAY_BUTTON: (0.7, 0.8, 0.95, 0.95),  # bottom-right
            OCRElement.VICTORY: (0.3, 0.3, 0.7, 0.7),      # centro
            OCRElement.DEFEAT: (0.3, 0.3, 0.7, 0.7),        # centro
            OCRElement.BRAWL_STARS: (0.3, 0.05, 0.7, 0.15), # top-center
            OCRElement.CONNECTION_LOST: (0.3, 0.4, 0.7, 0.6), # centro
        }
        
        logger.info("[OCR] Detector inicializado")
    
    def _get_reader(self):
        """Lazy loading do reader OCR."""
        if self._reader is None:
            try:
                import easyocr
                logger.info("[OCR] Carregando EasyOCR...")
                self._reader = easyocr.Reader(self.languages, gpu=self.enable_gpu)
                logger.info("[OCR] EasyOCR carregado com sucesso")
            except ImportError:
                logger.error("[OCR] EasyOCR não instalado. Instale com: pip install easyocr")
                return None
            except Exception as e:
                logger.error(f"[OCR] Erro ao carregar EasyOCR: {e}")
                return None
        
        return self._reader
    
    def detect_elements(
        self,
        screenshot: np.ndarray,
        elements: List[OCRElement] = None,
        use_cache: bool = True
    ) -> Dict[OCRElement, OCRDetection]:
        """
        Detecta múltiplos elementos via OCR.
        
        Args:
            screenshot: Imagem do jogo (BGR format)
            elements: Lista de elementos para detectar (None = todos)
            use_cache: Usar cache de detecções anteriores
        
        Returns:
            Dict com elementos detectados
        """
        
        if elements is None:
            elements = list(OCRElement)
        
        results = {}
        reader = self._get_reader()
        
        if reader is None:
            logger.warning("[OCR] Reader não disponível")
            return results
        
        for element in elements:
            # Verificar cache
            if use_cache and element in self.detection_cache:
                cached = self.detection_cache[element]
                if time.time() - cached.timestamp < self.cache_ttl:
                    results[element] = cached
                    logger.debug(f"[OCR] Cache hit para {element.value}")
                    continue
            
            # Detectar elemento
            detection = self._detect_element(screenshot, element, reader)
            
            if detection and detection.confidence >= self.confidence_threshold:
                results[element] = detection
                self.detection_cache[element] = detection
                logger.debug(f"[OCR] {element.value} detectado: {detection.confidence:.3f}")
        
        return results
    
    def _detect_element(
        self,
        screenshot: np.ndarray,
        element: OCRElement,
        reader
    ) -> Optional[OCRDetection]:
        """Detecta um elemento específico."""
        
        # Obter ROI se definida
        roi = self.rois.get(element)
        if roi:
            h, w = screenshot.shape[:2]
            x1, y1, x2, y2 = roi
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            roi_img = screenshot[y1:y2, x1:x2]
        else:
            roi_img = screenshot
            x1, y1 = 0, 0
        
        # Executar OCR
        try:
            ocr_results = reader.readtext(roi_img)
        except Exception as e:
            logger.error(f"[OCR] Erro no OCR: {e}")
            return None
        
        # Buscar padrões de texto
        patterns = self.text_patterns.get(element, [])
        
        for (bbox, text, confidence) in ocr_results:
            # Normalizar texto
            text_normalized = text.strip().lower()
            
            # Verificar se corresponde a algum padrão
            for pattern in patterns:
                if re.search(pattern, text_normalized, re.IGNORECASE):
                    # Calcular coordenadas globais (fora da ROI)
                    if roi:
                        global_bbox = (
                            x1 + bbox[0][0],
                            y1 + bbox[0][1],
                            x1 + bbox[2][0],
                            y1 + bbox[2][1]
                        )
                    else:
                        global_bbox = (
                            bbox[0][0], bbox[0][1],
                            bbox[2][0], bbox[2][1]
                        )
                    
                    # Calcular centro
                    center_x = int((global_bbox[0] + global_bbox[2]) / 2)
                    center_y = int((global_bbox[1] + global_bbox[3]) / 2)
                    
                    return OCRDetection(
                        element=element,
                        text=text,
                        confidence=confidence,
                        bbox=tuple(map(int, global_bbox)),
                        center=(center_x, center_y)
                    )
        
        return None
    
    def detect_state_from_text(self, screenshot: np.ndarray) -> Tuple[str, float]:
        """
        Determina o estado do jogo baseado em texto detectado.
        
        Returns:
            (estado, confiança)
        """
        
        # Detectar elementos-chave
        detections = self.detect_elements(screenshot, [
            OCRElement.PLAY_BUTTON,
            OCRElement.VICTORY,
            OCRElement.DEFEAT,
            OCRElement.CONNECTION_LOST,
            OCRElement.LOADING,
            OCRElement.PROCEED,
            OCRElement.PLAY_AGAIN
        ])
        
        # Prioridade de estados
        if OCRElement.VICTORY in detections:
            return "victory", detections[OCRElement.VICTORY].confidence
        
        elif OCRElement.DEFEAT in detections:
            return "defeat", detections[OCRElement.DEFEAT].confidence
        
        elif OCRElement.CONNECTION_LOST in detections:
            return "connection_lost", detections[OCRElement.CONNECTION_LOST].confidence
        
        elif OCRElement.LOADING in detections:
            return "loading", detections[OCRElement.LOADING].confidence
        
        elif OCRElement.PLAY_AGAIN in detections:
            return "match_end", detections[OCRElement.PLAY_AGAIN].confidence
        
        elif OCRElement.PROCEED in detections:
            return "match_end", detections[OCRElement.PROCEED].confidence
        
        elif OCRElement.PLAY_BUTTON in detections:
            return "lobby", detections[OCRElement.PLAY_BUTTON].confidence
        
        else:
            return "unknown", 0.0
    
    def detect_map_name(self, screenshot: np.ndarray) -> Optional[str]:
        """
        Tenta detectar o nome do mapa via OCR.
        
        Returns:
            Nome do mapa ou None
        """
        
        # ROI para nome do mapa (topo centro, abaixo do timer)
        h, w = screenshot.shape[:2]
        roi = (0.3, 0.1, 0.7, 0.2)  # 30%-70% width, 10%-20% height
        x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        roi_img = screenshot[y1:y2, x1:x2]
        
        reader = self._get_reader()
        if reader is None:
            return None
        
        try:
            ocr_results = reader.readtext(roi_img)
        except Exception as e:
            logger.error(f"[OCR] Erro ao detectar mapa: {e}")
            return None
        
        # Filtrar resultados com alta confiança
        high_conf_results = [
            text for (bbox, text, conf) in ocr_results 
            if conf > 0.7 and len(text) > 3
        ]
        
        if high_conf_results:
            # Retornar o texto mais longo (provavelmente o nome do mapa)
            map_name = max(high_conf_results, key=len)
            logger.info(f"[OCR] Mapa detectado: {map_name}")
            return map_name
        
        return None
    
    def detect_brawler_name(self, screenshot: np.ndarray) -> Optional[str]:
        """
        Tenta detectar o nome do brawler selecionado via OCR.
        
        Returns:
            Nome do brawler ou None
        """
        
        # ROI para nome do brawler (bottom center, acima do joystick)
        h, w = screenshot.shape[:2]
        roi = (0.4, 0.75, 0.6, 0.85)  # 40%-60% width, 75%-85% height
        x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        roi_img = screenshot[y1:y2, x1:x2]
        
        reader = self._get_reader()
        if reader is None:
            return None
        
        try:
            ocr_results = reader.readtext(roi_img)
        except Exception as e:
            logger.error(f"[OCR] Erro ao detectar brawler: {e}")
            return None
        
        # Filtrar resultados com alta confiança
        high_conf_results = [
            text for (bbox, text, conf) in ocr_results 
            if conf > 0.6 and len(text) > 2
        ]
        
        if high_conf_results:
            brawler_name = high_conf_results[0].strip()
            logger.info(f"[OCR] Brawler detectado: {brawler_name}")
            return brawler_name
        
        return None
    
    def invalidate_cache(self, element: Optional[OCRElement] = None):
        """Invalida cache de detecções."""
        if element:
            if element in self.detection_cache:
                del self.detection_cache[element]
                logger.debug(f"[OCR] Cache invalidado para {element.value}")
        else:
            self.detection_cache.clear()
            logger.debug("[OCR] Todo o cache invalidado")
    
    def get_detection_stats(self) -> dict:
        """Retorna estatísticas de detecção."""
        return {
            "cache_size": len(self.detection_cache),
            "reader_available": self._reader is not None,
            "languages": self.languages,
            "gpu_enabled": self.enable_gpu,
            "confidence_threshold": self.confidence_threshold
        }


# Função de conveniência para detecção híbrida
def hybrid_state_detection(
    screenshot: np.ndarray,
    pixel_detector,
    template_detector,
    ocr_detector: OCRStateDetector,
    weights: dict = None
) -> Tuple[str, float]:
    """
    Combina múltiplos métodos de detecção com pesos.
    
    Args:
        screenshot: Imagem do jogo
        pixel_detector: Detector baseado em pixels
        template_detector: Detector baseado em template
        ocr_detector: Detector OCR
        weights: Pesos para cada método (default: igual)
    
    Returns:
        (estado, confiança)
    """
    
    if weights is None:
        weights = {
            "pixel": 0.3,
            "template": 0.4,
            "ocr": 0.3
        }
    
    # Detectar com cada método
    pixel_state, pixel_conf = pixel_detector.detect(screenshot) if pixel_detector else ("unknown", 0.0)
    template_state, template_conf = template_detector.detect(screenshot) if template_detector else ("unknown", 0.0)
    ocr_state, ocr_conf = ocr_detector.detect_state_from_text(screenshot) if ocr_detector else ("unknown", 0.0)
    
    # Contar votos
    votes = {}
    total_weight = 0.0
    
    if pixel_conf > 0.5:
        votes[pixel_state] = votes.get(pixel_state, 0.0) + weights["pixel"] * pixel_conf
        total_weight += weights["pixel"]
    
    if template_conf > 0.5:
        votes[template_state] = votes.get(template_state, 0.0) + weights["template"] * template_conf
        total_weight += weights["template"]
    
    if ocr_conf > 0.5:
        votes[ocr_state] = votes.get(ocr_state, 0.0) + weights["ocr"] * ocr_conf
        total_weight += weights["ocr"]
    
    # Determinar estado vencedor
    if votes:
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / total_weight if total_weight > 0 else 0.0
        logger.debug(f"[HYBRID] Estado: {winner} (conf: {confidence:.3f}) - "
                    f"pixel={pixel_state}({pixel_conf:.2f}), "
                    f"template={template_state}({template_conf:.2f}), "
                    f"ocr={ocr_state}({ocr_conf:.2f})")
        return winner, confidence
    else:
        return "unknown", 0.0
