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

        # ROIs mais precisas para HUD — múltiplas janelas pequenas aumentam recall
        # sem sacrificar demasiada precisão quando o layout/resolução varia.
        self.hud_rois = {
            "match_timer_text": [
                (0.445, 0.006, 0.555, 0.060),
                (0.420, 0.000, 0.580, 0.074),
                (0.455, 0.010, 0.545, 0.052),
            ],
            "score_text": [
                (0.390, 0.035, 0.610, 0.125),
                (0.365, 0.020, 0.635, 0.140),
                (0.410, 0.045, 0.590, 0.115),
            ],
            "ability_attack": [
                (0.720, 0.745, 0.945, 0.990),
                (0.700, 0.725, 0.955, 0.995),
            ],
            "ability_super": [
                (0.055, 0.675, 0.275, 0.990),
                (0.040, 0.655, 0.290, 0.995),
            ],
            "ability_gadget": [
                (0.275, 0.675, 0.455, 0.990),
                (0.255, 0.655, 0.470, 0.995),
            ],
        }
        self._ocr_scale_factor = 2.0
        self._ocr_min_confidence = 0.35
        
        logger.info("[OCR] Detector inicializado")

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto OCR para reduzir ruído e inconsistências comuns."""
        text = (text or "").strip()
        if not text:
            return ""
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.upper()

    def _normalize_numeric_text(self, text: str) -> str:
        """Normaliza texto com foco em números/tempo, corrigindo confusões OCR típicas."""
        text = self._normalize_text(text)
        substitutions = str.maketrans({
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "|": "1",
            "S": "5",
            "B": "8",
        })
        return text.translate(substitutions)

    def _expand_roi(self, roi: Tuple[float, float, float, float], pad: float = 0.02) -> Tuple[float, float, float, float]:
        """Expande uma ROI normalizada sem sair dos limites [0, 1]."""
        x1, y1, x2, y2 = roi
        return (
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(1.0, x2 + pad),
            min(1.0, y2 + pad),
        )

    def _candidate_rois(self, key: str) -> List[Tuple[float, float, float, float]]:
        """Retorna ROIs candidatas para um HUD field, já ligeiramente alargadas."""
        candidates = self.hud_rois.get(key, [])
        if not candidates:
            return []
        return [self._expand_roi(roi, pad=0.015 if key.startswith("ability_") else 0.012) for roi in candidates]

    def _preprocess_crop_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        """Gera variantes pré-processadas para aumentar a chance de OCR correto."""
        if crop is None or crop.size == 0:
            return []

        variants: List[np.ndarray] = [crop]

        try:
            if crop.ndim == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            scaled = cv2.resize(
                gray,
                None,
                fx=self._ocr_scale_factor,
                fy=self._ocr_scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append(gray)
            variants.append(scaled)

            blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
            variants.append(blurred)

            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            variants.append(thresh)
        except Exception as e:
            logger.debug(f"[OCR] Preprocess fallback: {e}")

        return variants

    def _ocr_variants(self, reader, crop: np.ndarray):
        """Executa OCR em várias variantes do mesmo recorte e devolve resultados combinados."""
        results = []
        seen = set()
        for variant in self._preprocess_crop_variants(crop):
            try:
                for bbox, text, confidence in reader.readtext(variant):
                    key = (self._normalize_text(text), round(float(confidence), 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((bbox, text, float(confidence)))
            except Exception as e:
                logger.debug(f"[OCR] OCR variant failed: {e}")
        return results

    def parse_timer_text(self, text: str) -> Optional[float]:
        """Converte texto OCR de timer em segundos restantes."""
        normalized = self._normalize_numeric_text(text)
        if not normalized:
            return None

        match = re.search(r"(\d{1,2})\s*[:.,]\s*(\d{2})", normalized)
        if not match:
            digits = re.findall(r"\d+", normalized)
            if len(digits) >= 2:
                match = re.match(r"(?P<minutes>\d{1,2})", digits[0] + ":" + digits[1])
        if match and len(match.groups()) >= 2:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            if 0 <= seconds < 60:
                return float(minutes * 60 + seconds)
        return None

    def parse_score_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Converte texto OCR de score em uma tupla (time_a, time_b)."""
        normalized = self._normalize_numeric_text(text)
        if not normalized:
            return None

        match = re.search(r"(\d{1,2})\s*[-:/]\s*(\d{1,2})", normalized)
        if match:
            return int(match.group(1)), int(match.group(2))

        digits = re.findall(r"\d+", normalized)
        if len(digits) >= 2:
            return int(digits[0]), int(digits[1])
        return None

    def parse_ability_state(self, text: str) -> Optional[bool]:
        """Tenta inferir se um indicador de habilidade está pronto ou não."""
        normalized = self._normalize_text(text)
        if not normalized:
            return None

        ready_keywords = ("READY", "FULL", "CHARGED", "AVAILABLE", "ARMED", "ON")
        not_ready_keywords = ("COOLDOWN", "RECHARGING", "CHARGING", "FILLING", "WAIT", "OFF")

        if any(keyword in normalized for keyword in ready_keywords):
            return True
        if any(keyword in normalized for keyword in not_ready_keywords):
            return False
        return None
    
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
        
        # Executar OCR com variantes de pré-processamento para maximizar recall
        ocr_results = self._ocr_variants(reader, roi_img)
        
        # Buscar padrões de texto
        patterns = self.text_patterns.get(element, [])
        
        for (bbox, text, confidence) in ocr_results:
            # Normalizar texto
            text_normalized = self._normalize_text(text).lower()
            if confidence < self._ocr_min_confidence:
                continue
            
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
        
        ocr_results = self._ocr_variants(reader, roi_img)
        
        # Filtrar resultados com alta confiança
        high_conf_results = [
            self._normalize_text(text) for (bbox, text, conf) in ocr_results 
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
        
        ocr_results = self._ocr_variants(reader, roi_img)
        
        # Filtrar resultados com alta confiança
        high_conf_results = [
            self._normalize_text(text) for (bbox, text, conf) in ocr_results 
            if conf > 0.6 and len(text) > 2
        ]
        
        if high_conf_results:
            brawler_name = high_conf_results[0].strip()
            logger.info(f"[OCR] Brawler detectado: {brawler_name}")
            return brawler_name
        
        return None

    def extract_hud_text(self, screenshot: np.ndarray) -> Dict[str, object]:
        """Best-effort OCR for HUD fields such as timer, score and abilities."""
        result: Dict[str, object] = {
            "match_timer_text": None,
            "match_time_remaining": None,
            "match_time_seconds": None,
            "score_text": None,
            "match_score": None,
            "ability_texts": {},
            "ability_states": {},
        }

        reader = self._get_reader()
        if reader is None or screenshot is None or screenshot.size == 0:
            return result

        h, w = screenshot.shape[:2]

        def _crop_from_roi(roi):
            x1, y1, x2, y2 = roi
            rx1, ry1, rx2, ry2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            return screenshot[ry1:ry2, rx1:rx2]

        def _best_text_from_rois(
            rois: List[Tuple[float, float, float, float]],
            min_conf: float = 0.45,
            validator=None,
        ) -> List[Tuple[str, float]]:
            candidates: List[Tuple[str, float]] = []
            for roi in rois:
                crop = _crop_from_roi(roi)
                if crop.size == 0:
                    continue
                for _, text, conf in self._ocr_variants(reader, crop):
                    if conf < min_conf:
                        continue
                    normalized = self._normalize_text(text)
                    if normalized and (validator is None or validator(normalized)):
                        candidates.append((normalized, conf))
            candidates.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
            return candidates

        timer_candidates = _best_text_from_rois(
            self._candidate_rois("match_timer_text"),
            validator=lambda text: bool(re.search(r"\d", text)) and (":" in text or "." in text or "," in text),
        )
        score_candidates = _best_text_from_rois(
            self._candidate_rois("score_text"),
            validator=lambda text: bool(re.search(r"\d", text)) and bool(re.search(r"[-/]", text)),
        )

        if timer_candidates:
            timer_text, timer_conf = timer_candidates[0]
            parsed_timer = self.parse_timer_text(timer_text)
            result["match_timer_text"] = timer_text
            result["match_time_remaining"] = parsed_timer
            result["match_time_seconds"] = parsed_timer
            if parsed_timer is not None:
                result["match_time_seconds"] = parsed_timer

        if score_candidates:
            score_text, _ = score_candidates[0]
            parsed_score = self.parse_score_text(score_text)
            result["score_text"] = score_text
            result["match_score"] = parsed_score

        ability_texts: Dict[str, str] = {}
        ability_states: Dict[str, Optional[bool]] = {}
        for key in ("ability_attack", "ability_super", "ability_gadget"):
            candidates = _best_text_from_rois(self._candidate_rois(key), min_conf=0.4)
            if not candidates:
                continue
            text, _ = candidates[0]
            ability_texts[key] = text
            ability_states[key] = self.parse_ability_state(text)

        result["ability_texts"] = ability_texts
        result["ability_states"] = ability_states

        return result
    
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
