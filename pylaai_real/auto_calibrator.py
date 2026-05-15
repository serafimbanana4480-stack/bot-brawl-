"""
auto_calibrator.py

Sistema de recalibração automática de coordenadas.
Detecta dinamicamente posições de botões e elementos da UI usando visão computacional,
adaptando-se automaticamente a mudanças na interface do jogo.

Funcionalidades:
- Detecção automática de botões principais (Play, Brawlers, etc.)
- Template matching adaptativo com múltiplas escalas
- Sistema de fallback para coordenadas fixas se detecção falhar
- Cache de coordenadas para performance
- Validação de coordenadas detectadas
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class CalibratedCoords:
    """Coordenadas calibradas para um elemento."""
    element_name: str
    x: int
    y: int
    confidence: float
    method: str  # "template", "color", "ocr", "fallback"
    timestamp: float = field(default_factory=time.time)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


class AutoCalibrator:
    """
    Sistema de recalibração automática de coordenadas.
    
    Usa múltiplos métodos para detectar elementos:
    1. Template matching (mais preciso)
    2. Color detection (mais rápido)
    3. OCR (mais robusto para texto)
    4. Fallback para coordenadas fixas
    """
    
    def __init__(
        self,
        templates_dir: Path = None,
        cache_file: Path = None,
        enable_cache: bool = True,
        cache_ttl_hours: float = 24.0
    ):
        self.templates_dir = templates_dir or Path("images/templates")
        self.cache_file = cache_file or Path("data/coords_cache.json")
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl_hours * 3600  # converter para segundos
        
        # Cache de coordenadas
        self.coords_cache: Dict[str, CalibratedCoords] = {}
        
        # Carregar cache do disco
        if self.enable_cache and self.cache_file.exists():
            self._load_cache()
        
        # Carregar templates
        self.templates = self._load_templates()
        
        logger.info(f"[AUTOCAL] Inicializado: {len(self.templates)} templates, {len(self.coords_cache)} coords cacheadas")
    
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Carrega templates de imagens."""
        templates = {}
        
        if not self.templates_dir.exists():
            logger.warning(f"[AUTOCAL] Diretório de templates não encontrado: {self.templates_dir}")
            return templates
        
        # Templates comuns para Brawl Stars
        template_names = [
            "play_button",
            "play_button_hover",
            "brawl_stars_logo",
            "trophy_icon",
            "shop_icon",
            "battle_log_icon",
            "settings_icon",
            "x_button",  # para fechar popups
            "proceed_button",
            "play_again_button"
        ]
        
        for name in template_names:
            template_path = self.templates_dir / f"{name}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if template is not None:
                    templates[name] = template
                    logger.debug(f"[AUTOCAL] Template carregado: {name}")
        
        return templates
    
    def _load_cache(self):
        """Carrega cache de coordenadas do disco."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            now = time.time()
            for name, coords_data in data.items():
                # Verificar se cache ainda é válido
                if now - coords_data['timestamp'] < self.cache_ttl:
                    self.coords_cache[name] = CalibratedCoords(**coords_data)
            
            logger.info(f"[AUTOCAL] Cache carregado: {len(self.coords_cache)} coordenadas válidas")
        except Exception as e:
            logger.warning(f"[AUTOCAL] Falha ao carregar cache: {e}")
    
    def _save_cache(self):
        """Salva cache de coordenadas no disco."""
        if not self.enable_cache:
            return
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for name, coords in self.coords_cache.items():
                data[name] = {
                    "element_name": coords.element_name,
                    "x": coords.x,
                    "y": coords.y,
                    "confidence": coords.confidence,
                    "method": coords.method,
                    "timestamp": coords.timestamp,
                    "bbox": coords.bbox
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[AUTOCAL] Cache salvo: {len(data)} coordenadas")
        except Exception as e:
            logger.warning(f"[AUTOCAL] Falha ao salvar cache: {e}")
    
    def detect_element(
        self,
        screenshot: np.ndarray,
        element_name: str,
        method: str = "auto",
        fallback_coords: Optional[Tuple[int, int]] = None
    ) -> Optional[CalibratedCoords]:
        """
        Detecta posição de um elemento na screenshot.
        
        Args:
            screenshot: Imagem do jogo (BGR format do OpenCV)
            element_name: Nome do elemento para detectar
            method: Método de detecção ("auto", "template", "color", "ocr", "fallback")
            fallback_coords: Coordenadas fixas de fallback (x, y)
        
        Returns:
            CalibratedCoords com posição e confiança, ou None se falhar
        """
        
        # Verificar cache primeiro
        if element_name in self.coords_cache:
            cached = self.coords_cache[element_name]
            # Cache ainda válido?
            if time.time() - cached.timestamp < self.cache_ttl:
                logger.debug(f"[AUTOCAL] Cache hit para {element_name}")
                return cached
        
        # Tentar detecção
        result = None
        
        if method == "auto":
            # Tentar múltiplos métodos em ordem de precisão
            for try_method in ["template", "color", "ocr"]:
                result = self._detect_with_method(screenshot, element_name, try_method)
                if result and result.confidence > 0.6:
                    break
        else:
            result = self._detect_with_method(screenshot, element_name, method)
        
        # Se falhou, usar fallback
        if result is None and fallback_coords:
            logger.warning(f"[AUTOCAL] Detecção falhou para {element_name}, usando fallback")
            result = CalibratedCoords(
                element_name=element_name,
                x=fallback_coords[0],
                y=fallback_coords[1],
                confidence=0.5,
                method="fallback"
            )
        
        # Salvar no cache se sucesso
        if result and result.confidence > 0.5:
            self.coords_cache[element_name] = result
            self._save_cache()
        
        return result
    
    def _detect_with_method(
        self,
        screenshot: np.ndarray,
        element_name: str,
        method: str
    ) -> Optional[CalibratedCoords]:
        """Detecta elemento usando método específico."""
        
        if method == "template":
            return self._detect_template(screenshot, element_name)
        elif method == "color":
            return self._detect_color(screenshot, element_name)
        elif method == "ocr":
            return self._detect_ocr(screenshot, element_name)
        else:
            logger.warning(f"[AUTOCAL] Método desconhecido: {method}")
            return None
    
    def _detect_template(
        self,
        screenshot: np.ndarray,
        element_name: str
    ) -> Optional[CalibratedCoords]:
        """Detecta usando template matching multi-escala."""
        
        if element_name not in self.templates:
            logger.debug(f"[AUTOCAL] Template não encontrado: {element_name}")
            return None
        
        template = self.templates[element_name]
        h, w = template.shape[:2]
        
        # Tentar múltiplas escalas (0.8x a 1.2x)
        best_match = None
        best_confidence = 0.0
        
        for scale in np.linspace(0.8, 1.2, 9):
            # Redimensionar template
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            scaled_template = cv2.resize(template, (scaled_w, scaled_h))
            
            # Template matching
            result = cv2.matchTemplate(screenshot, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                # Calcular centro do match
                center_x = max_loc[0] + scaled_w // 2
                center_y = max_loc[1] + scaled_h // 2
                
                best_match = CalibratedCoords(
                    element_name=element_name,
                    x=center_x,
                    y=center_y,
                    confidence=max_val,
                    method="template",
                    bbox=(max_loc[0], max_loc[1], max_loc[0] + scaled_w, max_loc[1] + scaled_h)
                )
        
        if best_match and best_match.confidence > 0.7:
            logger.debug(f"[AUTOCAL] Template match para {element_name}: {best_match.confidence:.3f}")
            return best_match
        
        return None
    
    def _detect_color(
        self,
        screenshot: np.ndarray,
        element_name: str
    ) -> Optional[CalibratedCoords]:
        """Detecta usando análise de cor (para botões coloridos)."""
        
        # Converter para HSV
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
        
        # Definir ranges de cor para elementos comuns
        color_ranges = {
            "play_button": {
                "lower": np.array([0, 150, 150]),    # Vermelho/laranja
                "upper": np.array([20, 255, 255])
            },
            "x_button": {
                "lower": np.array([0, 0, 200]),      # Branco/cinza claro
                "upper": np.array([180, 50, 255])
            },
            "trophy_icon": {
                "lower": np.array([20, 100, 100]),   # Amarelo/dourado
                "upper": np.array([40, 255, 255])
            }
        }
        
        if element_name not in color_ranges:
            return None
        
        # Criar máscara
        lower = color_ranges[element_name]["lower"]
        upper = color_ranges[element_name]["upper"]
        mask = cv2.inRange(hsv, lower, upper)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Encontrar contorno maior
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular centro
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Calcular área para confiança
        area = cv2.contourArea(largest_contour)
        img_area = screenshot.shape[0] * screenshot.shape[1]
        confidence = min(1.0, area / (img_area * 0.01))  # Normalizar
        
        if confidence > 0.3:
            logger.debug(f"[AUTOCAL] Color detection para {element_name}: {confidence:.3f}")
            return CalibratedCoords(
                element_name=element_name,
                x=center_x,
                y=center_y,
                confidence=confidence,
                method="color"
            )
        
        return None
    
    def _detect_ocr(
        self,
        screenshot: np.ndarray,
        element_name: str
    ) -> Optional[CalibratedCoords]:
        """Detecta usando OCR (para botões com texto)."""
        
        try:
            import easyocr
        except ImportError:
            logger.warning("[AUTOCAL] EasyOCR não instalado, OCR não disponível")
            return None
        
        # Textos esperados para cada elemento
        text_patterns = {
            "play_button": ["PLAY", "Play"],
            "proceed_button": ["PROCEED", "Proceed"],
            "play_again_button": ["PLAY AGAIN", "Play Again"],
            "x_button": ["X", "✕"]
        }
        
        if element_name not in text_patterns:
            return None
        
        # Inicializar reader (lazy)
        if not hasattr(self, '_ocr_reader'):
            self._ocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Executar OCR
        results = self._ocr_reader.readtext(screenshot)
        
        # Buscar texto correspondente
        for (bbox, text, confidence) in results:
            for pattern in text_patterns[element_name]:
                if pattern.lower() in text.lower():
                    # Calcular centro do bbox
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[2]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    logger.debug(f"[AUTOCAL] OCR match para {element_name}: '{text}' ({confidence:.3f})")
                    return CalibratedCoords(
                        element_name=element_name,
                        x=center_x,
                        y=center_y,
                        confidence=confidence,
                        method="ocr",
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    )
        
        return None
    
    def invalidate_cache(self, element_name: Optional[str] = None):
        """Invalida cache para um elemento ou todos."""
        if element_name:
            if element_name in self.coords_cache:
                del self.coords_cache[element_name]
                logger.info(f"[AUTOCAL] Cache invalidado para {element_name}")
        else:
            self.coords_cache.clear()
            logger.info("[AUTOCAL] Todo o cache invalidado")
        
        self._save_cache()
    
    def get_all_cached_coords(self) -> Dict[str, CalibratedCoords]:
        """Retorna todas as coordenadas cacheadas."""
        return self.coords_cache.copy()
    
    def calibrate_all_elements(
        self,
        screenshot: np.ndarray,
        elements: List[str]
    ) -> Dict[str, CalibratedCoords]:
        """
        Calibra múltiplos elementos de uma vez.
        
        Útil para executar durante setup inicial ou quando a interface mudou.
        """
        results = {}
        
        for element in elements:
            result = self.detect_element(screenshot, element, method="auto")
            if result:
                results[element] = result
                logger.info(f"[AUTOCAL] {element}: ({result.x}, {result.y}) - {result.confidence:.3f}")
            else:
                logger.warning(f"[AUTOCAL] Falha ao detectar {element}")
        
        return results


# Função de conveniência para calibração interativa
def interactive_calibration_setup(
    screenshot_taker,
    elements_to_calibrate: List[str]
) -> Dict[str, Tuple[int, int]]:
    """
    Executa setup de calibração interativo.
    
    Mostra screenshot e pede ao usuário para clicar em cada elemento.
    """
    import cv2
    
    coords = {}
    
    for element in elements_to_calibrate:
        # Capturar screenshot
        screenshot = screenshot_taker.capture()
        
        # Mostrar imagem
        display = screenshot.copy()
        cv2.putText(
            display,
            f"Clique no botão: {element}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Calibração Interativa", display)
        
        # Esperar clique
        clicked = False
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                coords[element] = (x, y)
                clicked = True
                cv2.destroyAllWindows()
        
        cv2.setMouseCallback("Calibração Interativa", mouse_callback)
        
        # Timeout de 30 segundos
        start_time = time.time()
        while not clicked and time.time() - start_time < 30:
            cv2.waitKey(100)
        
        if not clicked:
            logger.warning(f"[AUTOCAL] Timeout para {element}")
    
    return coords
