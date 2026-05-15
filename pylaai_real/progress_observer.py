"""
progress_observer.py

Observa o progresso apos cada jogo usando OCR com fallback para template matching.

Fix Error #15: Graceful fallback when easyocr is not installed.
"""

import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ProgressObserver:
    """Le resultados das partidas usando OCR (com fallback para template matching)"""

    def __init__(self):
        self.crop_region = (0, 0, 400, 200)  # Top-left corner
        self.trophies = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.last_result = None
        self.reader = None
        self._ocr_available = False
        self._ocr_backend = "none"
        self._init_ocr()

    def _init_ocr(self):
        """Inicializa o leitor OCR com fallback gracioso (Fix Error #15)."""
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=True)
            self._ocr_available = True
            self._ocr_backend = "easyocr"
            logger.info("EasyOCR inicializado com sucesso")
        except ImportError:
            try:
                import pytesseract  # type: ignore
                self.reader = pytesseract
                self._ocr_available = True
                self._ocr_backend = "pytesseract"
                logger.info("pytesseract inicializado como fallback de OCR")
            except ImportError:
                logger.warning(
                    "Nenhum OCR instalado. Usando heurísticas por cor para o resultado da partida."
                )
                self._ocr_available = False
                self._ocr_backend = "none"
        except Exception as e:
            logger.error(f"Erro ao inicializar OCR: {e}")
            self._ocr_available = False
            self._ocr_backend = "none"

    def _candidate_crop_regions(self, screenshot: np.ndarray):
        """Gera regiões candidatas onde o resultado costuma aparecer."""
        h, w = screenshot.shape[:2]
        return [
            self.crop_region,
            (int(w * 0.22), int(h * 0.06), int(w * 0.78), int(h * 0.32)),
            (int(w * 0.30), int(h * 0.10), int(w * 0.70), int(h * 0.38)),
        ]

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Melhora contraste e reduz ruído antes do OCR."""
        pil_img = Image.fromarray(image)
        gray = ImageOps.grayscale(pil_img)
        gray = ImageOps.autocontrast(gray)
        return np.array(gray)

    def _ocr_text_candidates(self, screenshot: np.ndarray):
        """Tenta extrair texto com o backend OCR disponível."""
        if not self._ocr_available or self.reader is None:
            return []

        candidates = []
        for crop_region in self._candidate_crop_regions(screenshot):
            pil_img = Image.fromarray(screenshot)
            cropped = pil_img.crop(crop_region)
            array_screenshot = self._preprocess_for_ocr(np.array(cropped))

            try:
                if self._ocr_backend == "easyocr":
                    results = self.reader.readtext(array_screenshot)
                    for _, text, conf in results:
                        candidates.append((text, conf))
                elif self._ocr_backend == "pytesseract":
                    text = self.reader.image_to_string(array_screenshot)
                    if text:
                        candidates.append((text, 1.0))
            except Exception as e:
                logger.debug(f"OCR backend '{self._ocr_backend}' falhou na região {crop_region}: {e}")

        return candidates

    def find_game_result(self, screenshot: np.ndarray) -> bool:
        """
        Tenta ler o resultado da partida (VITORIA/DERROTA).
        Retorna True se encontrou resultado.
        """
        if screenshot is None:
            return False

        # Try OCR first, with backend fallback.
        if self._ocr_available and self.reader is not None:
            ocr_result = self._find_result_ocr(screenshot)
            if ocr_result:
                return True

        # Fallback: color-based heuristic
        return self._find_result_heuristic(screenshot)

    def _find_result_ocr(self, screenshot: np.ndarray) -> bool:
        """Find game result using OCR."""
        try:
            for text, conf in self._ocr_text_candidates(screenshot):
                if conf > 0.5:
                    game_result = self._closest_match(text)
                    if game_result:
                        self._add_result(game_result)
                        return True

            return False

        except Exception as e:
            logger.error(f"Erro OCR: {e}", exc_info=True)
            return False

    def _find_result_heuristic(self, screenshot: np.ndarray) -> bool:
        """
        Fallback: detect win/loss by dominant color in result area.
        Victory screens tend to be blue/gold, defeat screens tend to be red/dark.
        """
        try:
            h, w = screenshot.shape[:2]
            # Check center-top area where result text appears
            roi = screenshot[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)]

            if roi.size == 0:
                return False

            # Calculate mean color (BGR)
            mean_color = np.mean(roi, axis=(0, 1))

            # Blue/gold dominant = victory (Brawl Stars victory screen)
            blue_ratio = mean_color[0] / (mean_color[2] + 1)  # B/R ratio
            green_val = mean_color[1]

            if blue_ratio > 1.3 and green_val > 100:
                self._add_result('win')
                return True
            elif mean_color[2] > 150 and blue_ratio < 0.7:
                self._add_result('loss')
                return True

            # Fallback adicional: vitória costuma ter brilho e saturação maiores que derrota.
            brightness = float(np.mean(np.max(roi, axis=2)))
            if brightness > 165 and green_val > 80:
                self._add_result('win')
                return True

            return False
        except Exception as e:
            logger.debug(f"Heuristic result detection failed: {e}")
            return False

    def _closest_match(self, text: str) -> Optional[str]:
        """Encontra o resultado mais proximo do texto reconhecido."""
        text = text.upper().strip()

        win_keywords = ['VICTORY', 'VITORIA', 'WIN', 'VICTORIA', 'WINNER', '#1']
        loss_keywords = ['DEFEAT', 'DERROTA', 'LOSS', 'LOSER', 'ELIMINATED']
        draw_keywords = ['DRAW', 'EMPATE', 'TIE']

        for keyword in win_keywords:
            if keyword in text:
                return 'win'

        for keyword in loss_keywords:
            if keyword in text:
                return 'loss'

        for keyword in draw_keywords:
            if keyword in text:
                return 'draw'

        return None

    def _add_result(self, result: str):
        """Regista resultado da partida."""
        self.last_result = result
        if result == 'win':
            self.wins += 1
            self.trophies += 8
            logger.info(f"VITORIA! Trofeus: {self.trophies} (W:{self.wins}/L:{self.losses})")
        elif result == 'loss':
            self.losses += 1
            self.trophies -= 6
            logger.info(f"DERROTA! Trofeus: {self.trophies} (W:{self.wins}/L:{self.losses})")
        elif result == 'draw':
            self.draws += 1
            logger.info(f"EMPATE! Trofeus: {self.trophies} (W:{self.wins}/L:{self.losses})")

        # Notify API match history (Fix Error #22)
        try:
            from .._api_bridge import record_match_if_available
            record_match_if_available(result, self.trophies)
        except Exception:
            pass  # API bridge is optional

    def add_trophies(self, amount: int):
        """Adiciona trofeus manualmente."""
        self.trophies += amount

    def get_stats(self) -> Dict:
        """Retorna estatisticas."""
        total = self.wins + self.losses + self.draws
        return {
            'trophies': self.trophies,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': total,
            'win_rate': (self.wins / total * 100) if total > 0 else 0,
            'ocr_available': self._ocr_available,
        }

    def get_last_result(self):
        """Retorna o último resultado detectado, se houver."""
        return self.last_result

    def clear_last_result(self):
        """Limpa o último resultado detectado antes de uma nova partida."""
        self.last_result = None

    def should_switch_brawler(self, target_trophies: int, target_wins: int) -> bool:
        """Verifica se deve trocar de brawler baseado nas metas."""
        stats = self.get_stats()
        if stats['trophies'] >= target_trophies:
            return True
        if stats['wins'] >= target_wins:
            return True
        return False
