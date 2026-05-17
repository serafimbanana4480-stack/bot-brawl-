"""
vision/multimodal_pipeline.py

Pipeline multimodal unificado para o bot Brawl Stars.

Combina 3 camadas de visão:
1. Camada de Objetos (YOLO) — detect_main, detect_enemies
2. Camada de Texto (OCR) — OCRHudExtractor para valores do HUD
3. Camada de Pixel/Heurísticas — PlayerStateDetector para estado do jogador

Saída: GameState unificado com todos os campos.

Design:
- Execução sequencial (YOLO já foi executado antes, recebemos as detecções)
- OCR e PlayerStateDetector são opcionais (lazy)
- Latência total < 50ms alvo
- Métricas de qualidade por camada
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from vision.game_state import (
    GameState,
    HudValues,
    PlayerStatus,
    YoloDetection,
)


class MultimodalPipeline:
    """
    Pipeline que funde múltiplas fontes de visão num GameState coeso.

    Uso:
        pipeline = MultimodalPipeline(resolution=(1920, 1080))
        game_state = pipeline.process(
            screenshot=screenshot,
            yolo_detections=detections,
            game_state_hint="in_game",
            frame_id=42,
        )
    """

    # Mapeamento de classes YOLO → nomes
    # Ajustar conforme o modelo YOLO treinado
    CLASS_NAMES: Dict[int, str] = {
        0: "player",
        1: "enemy",
        2: "brawler",
        3: "cube",
        4: "gem",
        5: "powerup",
        6: "star",
    }

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        enable_ocr: bool = True,
        enable_player_state: bool = True,
        enable_hud: bool = True,
    ):
        self.resolution = resolution
        self.enable_ocr = enable_ocr
        self.enable_player_state = enable_player_state
        self.enable_hud = enable_hud

        # Lazy-loaded components
        self._ocr_extractor: Optional[Any] = None
        self._player_detector: Optional[Any] = None
        self._has_ocr = False
        self._has_player_detector = False

        # Stats
        self.frame_count = 0
        self.total_latency_ms = 0.0

        logger.info(
            "[PIPELINE] Inicializado: resolution=%s, ocr=%s, player_state=%s, hud=%s",
            resolution,
            enable_ocr,
            enable_player_state,
            enable_hud,
        )

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _ensure_ocr(self) -> bool:
        if not self.enable_ocr:
            return False
        if self._has_ocr and self._ocr_extractor is not None:
            return True
        if self._ocr_extractor is not None and not self._has_ocr:
            return False  # já tentamos e falhou
        try:
            from vision.ocr_hud_extractor import OCRHudExtractor

            self._ocr_extractor = OCRHudExtractor(resolution=self.resolution)
            self._has_ocr = True
            logger.info("[PIPELINE] OCRHudExtractor ligado")
            return True
        except Exception as exc:
            logger.warning("[PIPELINE] OCRHudExtractor indisponível: %s", exc)
            self._has_ocr = False
            return False

    def _ensure_player_detector(self) -> bool:
        if not self.enable_player_state:
            return False
        if self._has_player_detector and self._player_detector is not None:
            return True
        if self._player_detector is not None and not self._has_player_detector:
            return False
        try:
            from vision.player_state_detector import PlayerStateDetector

            self._player_detector = PlayerStateDetector(
                smoothing_frames=1, enable_ocr=False
            )  # OCR já é feito separadamente
            self._has_player_detector = True
            logger.info("[PIPELINE] PlayerStateDetector ligado")
            return True
        except Exception as exc:
            logger.warning("[PIPELINE] PlayerStateDetector indisponível: %s", exc)
            self._has_player_detector = False
            return False

    # ------------------------------------------------------------------
    # Conversão de detecções
    # ------------------------------------------------------------------
    def _convert_detections(
        self, raw_detections: List[Dict]
    ) -> Tuple[List[YoloDetection], Optional[YoloDetection], List[YoloDetection], List[YoloDetection]]:
        """Converte detecções YOLO raw para YoloDetection estruturado."""
        all_dets: List[YoloDetection] = []
        player_det: Optional[YoloDetection] = None
        enemies: List[YoloDetection] = []
        powerups: List[YoloDetection] = []

        w, h = self.resolution

        for det in raw_detections:
            cls = det.get("class", -1)
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("confidence", 0.0)

            # Normaliza
            x1, y1, x2, y2 = bbox
            nx1, ny1 = x1 / w, y1 / h
            nx2, ny2 = x2 / w, y2 / h
            cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2

            yd = YoloDetection(
                class_id=cls,
                class_name=self.CLASS_NAMES.get(cls, f"class_{cls}"),
                bbox=(nx1, ny1, nx2, ny2),
                confidence=conf,
                center=(cx, cy),
            )
            all_dets.append(yd)

            if cls == 0:
                player_det = yd
            elif cls in (1, 2):
                enemies.append(yd)
            elif cls in (3, 4, 5, 6):
                powerups.append(yd)

        return all_dets, player_det, enemies, powerups

    # ------------------------------------------------------------------
    # Camada HUD (OCR)
    # ------------------------------------------------------------------
    def _extract_hud(self, screenshot: np.ndarray) -> HudValues:
        """Extrai valores do HUD via OCR."""
        if not self._ensure_ocr() or self._ocr_extractor is None:
            return HudValues()

        try:
            t0 = time.time()
            hud_state = self._ocr_extractor.extract_all(screenshot)
            elapsed = (time.time() - t0) * 1000
            logger.debug("[PIPELINE] OCR extraiu em %.1f ms", elapsed)

            return HudValues(
                hp=hud_state.hp.parsed_value,
                hp_confidence=hud_state.hp.confidence,
                ammo=int(hud_state.ammo.parsed_value) if hud_state.ammo.parsed_value is not None else None,
                ammo_confidence=hud_state.ammo.confidence,
                super_charge=hud_state.super_charge.parsed_value,
                super_confidence=hud_state.super_charge.confidence,
                timer_seconds=hud_state.timer.parsed_value,
                timer_confidence=hud_state.timer.confidence,
                team_score=(0, 0) if hud_state.score.parsed_value is not None else None,
                score_confidence=hud_state.score.confidence,
                cube_count=int(hud_state.cubes.parsed_value) if hud_state.cubes.parsed_value is not None else None,
                cube_confidence=hud_state.cubes.confidence,
                gem_count=int(hud_state.gems.parsed_value) if hud_state.gems.parsed_value is not None else None,
                gem_confidence=hud_state.gems.confidence,
            )
        except Exception as exc:
            logger.warning("[PIPELINE] OCR falhou: %s", exc)
            return HudValues()

    # ------------------------------------------------------------------
    # Camada Player State
    # ------------------------------------------------------------------
    def _extract_player_state(
        self, screenshot: np.ndarray, detections: List[Dict]
    ) -> PlayerStatus:
        """Extrai estado do jogador via PlayerStateDetector."""
        if not self._ensure_player_detector() or self._player_detector is None:
            return PlayerStatus()

        try:
            t0 = time.time()
            psd_state, _ = self._player_detector.detect(
                screenshot=screenshot,
                detections=detections,
                frame_id=self.frame_count,
            )
            elapsed = (time.time() - t0) * 1000
            logger.debug("[PIPELINE] PlayerState em %.1f ms", elapsed)

            # Converte enums para strings
            from vision.player_state_detector import (
                LifeState,
                SuperState,
                ThreatState,
                VisibilityState,
            )

            life_map = {
                LifeState.ALIVE: "alive",
                LifeState.DEAD: "dead",
                LifeState.SPECTATING: "spectating",
            }
            threat_map = {
                ThreatState.SAFE: "safe",
                ThreatState.CAUTION: "caution",
                ThreatState.DANGER: "danger",
                ThreatState.CRITICAL: "critical",
            }

            return PlayerStatus(
                life=life_map.get(psd_state.life, "unknown"),
                super_ready=psd_state.super_state == SuperState.READY,
                gadget_ready=psd_state.gadget.name == "READY",
                in_bush=psd_state.visibility == VisibilityState.IN_BUSH,
                threat_level=threat_map.get(psd_state.threat, "unknown"),
                hp=psd_state.hp if psd_state.hp >= 0 else 1.0,
                ammo=psd_state.ammo if psd_state.ammo >= 0 else -1,
                super_charge=psd_state.super_charge if psd_state.super_charge >= 0 else -1.0,
                confidence=psd_state.confidence,
            )
        except Exception as exc:
            logger.warning("[PIPELINE] PlayerState falhou: %s", exc)
            return PlayerStatus()

    # ------------------------------------------------------------------
    # Merge HUD + Player State
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_hud_into_player(hud: HudValues, player: PlayerStatus) -> PlayerStatus:
        """Usa dados do HUD para enriquecer o estado do jogador."""
        # HP: prefere HUD se confiança alta
        if hud.hp is not None and hud.hp_confidence >= 0.5:
            player.hp = hud.hp

        # Ammo: prefere HUD
        if hud.ammo is not None and hud.ammo_confidence >= 0.5:
            player.ammo = hud.ammo

        # Super: prefere HUD
        if hud.super_charge is not None and hud.super_confidence >= 0.5:
            player.super_charge = hud.super_charge
            player.super_ready = hud.super_charge >= 0.99

        return player

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def process(
        self,
        screenshot: np.ndarray,
        yolo_detections: List[Dict],
        game_state_hint: str = "unknown",
        game_state_confidence: float = 0.0,
        frame_id: int = 0,
    ) -> GameState:
        """
        Processa screenshot + detecções YOLO e devolve GameState unificado.

        Args:
            screenshot: Imagem RGB do jogo
            yolo_detections: Lista de dicts com detecções YOLO
            game_state_hint: Estado do jogo vindo do UnifiedStateDetector
            game_state_confidence: Confiança do estado do jogo
            frame_id: ID do frame

        Returns:
            GameState completo
        """
        t0 = time.time()
        self.frame_count += 1

        # 1) Converte detecções
        all_dets, player_det, enemies, powerups = self._convert_detections(yolo_detections)

        # 2) Extrai HUD via OCR
        hud = HudValues()
        if self.enable_hud:
            hud = self._extract_hud(screenshot)

        # 3) Extrai estado do jogador
        player = PlayerStatus()
        if self.enable_player_state:
            player = self._extract_player_state(screenshot, yolo_detections)

        # 4) Merge HUD → Player
        player = self._merge_hud_into_player(hud, player)

        # 5) Monta GameState
        h, w = screenshot.shape[:2] if screenshot is not None and screenshot.size > 0 else (0, 0)

        game_state = GameState(
            frame_id=frame_id,
            game_state=game_state_hint,
            game_state_confidence=game_state_confidence,
            detections=all_dets,
            player_detection=player_det,
            enemy_detections=enemies,
            powerup_detections=powerups,
            hud=hud,
            player=player,
            resolution=self.resolution,
            raw_screenshot_shape=(h, w, 3) if screenshot is not None and screenshot.ndim == 3 else None,
        )

        # 6) Métricas
        elapsed = (time.time() - t0) * 1000
        game_state.latency_ms = elapsed
        self.total_latency_ms += elapsed

        avg_latency = self.total_latency_ms / self.frame_count if self.frame_count > 0 else 0
        logger.debug(
            "[PIPELINE] Frame %d em %.1f ms (avg=%.1f ms) | game=%s player=%s enemies=%d",
            frame_id,
            elapsed,
            avg_latency,
            game_state.game_state,
            game_state.player.life,
            len(game_state.enemy_detections),
        )

        return game_state

    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas do pipeline."""
        return {
            "frame_count": self.frame_count,
            "avg_latency_ms": self.total_latency_ms / self.frame_count if self.frame_count > 0 else 0,
            "ocr_available": self._has_ocr,
            "player_detector_available": self._has_player_detector,
        }

    def reset(self) -> None:
        """Reseta estatísticas e estado interno."""
        self.frame_count = 0
        self.total_latency_ms = 0.0
        if self._player_detector is not None:
            try:
                self._player_detector.reset()
            except Exception:
                pass
        logger.info("[PIPELINE] Resetado")
