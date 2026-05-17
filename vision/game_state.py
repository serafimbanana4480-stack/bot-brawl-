"""
vision/game_state.py

Estrutura de dados GameState unificada — visão multimodal.

Combina:
- Detecções YOLO (objetos, inimigos, brawlers, power-ups)
- Estado do jogador (vida, super, visibilidade, ameaça)
- Valores do HUD (HP, ammo, timer, score)
- Estado do jogo (lobby, in_game, victory, defeat, etc.)
- Mapa e modo de jogo
- Metadados do frame (timestamp, latência)

Design:
- Dataclass imutável (ou quase) para passagem entre subsistemas
- Serialização para JSON (dataset, logs, replay)
- Validação de campos obrigatórios
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class YoloDetection:
    """Uma detecção YOLO normalizada."""

    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalizados 0-1
    confidence: float
    center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "center": list(self.center),
        }


@dataclass
class HudValues:
    """Valores extraídos do HUD via OCR."""

    hp: Optional[float] = None  # 0.0–1.0
    hp_confidence: float = 0.0
    ammo: Optional[int] = None  # 0–3
    ammo_confidence: float = 0.0
    super_charge: Optional[float] = None  # 0.0–1.0
    super_confidence: float = 0.0
    timer_seconds: Optional[float] = None
    timer_confidence: float = 0.0
    team_score: Optional[Tuple[int, int]] = None  # (nosso, inimigo)
    score_confidence: float = 0.0
    cube_count: Optional[int] = None
    cube_confidence: float = 0.0
    gem_count: Optional[int] = None
    gem_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hp": self.hp,
            "hp_confidence": self.hp_confidence,
            "ammo": self.ammo,
            "ammo_confidence": self.ammo_confidence,
            "super_charge": self.super_charge,
            "super_confidence": self.super_confidence,
            "timer_seconds": self.timer_seconds,
            "timer_confidence": self.timer_confidence,
            "team_score": list(self.team_score) if self.team_score else None,
            "score_confidence": self.score_confidence,
            "cube_count": self.cube_count,
            "cube_confidence": self.cube_confidence,
            "gem_count": self.gem_count,
            "gem_confidence": self.gem_confidence,
        }


@dataclass
class PlayerStatus:
    """Estado do jogador inferido."""

    life: str = "unknown"  # alive, dead, spectating
    super_ready: bool = False
    gadget_ready: bool = False
    in_bush: bool = False
    threat_level: str = "unknown"  # safe, caution, danger, critical
    hp: float = 1.0
    ammo: int = -1
    super_charge: float = -1.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "life": self.life,
            "super_ready": self.super_ready,
            "gadget_ready": self.gadget_ready,
            "in_bush": self.in_bush,
            "threat_level": self.threat_level,
            "hp": self.hp,
            "ammo": self.ammo,
            "super_charge": self.super_charge,
            "confidence": self.confidence,
        }


@dataclass
class GameState:
    """
    Estado completo do jogo — saída do pipeline multimodal.

    Combina todas as fontes de visão numa única estrutura.
    """

    # Identificação
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)

    # Estado do jogo (do UnifiedStateDetector)
    game_state: str = "unknown"  # lobby, in_game, victory, defeat, loading, etc.
    game_state_confidence: float = 0.0

    # Modo e mapa
    game_mode: Optional[str] = None  # gem_grab, showdown, brawl_ball, etc.
    map_name: Optional[str] = None

    # Detecções YOLO
    detections: List[YoloDetection] = field(default_factory=list)
    player_detection: Optional[YoloDetection] = None
    enemy_detections: List[YoloDetection] = field(default_factory=list)
    powerup_detections: List[YoloDetection] = field(default_factory=list)

    # HUD (OCR)
    hud: HudValues = field(default_factory=HudValues)

    # Estado do jogador
    player: PlayerStatus = field(default_factory=PlayerStatus)

    # Metadados
    latency_ms: float = 0.0
    resolution: Tuple[int, int] = (1920, 1080)

    # Raw data (opcional, para debug)
    raw_screenshot_shape: Optional[Tuple[int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict (JSON-friendly)."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "game_state": self.game_state,
            "game_state_confidence": self.game_state_confidence,
            "game_mode": self.game_mode,
            "map_name": self.map_name,
            "detections": [d.to_dict() for d in self.detections],
            "player_detection": self.player_detection.to_dict() if self.player_detection else None,
            "enemy_detections": [d.to_dict() for d in self.enemy_detections],
            "powerup_detections": [d.to_dict() for d in self.powerup_detections],
            "hud": self.hud.to_dict(),
            "player": self.player.to_dict(),
            "latency_ms": self.latency_ms,
            "resolution": list(self.resolution),
            "raw_screenshot_shape": list(self.raw_screenshot_shape) if self.raw_screenshot_shape else None,
        }

    @property
    def is_in_game(self) -> bool:
        """True se estamos numa partida ativa."""
        return self.game_state in ("in_game", "in_game_countdown")

    @property
    def can_act(self) -> bool:
        """True se o jogador pode agir (vivo + em jogo)."""
        return self.is_in_game and self.player.life == "alive"

    @property
    def enemy_count(self) -> int:
        """Número de inimigos detetados."""
        return len(self.enemy_detections)

    @property
    def is_super_ready(self) -> bool:
        """True se o super está pronto."""
        return self.player.super_ready or (
            self.player.super_charge is not None and self.player.super_charge >= 0.99
        )

    @property
    def has_ammo(self) -> bool:
        """True se há ammo disponível. ammo=-1 significa desconhecido (assume ok)."""
        return self.player.ammo is None or self.player.ammo == -1 or self.player.ammo > 0
