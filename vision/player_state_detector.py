"""
vision/player_state_detector.py

Detecção de estado do jogador combinando múltiplas fontes de visão.

Fuses:
- YOLO (detect_main): posição do jogador, inimigos, cubos, power-ups
- OCR (OCRHudExtractor): HP, ammo, super, timer, score
- Pixel heurísticas (GameFeatureExtractor): paredes, arbustos, HP por cor

Estados detectados:
- alive / dead / spectating
- super_ready / super_charging / super_empty
- gadget_ready / gadget_used / gadget_unavailable
- in_bush / exposed
- combat / safe / danger

Design:
- Cada fonte vota com peso configurável
- Suavização temporal de estados (evita oscilação frame-a-frame)
- Eventos de transição emitidos para subsistemas (RL, observability)
- Lazy imports para não quebrar quando dependências faltam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums de estado
# ---------------------------------------------------------------------------

class LifeState(Enum):
    """Estado de vida do jogador."""
    UNKNOWN = auto()
    ALIVE = auto()
    DEAD = auto()
    SPECTATING = auto()


class SuperState(Enum):
    """Estado do super ability."""
    UNKNOWN = auto()
    READY = auto()
    CHARGING = auto()
    EMPTY = auto()


class GadgetState(Enum):
    """Estado do gadget."""
    UNKNOWN = auto()
    READY = auto()
    USED = auto()
    UNAVAILABLE = auto()


class VisibilityState(Enum):
    """Visibilidade do jogador."""
    UNKNOWN = auto()
    IN_BUSH = auto()
    EXPOSED = auto()


class ThreatState(Enum):
    """Nível de ameaça iminente."""
    UNKNOWN = auto()
    SAFE = auto()
    CAUTION = auto()
    DANGER = auto()
    CRITICAL = auto()


# ---------------------------------------------------------------------------
# PlayerState
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Estado completo do jogador num frame."""

    life: LifeState = LifeState.UNKNOWN
    super_state: SuperState = SuperState.UNKNOWN
    gadget: GadgetState = GadgetState.UNKNOWN
    visibility: VisibilityState = VisibilityState.UNKNOWN
    threat: ThreatState = ThreatState.UNKNOWN

    # Valores brutos (quando disponíveis)
    hp: float = 1.0  # 0.0–1.0
    ammo: int = -1  # 0–3, -1 = desconhecido
    super_charge: float = -1.0  # 0.0–1.0
    enemy_count_nearby: int = 0
    enemy_distance_closest: float = float("inf")

    # Metadados
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    confidence: float = 0.0  # confiança global do estado

    def to_dict(self) -> dict[str, Any]:
        return {
            "life": self.life.name,
            "super": self.super_state.name,
            "gadget": self.gadget.name,
            "visibility": self.visibility.name,
            "threat": self.threat.name,
            "hp": self.hp,
            "ammo": self.ammo,
            "super_charge": self.super_charge,
            "enemy_count_nearby": self.enemy_count_nearby,
            "enemy_distance_closest": self.enemy_distance_closest,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "confidence": self.confidence,
        }

    @property
    def can_attack(self) -> bool:
        """True se pode atacar (vivo + ammo > 0)."""
        return self.life == LifeState.ALIVE and self.ammo != 0

    @property
    def can_super(self) -> bool:
        """True se pode usar super."""
        return self.life == LifeState.ALIVE and self.super_state == SuperState.READY

    @property
    def is_vulnerable(self) -> bool:
        """True se vulnerável (exposto + perigo/crítico)."""
        return (
            self.visibility == VisibilityState.EXPOSED
            and self.threat in (ThreatState.DANGER, ThreatState.CRITICAL)
        )


# ---------------------------------------------------------------------------
# Transition event
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Evento de transição de estado (ex: ALIVE → DEAD)."""

    field: str  # nome do campo que mudou
    old: str
    new: str
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0


# ---------------------------------------------------------------------------
# PlayerStateDetector
# ---------------------------------------------------------------------------

class PlayerStateDetector:
    """
    Detector de estado do jogador com fusão multi-fonte.

    Pesos de confiança por fonte (configuráveis):
    - yolo: 0.40 (deteção direta de objetos)
    - ocr:  0.35 (dados do HUD)
    - pixel: 0.25 (heurísticas de cor/posição)
    """

    # Thresholds de ameaça
    DANGER_ENEMY_DIST = 250  # pixels — dentro disto = perigo
    CRITICAL_ENEMY_DIST = 120
    CAUTION_ENEMY_DIST = 500

    # Smoothing — número de frames para confirmar mudança de estado
    SMOOTHING_FRAMES = 3

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        danger_distance: int = 250,
        critical_distance: int = 120,
        smoothing_frames: int = 3,
        enable_ocr: bool = True,
    ):
        self.weights = weights or {"yolo": 0.40, "ocr": 0.35, "pixel": 0.25}
        self.danger_distance = danger_distance
        self.critical_distance = critical_distance
        self.smoothing_frames = smoothing_frames
        self.enable_ocr = enable_ocr

        # Estado anterior (para suavização e eventos)
        self._prev_state: PlayerState | None = None
        self._state_history: list[PlayerState] = []
        self._transition_callbacks: list[Any] = []

        # Lazy imports — preenchidos na primeira chamada
        self._ocr_extractor: Any | None = None
        self._has_ocr = False

        logger.info(
            "[PSD] Inicializado: weights=%s, danger=%dpx, smoothing=%d frames, ocr=%s",
            self.weights,
            danger_distance,
            smoothing_frames,
            enable_ocr,
        )

    # ------------------------------------------------------------------
    # Lazy OCR
    # ------------------------------------------------------------------
    def _ensure_ocr(self):
        if self._ocr_extractor is not None or self._has_ocr:
            return
        try:
            from vision.ocr_hud_extractor import OCRHudExtractor

            self._ocr_extractor = OCRHudExtractor()
            self._has_ocr = True
            logger.info("[PSD] OCRHudExtractor ligado")
        except Exception as exc:
            logger.warning("[PSD] OCRHudExtractor indisponível: %s", exc)
            self._has_ocr = False

    # ------------------------------------------------------------------
    # Fonte: YOLO detections
    # ------------------------------------------------------------------
    def _source_yolo(
        self, detections: list[dict], player_class_id: int = 0
    ) -> tuple[dict[str, Any], float]:
        """
        Analisa detecções YOLO e infere estado.

        Returns:
            (state_dict, confidence)
        """
        state: dict[str, Any] = {
            "life": LifeState.UNKNOWN,
            "enemy_count_nearby": 0,
            "enemy_distance_closest": float("inf"),
        }

        if not detections:
            return state, 0.0

        player_pos = None
        enemies = []

        for det in detections:
            cls = det.get("class", -1)
            bbox = det.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if cls == player_class_id:
                player_pos = (cx, cy)
            elif cls in (1, 2, 3):  # inimigos / outros brawlers
                enemies.append((cx, cy))

        if player_pos is not None:
            state["life"] = LifeState.ALIVE
            # Calcula distâncias aos inimigos
            for ex, ey in enemies:
                dist = np.hypot(ex - player_pos[0], ey - player_pos[1])
                if dist < state["enemy_distance_closest"]:
                    state["enemy_distance_closest"] = dist
                if dist < self.danger_distance:
                    state["enemy_count_nearby"] += 1
        else:
            # Sem deteção do jogador — pode estar morto ou fora de frame
            state["life"] = LifeState.UNKNOWN

        confidence = 0.7 if player_pos is not None else 0.3
        return state, confidence

    # ------------------------------------------------------------------
    # Fonte: OCR (HUD)
    # ------------------------------------------------------------------
    def _source_ocr(self, screenshot: np.ndarray) -> tuple[dict[str, Any], float]:
        """Extrai estado via OCR do HUD."""
        state: dict[str, Any] = {
            "hp": -1.0,
            "ammo": -1,
            "super_charge": -1.0,
            "super_state": SuperState.UNKNOWN,
        }

        if not self.enable_ocr:
            return state, 0.0

        self._ensure_ocr()
        if not self._has_ocr or self._ocr_extractor is None:
            return state, 0.0

        try:
            hud = self._ocr_extractor.extract_all(screenshot)

            if hud.hp.is_valid and hud.hp.parsed_value is not None:
                state["hp"] = hud.hp.parsed_value

            if hud.ammo.is_valid and hud.ammo.parsed_value is not None:
                state["ammo"] = int(hud.ammo.parsed_value)

            if hud.super_charge.is_valid and hud.super_charge.parsed_value is not None:
                state["super_charge"] = hud.super_charge.parsed_value
                if state["super_charge"] >= 0.99:
                    state["super_state"] = SuperState.READY
                elif state["super_charge"] > 0.01:
                    state["super_state"] = SuperState.CHARGING
                else:
                    state["super_state"] = SuperState.EMPTY

            confidence = (
                hud.hp.confidence * 0.4
                + hud.ammo.confidence * 0.3
                + hud.super_charge.confidence * 0.3
            )
            return state, confidence
        except Exception as exc:
            logger.debug("[PSD] OCR source error: %s", exc)
            return state, 0.0

    # ------------------------------------------------------------------
    # Fonte: Pixel heurísticas
    # ------------------------------------------------------------------
    def _source_pixel(
        self, screenshot: np.ndarray, detections: list[dict]
    ) -> tuple[dict[str, Any], float]:
        """Heurísticas de pixel para estado."""
        state: dict[str, Any] = {"visibility": VisibilityState.UNKNOWN}

        if screenshot is None or screenshot.size == 0:
            return state, 0.0

        try:
            # Deteta se jogador está em arbusto pela cor na posição do jogador
            player_pos = None
            for det in detections:
                if det.get("class", -1) == 0:
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    player_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    break

            if player_pos is not None:
                px, py = int(player_pos[0]), int(player_pos[1])
                h, w = screenshot.shape[:2]
                if 0 <= px < w and 0 <= py < h:
                    # Arbusto = verde escuro saturado
                    pixel = screenshot[py, px]
                    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                    # Bush: verde dominante, médio a escuro
                    if g > r + 20 and g > b + 20 and 40 < g < 140:
                        state["visibility"] = VisibilityState.IN_BUSH
                    else:
                        state["visibility"] = VisibilityState.EXPOSED

            confidence = 0.5
            return state, confidence
        except Exception as exc:
            logger.debug("[PSD] Pixel source error: %s", exc)
            return state, 0.0

    # ------------------------------------------------------------------
    # Fusão
    # ------------------------------------------------------------------
    def _fuse(
        self,
        yolo_state: dict[str, Any],
        yolo_conf: float,
        ocr_state: dict[str, Any],
        ocr_conf: float,
        pixel_state: dict[str, Any],
        pixel_conf: float,
    ) -> PlayerState:
        """Fusa os estados das 3 fontes num PlayerState final."""

        state = PlayerState()

        # Normaliza confianças para pesos
        total_conf = yolo_conf + ocr_conf + pixel_conf
        if total_conf < 0.01:
            return state  # tudo falhou — estado desconhecido

        w_y = (yolo_conf / total_conf) * self.weights["yolo"]
        w_o = (ocr_conf / total_conf) * self.weights["ocr"]
        w_p = (pixel_conf / total_conf) * self.weights["pixel"]
        w_sum = w_y + w_o + w_p

        # --- Life state ---
        life_votes: dict[LifeState, float] = {}
        if "life" in yolo_state:
            life_votes[yolo_state["life"]] = life_votes.get(yolo_state["life"], 0) + w_y
        # OCR não vota em life (a não ser via HP = 0)
        if "hp" in ocr_state and ocr_state["hp"] == 0.0:
            life_votes[LifeState.DEAD] = life_votes.get(LifeState.DEAD, 0) + w_o * 0.5
        if "life" in pixel_state:
            life_votes[pixel_state["life"]] = life_votes.get(pixel_state["life"], 0) + w_p

        if life_votes:
            state.life = max(life_votes, key=life_votes.get)

        # --- HP ---
        hp_values = []
        if "hp" in ocr_state and ocr_state["hp"] >= 0:
            hp_values.append((ocr_state["hp"], w_o))
        # YOLO não fornece HP numérico diretamente
        if hp_values:
            state.hp = sum(v * w for v, w in hp_values) / sum(w for _, w in hp_values)

        # --- Ammo ---
        if "ammo" in ocr_state and ocr_state["ammo"] >= 0:
            state.ammo = ocr_state["ammo"]

        # --- Super ---
        super_votes: dict[SuperState, float] = {}
        if "super_state" in ocr_state and ocr_state["super_state"] != SuperState.UNKNOWN:
            super_votes[ocr_state["super_state"]] = super_votes.get(ocr_state["super_state"], 0) + w_o
        if "super_charge" in ocr_state and ocr_state["super_charge"] >= 0:
            # Voto baseado no valor numérico
            sc = ocr_state["super_charge"]
            if sc >= 0.99:
                super_votes[SuperState.READY] = super_votes.get(SuperState.READY, 0) + w_o * 0.5
            elif sc > 0.01:
                super_votes[SuperState.CHARGING] = super_votes.get(SuperState.CHARGING, 0) + w_o * 0.5
        if super_votes:
            state.super_state = max(super_votes, key=super_votes.get)
        else:
            state.super_state = SuperState.UNKNOWN

        state.super_charge = ocr_state.get("super_charge", -1.0)

        # --- Visibility ---
        vis_votes: dict[VisibilityState, float] = {}
        if "visibility" in pixel_state:
            vis_votes[pixel_state["visibility"]] = vis_votes.get(pixel_state["visibility"], 0) + w_p
        if vis_votes:
            state.visibility = max(vis_votes, key=vis_votes.get)

        # --- Threat ---
        if "enemy_count_nearby" in yolo_state:
            nearby_count = yolo_state["enemy_count_nearby"]
            dist = yolo_state.get("enemy_distance_closest", float("inf"))
            has_enemies = dist != float("inf")
            if not has_enemies:
                state.threat = ThreatState.SAFE
            elif dist < self.critical_distance:
                state.threat = ThreatState.CRITICAL
            elif dist < self.danger_distance:
                state.threat = ThreatState.DANGER
            elif nearby_count == 0 and has_enemies:
                # Inimigos existem mas estão longe (fora do danger_distance)
                state.threat = ThreatState.CAUTION
            elif nearby_count > 0:
                state.threat = ThreatState.CAUTION
            else:
                state.threat = ThreatState.SAFE

        state.enemy_count_nearby = yolo_state.get("enemy_count_nearby", 0)
        state.enemy_distance_closest = yolo_state.get("enemy_distance_closest", float("inf"))

        # Confiança global = média ponderada das confianças das fontes
        state.confidence = (yolo_conf * w_y + ocr_conf * w_o + pixel_conf * w_p) / w_sum
        if w_sum < 0.01:
            state.confidence = 0.0

        return state

    # ------------------------------------------------------------------
    # Suavização temporal
    # ------------------------------------------------------------------
    def _smooth(self, new_state: PlayerState) -> PlayerState:
        """Suaviza transições bruscas usando histórico."""
        self._state_history.append(new_state)
        if len(self._state_history) > self.smoothing_frames:
            self._state_history.pop(0)

        if len(self._state_history) < self.smoothing_frames:
            return new_state  # ainda sem histórico suficiente

        # Votação majoritária para estados discretos
        def _mode(values):
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            return max(counts, key=counts.get) if counts else values[-1]

        def _safe_mean(values, default=0.0, cast=None):
            """Média segura que retorna default quando lista vazia."""
            if not values:
                return default
            m = np.mean(values)
            if cast is not None:
                return cast(round(m))
            return m

        smoothed = PlayerState()
        smoothed.life = _mode([s.life for s in self._state_history])
        smoothed.super_state = _mode([s.super_state for s in self._state_history])
        smoothed.gadget = _mode([s.gadget for s in self._state_history])
        smoothed.visibility = _mode([s.visibility for s in self._state_history])
        smoothed.threat = _mode([s.threat for s in self._state_history])

        # Valores contínuos = média (com fallback para default quando lista vazia)
        smoothed.hp = _safe_mean([s.hp for s in self._state_history if s.hp >= 0], default=1.0)
        smoothed.ammo = _safe_mean([s.ammo for s in self._state_history if s.ammo >= 0], default=3, cast=int)
        smoothed.super_charge = _safe_mean([s.super_charge for s in self._state_history if s.super_charge >= 0], default=0.0)
        smoothed.enemy_count_nearby = _safe_mean([s.enemy_count_nearby for s in self._state_history], default=0, cast=int)
        smoothed.enemy_distance_closest = _safe_mean([s.enemy_distance_closest for s in self._state_history], default=float("inf"))

        smoothed.confidence = new_state.confidence
        smoothed.timestamp = new_state.timestamp
        smoothed.frame_id = new_state.frame_id

        return smoothed

    # ------------------------------------------------------------------
    # Transições
    # ------------------------------------------------------------------
    def _detect_transitions(
        self, prev: PlayerState | None, curr: PlayerState
    ) -> list[StateTransition]:
        """Compara estado anterior com atual e retorna transições."""
        if prev is None:
            return []
        transitions = []
        fields = [
            ("life", lambda s: s.life.name),
            ("super_state", lambda s: s.super_state.name),
            ("gadget", lambda s: s.gadget.name),
            ("visibility", lambda s: s.visibility.name),
            ("threat", lambda s: s.threat.name),
        ]
        for field_name, getter in fields:
            old = getter(prev)
            new = getter(curr)
            if old != new:
                transitions.append(
                    StateTransition(
                        field=field_name,
                        old=old,
                        new=new,
                        timestamp=curr.timestamp,
                        frame_id=curr.frame_id,
                    )
                )
        return transitions

    def register_transition_callback(self, callback) -> None:
        """Regista callback para ser chamado quando há transições."""
        self._transition_callbacks.append(callback)

    def _emit_transitions(self, transitions: list[StateTransition]) -> None:
        for cb in self._transition_callbacks:
            try:
                cb(transitions)
            except Exception as exc:
                logger.warning("[PSD] Transition callback error: %s", exc)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def detect(
        self,
        screenshot: np.ndarray,
        detections: list[dict],
        frame_id: int = 0,
        return_transitions: bool = False,
    ) -> tuple[PlayerState, list[StateTransition] | None]:
        """
        Detecta estado do jogador a partir de screenshot + detecções YOLO.

        Args:
            screenshot: Imagem do jogo (RGB)
            detections: Lista de detecções YOLO
            frame_id: ID do frame (para tracking)
            return_transitions: Se True, retorna também lista de transições

        Returns:
            (PlayerState, transitions) ou (PlayerState, None)
        """
        t0 = time.time()

        # 1) Coleta de cada fonte
        yolo_state, yolo_conf = self._source_yolo(detections)
        ocr_state, ocr_conf = self._source_ocr(screenshot)
        pixel_state, pixel_conf = self._source_pixel(screenshot, detections)

        # 2) Fusão
        fused = self._fuse(yolo_state, yolo_conf, ocr_state, ocr_conf, pixel_state, pixel_conf)
        fused.timestamp = time.time()
        fused.frame_id = frame_id

        # 3) Suavização temporal
        smoothed = self._smooth(fused)

        # 4) Transições
        transitions = self._detect_transitions(self._prev_state, smoothed)
        if transitions:
            self._emit_transitions(transitions)
            for tr in transitions:
                logger.info(
                    "[PSD] Transição: %s: %s → %s (frame %d)",
                    tr.field, tr.old, tr.new, frame_id,
                )

        self._prev_state = smoothed

        elapsed = (time.time() - t0) * 1000
        logger.debug(
            "[PSD] Detect completo em %.1f ms | life=%s threat=%s conf=%.2f",
            elapsed,
            smoothed.life.name,
            smoothed.threat.name,
            smoothed.confidence,
        )

        if return_transitions:
            return smoothed, transitions
        return smoothed, None

    def reset(self) -> None:
        """Reseta estado interno (útil entre partidas)."""
        self._prev_state = None
        self._state_history.clear()
        logger.info("[PSD] Estado resetado")
