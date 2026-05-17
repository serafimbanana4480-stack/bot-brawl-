"""
core/smart_frame_skipper.py

Frame Skipping Inteligente — adaptativo baseado em contexto.

Decide dinamicamente se deve processar (inferência YOLO, OCR, decisão)
ou pular o frame atual, balanceando performance e qualidade.

Regras:
- Estado de jogo: mais frames (cada frame conta)
- Estado de lobby: menos frames (nada muda)
- Degradação ativa: skip mais agressivo
- Em combate: nunca skip (tempo real)
- APM alto: skip para reduzir carga
"""

import time
import logging
from typing import Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class SmartFrameSkipper:
    """
    Frame skipper adaptativo com múltiplas estratégias.
    """

    # Regras por estado do jogo
    STATE_SKIP_RULES = {
        "in_game": {"skip_ratio": 0, "max_skip": 0, "reason": "combat"},
        "in_game_countdown": {"skip_ratio": 0, "max_skip": 0, "reason": "countdown"},
        "lobby": {"skip_ratio": 4, "max_skip": 4, "reason": "static"},
        "brawler_select": {"skip_ratio": 2, "max_skip": 2, "reason": "selection"},
        "match_loading": {"skip_ratio": 3, "max_skip": 3, "reason": "loading"},
        "mission": {"skip_ratio": 5, "max_skip": 5, "reason": "menu"},
        "news": {"skip_ratio": 5, "max_skip": 5, "reason": "menu"},
        "shop": {"skip_ratio": 5, "max_skip": 5, "reason": "menu"},
        "settings": {"skip_ratio": 5, "max_skip": 5, "reason": "menu"},
        "unknown": {"skip_ratio": 2, "max_skip": 2, "reason": "uncertain"},
    }

    # Regras por modo de degradação
    DEGRADATION_MULTIPLIERS = {
        "full_quality": 1.0,
        "degraded": 2.0,
        "minimal": 4.0,
        "emergency": 10.0,
    }

    def __init__(self, max_history: int = 100):
        self._history: deque = deque(maxlen=max_history)
        self._skip_counter = 0
        self._frame_counter = 0
        self._last_state = "unknown"
        self._current_skip_interval = 0

    def should_process_frame(
        self,
        frame_counter: int,
        current_state: str,
        degradation_mode: str = "full_quality",
        apm: Optional[int] = None,
        combat_active: bool = False,
    ) -> bool:
        """
        Decide se deve processar o frame atual.
        Retorna True se deve processar, False se deve skip.
        """
        self._frame_counter = frame_counter

        # Regra 1: Em combate ativo, NUNCA skip
        if combat_active:
            return True

        # Regra 2: Estado mudou → processar (para detectar transição)
        if current_state != self._last_state:
            self._last_state = current_state
            self._skip_counter = 0
            return True

        # Regra 3: APM muito alto → skip para reduzir carga
        if apm and apm > 50:
            # Skip 50% dos frames quando APM > 50
            if frame_counter % 2 != 0:
                return False

        # Regra 4: Baseado no estado do jogo
        rule = self.STATE_SKIP_RULES.get(current_state, {"skip_ratio": 1, "max_skip": 1})
        base_skip = rule["skip_ratio"]

        # Regra 5: Multiplicador por degradação
        mult = self.DEGRADATION_MULTIPLIERS.get(degradation_mode, 1.0)
        effective_skip = int(base_skip * mult)

        # Se skip == 0, processa todos
        if effective_skip <= 0:
            return True

        # Processa a cada (effective_skip + 1) frames
        should_process = (frame_counter % (effective_skip + 1)) == 0

        self._history.append({
            "frame": frame_counter,
            "state": current_state,
            "degradation": degradation_mode,
            "processed": should_process,
            "timestamp": time.time(),
        })

        return should_process

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de skip."""
        if not self._history:
            return {"processed_ratio": 1.0, "total_frames": 0}

        processed = sum(1 for h in self._history if h["processed"])
        total = len(self._history)
        return {
            "processed_ratio": round(processed / total, 2) if total else 1.0,
            "total_frames": self._frame_counter,
            "last_state": self._last_state,
        }
