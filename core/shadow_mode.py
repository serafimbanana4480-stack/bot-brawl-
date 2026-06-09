"""
core/shadow_mode.py

Shadow Mode — Testar estratégias sem risco.

Executa uma segunda instância da lógica de decisão "em paralelo"
com a principal, mas sem enviar comandos ao emulador. Compara
as decisões do shadow vs real e gera métricas de performance.

Benefícios:
- Testar novas estratégias sem perder partidas
- A/B test de algoritmos em tempo real
- Detectar regressões antes de ativar

Uso:
    shadow = ShadowMode(real_play_logic)
    shadow.set_strategy("aggressive_moo")
    shadow.on_cycle(frame, detections)  # não envia inputs
    report = shadow.compare_with_real()
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Registro de uma decisão (real ou shadow)."""
    timestamp: float
    state_hash: str
    action: str
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = ""  # "real" ou shadow strategy name


class ShadowMode:
    """
    Executor shadow de estratégias de decisão.
    """

    def __init__(self, real_decider: Optional[Any] = None, max_history: int = 1000):
        self.real_decider = real_decider
        self._strategies: Dict[str, Callable[[Dict], str]] = {}
        self._active_strategy: Optional[str] = None
        self._real_history: deque = deque(maxlen=max_history)
        self._shadow_history: deque = deque(maxlen=max_history)
        self._enabled = False
        self._agreement_count = 0
        self._disagreement_count = 0

    def register_strategy(self, name: str, decide_fn: Callable[[Dict], str]):
        """Registra uma estratégia shadow."""
        self._strategies[name] = decide_fn
        logger.info("[SHADOW] Estratégia registrada: %s", name)

    def set_strategy(self, name: str):
        """Ativa uma estratégia shadow."""
        if name not in self._strategies:
            raise ValueError(f"Estratégia '{name}' não registrada")
        self._active_strategy = name
        self._enabled = True
        logger.info("[SHADOW] Estratégia ativa: %s", name)

    def disable(self):
        """Desativa shadow mode."""
        self._enabled = False
        logger.info("[SHADOW] Desativado")

    def on_cycle(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Executa ciclo shadow. Retorna ação que o shadow tomaria.
        NÃO envia comandos ao emulador.
        """
        if not self._enabled or not self._active_strategy:
            return None

        strategy_fn = self._strategies[self._active_strategy]
        try:
            shadow_action = strategy_fn(context)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug("[SHADOW] Erro na estratégia %s: %s", self._active_strategy, e)
            return None

        self._shadow_history.append(DecisionRecord(
            timestamp=time.time(),
            state_hash=self._hash_context(context),
            action=shadow_action,
            source=self._active_strategy,
        ))

        # Comparar com real se disponível
        if self.real_decider and hasattr(self.real_decider, "last_action"):
            real_action = self.real_decider.last_action
            self._real_history.append(DecisionRecord(
                timestamp=time.time(),
                state_hash=self._hash_context(context),
                action=real_action,
                source="real",
            ))

            if real_action == shadow_action:
                self._agreement_count += 1
            else:
                self._disagreement_count += 1

        return shadow_action

    def compare_with_real(self) -> Dict[str, Any]:
        """Compara decisões shadow vs real."""
        total = self._agreement_count + self._disagreement_count
        if total == 0:
            return {"status": "no_data"}

        agreement_rate = self._agreement_count / total

        # Analisar divergências recentes
        recent_shadow = list(self._shadow_history)[-100:]
        recent_real = list(self._real_history)[-100:]

        divergence_by_state: Dict[str, int] = {}
        for s, r in zip(recent_shadow, recent_real):
            if s.state_hash == r.state_hash and s.action != r.action:
                key = f"{r.action}_vs_{s.action}"
                divergence_by_state[key] = divergence_by_state.get(key, 0) + 1

        return {
            "strategy": self._active_strategy,
            "agreement_rate": round(agreement_rate, 3),
            "agreements": self._agreement_count,
            "disagreements": self._disagreement_count,
            "shadow_decisions": len(self._shadow_history),
            "real_decisions": len(self._real_history),
            "top_divergences": sorted(divergence_by_state.items(), key=lambda x: -x[1])[:5],
        }

    def get_shadow_recommendation(self) -> Optional[str]:
        """
        Retorna recomendação: se shadow está performando melhor,
        sugere ativar como real.
        """
        comparison = self.compare_with_real()
        if comparison.get("status") == "no_data":
            return None

        # Simplificado: se agreement > 90%, shadow é muito similar
        # Se agreement < 50%, shadow é muito diferente (pode ser melhor ou pior)
        agreement = comparison.get("agreement_rate", 1.0)
        if agreement < 0.3 and len(self._shadow_history) > 100:
            return f"Shadow '{self._active_strategy}' diverge significativamente. Requer avaliação manual."
        return None

    @staticmethod
    def _hash_context(ctx: Dict) -> str:
        """Hash simples do contexto para comparação."""
        # Usar apenas chaves relevantes
        keys = ["player_hp", "enemy_count", "game_state", "brawler"]
        parts = [str(ctx.get(k, "")) for k in keys]
        return "|".join(parts)

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "active_strategy": self._active_strategy,
            "registered_strategies": list(self._strategies.keys()),
            "history_size": len(self._shadow_history),
            **self.compare_with_real(),
        }
