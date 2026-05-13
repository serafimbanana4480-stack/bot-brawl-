"""
core/anti_ban.py

Estratégia anti-ban robusta para o bot Brawl Stars.

Funcionalidades:
- Deteção de padrões repetitivos (sinais de botting)
- Limitador de win rate (evita taxas suspeitas)
- Randomização de schedule (horários de jogo variados)
- Obfuscation de ações (ruído e "erros humanos")
- Fingerprint randomization (comportamento único por sessão)
"""

import logging
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

__all__ = [
    "AntiBanSystem",
    "AntiBanConfig",
    "PatternDetector",
    "WinRateLimiter",
    "ScheduleRandomizer",
    "ActionObfuscator",
    "FingerprintRandomizer",
]

logger = logging.getLogger(__name__)


@dataclass
class AntiBanConfig:
    """Configurações da estratégia anti-ban."""
    enabled: bool = True
    max_win_rate: float = 0.75  # Win rate máximo aceitável antes de throttle
    min_win_rate: float = 0.20  # Win rate mínimo para não parecer intencional
    pattern_window_size: int = 20  # Janela para deteção de padrões
    pattern_similarity_threshold: float = 0.85  # Threshold de similaridade
    schedule_variance_minutes: int = 30  # Variação de horários de início
    action_noise_probability: float = 0.05  # Prob. de adicionar delay extra
    action_noise_max_delay: float = 2.0  # Delay máximo de ruído
    session_fingerprint_change_interval_hours: float = 24.0
    max_matches_per_hour: int = 8  # Limite de partidas por hora
    min_break_between_matches_sec: float = 15.0  # Pausa mínima entre partidas


class PatternDetector:
    """Detecta padrões repetitivos nas ações do bot."""

    def __init__(self, window_size: int = 20, threshold: float = 0.85):
        self.window_size = window_size
        self.threshold = threshold
        self.actions: deque = deque(maxlen=window_size * 2)

    def record_action(self, action_type: str, coordinates: Optional[tuple] = None):
        """Registra uma ação para análise de padrão."""
        self.actions.append((action_type, coordinates, time.time()))

    def detect_repetitive_pattern(self) -> bool:
        """Retorna True se um padrão repetitivo for detetado."""
        if len(self.actions) < self.window_size:
            return False

        # Simplificação: compara tipos de ação em janelas deslizantes
        recent = [a[0] for a in list(self.actions)[-self.window_size:]]
        # Contar frequência do tipo mais comum
        counts = Counter(recent)
        most_common_ratio = counts.most_common(1)[0][1] / len(recent)
        repetitive = most_common_ratio >= self.threshold
        if repetitive:
            logger.warning(f"[ANTI-BAN] Padrão repetitivo detetado (ratio={most_common_ratio:.2f})")
        return repetitive


class WinRateLimiter:
    """Limita win rate para evitar deteção."""

    def __init__(self, max_rate: float = 0.75, min_rate: float = 0.20):
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.results: deque = deque(maxlen=100)

    def record_result(self, result: str):
        """Registra resultado de uma partida."""
        self.results.append(result)

    def get_current_win_rate(self) -> float:
        if not self.results:
            return 0.0
        wins = sum(1 for r in self.results if r == "win")
        return wins / len(self.results)

    def should_throttle(self) -> bool:
        """Retorna True se deve reduzir taxa de vitórias (jogar menos eficiente)."""
        rate = self.get_current_win_rate()
        if rate > self.max_rate:
            logger.warning(f"[ANTI-BAN] Win rate alto ({rate:.2f}), throttling recomendado")
            return True
        return False

    def should_intensify(self) -> bool:
        """Retorna True se win rate está muito baixo (pode parecer intencional)."""
        rate = self.get_current_win_rate()
        if 0 < len(self.results) < 10 and rate < self.min_rate:
            logger.info(f"[ANTI-BAN] Win rate baixo ({rate:.2f}), intensificação permitida")
            return True
        return False


class ScheduleRandomizer:
    """Randomiza horários de jogo para parecer mais humano."""

    def __init__(self, variance_minutes: int = 30):
        self.variance_minutes = variance_minutes

    def get_next_start_time(self, base_hour: int = 10) -> datetime:
        """Calcula próximo horário de início com variação aleatória."""
        now = datetime.now()
        base = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
        if base < now:
            base += timedelta(days=1)
        variance = random.randint(-self.variance_minutes, self.variance_minutes)
        return base + timedelta(minutes=variance)

    def should_play_now(self) -> bool:
        """Retorna True com probabilidade que varia ao longo do dia."""
        hour = datetime.now().hour
        # Menos provável jogar de madrugada (0-6h)
        if 0 <= hour < 6:
            return random.random() < 0.05
        # Horário comum (10h-22h)
        if 10 <= hour <= 22:
            return random.random() < 0.8
        # Transições
        return random.random() < 0.3


class ActionObfuscator:
    """Adiciona ruído e "erros humanos" às ações."""

    def __init__(self, noise_probability: float = 0.05, max_delay: float = 2.0):
        self.noise_probability = noise_probability
        self.max_delay = max_delay

    def apply(self, action_type: str, original_delay: float) -> float:
        """Retorna delay modificado com possível ruído."""
        if random.random() < self.noise_probability:
            noise = random.uniform(0.0, self.max_delay)
            new_delay = original_delay + noise
            logger.debug(f"[ANTI-BAN] Ruído aplicado a {action_type}: {original_delay:.2f}s -> {new_delay:.2f}s")
            return new_delay
        return original_delay

    def maybe_missclick(self) -> bool:
        """Retorna True ocasionalmente para simular missclick."""
        return random.random() < (self.noise_probability * 0.2)


class FingerprintRandomizer:
    """Muda comportamento entre sessões para evitar fingerprinting."""

    def __init__(self, change_interval_hours: float = 24.0):
        self.change_interval_hours = change_interval_hours
        self.last_change = time.time()
        self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> Dict:
        """Gera um fingerprint aleatório de comportamento."""
        return {
            "delay_multiplier": random.uniform(0.8, 1.2),
            "reaction_variance": random.uniform(0.8, 1.3),
            "aggression_bias": random.uniform(-0.2, 0.2),
            "preferred_quadrant": random.choice(["top-left", "top-right", "bottom-left", "bottom-right"]),
        }

    def get_fingerprint(self) -> Dict:
        """Retorna fingerprint atual, regenerando se necessário."""
        elapsed_hours = (time.time() - self.last_change) / 3600.0
        if elapsed_hours >= self.change_interval_hours:
            self.fingerprint = self._generate_fingerprint()
            self.last_change = time.time()
            logger.info(f"[ANTI-BAN] Novo fingerprint gerado: {self.fingerprint}")
        return self.fingerprint


class AntiBanSystem:
    """Sistema unificado anti-ban."""

    def __init__(self, config: Optional[AntiBanConfig] = None):
        self.config = config or AntiBanConfig()
        self.pattern_detector = PatternDetector(
            window_size=self.config.pattern_window_size,
            threshold=self.config.pattern_similarity_threshold,
        )
        self.win_rate_limiter = WinRateLimiter(
            max_rate=self.config.max_win_rate,
            min_rate=self.config.min_win_rate,
        )
        self.schedule_randomizer = ScheduleRandomizer(
            variance_minutes=self.config.schedule_variance_minutes,
        )
        self.action_obfuscator = ActionObfuscator(
            noise_probability=self.config.action_noise_probability,
            max_delay=self.config.action_noise_max_delay,
        )
        self.fingerprint_randomizer = FingerprintRandomizer(
            change_interval_hours=self.config.session_fingerprint_change_interval_hours,
        )
        self.last_match_time: Optional[float] = None
        self.matches_this_hour = 0
        self.hour_start = time.time()

    def record_action(self, action_type: str, coordinates: Optional[tuple] = None):
        if not self.config.enabled:
            return
        self.pattern_detector.record_action(action_type, coordinates)

    def record_match_result(self, result: str):
        if not self.config.enabled:
            return
        self.win_rate_limiter.record_result(result)
        self.last_match_time = time.time()
        self.matches_this_hour += 1

    def check_throttle(self) -> bool:
        """Verifica se deve reduzir eficiência."""
        if not self.config.enabled:
            return False
        return self.win_rate_limiter.should_throttle()

    def check_pattern(self) -> bool:
        """Verifica se padrão repetitivo foi detetado."""
        if not self.config.enabled:
            return False
        return self.pattern_detector.detect_repetitive_pattern()

    def apply_action_delay(self, action_type: str, base_delay: float) -> float:
        """Aplica obfuscation a um delay de ação."""
        if not self.config.enabled:
            return base_delay
        return self.action_obfuscator.apply(action_type, base_delay)

    def should_start_match(self) -> bool:
        """Verifica se deve iniciar uma nova partida agora."""
        if not self.config.enabled:
            return True
        # Verificar limite de partidas por hora
        if time.time() - self.hour_start > 3600:
            self.hour_start = time.time()
            self.matches_this_hour = 0
        if self.matches_this_hour >= self.config.max_matches_per_hour:
            logger.warning("[ANTI-BAN] Limite de partidas/hora atingido")
            return False
        # Verificar pausa mínima entre partidas
        if self.last_match_time and (time.time() - self.last_match_time) < self.config.min_break_between_matches_sec:
            logger.info("[ANTI-BAN] Pausa mínima entre partidas não decorrida")
            return False
        # Verificar schedule
        if not self.schedule_randomizer.should_play_now():
            logger.info("[ANTI-BAN] Schedule indica pausa")
            return False
        return True

    def get_fingerprint(self) -> Dict:
        return self.fingerprint_randomizer.get_fingerprint()

    def get_status(self) -> Dict:
        """Retorna status atual do sistema anti-ban."""
        return {
            "enabled": self.config.enabled,
            "win_rate": self.win_rate_limiter.get_current_win_rate(),
            "pattern_detected": self.check_pattern(),
            "throttle_active": self.check_throttle(),
            "matches_this_hour": self.matches_this_hour,
            "fingerprint": self.get_fingerprint(),
        }
