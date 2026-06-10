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
        """Retorna True se o horário atual é compatível com padrões humanos realistas.

        Regras:
        - Não joga entre 01:00-07:00 (sono)
        - Não joga durante pausas de refeição (almoço ~12h, jantar ~19h)
        - Variação de horário de início por dia
        - Sessões mais curtas durante a semana, mais longas ao fim de semana
        """
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

        # Sleep window: 01:00 - 07:00 (with some variance)
        sleep_start = 1.0 + random.uniform(-0.5, 0.5)
        sleep_end = 7.0 + random.uniform(-0.5, 1.0)
        if sleep_start <= hour <= sleep_end:
            logger.debug("[ANTI-BAN] Schedule: sleep window")
            return False

        # Meal breaks with variance
        lunch_start = 12.0 + random.uniform(-0.5, 0.5)
        lunch_end = lunch_start + random.uniform(0.5, 1.5)
        dinner_start = 19.0 + random.uniform(-0.5, 0.5)
        dinner_end = dinner_start + random.uniform(0.75, 1.5)

        if lunch_start <= hour <= lunch_end:
            logger.debug("[ANTI-BAN] Schedule: lunch break")
            return False
        if dinner_start <= hour <= dinner_end:
            logger.debug("[ANTI-BAN] Schedule: dinner break")
            return False

        # Weekend bonus: slightly more lenient on Fri/Sat/Sun evenings
        weekday = now.weekday()
        if weekday >= 4 and hour >= 20.0:
            return True

        return True


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
        """Gera um fingerprint comportamental de 12 dimensões."""
        return {
            "delay_multiplier": random.uniform(0.8, 1.2),
            "reaction_variance": random.uniform(0.8, 1.3),
            "aggression_bias": random.uniform(-0.2, 0.2),
            "preferred_quadrant": random.choice(["top-left", "top-right", "bottom-left", "bottom-right"]),
            "apm_target": random.uniform(25, 55),  # human APM range
            "curvature_preference": random.uniform(-0.3, 0.3),  # CW vs CCW bias
            "pause_frequency": random.uniform(0.05, 0.25),  # micro-pause probability
            "decision_pattern": random.choice(["deliberate", "impulsive", "balanced"]),
            "overshoot_tendency": random.uniform(0.02, 0.08),  # 2-8% overshoot rate
            "typing_speed_wpm": random.uniform(35, 75),
            "hesitation_base": random.uniform(0.15, 0.45),
            "fatigue_recovery_rate": random.uniform(0.3, 0.7),
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
        
        # Session fatigue tracking
        self.session_start = time.time()
        self.session_actions = 0
        self._fatigue_level = 0.0
        
        # Session schedule with realistic breaks
        self.session_schedule = SessionSchedule()
        
        # Click heatmap for anti-fingerprinting
        self.click_heatmap = ClickHeatmap(grid_size=50)

    def _update_fatigue(self) -> float:
        """Atualiza e retorna nível de fadiga da sessão (0.0-1.0)."""
        elapsed_hours = (time.time() - self.session_start) / 3600.0
        # Fadiga acumula com tempo e ações
        time_fatigue = min(1.0, elapsed_hours / 3.0)  # 3h = max fatigue
        action_fatigue = min(1.0, self.session_actions / 2000.0)  # 2000 actions = max
        self._fatigue_level = 0.6 * time_fatigue + 0.4 * action_fatigue
        return self._fatigue_level

    def get_fatigue_factor(self) -> float:
        """Retorna fator de fadiga para ajustar delays e precisão."""
        return self._update_fatigue()

    def record_action(self, action_type: str, coordinates: Optional[tuple] = None):
        if not self.config.enabled:
            return
        self.pattern_detector.record_action(action_type, coordinates)
        self.session_actions += 1
        # Record click in heatmap
        if coordinates and action_type in ("tap", "click"):
            self.click_heatmap.record_click(coordinates[0], coordinates[1])
        logger.debug(
            "[ANTI-BAN] Action recorded: %s, fatigue=%.3f, actions=%d",
            action_type, self._update_fatigue(), self.session_actions
        )

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
        # Check session schedule (lunch, dinner, sleep breaks)
        if not self.session_schedule.should_play_now():
            logger.info("[ANTI-BAN] Session schedule indica pausa (refeição/sono)")
            return False
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

    def get_click_heatmap_stats(self) -> Dict:
        """Get click heatmap statistics for anti-fingerprinting analysis."""
        return self.click_heatmap.get_stats()

    def get_jittered_coordinates(self, x: float, y: float) -> tuple:
        """Apply heatmap-aware jitter to coordinates to avoid click fingerprinting.
        
        If too many clicks are in the same grid cell, adds extra jitter.
        """
        fp = self.get_fingerprint()
        base_jitter = 3  # Default jitter range
        
        # Check if this grid cell has too many clicks
        cell_count = self.click_heatmap.get_cell_count(x, y)
        if cell_count > 10:
            # Add extra jitter proportional to click density
            extra_jitter = min(cell_count - 10, 15)
            base_jitter += extra_jitter
            logger.debug(f"[ANTI-BAN] Extra jitter ({extra_jitter}px) for over-clicked area at ({x:.0f}, {y:.0f})")
        
        # Apply fingerprint-based jitter direction bias
        quadrant = fp.get("preferred_quadrant", "top-left")
        bias_x, bias_y = 0, 0
        if "top" in quadrant:
            bias_y = -1
        else:
            bias_y = 1
        if "left" in quadrant:
            bias_x = -1
        else:
            bias_x = 1
        
        jx = x + random.randint(-base_jitter, base_jitter) + bias_x * random.randint(0, 2)
        jy = y + random.randint(-base_jitter, base_jitter) + bias_y * random.randint(0, 2)
        return (jx, jy)

    def get_status(self) -> Dict:
        """Retorna status atual do sistema anti-ban."""
        return {
            "enabled": self.config.enabled,
            "win_rate": self.win_rate_limiter.get_current_win_rate(),
            "pattern_detected": self.check_pattern(),
            "throttle_active": self.check_throttle(),
            "matches_this_hour": self.matches_this_hour,
            "fingerprint": self.get_fingerprint(),
            "session_schedule": self.session_schedule.get_status(),
            "click_heatmap": self.click_heatmap.get_stats(),
            "session_fatigue": round(self._update_fatigue(), 3),
            "session_duration_hours": round((time.time() - self.session_start) / 3600, 2),
            "total_actions": self.session_actions,
        }


class SessionSchedule:
    """Manages realistic session schedule with breaks for meals and sleep."""

    def __init__(self):
        self.session_start = time.time()
        self._break_until: Optional[float] = None
        self._total_play_time = 0.0
        
        # Realistic break schedule (hours in day, duration in minutes)
        self._break_schedule = [
            {"name": "lunch", "around_hour": 12, "variance_min": 60, "duration_min": (30, 60)},
            {"name": "dinner", "around_hour": 19, "variance_min": 60, "duration_min": (30, 60)},
            {"name": "sleep", "around_hour": 23, "variance_min": 60, "duration_min": (420, 540)},  # 7-9 hours
        ]
        # Pre-generate today's break times
        self._today_breaks = self._generate_break_times()

    def _generate_break_times(self) -> List[Dict]:
        """Generate break times for today with randomization."""
        from datetime import datetime, timedelta
        now = datetime.now()
        breaks = []
        for brk in self._break_schedule:
            hour = brk["around_hour"] + random.uniform(
                -brk["variance_min"] / 60, brk["variance_min"] / 60
            )
            break_time = now.replace(hour=int(hour), minute=int((hour % 1) * 60), second=0, microsecond=0)
            duration_min = random.uniform(*brk["duration_min"])
            breaks.append({
                "name": brk["name"],
                "start": break_time.timestamp(),
                "end": (break_time + timedelta(minutes=duration_min)).timestamp(),
            })
        return breaks

    def should_play_now(self) -> bool:
        """Check if currently in a break period."""
        now = time.time()
        
        # Check explicit break
        if self._break_until and now < self._break_until:
            return False
        
        # Check scheduled breaks
        for brk in self._today_breaks:
            if brk["start"] <= now <= brk["end"]:
                logger.info(f"[ANTI-BAN] Scheduled break: {brk['name']}")
                return False
        
        # Regenerate breaks if day changed
        from datetime import datetime
        if datetime.now().hour < 6:  # New day
            self._today_breaks = self._generate_break_times()
        
        return True

    def take_break(self, duration_seconds: float):
        """Force a break for the specified duration."""
        self._break_until = time.time() + duration_seconds
        logger.info(f"[ANTI-BAN] Taking break for {duration_seconds:.0f}s")

    def get_status(self) -> Dict:
        now = time.time()
        current_break = None
        for brk in self._today_breaks:
            if brk["start"] <= now <= brk["end"]:
                current_break = brk["name"]
                break
        return {
            "on_break": not self.should_play_now(),
            "current_break": current_break,
            "forced_break_remaining": max(0, (self._break_until or 0) - now),
            "session_duration_hours": (now - self.session_start) / 3600,
        }


class ClickHeatmap:
    """Tracks click positions to detect and prevent click fingerprinting."""

    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
        self._clicks: Dict[tuple, int] = {}  # (grid_x, grid_y) -> count
        self._total_clicks = 0

    def record_click(self, x: float, y: float):
        """Record a click at the given coordinates."""
        gx = int(x / self.grid_size)
        gy = int(y / self.grid_size)
        key = (gx, gy)
        self._clicks[key] = self._clicks.get(key, 0) + 1
        self._total_clicks += 1

    def get_cell_count(self, x: float, y: float) -> int:
        """Get click count for the grid cell containing (x, y)."""
        gx = int(x / self.grid_size)
        gy = int(y / self.grid_size)
        return self._clicks.get((gx, gy), 0)

    def get_stats(self) -> Dict:
        """Get heatmap statistics."""
        if not self._clicks:
            return {"total_clicks": 0, "unique_cells": 0, "max_cell_clicks": 0, "concentration": 0.0}
        max_clicks = max(self._clicks.values())
        total = sum(self._clicks.values())
        concentration = max_clicks / total if total > 0 else 0.0
        return {
            "total_clicks": self._total_clicks,
            "unique_cells": len(self._clicks),
            "max_cell_clicks": max_clicks,
            "concentration": f"{concentration:.2%}",
            "overclicked_cells": sum(1 for v in self._clicks.values() if v > 10),
        }
