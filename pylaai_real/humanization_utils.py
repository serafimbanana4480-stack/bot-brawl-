"""
humanization_utils.py

Utilitarios para humanizar o comportamento do bot:
- Jitter em tempos e coordenadas
- Curvas Bezier para swipes
- Simulacao de pausas naturais
- Variacao de APM
"""

import time
import random
import math
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def human_delay(base_seconds: float, jitter_percent: float = 0.25, min_val: float = 0.05) -> float:
    """
    Retorna um tempo com variacao aleatoria para parecer humano.

    Ex: human_delay(0.5, 0.3) -> 0.35-0.65s
    """
    jitter = base_seconds * jitter_percent
    result = base_seconds + random.uniform(-jitter, jitter)
    return max(min_val, result)


def jitter_coords(x: int, y: int, max_jitter: int = 10) -> Tuple[int, int]:
    """Adiciona jitter aleatorio a coordenadas."""
    return (x + random.randint(-max_jitter, max_jitter),
            y + random.randint(-max_jitter, max_jitter))


def jitter_value(base: float, jitter_percent: float = 0.15, min_val: float = None) -> float:
    """Adiciona variacao percentual a um valor."""
    jitter = base * jitter_percent
    result = base * random.uniform(1 - jitter_percent, 1 + jitter_percent)
    if min_val is not None:
        result = max(min_val, result)
    return result


def should_missclick(probability: float = 0.05) -> bool:
    """Retorna True com uma probabilidade (simula missclick humano)."""
    return random.random() < probability


def bezier_curve_points(x1: int, y1: int, x2: int, y2: int,
                        curvature: float = 0.3, num_points: int = 20) -> List[Tuple[int, int]]:
    """
    Gera pontos ao longo de uma curva de Bezier quadratica.
    Usado para swipes que parecem humanos (nao sao linhas retas).
    """
    # Ponto de controle aleatorio para criar curva
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    # Offset perpendicular a direcao do movimento
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx**2 + dy**2) or 1
    # Perpendicular
    perp_x = -dy / dist
    perp_y = dx / dist
    # Offset aleatorio
    offset = random.uniform(-dist * curvature, dist * curvature)
    ctrl_x = mid_x + perp_x * offset
    ctrl_y = mid_y + perp_y * offset

    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Bezier quadratica: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * ctrl_x + t**2 * x2
        by = (1 - t)**2 * y1 + 2 * (1 - t) * t * ctrl_y + t**2 * y2
        # Adicionar micro-jitter para nao ser perfeito
        bx += random.uniform(-2, 2)
        by += random.uniform(-2, 2)
        points.append((int(bx), int(by)))

    return points


class HumanPauseSimulator:
    """
    Simula pausas naturais que humanos fazem durante o jogo:
    - Hesitacao antes de decisoes importantes
    - Pausas para "avaliar" a situacao
    - Micro-pausas entre acoes rapidas
    """

    def __init__(self):
        self.last_major_pause = time.time()
        self.major_pause_interval = random.uniform(20, 60)
        self.last_micro_pause = 0
        self.micro_pause_interval = random.uniform(5, 15)
        self._paused = False

    def check_major_pause(self) -> bool:
        """
        Verifica se deve fazer uma pausa maior (hesitacao estrategica).
        Retorna True se deve pausar.
        """
        if time.time() - self.last_major_pause > self.major_pause_interval:
            pause_dur = random.uniform(0.3, 1.5)
            logger.debug(f"[HUMAN] Pausa estrategica: {pause_dur:.2f}s")
            time.sleep(pause_dur)
            self.last_major_pause = time.time()
            self.major_pause_interval = random.uniform(20, 60)
            return True
        return False

    def check_micro_pause(self) -> bool:
        """Pausa micro entre acoes rapidas."""
        if time.time() - self.last_micro_pause > self.micro_pause_interval:
            pause_dur = random.uniform(0.05, 0.2)
            time.sleep(pause_dur)
            self.last_micro_pause = time.time()
            self.micro_pause_interval = random.uniform(5, 15)
            return True
        return False

    def reaction_delay(self, base_ms: int = 150) -> float:
        """
        Delay de reacao humano: tempo entre ver algo e reagir.
        Humanos tipicamente: 150-400ms para reacoes simples.
        """
        delay_ms = random.gauss(base_ms, 80)  # Distribuicao normal
        delay_ms = max(80, min(600, delay_ms))  # Clamp 80-600ms
        return delay_ms / 1000.0


class APMController:
    """
    Controla Actions Per Minute para parecer humano.
    Humanos jogam Brawl Stars a ~25-45 APM.
    """

    def __init__(self, target_apm: float = 35.0):
        self.target_apm = target_apm
        self.action_times = []
        self.window_seconds = 60
        self._last_adjustment = time.time()

    def record_action(self):
        """Registra uma acao para calcular APM atual."""
        now = time.time()
        self.action_times.append(now)
        # Limpar acoes antigas
        cutoff = now - self.window_seconds
        self.action_times = [t for t in self.action_times if t > cutoff]

    def get_current_apm(self) -> float:
        """APM nos ultimos 60 segundos."""
        now = time.time()
        cutoff = now - self.window_seconds
        recent = [t for t in self.action_times if t > cutoff]
        return len(recent) * (60.0 / self.window_seconds)

    def should_delay_for_apm(self) -> float:
        """
        Se APM esta muito alto, retorna um delay para abrandar.
        Retorna 0 se APM esta OK.
        """
        current = self.get_current_apm()
        if current > self.target_apm * 1.3:
            excess = current - self.target_apm
            delay = excess * 0.05  # 50ms por APM acima do alvo
            return min(delay, 2.0)  # Max 2s delay
        return 0.0

    def adjust_target(self, win_rate: float = None):
        """Ajusta APM alvo baseado em win rate (mais alto = mais confianca)."""
        if win_rate is None:
            return
        if win_rate > 0.6:
            self.target_apm = min(50, self.target_apm + 2)
        elif win_rate < 0.3:
            self.target_apm = max(20, self.target_apm - 3)
