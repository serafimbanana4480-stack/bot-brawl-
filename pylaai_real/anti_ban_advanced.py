"""
anti_ban_advanced.py

Sistema Anti-Ban Ultra-Avançado para Soberana Omega.

Arquitetura (10 pilares):
1. BehavioralFingerprint    — fingerprint dinâmico do estilo de jogo
2. InputRandomizer          — randomização de inputs com distribuições realistas
3. MicroHumanizer           — micro-pausas, hesitações, "pensamentos"
4. SessionOrchestrator      — gestão inteligente de sessões (horários, durações, breaks)
5. PlaystyleProfile         — perfil adaptativo do jogador (estilo único por sessão)
6. BanRiskScorer            — scoring de risco em tempo real (0-1000)
7. AutoRemediator           — auto-correção quando risco > threshold
8. JoystickMovementAnalyzer — deteção de movimento robótico (círculos perfeitos, linhas retas)
9. AimFatigueSimulator      — fadiga de mira ao longo da sessão
10. SocialBehaviorEmulator  — emulação de comportamentos sociais (emotes, rotação de brawler)

Tudo integrado na classe principal AdvancedAntiBanSystem.
"""

import time
import math
import random
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _gaussian_noise(mean: float, std: float) -> float:
    """Box-Muller transform para distribuição gaussiana."""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + z * std


# ---------------------------------------------------------------------------
# 1. BEHAVIORAL FINGERPRINT
# ---------------------------------------------------------------------------

@dataclass
class FingerprintDimensions:
    """Dimensões mensuráveis do fingerprint comportamental."""
    tap_cluster_tightness: float = 0.0      # 0=espalhado, 1=cluster perfeito
    swipe_curvature_bias: float = 0.0       # preferência por curvas CW vs CCW
    decision_speed: float = 0.5              # 0=lento/cauteloso, 1=instantâneo
    aggression_oscillation: float = 0.3       # amplitude de oscilação de agressividade
    break_frequency: float = 0.5            # frequência de micro-pausas
    aim_precision_drift: float = 0.0         # drift de precisão ao longo do tempo
    reaction_time_base: float = 0.25        # tempo de reação base em segundos
    joystick_preference: str = "mixed"      # "tap", "swipe", "mixed"
    super_usage_timing: str = "opportunist" # "early", "late", "opportunist"
    movement_style: str = "direct"          # "direct", "erratic", "cautious"


class BehavioralFingerprint:
    """
    Gera e mantém um fingerprint comportamental dinâmico.
    Muda sutilmente ao longo de uma sessão (simula "estar mais confortável").
    """

    def __init__(self, seed: Optional[str] = None):
        self.seed = seed or hashlib.sha256(
            (str(time.time()) + str(random.random())).encode()
        ).hexdigest()[:16]
        # Reproducibilidade parcial para consistência dentro da sessão
        self._rng = random.Random(self.seed)
        self.dimensions = self._generate_dimensions()
        self.creation_time = time.time()
        self._evolution_rate = self._rng.uniform(0.02, 0.08)  # mudança por hora

    def _generate_dimensions(self) -> FingerprintDimensions:
        d = FingerprintDimensions()
        d.tap_cluster_tightness = self._rng.gauss(0.35, 0.15)
        d.swipe_curvature_bias = self._rng.gauss(0.0, 0.3)
        d.decision_speed = _clamp(self._rng.gauss(0.55, 0.2), 0.1, 0.95)
        d.aggression_oscillation = _clamp(self._rng.gauss(0.25, 0.15), 0.05, 0.6)
        d.break_frequency = _clamp(self._rng.gauss(0.45, 0.2), 0.1, 0.9)
        d.aim_precision_drift = self._rng.gauss(0.0, 0.1)
        d.reaction_time_base = _clamp(self._rng.gauss(0.28, 0.08), 0.12, 0.55)
        d.joystick_preference = self._rng.choice(["tap", "swipe", "mixed"])
        d.super_usage_timing = self._rng.choice(["early", "late", "opportunist"])
        d.movement_style = self._rng.choice(["direct", "erratic", "cautious"])
        return d

    def evolve(self, elapsed_hours: float):
        """Evolui o fingerprint lentamente (jogador "aquece")."""
        delta = elapsed_hours * self._evolution_rate
        d = self.dimensions
        d.decision_speed = _clamp(d.decision_speed + delta * 0.1, 0.1, 0.95)
        d.reaction_time_base = _clamp(d.reaction_time_base - delta * 0.02, 0.12, 0.55)
        d.aggression_oscillation = _clamp(d.aggression_oscillation - delta * 0.03, 0.05, 0.6)

    def get_delay_multiplier(self) -> float:
        """Baseado em decision_speed: 0.6x a 1.4x."""
        return 1.6 - self.dimensions.decision_speed

    def get_tap_jitter_radius(self) -> float:
        """Raio de jitter para taps (baseado em tap_cluster_tightness)."""
        # Mais tightness = menos jitter (hábitos de toque mais fixos)
        base = 3.0
        return base * (1.0 - self.dimensions.tap_cluster_tightness * 0.5)

    def summary(self) -> Dict:
        return {"seed": self.seed, "age_hours": (time.time() - self.creation_time) / 3600,
                **asdict(self.dimensions)}


# ---------------------------------------------------------------------------
# 2. INPUT RANDOMIZER
# ---------------------------------------------------------------------------

class InputRandomizer:
    """
    Randomiza inputs com distribuições realistas (não uniformes).

    - Taps: offset com distribuição normal (humanos têm bias para o centro do dedo)
    - Swipes: curvas de Bézier com controlo aleatório, velocidade variável
    - Timing: distribuição log-normal (mais ações curtas, algumas longas)
    """

    def __init__(self, fingerprint: BehavioralFingerprint):
        self.fp = fingerprint
        self._tap_history: deque = deque(maxlen=200)
        self._hotspot = (None, None)  # centro de cluster preferido

    def randomize_tap(self, x: float, y: float) -> Tuple[float, float]:
        """Adiciona offset realista a um tap."""
        radius = self.fp.get_tap_jitter_radius()
        # Distribuição normal 2D (humanos erram mais em Y do que em X)
        dx = _gaussian_noise(0, radius * 1.0)
        dy = _gaussian_noise(0, radius * 1.3)
        new_x, new_y = x + dx, y + dy
        self._tap_history.append((new_x, new_y, time.time()))
        return new_x, new_y

    def randomize_swipe_control_points(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> List[Tuple[float, float]]:
        """Gera pontos de controlo para curva de Bézier humanizada."""
        # Swipes humanos têm 1-3 pontos de controlo, geralmente formando S suave
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        # Offset do ponto médio baseado no fingerprint
        bias = self.fp.dimensions.swipe_curvature_bias
        offset_x = _gaussian_noise(bias * 30, 25)
        offset_y = _gaussian_noise(0, 20)
        cp1 = (mid_x + offset_x * 0.5, mid_y + offset_y * 0.3)
        cp2 = (mid_x + offset_x * 0.8, mid_y + offset_y * 0.7)
        return [cp1, cp2]

    def randomize_delay(self, base_delay: float, action_type: str = "generic") -> float:
        """
        Adiciona variação realista a delays.
        Distribuição log-normal: a maioria das ações é rápida, algumas lentas.
        """
        sigma = 0.5  # variabilidade
        log_normal = math.exp(_gaussian_noise(0, sigma))
        multiplier = 0.7 + log_normal * 0.4  # 0.7x a ~2.5x
        # Ações de combate têm menos variação (foco)
        if action_type in ("attack", "super", "dodge"):
            multiplier = 0.85 + log_normal * 0.25
        # Ações de lobby têm mais variação (distração)
        elif action_type in ("lobby_click", "menu_nav"):
            multiplier = 0.6 + log_normal * 0.6
        return base_delay * multiplier

    def should_add_micro_pause(self) -> bool:
        """Probabilidade de micro-pausa baseada no fingerprint."""
        return random.random() < self.fp.dimensions.break_frequency * 0.15

    def get_micro_pause_duration(self) -> float:
        """Duração de micro-pausa (0.3-2.0s)."""
        return random.expovariate(2.0) + 0.3


# ---------------------------------------------------------------------------
# 3. MICRO HUMANIZER
# ---------------------------------------------------------------------------

class MicroHumanizer:
    """
    Simula micro-comportamentos humanos:
    - Hesitação antes de decisões importantes
    - "Pensamento estratégico" (pausa antes de usar super)
    - Correção de trajetória (muda de ideia a meio)
    - Double-tap ocasional (humanos tocam 2x sem querer)
    """

    def __init__(self, fingerprint: BehavioralFingerprint):
        self.fp = fingerprint
        self._hesitation_cooldown = 0.0
        self._last_hesitation = 0.0

    def hesitate_before_action(self, action_importance: float = 0.5) -> float:
        """
        Retorna tempo de hesitação extra antes de uma ação.
        importance: 0=trivial, 1=crítica (ex: usar super no clutch)
        """
        now = time.time()
        if now < self._hesitation_cooldown:
            return 0.0
        # Probabilidade de hesitar baseada na importância e no fingerprint
        base_prob = action_importance * 0.4
        if random.random() > base_prob:
            return 0.0
        # Duração da hesitação
        base_reaction = self.fp.dimensions.reaction_time_base
        hesitation = base_reaction * (0.5 + action_importance * 1.5)
        hesitation += random.expovariate(3.0)  # cauda longa
        self._hesitation_cooldown = now + hesitation + 1.0
        self._last_hesitation = now
        logger.debug(f"[ANTI-BAN] Hesitação: {hesitation:.3f}s (importance={action_importance:.2f})")
        return hesitation

    def maybe_double_tap(self) -> bool:
        """Humanos ocasionalmente dão double-tap (especialmente em botões pequenos)."""
        return random.random() < 0.03

    def maybe_correct_trajectory(self, original_target: Tuple[float, float]) -> Tuple[float, float]:
        """Simula mudança de ideia a meio do movimento."""
        if random.random() > 0.05:
            return original_target
        # Ajusta target em 10-30%
        tx, ty = original_target
        dx = _gaussian_noise(0, abs(tx) * 0.15 + 10)
        dy = _gaussian_noise(0, abs(ty) * 0.15 + 10)
        logger.debug(f"[ANTI-BAN] Correção de trajetória")
        return (tx + dx, ty + dy)

    def maybe_pause_to_think(self, situation_complexity: float = 0.5) -> float:
        """
        Simula "pausa para pensar" em situações complexas.
        complexity: número de inimigos, disponibilidade de super, etc. normalizado 0-1
        """
        prob = complexity * 0.25 * (1.0 - self.fp.dimensions.decision_speed)
        if random.random() > prob:
            return 0.0
        pause = 0.5 + random.expovariate(1.5) * complexity
        logger.debug(f"[ANTI-BAN] Pausa para pensar: {pause:.3f}s")
        return pause


# ---------------------------------------------------------------------------
# 4. SESSION ORCHESTRATOR
# ---------------------------------------------------------------------------

@dataclass
class SessionPlan:
    target_duration_min: float = 45.0
    target_duration_max: float = 120.0
    break_interval_min: float = 15.0
    break_interval_max: float = 35.0
    break_duration_min: float = 3.0
    break_duration_max: float = 8.0
    max_matches_per_hour: int = 7
    warmup_matches: int = 2
    peak_performance_matches: int = 4
    fatigue_start_match: int = 6


class SessionOrchestrator:
    """
    Orquestra uma sessão como um humano real a jogar:
    - Warm-up (primeiras partidas mais lentas)
    - Peak performance (meio da sessão)
    - Fatigue (fim da sessão - mais erros, mais pausas)
    - Breaks naturais entre clusters de partidas
    """

    def __init__(self, plan: Optional[SessionPlan] = None):
        self.plan = plan or SessionPlan()
        self.session_start = time.time()
        self.matches_played = 0
        self._next_break_at = time.time() + random.uniform(
            self.plan.break_interval_min * 60, self.plan.break_interval_max * 60
        )
        self._in_break = False
        self._break_end_time = 0.0
        self._session_end_time = time.time() + random.uniform(
            self.plan.target_duration_min * 60, self.plan.target_duration_max * 60
        )
        self._match_times: deque = deque(maxlen=20)

    def record_match_start(self):
        self.matches_played += 1
        self._match_times.append(time.time())

    def get_fatigue_factor(self) -> float:
        """0=sem fadiga, 1=máxima fadiga."""
        if self.matches_played < self.plan.warmup_matches:
            return 0.0
        if self.matches_played >= self.plan.fatigue_start_match:
            fatigue = (self.matches_played - self.plan.fatigue_start_match) / 4.0
            return _clamp(fatigue, 0.0, 1.0)
        return 0.0

    def get_warmup_factor(self) -> float:
        """1=ainda em warmup, 0=totalmente aquecido."""
        if self.matches_played >= self.plan.warmup_matches:
            return 0.0
        return 1.0 - self.matches_played / self.plan.warmup_matches

    def should_take_break(self) -> bool:
        now = time.time()
        if self._in_break and now >= self._break_end_time:
            self._in_break = False
            self._schedule_next_break()
            return False
        if self._in_break:
            return True
        if now >= self._next_break_at:
            self._start_break()
            return True
        return False

    def _start_break(self):
        duration = random.uniform(self.plan.break_duration_min, self.plan.break_duration_max)
        self._break_end_time = time.time() + duration * 60
        self._in_break = True
        logger.info(f"[ANTI-BAN] Break natural: {duration:.1f}min")

    def _schedule_next_break(self):
        self._next_break_at = time.time() + random.uniform(
            self.plan.break_interval_min * 60, self.plan.break_interval_max * 60
        )

    def should_end_session(self) -> bool:
        return time.time() >= self._session_end_time

    def get_matches_this_hour(self) -> int:
        cutoff = time.time() - 3600
        return sum(1 for t in self._match_times if t > cutoff)

    def get_session_pressure(self) -> float:
        """Retorna uma pressão agregada da sessão para pacing adaptativo."""
        fatigue = self.get_fatigue_factor()
        warmup = self.get_warmup_factor()
        matches_per_hour = self.get_matches_this_hour()
        break_urgency = 1.0 if self.should_take_break() else 0.0
        density_pressure = _clamp(matches_per_hour / max(1, self.plan.max_matches_per_hour), 0.0, 1.0)
        return _clamp(
            fatigue * 0.45 + warmup * 0.20 + density_pressure * 0.20 + break_urgency * 0.60,
            0.0,
            1.0,
        )

    def recommend_pacing(self, base_delay: float, action_type: str = "generic") -> float:
        """Ajusta um delay base para refletir fase da sessão e densidade recente."""
        pressure = self.get_session_pressure()
        delay = base_delay

        # Warm-up: mais lento no começo para parecer humano e não estourar APM cedo.
        delay *= 1.0 + self.get_warmup_factor() * 0.35

        # Fadiga: mais lento conforme a sessão avança.
        delay *= 1.0 + self.get_fatigue_factor() * 0.45

        # Break urgency: aumenta significativamente a cadência quando uma pausa está próxima.
        if self.should_take_break():
            delay *= 1.35

        # Densidade de partidas: se a janela horária está agressiva, reduza ritmo.
        matches_ratio = _clamp(self.get_matches_this_hour() / max(1, self.plan.max_matches_per_hour), 0.0, 1.5)
        delay *= 1.0 + matches_ratio * 0.12

        # Micro-ajuste por tipo de ação: menus/lobby devem parecer menos mecânicos.
        if action_type in ("lobby_click", "menu_nav"):
            delay *= 1.08 + pressure * 0.08
        elif action_type in ("attack", "super", "dodge"):
            delay *= 0.92 + pressure * 0.05

        return _clamp(delay, base_delay * 0.5, base_delay * 2.75)

    def can_start_match(self) -> bool:
        if self.should_take_break():
            return False
        if self.should_end_session():
            return False
        if self.get_matches_this_hour() >= self.plan.max_matches_per_hour:
            return False
        return True

    def get_status(self) -> Dict:
        elapsed = (time.time() - self.session_start) / 60
        return {
            "matches_played": self.matches_played,
            "elapsed_minutes": round(elapsed, 1),
            "fatigue_factor": round(self.get_fatigue_factor(), 2),
            "warmup_factor": round(self.get_warmup_factor(), 2),
            "session_pressure": round(self.get_session_pressure(), 2),
            "in_break": self._in_break,
            "session_should_end": self.should_end_session(),
            "can_start_match": self.can_start_match(),
            "matches_this_hour": self.get_matches_this_hour(),
        }


# ---------------------------------------------------------------------------
# 5. PLAYSTYLE PROFILE
# ---------------------------------------------------------------------------

class PlaystyleProfile:
    """
    Perfil de jogador persistente e adaptativo.
    Cada sessão tem um estilo único que evolui com base no desempenho.
    """

    def __init__(self, save_path: Path = Path("data/playstyle_profile.json")):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.traits = {
            "aggression_baseline": 0.5,
            "risk_tolerance": 0.5,
            "super_hoarding": 0.3,
            "bush_camping": 0.2,
            "team_awareness": 0.5,
            "rotation_preference": "clockwise",  # ou "counter", "mixed"
            "favorite_area": "center",  # "center", "edges", "top", "bottom"
            "reaction_to_low_hp": "aggressive",  # "aggressive", "defensive", "panic"
        }
        self._session_history: deque = deque(maxlen=50)
        self._load()

    def _load(self):
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.traits.update(data.get("traits", {}))
            self._session_history.extend(data.get("history", []))
        except Exception as e:
            logger.debug(f"[ANTI-BAN] Playstyle load error: {e}")

    def save(self):
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "traits": self.traits,
                    "history": list(self._session_history),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"[ANTI-BAN] Playstyle save error: {e}")

    def adapt_from_match(self, won: bool, kills: int, deaths: int, damage_dealt: float,
                         survival_time: float, brawler: str, map_name: str):
        """Adaptra o perfil com base no resultado da partida."""
        entry = {
            "won": won, "kills": kills, "deaths": deaths,
            "damage_dealt": damage_dealt, "survival_time": survival_time,
            "brawler": brawler, "map": map_name,
            "timestamp": time.time(),
        }
        self._session_history.append(entry)

        # Adaptação sutil (evitar mudanças bruscas)
        alpha = 0.05  # learning rate lento
        if won:
            self.traits["aggression_baseline"] = _clamp(
                self.traits["aggression_baseline"] + alpha * 0.1, 0.1, 0.9
            )
        else:
            self.traits["aggression_baseline"] = _clamp(
                self.traits["aggression_baseline"] - alpha * 0.15, 0.1, 0.9
            )

        # Super hoarding: se morreu com super carregado, aumenta hoarding
        if deaths > 0 and damage_dealt > 2000:
            self.traits["super_hoarding"] = _clamp(
                self.traits["super_hoarding"] + alpha * 0.2, 0.0, 0.8
            )

        self.save()

    def get_aggression(self, session_fatigue: float = 0.0) -> float:
        """Retorna agressividade atual considerando fadiga."""
        base = self.traits["aggression_baseline"]
        # Fadiga reduz agressão (jogadores cansados jogam mais seguro)
        fatigue_effect = session_fatigue * -0.3
        return _clamp(base + fatigue_effect, 0.1, 0.9)

    def should_use_super(self, hp_ratio: float, enemies_nearby: int) -> bool:
        """Decisão de usar super baseada no perfil."""
        hoarding = self.traits["super_hoarding"]
        # Super hoarders esperam mais
        threshold = 0.4 + hoarding * 0.4
        urgency = (1.0 - hp_ratio) * 0.5 + enemies_nearby * 0.15
        return urgency > threshold

    def summary(self) -> Dict:
        return dict(self.traits)


# ---------------------------------------------------------------------------
# 6. BAN RISK SCORER
# ---------------------------------------------------------------------------

class BanRiskScorer:
    """
    Sistema de scoring de risco de ban em tempo real.
    Múltiplas dimensões de risco, cada uma com peso.

    Dimensões:
    - Timing consistency (intervalos perfeitos = suspeito)
    - Spatial consistency (toques no mesmo pixel = suspeito)
    - Movement robotic (joystick círculos perfeitos)
    - Performance anomaly (win rate fora da média humana)
    - Session behavior (sessões muito longas, sem breaks)
    - APM outlier (APM inconsistente com humanos)
    """

    def __init__(self):
        self._timing_intervals: deque = deque(maxlen=500)
        self._tap_positions: deque = deque(maxlen=500)
        self._swipe_angles: deque = deque(maxlen=200)
        self._win_history: deque = deque(maxlen=50)
        self._apm_samples: deque = deque(maxlen=60)
        self._risk_history: deque = deque(maxlen=100)
        self._peak_risk = 0.0

    def record_timing(self, interval: float):
        self._timing_intervals.append(interval)

    def record_tap(self, x: float, y: float):
        self._tap_positions.append((x, y))

    def record_swipe_angle(self, angle: float):
        self._swipe_angles.append(angle)

    def record_apm(self, apm: int):
        self._apm_samples.append(apm)

    def record_match_result(self, won: bool):
        self._win_history.append(won)

    def _score_timing_consistency(self) -> float:
        """0-100. Intervalos perfeitos = alto risco."""
        if len(self._timing_intervals) < 20:
            return 0.0
        intervals = list(self._timing_intervals)
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        cv = math.sqrt(variance) / mean if mean > 0 else 0  # coefficient of variation
        # Humanos têm CV ~0.3-0.8. Bots têm CV < 0.1
        if cv < 0.05:
            return 100.0
        if cv < 0.15:
            return 70.0
        if cv < 0.3:
            return 40.0
        return 0.0

    def _score_spatial_consistency(self) -> float:
        """0-100. Pixel-perfect repetition = alto risco."""
        if len(self._tap_positions) < 30:
            return 0.0
        # Contar posições únicas (arredondadas a 5px)
        rounded = set((round(x / 5), round(y / 5)) for x, y in self._tap_positions)
        unique_ratio = len(rounded) / len(self._tap_positions)
        # Humanos têm >70% únicos. Bots têm <30%
        if unique_ratio < 0.2:
            return 100.0
        if unique_ratio < 0.4:
            return 60.0
        if unique_ratio < 0.6:
            return 30.0
        return 0.0

    def _score_movement_robotic(self) -> float:
        """0-100. Ângulos perfeitamente distribuídos = suspeito."""
        if len(self._swipe_angles) < 20:
            return 0.0
        angles = list(self._swipe_angles)
        # Humanos têm bias em certas direções (ex: preferem swipes horizontais)
        # Distribuição uniforme = robótico
        bucket_size = math.pi / 6  # 30 graus
        buckets = defaultdict(int)
        for a in angles:
            buckets[int((a % (2 * math.pi)) / bucket_size)] += 1
        if not buckets:
            return 0.0
        max_bucket = max(buckets.values())
        uniformity = max_bucket / len(angles)
        # Humanos: uniformity 0.3-0.6. Bots uniformes: ~0.15 (demasiado espalhado)
        # ou bots com padrões fixos: >0.8
        if uniformity > 0.75 or uniformity < 0.1:
            return 80.0
        return 0.0

    def _score_performance_anomaly(self) -> float:
        """0-100. Win rate suspeito."""
        if len(self._win_history) < 10:
            return 0.0
        wins = sum(1 for w in self._win_history if w)
        wr = wins / len(self._win_history)
        # Win rates humanos típicos: 0.45-0.65
        if wr > 0.85:
            return 90.0
        if wr > 0.75:
            return 50.0
        if wr < 0.15 and len(self._win_history) > 20:
            return 40.0  # Intencionalmente a perder?
        return 0.0

    def _score_apm_outlier(self) -> float:
        """0-100. APM inconsistente."""
        if len(self._apm_samples) < 10:
            return 0.0
        apms = list(self._apm_samples)
        avg = sum(apms) / len(apms)
        # APM humano: 25-50. APM bot típico: 80-150
        if avg > 100:
            return 80.0
        if avg > 70:
            return 50.0
        if avg > 55:
            return 25.0
        return 0.0

    def calculate_risk(self) -> Dict:
        """Calcula risco total e por dimensão."""
        scores = {
            "timing": self._score_timing_consistency(),
            "spatial": self._score_spatial_consistency(),
            "movement": self._score_movement_robotic(),
            "performance": self._score_performance_anomaly(),
            "apm": self._score_apm_outlier(),
        }
        weights = {
            "timing": 0.25,
            "spatial": 0.20,
            "movement": 0.15,
            "performance": 0.20,
            "apm": 0.20,
        }
        total = sum(scores[k] * weights[k] for k in scores)
        self._risk_history.append({"t": time.time(), "score": total, "dims": scores})
        self._peak_risk = max(self._peak_risk, total)
        return {
            "total": round(total, 1),
            "peak": round(self._peak_risk, 1),
            "dimensions": {k: round(v, 1) for k, v in scores.items()},
            "risk_level": "critical" if total > 75 else "high" if total > 50 else "medium" if total > 25 else "low",
        }

    def get_trend(self, window_sec: float = 300.0) -> str:
        """Retorna tendência do risco nos últimos N segundos."""
        cutoff = time.time() - window_sec
        recent = [r["score"] for r in self._risk_history if r["t"] > cutoff]
        if len(recent) < 5:
            return "stable"
        first = sum(recent[:len(recent)//3]) / max(1, len(recent)//3)
        last = sum(recent[-len(recent)//3:]) / max(1, len(recent)//3)
        delta = last - first
        if delta > 10:
            return "rising"
        if delta < -10:
            return "falling"
        return "stable"


# ---------------------------------------------------------------------------
# 7. AUTO REMEDIATOR
# ---------------------------------------------------------------------------

class AutoRemediator:
    """
    Quando o risco de ban é alto, aplica automaticamente medidas corretivas.
    Medidas progressivas (leve -> moderada -> severa).
    """

    def __init__(self):
        self._last_remediation = 0.0
        self._cooldown = 10.0
        self._consecutive_triggers = 0
        self._remediation_log: deque = deque(maxlen=50)

    def evaluate_and_remediate(self, risk: Dict, fingerprint: BehavioralFingerprint,
                                input_randomizer: InputRandomizer,
                                micro_humanizer: MicroHumanizer) -> Dict:
        """
        Avalia risco e aplica remediação. Retorna dict com ações tomadas.
        """
        now = time.time()
        if now - self._last_remediation < self._cooldown:
            return {"action": "none", "reason": "cooldown"}

        total = risk["total"]
        level = risk["risk_level"]

        if level in ("low", "medium"):
            self._consecutive_triggers = 0
            return {"action": "none", "reason": "risk_acceptable"}

        actions_taken = []
        self._consecutive_triggers += 1
        severity = self._consecutive_triggers

        # MEDIDA 1 (leve): Aumentar jitter e delays
        if total > 40 or severity >= 1:
            # Aumentar break_frequency temporariamente
            fingerprint.dimensions.break_frequency = _clamp(
                fingerprint.dimensions.break_frequency + 0.15, 0.1, 0.9
            )
            actions_taken.append("increased_micro_pauses")

        # MEDIDA 2 (moderada): Introduzir missclicks e hesitações
        if total > 55 or severity >= 2:
            fingerprint.dimensions.reaction_time_base = _clamp(
                fingerprint.dimensions.reaction_time_base + 0.05, 0.12, 0.55
            )
            fingerprint.dimensions.decision_speed = _clamp(
                fingerprint.dimensions.decision_speed - 0.1, 0.1, 0.95
            )
            actions_taken.append("slowed_reactions")
            actions_taken.append("increased_hesitation")

        # MEDIDA 3 (severa): Forçar break e alterar fingerprint drasticamente
        if total > 70 or severity >= 3:
            fingerprint.dimensions.tap_cluster_tightness = _clamp(
                fingerprint.dimensions.tap_cluster_tightness - 0.2, 0.0, 1.0
            )
            actions_taken.append("dispersed_tap_pattern")
            actions_taken.append("force_break_recommended")

        self._last_remediation = now
        self._cooldown = min(60.0, 10.0 * severity)

        entry = {
            "timestamp": now,
            "risk_total": total,
            "severity": severity,
            "actions": actions_taken,
        }
        self._remediation_log.append(entry)
        logger.warning(f"[ANTI-BAN] Remediação aplicada (risk={total:.1f}, severity={severity}): {actions_taken}")

        return {
            "action": "remediated",
            "severity": severity,
            "actions": actions_taken,
            "force_break": total > 70 or severity >= 3,
        }

    def get_remediation_history(self) -> List[Dict]:
        return list(self._remediation_log)


# ---------------------------------------------------------------------------
# 8. JOYSTICK MOVEMENT ANALYZER
# ---------------------------------------------------------------------------

class JoystickMovementAnalyzer:
    """
    Analisa movimentos do joystick para detetar padrões robóticos:
    - Círculos perfeitos (bots usam math.cos/sin)
    - Linhas retas perfeitas (0° ou 90° exatos)
    - Velocidade angular constante
    """

    def __init__(self):
        self._direction_history: deque = deque(maxlen=100)
        self._speed_history: deque = deque(maxlen=100)

    def record_joystick_input(self, dx: float, dy: float, duration: float):
        angle = math.atan2(dy, dx)
        speed = math.sqrt(dx*dx + dy*dy) / max(duration, 0.001)
        self._direction_history.append(angle)
        self._speed_history.append(speed)

    def detect_perfect_circles(self) -> bool:
        """Deteta se o bot está a fazer círculos perfeitos."""
        if len(self._direction_history) < 30:
            return False
        angles = list(self._direction_history)
        # Círculo perfeito: variação angular constante
        deltas = [(angles[i] - angles[i-1]) % (2*math.pi) for i in range(1, len(angles))]
        if len(deltas) < 10:
            return False
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta)**2 for d in deltas) / len(deltas)
        # Círculo perfeito: variância muito baixa na variação angular
        return variance < 0.001 and abs(mean_delta) > 0.05

    def detect_perfect_lines(self) -> bool:
        """Deteta linhas retas perfeitas (ângulos múltiplos de 45°)."""
        if len(self._direction_history) < 15:
            return False
        angles = list(self._direction_history)
        # Verificar se ângulos são múltiplos de 45° com precisão extrema
        perfect_count = 0
        for a in angles:
            deg = math.degrees(a) % 360
            nearest_45 = round(deg / 45) * 45
            if abs(deg - nearest_45) < 1.0:
                perfect_count += 1
        ratio = perfect_count / len(angles)
        return ratio > 0.7

    def detect_constant_speed(self) -> bool:
        """Deteta velocidade angular constante (muito robótico)."""
        if len(self._speed_history) < 20:
            return False
        speeds = list(self._speed_history)
        mean = sum(speeds) / len(speeds)
        variance = sum((s - mean)**2 for s in speeds) / len(speeds)
        cv = math.sqrt(variance) / mean if mean > 0 else 0
        return cv < 0.05

    def get_robotic_score(self) -> float:
        """0-100. Score de "roboticness"."""
        score = 0.0
        if self.detect_perfect_circles():
            score += 40.0
        if self.detect_perfect_lines():
            score += 30.0
        if self.detect_constant_speed():
            score += 30.0
        return min(100.0, score)

    def suggest_humanized_direction(self, target_dx: float, target_dy: float) -> Tuple[float, float]:
        """Adiciona ruído à direção do joystick para evitar linhas/círculos perfeitos."""
        angle = math.atan2(target_dy, target_dx)
        # Adicionar ruído angular (humanos não mantêm ângulo perfeito)
        angle_noise = _gaussian_noise(0, 0.08)  # ~5 graus de desvio
        new_angle = angle + angle_noise
        mag = math.sqrt(target_dx**2 + target_dy**2)
        # Variação de magnitude
        mag_noise = _gaussian_noise(0, mag * 0.05)
        new_mag = _clamp(mag + mag_noise, mag * 0.7, mag * 1.3)
        return (new_mag * math.cos(new_angle), new_mag * math.sin(new_angle))


# ---------------------------------------------------------------------------
# 9. AIM FATIGUE SIMULATOR
# ---------------------------------------------------------------------------

class AimFatigueSimulator:
    """
    Simula fadiga de mira ao longo da sessão:
    - Precisão diminui com o tempo
    - Reação ao stress (inimigos próximos) melhora brevemente (adrenalina)
    - Recuperação durante breaks
    """

    def __init__(self):
        self._session_start = time.time()
        self._last_break_time = time.time()
        self._precision_base = 1.0
        self._stress_buffer = 0.0  # adrenalina temporária
        self._shots_fired = 0
        self._hits = 0

    def record_shot(self, hit: bool):
        self._shots_fired += 1
        if hit:
            self._hits += 1

    def record_break(self):
        self._last_break_time = time.time()
        self._stress_buffer = 0.0

    def record_stress_event(self, intensity: float = 0.5):
        """Inimigo próximo, HP baixo, etc."""
        self._stress_buffer = _clamp(self._stress_buffer + intensity * 0.3, 0.0, 0.5)

    def get_precision_factor(self) -> float:
        """
        1.0 = precisão total, 0.5 = metade da precisão.
        Fadiga acumula ao longo da sessão, resets após breaks.
        """
        elapsed = (time.time() - self._last_break_time) / 3600  # horas desde break
        # Fadiga: perde ~15% de precisão por hora
        fatigue = elapsed * 0.15
        # Adrenalina compensa temporariamente
        adrenaline = self._stress_buffer * 0.2
        # Precisão baseada no ratio acertos/tiros (confidence)
        confidence = 0.0
        if self._shots_fired > 5:
            confidence = (self._hits / self._shots_fired - 0.5) * 0.1
        return _clamp(self._precision_base - fatigue + adrenaline + confidence, 0.4, 1.1)

    def get_aim_offset(self, target_distance: float) -> Tuple[float, float]:
        """Offset de mira baseado na fadiga e distância."""
        precision = self.get_precision_factor()
        # Quanto mais longe, mais difícil (humanos reais)
        distance_factor = min(target_distance / 500, 1.0)
        base_error = (1.0 - precision) * 50 + distance_factor * 20
        dx = _gaussian_noise(0, base_error)
        dy = _gaussian_noise(0, base_error * 1.2)
        return dx, dy


# ---------------------------------------------------------------------------
# 10. SOCIAL BEHAVIOR EMULATOR
# ---------------------------------------------------------------------------

class SocialBehaviorEmulator:
    """
    Emula comportamentos sociais de jogadores humanos:
    - Uso ocasional de emotes
    - Rotação de brawlers (não jogar sempre o mesmo)
    - Reação a eventos do jogo (vitória/derrota)
    """

    def __init__(self):
        self._brawler_history: deque = deque(maxlen=20)
        self._emote_cooldown = 0.0
        self._last_emote_time = 0.0

    def record_brawler_used(self, brawler: str):
        self._brawler_history.append(brawler)

    def should_rotate_brawler(self) -> bool:
        """Sugere rotação se o mesmo brawler foi usado demasiadas vezes seguidas."""
        if len(self._brawler_history) < 5:
            return False
        recent = list(self._brawler_history)[-5:]
        return len(set(recent)) == 1

    def should_emote(self, situation: str = "generic") -> bool:
        """Situações: 'kill', 'death', 'victory', 'defeat', 'generic'."""
        now = time.time()
        if now < self._emote_cooldown:
            return False
        probs = {
            "kill": 0.08,
            "death": 0.03,
            "victory": 0.25,
            "defeat": 0.05,
            "generic": 0.01,
        }
        if random.random() < probs.get(situation, 0.01):
            self._emote_cooldown = now + random.uniform(15, 45)
            return True
        return False

    def get_preferred_emote(self) -> str:
        """Retorna emote aleatório (só os nomes, execução fica a cargo do caller)."""
        return random.choice(["thumbs_up", "angry", "sad", "happy", "gg"])

    def get_recommended_brawler_rotation(self, available: List[str]) -> Optional[str]:
        """Sugere brawler diferente do último."""
        if not available or not self._brawler_history:
            return None
        last = self._brawler_history[-1]
        candidates = [b for b in available if b != last]
        return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# SISTEMA PRINCIPAL
# ---------------------------------------------------------------------------

class AdvancedAntiBanSystem:
    """
    Sistema anti-ban ultra-avançado que orquestra todos os 10 pilares.
    Drop-in replacement para AntiBanSystem existente.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

        # Inicializar pilares
        self.fingerprint = BehavioralFingerprint()
        self.input_randomizer = InputRandomizer(self.fingerprint)
        self.micro_humanizer = MicroHumanizer(self.fingerprint)
        self.session_orchestrator = SessionOrchestrator()
        self.playstyle = PlaystyleProfile()
        self.risk_scorer = BanRiskScorer()
        self.remediator = AutoRemediator()
        self.joystick_analyzer = JoystickMovementAnalyzer()
        self.aim_fatigue = AimFatigueSimulator()
        self.social = SocialBehaviorEmulator()

        # Estado interno
        self._last_action_time = time.time()
        self._action_count = 0
        self._apm_window_start = time.time()

        logger.info("[ANTI-BAN-ADVANCED] Sistema inicializado com 10 pilares ativos")

    # ------------------------------------------------------------------
    # API pública (compatível com AntiBanSystem anterior)
    # ------------------------------------------------------------------

    def record_action(self, action_type: str = "generic", coordinates: Optional[Tuple] = None,
                      interval: Optional[float] = None):
        if not self.enabled:
            return
        now = time.time()
        self._action_count += 1

        # Timing
        if interval is None:
            interval = now - self._last_action_time
        self.risk_scorer.record_timing(interval)
        self._last_action_time = now

        # Spatial
        if coordinates:
            self.risk_scorer.record_tap(*coordinates)
            self.input_randomizer.randomize_tap(*coordinates)

        # APM
        if now - self._apm_window_start >= 60:
            self.risk_scorer.record_apm(self._action_count)
            self._action_count = 0
            self._apm_window_start = now

    def record_swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float):
        if not self.enabled:
            return
        angle = math.atan2(y2 - y1, x2 - x1)
        self.risk_scorer.record_swipe_angle(angle)
        self.joystick_analyzer.record_joystick_input(x2 - x1, y2 - y1, duration)

    def record_match_result(self, result: str, brawler: str = "unknown",
                            map_name: str = "unknown", kills: int = 0,
                            deaths: int = 0, damage_dealt: float = 0.0,
                            survival_time: float = 0.0):
        if not self.enabled:
            return
        won = result == "win"
        self.risk_scorer.record_match_result(won)
        self.session_orchestrator.record_match_start()
        self.social.record_brawler_used(brawler)
        self.playstyle.adapt_from_match(won, kills, deaths, damage_dealt,
                                         survival_time, brawler, map_name)
        self.aim_fatigue.record_break()  # entre partidas recupera um pouco

        # Evoluir fingerprint
        elapsed_hours = (time.time() - self.fingerprint.creation_time) / 3600
        self.fingerprint.evolve(elapsed_hours)

    def record_shot(self, hit: bool):
        self.aim_fatigue.record_shot(hit)

    def record_stress(self, intensity: float = 0.5):
        self.aim_fatigue.record_stress_event(intensity)

    def should_start_match(self) -> bool:
        if not self.enabled:
            return True
        return self.session_orchestrator.can_start_match()

    def check_throttle(self) -> bool:
        if not self.enabled:
            return False
        risk = self.risk_scorer.calculate_risk()
        return risk["risk_level"] in ("high", "critical")

    def check_pattern(self) -> bool:
        if not self.enabled:
            return False
        risk = self.risk_scorer.calculate_risk()
        return risk["total"] > 60

    def get_adaptive_pacing(self, base_delay: float, action_type: str = "generic") -> float:
        """Calcula pacing adaptativo combinando sessão, risco e remediação."""
        if not self.enabled:
            return base_delay

        delay = self.session_orchestrator.recommend_pacing(base_delay, action_type)

        risk = self.risk_scorer.calculate_risk()
        risk_total = risk.get("total", 0.0)
        risk_level = risk.get("risk_level", "low")

        if risk_level == "medium":
            delay *= 1.08
        elif risk_level == "high":
            delay *= 1.20
        elif risk_level == "critical":
            delay *= 1.40

        # Se o scoring estiver a subir, desacelerar levemente.
        if self.risk_scorer.get_trend() == "rising":
            delay *= 1.10

        remediation = self.remediator.evaluate_and_remediate(
            risk, self.fingerprint, self.input_randomizer, self.micro_humanizer
        )
        if remediation.get("force_break"):
            delay *= 1.25

        # Puxar um pouco para o fingerprint atual para manter consistência de sessão.
        delay *= self.fingerprint.get_delay_multiplier()

        return self.session_orchestrator.recommend_pacing(delay, action_type)

    def apply_action_delay(self, action_type: str, base_delay: float) -> float:
        if not self.enabled:
            return base_delay
        # 1. Randomização de delay
        delay = self.get_adaptive_pacing(base_delay, action_type)
        delay = self.input_randomizer.randomize_delay(delay, action_type)
        # 2. Hesitação
        importance = 0.3 if action_type in ("attack", "move") else 0.7
        hesitation = self.micro_humanizer.hesitate_before_action(importance)
        delay += hesitation
        # 3. Micro-pausa
        if self.input_randomizer.should_add_micro_pause():
            delay += self.input_randomizer.get_micro_pause_duration()
        # 4. Fadiga de sessão
        fatigue = self.session_orchestrator.get_fatigue_factor()
        delay *= (1.0 + fatigue * 0.2)
        # 5. Warmup (primeiras partidas mais lentas)
        warmup = self.session_orchestrator.get_warmup_factor()
        delay *= (1.0 + warmup * 0.15)
        return delay

    def randomize_tap(self, x: float, y: float) -> Tuple[float, float]:
        if not self.enabled:
            return x, y
        return self.input_randomizer.randomize_tap(x, y)

    def randomize_swipe(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        if not self.enabled:
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "control_points": []}
        # Aplicar correção de trajetória
        tx, ty = self.micro_humanizer.maybe_correct_trajectory((x2, y2))
        cps = self.input_randomizer.randomize_swipe_control_points(x1, y1, tx, ty)
        return {"x1": x1, "y1": y1, "x2": tx, "y2": ty, "control_points": cps}

    def get_aim_offset(self, target_distance: float) -> Tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        return self.aim_fatigue.get_aim_offset(target_distance)

    def get_fingerprint(self) -> Dict:
        return self.fingerprint.summary()

    def get_recommended_aggression(self) -> float:
        fatigue = self.session_orchestrator.get_fatigue_factor()
        return self.playstyle.get_aggression(fatigue)

    def should_use_super(self, hp_ratio: float, enemies_nearby: int) -> bool:
        return self.playstyle.should_use_super(hp_ratio, enemies_nearby)

    def should_emote(self, situation: str = "generic") -> bool:
        return self.social.should_emote(situation)

    def should_rotate_brawler(self) -> bool:
        return self.social.should_rotate_brawler()

    def get_status(self) -> Dict:
        risk = self.risk_scorer.calculate_risk()
        remediation = self.remediator.evaluate_and_remediate(
            risk, self.fingerprint, self.input_randomizer, self.micro_humanizer
        )
        return {
            "enabled": self.enabled,
            "risk": risk,
            "risk_trend": self.risk_scorer.get_trend(),
            "session": self.session_orchestrator.get_status(),
            "adaptive_pacing": {
                "next_attack_delay": round(self.get_adaptive_pacing(0.18, "attack"), 3),
                "next_menu_delay": round(self.get_adaptive_pacing(0.28, "menu_nav"), 3),
            },
            "fingerprint": self.fingerprint.summary(),
            "playstyle": self.playstyle.summary(),
            "aim_precision": round(self.aim_fatigue.get_precision_factor(), 3),
            "joystick_robotic_score": round(self.joystick_analyzer.get_robotic_score(), 1),
            "remediation": remediation,
        }
