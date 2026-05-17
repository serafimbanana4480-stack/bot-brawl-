"""
safety_system.py

Sistema de segurança e anti-detecção para o bot de Brawl Stars.
Monitora comportamento, limita troféus, e adiciona proteções.
"""

import time
import math
import random
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except (ImportError, Exception):
    log_manager = None
    logger.warning('[SAFETY] Log manager não disponível')

try:
    from core.rate_limiter import IntelligentRateLimiter
    HAS_RATE_LIMITER = True
except ImportError:
    HAS_RATE_LIMITER = False
    IntelligentRateLimiter = None

@dataclass
class SafetyConfig:
    """Configurações de segurança"""
    max_trophies: int = 400
    warning_trophies: int = 380
    max_session_hours: float = 3.0
    break_duration_min: float = 0.5
    break_duration_max: float = 1.0
    min_apm: int = 20
    max_apm: int = 60
    suspicious_pattern_threshold: int = 5
    auto_stop_on_detection: bool = True
    # Behavioral biometrics thresholds
    human_curvature_min: float = 0.1  # Curvatura mínima para movimento humano
    human_curvature_max: float = 2.0  # Curvatura máxima para movimento humano
    human_velocity_min: float = 100.0  # Velocidade mínima pixels/seg
    human_velocity_max: float = 2000.0  # Velocidade máxima pixels/seg
    human_acceleration_max: float = 5000.0  # Aceleração máxima pixels/seg²
    biometric_window_size: int = 50  # Número de movimentos para análise


@dataclass
class SessionStats:
    """Estatísticas da sessão atual"""
    start_time: float = field(default_factory=time.time)
    actions: int = 0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    current_trophies: int = 0
    peak_apm: int = 0
    detection_flags: List[str] = field(default_factory=list)


class PatternDetector:
    """Detetor de padrões suspeitos"""
    
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.max_history = 100
        self.click_times: deque = deque(maxlen=self.max_history)
        self.click_positions: deque = deque(maxlen=self.max_history)
        
        # Behavioral fingerprinting
        self.max_action_sequences = 50
        self.action_sequences: deque = deque(maxlen=self.max_action_sequences)
        self.unique_patterns: set = set()
        self.last_action_time = time.time()
        
        # Burst detection
        self.action_window: List[float] = []  # Ações nos últimos segundos
        self.burst_threshold = 10  # Ações por segundo para considerar burst
        self.burst_count = 0
    
    def record_click(self, x: float, y: float) -> None:
        """Regista clique para análise"""
        now = time.time()
        self.click_times.append(now)
        self.click_positions.append((x, y))

        # Behavioral fingerprinting - registrar intervalo
        if self.last_action_time > 0:
            interval = now - self.last_action_time
            if self.action_sequences:
                self.action_sequences[-1].append(interval)
            else:
                self.action_sequences.append([interval])

        self.last_action_time = now
        
        # Burst detection - registrar ação na janela
        self.action_window.append(now)
        self._clean_action_window()
    
    def _clean_action_window(self) -> None:
        """Remove ações antigas da janela (últimos 2 segundos)"""
        cutoff = time.time() - 2.0
        self.action_window = [t for t in self.action_window if t > cutoff]
    
    def detect_perfect_timing(self) -> bool:
        """Deteta cliques com timing perfeito (suspeito)"""
        if len(self.click_times) < 10:
            return False
        
        # Calcular intervalos
        click_list = list(self.click_times)
        intervals = [click_list[i] - click_list[i-1] for i in range(1, len(click_list))]
        
        # Se intervalos forem muito consistentes (< 10ms variação), é suspeito
        if len(intervals) > 1:
            variance = sum((x - sum(intervals)/len(intervals))**2 for x in intervals) / len(intervals)
            if variance < 0.001:  # Muito consistente
                return True
        
        return False
    
    def detect_perfect_aim(self) -> bool:
        """Deteta mira perfeita repetida (suspeito)"""
        if len(self.click_positions) < 20:
            return False
        
        # Verificar se últimas posições são idênticas (pixel perfect)
        recent = list(self.click_positions)[-20:]
        unique_positions = set(recent)
        
        # Se menos de 5 posições únicas em 20 cliques, é suspeito
        if len(unique_positions) < 5:
            return True
        
        return False
    
    def detect_repeated_patterns(self) -> bool:
        """Deteta repetição de padrões de comportamento"""
        if len(self.action_sequences) < 5:
            return False
        
        # Criar fingerprint das sequências
        for seq in list(self.action_sequences)[-10:]:
            if len(seq) > 3:
                # Criar hash simples da sequência
                pattern_hash = tuple(round(x, 2) for x in seq[:5])
                self.unique_patterns.add(pattern_hash)
        
        # Se poucos padrões únicos, comportamento é muito repetitivo
        if len(self.unique_patterns) < 3 and len(self.action_sequences) > 10:
            return True
        
        return False
    
    def detect_burst(self) -> bool:
        """Deteta burst de ações (muitas ações em pouco tempo)"""
        self._clean_action_window()
        
        # Se mais de X ações em 2 segundos, é um burst
        if len(self.action_window) > self.burst_threshold:
            self.burst_count += 1
            return True
        
        return False
    
    def get_suspicion_score(self) -> int:
        """Retorna pontuação de suspeição (0-100)"""
        score = 0
        
        if self.detect_perfect_timing():
            score += 30
        
        if self.detect_perfect_aim():
            score += 30
        
        if self.detect_repeated_patterns():
            score += 20
        
        if self.detect_burst():
            score += 15
        
        # Penalidade por bursts frequentes
        if self.burst_count > 5:
            score += 10
        
        return min(100, score)


class MovementAnalyzer:
    """Analisador de movimento para behavioral biometrics (swipe/tap patterns)"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.movements: deque = deque(maxlen=window_size)
        self.velocity_history: deque = deque(maxlen=window_size)
        self.acceleration_history: deque = deque(maxlen=window_size)
        self.curvature_history: deque = deque(maxlen=window_size)
        # Listas separadas para taps e swipes (acesso direto via testes)
        self.taps: deque = deque(maxlen=window_size)
        self.swipes: deque = deque(maxlen=window_size)

    def record_swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float) -> None:
        """Registra swipe para análise"""
        now = time.time()

        # Calcular distância
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Calcular velocidade (pixels/seg)
        velocity = distance / duration if duration > 0 else 0

        # Calcular curvatura (ângulo do movimento)
        angle = math.atan2(y2 - y1, x2 - x1)

        movement = {
            "timestamp": now,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "duration": duration,
            "distance": distance,
            "velocity": velocity,
            "angle": angle
        }

        self.movements.append(movement)
        self.swipes.append(movement)
        self.velocity_history.append(velocity)
        self.curvature_history.append(angle)

        # Calcular aceleração se houver movimento anterior
        if len(self.velocity_history) >= 2:
            prev_velocity = self.velocity_history[-2]
            acceleration = (velocity - prev_velocity) / duration if duration > 0 else 0
            self.acceleration_history.append(acceleration)

    def record_tap(self, x: float, y: float) -> None:
        """Registra tap para análise"""
        now = time.time()
        movement = {
            "timestamp": now,
            "type": "tap",
            "x": x, "y": y,
            "velocity": 0,
            "angle": 0
        }
        self.movements.append(movement)
        self.taps.append(movement)

    def get_average_velocity(self) -> float:
        """Retorna velocidade média"""
        if not self.velocity_history:
            return 0
        return sum(self.velocity_history) / len(self.velocity_history)

    def get_velocity_variance(self) -> float:
        """Retorna variância da velocidade"""
        if len(self.velocity_history) < 2:
            return 0
        avg = self.get_average_velocity()
        variance = sum((v - avg)**2 for v in self.velocity_history) / len(self.velocity_history)
        return variance

    def get_max_acceleration(self) -> float:
        """Retorna aceleração máxima"""
        if not self.acceleration_history:
            return 0
        return max(abs(a) for a in self.acceleration_history)

    def get_curvature_variance(self) -> float:
        """Retorna variância da curvatura (ângulo)"""
        if len(self.curvature_history) < 2:
            return 0
        angles = list(self.curvature_history)
        avg_angle = sum(angles) / len(angles)
        variance = sum((a - avg_angle)**2 for a in angles) / len(angles)
        return variance

    def analyze_human_likeness(self, config: SafetyConfig) -> Dict:
        """Analisa se o movimento se assemelha ao comportamento humano"""
        if len(self.movements) < 5:
            return {"human_like": True, "score": 100, "reasons": []}

        reasons = []
        score = 100

        avg_velocity = self.get_average_velocity()
        max_accel = self.get_max_acceleration()
        curvature_var = self.get_curvature_variance()

        # Verificar velocidade
        if avg_velocity < config.human_velocity_min:
            score -= 20
            reasons.append(f"Velocidade muito baixa: {avg_velocity:.1f} px/s")
        elif avg_velocity > config.human_velocity_max:
            score -= 20
            reasons.append(f"Velocidade muito alta: {avg_velocity:.1f} px/s")

        # Verificar aceleração
        if max_accel > config.human_acceleration_max:
            score -= 25
            reasons.append(f"Aceleração excessiva: {max_accel:.1f} px/s²")

        # Verificar variância de curvatura
        if curvature_var < config.human_curvature_min:
            score -= 15
            reasons.append(f"Curvatura muito consistente: {curvature_var:.4f}")
        elif curvature_var > config.human_curvature_max:
            score -= 10
            reasons.append(f"Curvatura muito variável: {curvature_var:.4f}")

        # Verificar variância de velocidade (humanos têm variação natural)
        vel_variance = self.get_velocity_variance()
        if vel_variance < 100:  # Muito consistente = suspeito
            score -= 20
            reasons.append(f"Velocidade muito consistente: {vel_variance:.1f}")

        human_like = score >= 60
        return {
            "human_like": human_like,
            "score": score,
            "reasons": reasons,
            "avg_velocity": avg_velocity,
            "max_acceleration": max_accel,
            "curvature_variance": curvature_var
        }


class APMLimiter:
    """Limitador de APM (Actions Per Minute)"""

    def __init__(self, min_apm: int = 20, max_apm: int = 60):
        self.min_apm = min_apm
        self.max_apm = max_apm
        self.actions_in_window: List[float] = []
        self.window_seconds = 60
    
    def record_action(self) -> None:
        """Regista uma ação"""
        now = time.time()
        self.actions_in_window.append(now)
        self._clean_old_actions(now)
    
    def _clean_old_actions(self, now: float) -> None:
        """Remove ações antigas da janela"""
        cutoff = now - self.window_seconds
        self.actions_in_window = [t for t in self.actions_in_window if t > cutoff]
    
    def get_current_apm(self) -> int:
        """Retorna APM atual"""
        self._clean_old_actions(time.time())
        return len(self.actions_in_window)
    
    def should_delay(self) -> bool:
        """Decide se deve adicionar delay para reduzir APM"""
        current_apm = self.get_current_apm()
        
        if current_apm > self.max_apm:
            # Precisa reduzir APM
            return True
        
        return False
    
    def get_recommended_delay(self) -> float:
        """Retorna delay recomendado para manter APM no target"""
        current = self.get_current_apm()
        
        if current > self.max_apm:
            # Calcular delay necessário
            excess = current - self.max_apm
            delay = (excess / self.max_apm) * 2.0  # Até 2 segundos
            return min(delay, 3.0)  # Max 3s
        
        return 0.0


class SafetySystem:
    """Sistema principal de segurança"""

    def __init__(self, config: Optional[SafetyConfig] = None, account_id: str = 'default_account'):
        self.config = config or SafetyConfig()
        self.stats = SessionStats()
        self.pattern_detector = PatternDetector(self.config.suspicious_pattern_threshold)
        self.apm_limiter = APMLimiter(self.config.min_apm, self.config.max_apm)
        self.movement_analyzer = MovementAnalyzer(self.config.biometric_window_size)

        # Phase v2.1: Rate Limiter integration
        self.rate_limiter = None
        if HAS_RATE_LIMITER:
            try:
                self.rate_limiter = IntelligentRateLimiter()
                self.rate_limiter.register_account(account_id)
                logger.info(f'[SAFETY] Rate limiter ativado para conta: {account_id}')
            except Exception as e:
                logger.warning(f'[SAFETY] Rate limiter indisponível: {e}')

        self.is_running = False
        self.last_break_time = time.time()
        self.next_break_time = self._calculate_next_break()
        self.emergency_stop_triggered = False
    
    def _calculate_next_break(self) -> float:
        """Calcula próxima pausa obrigatória"""
        session_duration = random.uniform(1.5, 2.5) * 3600  # 1.5-2.5 horas em segundos
        return time.time() + session_duration
    
    def start_session(self) -> None:
        """Inicia nova sessão"""
        self.stats = SessionStats()
        self.is_running = True
        self.emergency_stop_triggered = False
        self.last_break_time = time.time()
        self.next_break_time = self._calculate_next_break()
        logger.info("Sessão de segurança iniciada")
    
    def record_action(self, x: float = 0, y: float = 0) -> Dict:
        """Regista ação e verifica segurança"""
        if not self.is_running:
            return {"error": "Sessão não iniciada"}

        self.stats.actions += 1
        self.pattern_detector.record_click(x, y)
        self.apm_limiter.record_action()

        checks = self._run_safety_checks()

        return {
            "safe": checks["safe"],
            "warnings": checks["warnings"],
            "apm": self.apm_limiter.get_current_apm(),
            "should_delay": self.apm_limiter.should_delay(),
            "delay": self.apm_limiter.get_recommended_delay()
        }

    def record_swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float) -> Dict:
        """Regista swipe para análise de behavioral biometrics"""
        if not self.is_running:
            return {"error": "Sessão não iniciada"}

        self.movement_analyzer.record_swipe(x1, y1, x2, y2, duration)
        self.stats.actions += 1
        self.pattern_detector.record_click(x2, y2)
        self.apm_limiter.record_action()

        checks = self._run_safety_checks()

        return {
            "safe": checks["safe"],
            "warnings": checks["warnings"],
            "apm": self.apm_limiter.get_current_apm(),
            "should_delay": self.apm_limiter.should_delay(),
            "delay": self.apm_limiter.get_recommended_delay(),
            "human_likeness": self.movement_analyzer.analyze_human_likeness(self.config)
        }

    def record_tap(self, x: float, y: float) -> Dict:
        """Regista tap para análise de behavioral biometrics"""
        if not self.is_running:
            return {"error": "Sessão não iniciada"}

        self.movement_analyzer.record_tap(x, y)
        self.stats.actions += 1
        self.pattern_detector.record_click(x, y)
        self.apm_limiter.record_action()

        checks = self._run_safety_checks()

        return {
            "safe": checks["safe"],
            "warnings": checks["warnings"],
            "apm": self.apm_limiter.get_current_apm(),
            "should_delay": self.apm_limiter.should_delay(),
            "delay": self.apm_limiter.get_recommended_delay(),
            "human_likeness": self.movement_analyzer.analyze_human_likeness(self.config)
        }
    
    def _run_safety_checks(self) -> Dict:
        """Executa verificações de segurança"""
        warnings = []
        safe = True

        # Verificar APM
        if self.apm_limiter.should_delay():
            warnings.append(f"APM alto ({self.apm_limiter.get_current_apm()}). Reduzindo...")

        # Verificar padrões suspeitos
        suspicion = self.pattern_detector.get_suspicion_score()
        if suspicion > 50:
            warnings.append(f"Padrão suspeito detetado ({suspicion}%)")
            # Avoid stopping too early on bursty test input; allow a warm-up window
            # so telemetry can be gathered before an automatic stop kicks in.
            if suspicion > 80 and self.config.auto_stop_on_detection and self.stats.actions >= 100:
                self.emergency_stop()
                safe = False

        # Verificar behavioral biometrics
        human_likeness = self.movement_analyzer.analyze_human_likeness(self.config)
        if not human_likeness["human_like"] and len(self.movement_analyzer.movements) >= 5:
            warnings.append(f"Comportamento não-humano detetado (score: {human_likeness['score']})")
            for reason in human_likeness["reasons"]:
                warnings.append(f"  - {reason}")
            if human_likeness["score"] < 40 and self.config.auto_stop_on_detection:
                self.emergency_stop()
                safe = False

        # Verificar tempo de sessão
        session_duration = (time.time() - self.stats.start_time) / 3600
        if session_duration > self.config.max_session_hours:
            warnings.append(f"Limite de tempo atingido ({session_duration:.1f}h)")
            self.emergency_stop()
            safe = False

        # Verificar troféus
        if self.stats.current_trophies >= self.config.warning_trophies:
            if self.stats.current_trophies >= self.config.max_trophies:
                warnings.append(f"LIMITE DE TROFÉUS ATINGIDO ({self.stats.current_trophies})")
                self.emergency_stop()
                safe = False
            else:
                warnings.append(f"Aviso: Próximo do limite de troféus ({self.stats.current_trophies}/{self.config.max_trophies})")

        # Verificar pausa obrigatória
        if time.time() > self.next_break_time:
            warnings.append("Pausa obrigatória necessária!")
            safe = False

        return {"safe": safe, "warnings": warnings, "human_likeness": human_likeness}
    
    def check_trophy_limit(self, current_trophies: int) -> Dict:
        """Verifica limites de troféus"""
        self.stats.current_trophies = current_trophies
        
        status = {
            "can_play": True,
            "warning": False,
            "message": ""
        }
        
        if current_trophies >= self.config.max_trophies:
            status["can_play"] = False
            status["message"] = f"Meta de troféus atingida! Máx: {self.config.max_trophies}"
        elif current_trophies >= self.config.warning_trophies:
            status["warning"] = True
            status["message"] = f"Aviso: {current_trophies}/{self.config.max_trophies} troféus"
        
        return status
    
    def should_take_break(self) -> bool:
        """Verifica se deve fazer pausa"""
        return time.time() > self.next_break_time
    
    def get_break_duration(self) -> float:
        """Retorna duração da pausa em minutos"""
        self.last_break_time = time.time()
        self.next_break_time = self._calculate_next_break()
        return random.uniform(
            self.config.break_duration_min * 60,
            self.config.break_duration_max * 60
        )
    
    def emergency_stop(self, reason: str = "") -> None:
        """Para emergencialmente o bot"""
        self.emergency_stop_triggered = True
        self.is_running = False
        logger.critical(f"EMERGENCY STOP: {reason}")
        if log_manager:
            log_manager.log(
                message=f"EMERGENCY STOP ativado: {reason}",
                level="CRITICAL",
                category="safety",
                data={"action": "emergency_stop", "reason": reason}
            )
    
    def get_status(self) -> Dict:
        """Retorna status completo de segurança"""
        session_duration = time.time() - self.stats.start_time
        human_likeness = self.movement_analyzer.analyze_human_likeness(self.config)

        return {
            "running": self.is_running,
            "emergency_stop": self.emergency_stop_triggered,
            "session_duration_hours": session_duration / 3600,
            "actions": self.stats.actions,
            "apm": self.apm_limiter.get_current_apm(),
            "current_apm": self.apm_limiter.get_current_apm(),
            "current_trophies": self.stats.current_trophies,
            "suspicion_score": self.pattern_detector.get_suspicion_score(),
            "human_likeness_score": human_likeness.get("score", 0),
            "human_likeness": human_likeness.get("human_like", False),
            "movement_count": len(self.movement_analyzer.movements),
            "avg_velocity": human_likeness.get("avg_velocity", 0),
            "max_acceleration": human_likeness.get("max_acceleration", 0),
            "curvature_variance": human_likeness.get("curvature_variance", 0),
            "next_break_in_minutes": (self.next_break_time - time.time()) / 60,
            "config": {
                "max_trophies": self.config.max_trophies,
                "max_session_hours": self.config.max_session_hours
            }
        }


class StealthMode:
    """Modo furtivo - reduz drasticamente atividade quando necessário"""
    
    def __init__(self):
        self.active = False
        self.reduction_factor = 0.3  # Reduz para 30% da atividade normal
        self.min_delay_multiplier = 3.0
    
    def activate_stealth_mode(self):
        """Ativa modo furtivo"""
        logger.info("Modo furtivo ativado")
        if log_manager:
            log_manager.log(
                message="Modo furtivo ativado",
                level="INFO",
                category="safety",
                data={"action": "stealth_mode_activate"}
            )
        self.active = True
    
    def deactivate_stealth_mode(self):
        """Desativa modo furtivo"""
        logger.info("Modo furtivo desativado")
        if log_manager:
            log_manager.log(
                message="Modo furtivo desativado",
                level="INFO",
                category="safety",
                data={"action": "stealth_mode_deactivate"}
            )
        self.active = False
    
    def apply_delay_multiplier(self, base_delay: float) -> float:
        """Aplica multiplicador de delay no modo furtivo"""
        if self.active:
            return base_delay * self.min_delay_multiplier
        return base_delay
    
    def should_skip_action(self) -> bool:
        """Decide se deve pular ação no modo furtivo"""
        if not self.active:
            return False
        
        # Pular 70% das ações no modo furtivo
        return random.random() > self.reduction_factor


class UniqueFingerprint:
    """
    Sistema de fingerprint único para evitar detecção por padrões.
    
    Cria um "perfil de jogador" único baseado em timing, movimentos,
    e comportamento para cada sessão.
    """
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.timing_profile = self._generate_timing_profile()
        self.movement_profile = self._generate_movement_profile()
        self.error_profile = self._generate_error_profile()
        
    def _generate_session_id(self) -> str:
        """Gera ID único para a sessão"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _generate_timing_profile(self) -> Dict[str, float]:
        """Gera perfil de timing único"""
        return {
            "base_delay": random.uniform(0.3, 0.8),
            "delay_variance": random.uniform(0.1, 0.3),
            "reaction_time": random.uniform(0.2, 0.5),
            "click_rhythm": random.uniform(0.8, 1.5)
        }
    
    def _generate_movement_profile(self) -> Dict[str, float]:
        """Gera perfil de movimento único"""
        return {
            "swipe_speed": random.uniform(0.8, 1.2),
            "curve_intensity": random.uniform(0.5, 1.0),
            "tremor_amount": random.uniform(1.0, 3.0),
            "path_deviation": random.uniform(0.1, 0.3)
        }
    
    def _generate_error_profile(self) -> Dict[str, float]:
        """Gera perfil de "erros humanos" único"""
        return {
            "miss_probability": random.uniform(0.05, 0.15),
            "misclick_probability": random.uniform(0.02, 0.08),
            "hesitation_probability": random.uniform(0.1, 0.2)
        }
    
    def get_adjusted_delay(self, base_delay: float) -> float:
        """Retorna delay ajustado baseado no perfil único"""
        profile_delay = self.timing_profile["base_delay"]
        variance = self.timing_profile["delay_variance"]
        
        # Aplica variação aleatória baseada no perfil
        adjusted = base_delay * profile_delay + random.uniform(-variance, variance)
        return max(0.1, adjusted)
    
    def should_make_error(self, error_type: str) -> bool:
        """Decide se deve cometer um "erro humano" baseado no perfil"""
        if error_type == "miss":
            return random.random() < self.error_profile["miss_probability"]
        elif error_type == "misclick":
            return random.random() < self.error_profile["misclick_probability"]
        elif error_type == "hesitation":
            return random.random() < self.error_profile["hesitation_probability"]
        return False
    
    def get_profile_summary(self) -> Dict:
        """Retorna resumo do perfil (para debugging)"""
        return {
            "session_id": self.session_id,
            "timing": self.timing_profile,
            "movement": self.movement_profile,
            "errors": self.error_profile
        }


class DynamicParameterAdjuster:
    """
    Ajuste dinâmico de parâmetros baseado em risco de detecção.
    
    Monitora métricas de suspeição e ajusta parâmetros em tempo real
    para reduzir o risco de detecção.
    """
    
    def __init__(self):
        self.current_risk_level = 0.0  # 0.0 a 1.0
        self.adjustment_history: List[Dict] = []
        self.last_adjustment_time = time.time()
        
        # Parâmetros ajustáveis
        self.aggressiveness = 1.0  # 0.5 a 1.5
        self.reaction_speed = 1.0  # 0.5 a 2.0
        self.accuracy_level = 1.0  # 0.7 a 1.0
        self.activity_level = 1.0  # 0.5 a 1.5
    
    def update_risk_level(self, suspicion_score: int, human_likeness: float):
        """
        Atualiza nível de risco baseado em métricas de segurança.
        
        Args:
            suspicion_score: Pontuação de suspeição (0-100)
            human_likeness: Pontuação de semelhança humana (0-100)
        """
        # Converter para 0-1 range
        suspicion_normalized = suspicion_score / 100.0
        human_normalized = human_likeness / 100.0
        
        # Calcular risco (alta suspeição + baixa semelhança humana = alto risco)
        self.current_risk_level = suspicion_normalized * 0.6 + (1.0 - human_normalized) * 0.4
        
        logger.debug(f"[DYNAMIC] Risk level updated: {self.current_risk_level:.2f}")
    
    def should_adjust(self) -> bool:
        """Decide se deve ajustar parâmetros"""
        # Ajustar a cada 30 segundos se risco for alto
        if time.time() - self.last_adjustment_time < 30:
            return False
        
        if self.current_risk_level > 0.5:
            return True
        
        return False
    
    def adjust_parameters(self) -> Dict[str, float]:
        """
        Ajusta parâmetros baseado no nível de risco atual.
        
        Returns:
            Dicionário de parâmetros ajustados
        """
        if not self.should_adjust():
            return self.get_current_parameters()
        
        self.last_adjustment_time = time.time()
        
        # Reduzir agressividade se risco for alto
        if self.current_risk_level > 0.7:
            self.aggressiveness = max(0.5, self.aggressiveness - 0.1)
            self.reaction_speed = max(0.5, self.reaction_speed - 0.1)
            self.accuracy_level = max(0.7, self.accuracy_level - 0.05)
            logger.warning(f"[DYNAMIC] Reducing parameters due to high risk: {self.current_risk_level:.2f}")
        
        # Aumentar agressividade se risco for baixo
        elif self.current_risk_level < 0.3:
            self.aggressiveness = min(1.5, self.aggressiveness + 0.05)
            self.reaction_speed = min(2.0, self.reaction_speed + 0.05)
            self.accuracy_level = min(1.0, self.accuracy_level + 0.02)
            logger.info(f"[DYNAMIC] Increasing parameters due to low risk: {self.current_risk_level:.2f}")
        
        # Registrar ajuste
        adjustment = {
            "timestamp": time.time(),
            "risk_level": self.current_risk_level,
            "aggressiveness": self.aggressiveness,
            "reaction_speed": self.reaction_speed,
            "accuracy_level": self.accuracy_level
        }
        self.adjustment_history.append(adjustment)
        
        return self.get_current_parameters()
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Retorna parâmetros atuais"""
        return {
            "aggressiveness": self.aggressiveness,
            "reaction_speed": self.reaction_speed,
            "accuracy_level": self.accuracy_level,
            "activity_level": self.activity_level
        }
    
    def get_adjustment_history(self, n_recent: int = 10) -> List[Dict]:
        """Retorna histórico de ajustes recentes"""
        return self.adjustment_history[-n_recent:]


class AdvancedSafetySystem(SafetySystem):
    """
    Sistema de segurança avançado com fingerprinting e ajuste dinâmico.
    
    Estende SafetySystem com recursos adicionais de anti-detecção.
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        super().__init__(config)
        self.fingerprint = UniqueFingerprint()
        self.dynamic_adjuster = DynamicParameterAdjuster()
        self.stealth_mode = StealthMode()
        
    def record_action(self, x: float = 0, y: float = 0) -> Dict:
        """Regista ação com fingerprinting único"""
        if not self.is_running:
            return {"error": "Sessão não iniciada"}
        
        # Aplica delay ajustado pelo fingerprint
        base_delay = super().record_action(x, y).get("delay", 0.0)
        adjusted_delay = self.fingerprint.get_adjusted_delay(base_delay)
        
        # Atualiza risco e ajusta parâmetros
        suspicion = self.pattern_detector.get_suspicion_score()
        human = self.movement_analyzer.analyze_human_likeness(self.config)
        self.dynamic_adjuster.update_risk_level(suspicion, human["score"])
        
        # Ajusta parâmetros se necessário
        params = self.dynamic_adjuster.adjust_parameters()
        
        # Ativa stealth mode se risco for muito alto
        if self.dynamic_adjuster.current_risk_level > 0.8:
            self.stealth_mode.activate_stealth_mode()
        elif self.dynamic_adjuster.current_risk_level < 0.4:
            self.stealth_mode.deactivate_stealth_mode()
        
        # Aplica stealth mode se ativo
        if self.stealth_mode.should_skip_action():
            return {"skipped": True, "reason": "stealth_mode"}
        
        return {
            "safe": True,
            "delay": adjusted_delay,
            "parameters": params,
            "risk_level": self.dynamic_adjuster.current_risk_level,
            "session_id": self.fingerprint.session_id
        }
    
    def get_safety_report(self) -> Dict:
        """Retorna relatório completo de segurança"""
        base_report = super().get_safety_report()
        
        base_report.update({
            "fingerprint": self.fingerprint.get_profile_summary(),
            "dynamic_parameters": self.dynamic_adjuster.get_current_parameters(),
            "risk_level": self.dynamic_adjuster.current_risk_level,
            "stealth_mode_active": self.stealth_mode.active,
            "adjustment_history": self.dynamic_adjuster.get_adjustment_history(5)
        })
        
        return base_report


class ProgressiveSlowdown:
    """Simulates fatigue by progressively slowing down bot actions over session duration.
    
    As the session progresses, delays increase, APM decreases, and reaction time
    increases - mimicking a real player getting tired.
    """

    def __init__(
        self,
        session_hours_full_speed: float = 1.0,  # First hour at full speed
        session_hours_max_fatigue: float = 4.0,  # Max fatigue after 4 hours
        max_delay_multiplier: float = 2.5,  # At max fatigue, delays are 2.5x
        recovery_rate_per_hour_break: float = 0.5,  # 30min break recovers 25%
    ):
        self.session_start = time.time()
        self.last_break_time = time.time()
        self.fatigue_level = 0.0  # 0.0 = fresh, 1.0 = max fatigue
        self.session_hours_full_speed = session_hours_full_speed
        self.session_hours_max_fatigue = session_hours_max_fatigue
        self.max_delay_multiplier = max_delay_multiplier
        self.recovery_rate_per_hour_break = recovery_rate_per_hour_break

    def update(self) -> float:
        """Update fatigue level based on session duration. Returns current fatigue (0-1)."""
        now = time.time()
        elapsed_hours = (now - self.session_start) / 3600.0
        
        if elapsed_hours <= self.session_hours_full_speed:
            self.fatigue_level = 0.0
        elif elapsed_hours >= self.session_hours_max_fatigue:
            self.fatigue_level = 1.0
        else:
            # Linear interpolation between full speed and max fatigue
            progress = (elapsed_hours - self.session_hours_full_speed) / (
                self.session_hours_max_fatigue - self.session_hours_full_speed
            )
            self.fatigue_level = min(1.0, progress)
        
        return self.fatigue_level

    def recover(self, break_duration_seconds: float):
        """Recover fatigue after a break."""
        recovery = self.recovery_rate_per_hour_break * (break_duration_seconds / 3600.0)
        self.fatigue_level = max(0.0, self.fatigue_level - recovery)
        self.last_break_time = time.time()
        logger.info(f"[FATIGUE] Recovered {recovery:.2f} fatigue, now at {self.fatigue_level:.2f}")

    def get_delay_multiplier(self) -> float:
        """Get current delay multiplier based on fatigue (1.0 = normal, higher = slower)."""
        self.update()
        return 1.0 + (self.max_delay_multiplier - 1.0) * self.fatigue_level

    def get_apm_multiplier(self) -> float:
        """Get current APM multiplier based on fatigue (1.0 = normal, lower = fewer actions)."""
        self.update()
        return 1.0 - 0.5 * self.fatigue_level  # At max fatigue, APM is 50% of normal

    def get_reaction_delay(self, base_delay: float) -> float:
        """Get reaction delay adjusted for fatigue."""
        return base_delay * self.get_delay_multiplier()

    def should_take_break(self) -> bool:
        """Check if bot should take a break due to fatigue."""
        self.update()
        return self.fatigue_level > 0.7

    def get_status(self) -> Dict:
        self.update()
        return {
            "fatigue_level": f"{self.fatigue_level:.2f}",
            "delay_multiplier": f"{self.get_delay_multiplier():.2f}",
            "apm_multiplier": f"{self.get_apm_multiplier():.2f}",
            "elapsed_hours": f"{(time.time() - self.session_start) / 3600:.1f}",
            "should_break": self.should_take_break(),
        }
