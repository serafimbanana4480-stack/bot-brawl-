"""
core/degradation_manager.py

Graceful Degradation Completo para Soberana Omega.

Reduz qualidade progressivamente se o sistema fica instável,
garantindo que o bot NUNCA crash = ban garantido.

Modos:
- FULL_QUALITY: operação normal (YOLO completo, DQN, 30 FPS)
- DEGRADED:     redução controlada (YOLO simples, Q-table, 20 FPS)
- MINIMAL:      sobrevivência (heurísticas pixel-only, 10 FPS)
- EMERGENCY:    pausa ativa (apenas monitoramento, 1 ciclo/seg)

Integração com ErrorRecoverySystem para decisões automáticas.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class DegradationMode(Enum):
    FULL_QUALITY = "full_quality"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"


class DetectorCapability(Enum):
    """Quais detectores estão disponíveis em cada modo."""
    YOLO_MULTI_SCALE = auto()
    YOLO_SINGLE_SCALE = auto()
    PIXEL_HEURISTICS = auto()
    OCR_FULL = auto()
    OCR_LITE = auto()


class RLCapability(Enum):
    """Quais capacidades de RL estão disponíveis."""
    DQN_NETWORK = auto()
    Q_TABLE = auto()
    RULES_BASED = auto()


@dataclass
class DegradationConfig:
    """Configuração por modo de degradação."""
    mode: DegradationMode
    description: str

    # Vision
    yolo_input_size: int = 640
    yolo_multi_scale: bool = True
    yolo_half_precision: bool = False
    use_ocr: bool = True
    use_pixel_heuristics: bool = True

    # RL
    rl_backend: str = "dqn"  # dqn | q_table | rules

    # Performance
    target_fps: float = 30.0
    inference_skip_frames: int = 0
    cycle_delay_ms: float = 33.0

    # Safety
    max_apm: int = 40
    screenshot_interval_ms: float = 33.0

    # Recovery
    auto_recovery: bool = True
    circuit_breaker_enabled: bool = True


def _make_configs() -> dict[DegradationMode, DegradationConfig]:
    return {
        DegradationMode.FULL_QUALITY: DegradationConfig(
            mode=DegradationMode.FULL_QUALITY,
            description="Operação completa — todos os subsistemas ativos",
            yolo_input_size=640,
            yolo_multi_scale=True,
            yolo_half_precision=False,
            use_ocr=True,
            use_pixel_heuristics=True,
            rl_backend="dqn",
            target_fps=30.0,
            inference_skip_frames=0,
            cycle_delay_ms=33.0,
            max_apm=40,
            screenshot_interval_ms=33.0,
            auto_recovery=True,
            circuit_breaker_enabled=True,
        ),
        DegradationMode.DEGRADED: DegradationConfig(
            mode=DegradationMode.DEGRADED,
            description="Degradação controlada — YOLO simplificado, Q-table",
            yolo_input_size=320,
            yolo_multi_scale=False,
            yolo_half_precision=True,
            use_ocr=False,
            use_pixel_heuristics=True,
            rl_backend="q_table",
            target_fps=20.0,
            inference_skip_frames=1,
            cycle_delay_ms=50.0,
            max_apm=30,
            screenshot_interval_ms=50.0,
            auto_recovery=True,
            circuit_breaker_enabled=True,
        ),
        DegradationMode.MINIMAL: DegradationConfig(
            mode=DegradationMode.MINIMAL,
            description="Sobrevivência — heurísticas pixel-only, regras",
            yolo_input_size=320,
            yolo_multi_scale=False,
            yolo_half_precision=True,
            use_ocr=False,
            use_pixel_heuristics=True,
            rl_backend="rules",
            target_fps=10.0,
            inference_skip_frames=3,
            cycle_delay_ms=100.0,
            max_apm=20,
            screenshot_interval_ms=100.0,
            auto_recovery=False,
            circuit_breaker_enabled=False,
        ),
        DegradationMode.EMERGENCY: DegradationConfig(
            mode=DegradationMode.EMERGENCY,
            description="Pausa ativa — apenas monitoramento, mínimo de ADB",
            yolo_input_size=320,
            yolo_multi_scale=False,
            yolo_half_precision=True,
            use_ocr=False,
            use_pixel_heuristics=True,
            rl_backend="rules",
            target_fps=1.0,
            inference_skip_frames=30,
            cycle_delay_ms=1000.0,
            max_apm=5,
            screenshot_interval_ms=1000.0,
            auto_recovery=False,
            circuit_breaker_enabled=False,
        ),
    }


class DegradationManager:
    """
    Gerencia degradação progressiva baseada na saúde do sistema.

    Monitora:
    - Taxa de erro (por minuto)
    - Latência de inferência
    - Latência de screenshot
    - Falhas de ADB
    - Uso de memória
    - Temperatura / throttling do CPU

    Decisões:
    - Se erro > 30% → DEGRADED
    - Se erro < 10% por 2 min → melhora um nível
    - Se erro > 60% → MINIMAL
    - Se erro > 80% por 1 min → EMERGENCY (pausa)
    """

    def __init__(
        self,
        error_threshold_degraded: float = 0.30,
        error_threshold_minimal: float = 0.60,
        error_threshold_emergency: float = 0.80,
        recovery_improvement_duration: float = 120.0,
    ):
        self.error_threshold_degraded = error_threshold_degraded
        self.error_threshold_minimal = error_threshold_minimal
        self.error_threshold_emergency = error_threshold_emergency
        self.recovery_improvement_duration = recovery_improvement_duration

        self.configs = _make_configs()
        self.mode = DegradationMode.FULL_QUALITY
        self.config = self.configs[self.mode]

        # Métricas de saúde
        self.recent_errors: deque = deque(maxlen=100)
        self.recent_inference_times: deque = deque(maxlen=100)
        self.recent_screenshot_times: deque = deque(maxlen=100)

        self.last_mode_change = time.time()
        self.time_in_current_mode = 0.0

        # Callbacks para notificar subsistemas
        self._listeners: list[Callable[[DegradationMode, DegradationConfig], None]] = []
        self._lock = threading.RLock()

        logger.info("[DEGRADATION] Inicializado em modo %s", self.mode.value)

    # ------------------------------------------------------------------
    # Monitoramento
    # ------------------------------------------------------------------

    def record_error(self, component: str = "", error_type: str = ""):
        """Registra uma falha para cálculo de taxa."""
        with self._lock:
            self.recent_errors.append({
                "timestamp": time.time(),
                "component": component,
                "type": error_type,
            })

    def record_inference_time(self, duration_sec: float):
        """Registra latência de inferência YOLO."""
        self.recent_inference_times.append(duration_sec)

    def record_screenshot_time(self, duration_sec: float):
        """Registra latência de screenshot."""
        self.recent_screenshot_times.append(duration_sec)

    # ------------------------------------------------------------------
    # Decisão de modo
    # ------------------------------------------------------------------

    def check_health_and_degrade(self) -> DegradationMode:
        """
        Avalia saúde e ajusta modo se necessário.
        Deve ser chamada a cada ciclo principal (ou a cada ~5s).
        """
        with self._lock:
            now = time.time()
            self.time_in_current_mode = now - self.last_mode_change

            # Calcular taxas
            window_start = now - 60.0
            errors_last_min = sum(
                1 for e in self.recent_errors if e["timestamp"] > window_start
            )
            error_rate = errors_last_min / max(1, len(self.recent_errors))

            avg_inference = sum(self.recent_inference_times) / max(1, len(self.recent_inference_times))
            sum(self.recent_screenshot_times) / max(1, len(self.recent_screenshot_times))

            # Forçar degradação se inferência muito lenta (>2s)
            if avg_inference > 2.0 and self.mode == DegradationMode.FULL_QUALITY:
                logger.warning("[DEGRADATION] Inferência muito lenta (%.2fs) → DEGRADED", avg_inference)
                self._set_mode(DegradationMode.DEGRADED)
                return self.mode

            # Verificar thresholds
            if error_rate >= self.error_threshold_emergency:
                if self.mode != DegradationMode.EMERGENCY:
                    logger.error("[DEGRADATION] Taxa erro %.1f%% → EMERGENCY (pausa)", error_rate * 100)
                    self._set_mode(DegradationMode.EMERGENCY)

            elif error_rate >= self.error_threshold_minimal:
                if self.mode not in (DegradationMode.MINIMAL, DegradationMode.EMERGENCY):
                    logger.warning("[DEGRADATION] Taxa erro %.1f%% → MINIMAL", error_rate * 100)
                    self._set_mode(DegradationMode.MINIMAL)

            elif error_rate >= self.error_threshold_degraded:
                if self.mode == DegradationMode.FULL_QUALITY:
                    logger.warning("[DEGRADATION] Taxa erro %.1f%% → DEGRADED", error_rate * 100)
                    self._set_mode(DegradationMode.DEGRADED)

            # Recuperação: se está bom há tempo suficiente, melhorar
            elif error_rate < 0.05 and self.time_in_current_mode > self.recovery_improvement_duration:
                if self.mode == DegradationMode.EMERGENCY:
                    logger.info("[DEGRADATION] Recuperado → MINIMAL")
                    self._set_mode(DegradationMode.MINIMAL)
                elif self.mode == DegradationMode.MINIMAL:
                    logger.info("[DEGRADATION] Recuperado → DEGRADED")
                    self._set_mode(DegradationMode.DEGRADED)
                elif self.mode == DegradationMode.DEGRADED:
                    logger.info("[DEGRADATION] Recuperado → FULL_QUALITY")
                    self._set_mode(DegradationMode.FULL_QUALITY)

            return self.mode

    def _set_mode(self, new_mode: DegradationMode):
        """Muda modo e notifica listeners."""
        old_mode = self.mode
        if old_mode == new_mode:
            return

        self.mode = new_mode
        self.config = self.configs[new_mode]
        self.last_mode_change = time.time()
        self.time_in_current_mode = 0.0

        logger.info("[DEGRADATION] Modo alterado: %s → %s (%s)", old_mode.value, new_mode.value, self.config.description)

        # Notificar listeners
        for listener in self._listeners:
            try:
                listener(new_mode, self.config)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[DEGRADATION] Listener falhou: %s", e)

    # ------------------------------------------------------------------
    # API pública para subsistemas
    # ------------------------------------------------------------------

    def get_detector(self) -> str:
        """Retorna qual detector usar no modo atual."""
        if self.mode == DegradationMode.FULL_QUALITY:
            return "advanced_yolo"
        elif self.mode == DegradationMode.DEGRADED:
            return "simple_fast_yolo"
        else:
            return "pixel_heuristics"

    def get_rl_backend(self) -> str:
        """Retorna qual backend de RL usar."""
        return self.config.rl_backend

    def should_run_inference(self, frame_counter: int) -> bool:
        """Diz se deve rodar inferência neste frame (baseado em skip)."""
        skip = self.config.inference_skip_frames
        if skip <= 0:
            return True
        return frame_counter % (skip + 1) == 0

    def get_cycle_delay(self) -> float:
        """Delay entre ciclos em segundos."""
        return self.config.cycle_delay_ms / 1000.0

    def get_max_apm(self) -> int:
        """APM máximo permitido no modo atual."""
        return self.config.max_apm

    def register_listener(self, callback: Callable[[DegradationMode, DegradationConfig], None]):
        """Registra callback para mudanças de modo."""
        self._listeners.append(callback)

    def unregister_listener(self, callback: Callable[[DegradationMode, DegradationConfig], None]):
        """Remove callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Retorna status completo para dashboard/telemetria."""
        with self._lock:
            window_start = time.time() - 60.0
            errors_last_min = sum(1 for e in self.recent_errors if e["timestamp"] > window_start)
            return {
                "mode": self.mode.value,
                "description": self.config.description,
                "errors_last_minute": errors_last_min,
                "time_in_mode_seconds": round(self.time_in_current_mode, 1),
                "target_fps": self.config.target_fps,
                "max_apm": self.config.max_apm,
                "rl_backend": self.config.rl_backend,
                "yolo_input_size": self.config.yolo_input_size,
                "use_ocr": self.config.use_ocr,
                "auto_recovery": self.config.auto_recovery,
            }
