"""
core/error_recovery.py

Sistema avançado de tratamento de erros com recovery automático para o wrapper.
Implementa estratégias de recuperação granulares dependendo do tipo de erro.

Funcionalidades:
- Classificação de erros por tipo e severidade
- Estratégias de recovery específicas por tipo de erro
- Circuit breakers para evitar loops infinitos
- Graceful degradation
- Logging detalhado de erros
- Contadores de erros para detecção de problemas sistêmicos
- Recovery automático com fallback progressivo
"""

import threading
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tenacity-based circuit breakers
# ---------------------------------------------------------------------------
try:
    from tenacity import retry, retry_if_exception_type, retry_if_result, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:  # pragma: no cover
    HAS_TENACITY = False


def _is_false_or_none(val):
    return val is False or val is None


def _make_retry(min_wait: float, max_wait: float):
    if HAS_TENACITY:
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=min_wait, max=max_wait),
            retry=(retry_if_result(_is_false_or_none) | retry_if_exception_type(Exception)),
        )
    else:
        def _identity(func):
            return func
        return _identity


adb_retry = _make_retry(1, 10)
screenshot_retry = _make_retry(0.5, 5)
inference_retry = _make_retry(1, 10)


class ErrorType(Enum):
    """Tipos de erro que podem ocorrer."""
    SCREENSHOT_FAILURE = "screenshot_failure"
    STATE_DETECTION_FAILURE = "state_detection_failure"
    ADB_FAILURE = "adb_failure"
    YOLO_FAILURE = "yolo_failure"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severidade do erro."""
    LOW = "low"           # Recuperável, não impacta operação
    MEDIUM = "medium"     # Requer recovery, mas operação continua
    HIGH = "high"         # Requer recovery imediato
    CRITICAL = "critical" # Requer reinicialização do componente


class RecoveryStrategy(Enum):
    """Estratégias de recovery disponíveis."""
    RETRY = "retry"                    # Tentar novamente imediatamente
    RETRY_WITH_DELAY = "retry_delay"   # Tentar novamente com delay
    FALLBACK = "fallback"              # Usar método alternativo
    RESTART_COMPONENT = "restart"      # Reiniciar componente
    GRACEFUL_DEGRADE = "degrade"      # Reduzir funcionalidade
    SKIP = "skip"                     # Pular esta operação
    EMERGENCY_STOP = "emergency_stop"  # Parar tudo


@dataclass
class ErrorContext:
    """Contexto de um erro."""
    error_type: ErrorType
    severity: ErrorSeverity
    exception: Exception
    traceback_str: str
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    operation: str = ""
    additional_info: dict = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Ação de recovery a ser executada."""
    strategy: RecoveryStrategy
    description: str
    handler: Callable | None = None
    params: dict = field(default_factory=dict)
    max_attempts: int = 3
    delay_between_attempts: float = 1.0


@dataclass
class ErrorStats:
    """Estatísticas de erros."""
    total_errors: int = 0
    errors_by_type: dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    recovery_success: int = 0
    recovery_failure: int = 0
    last_error_time: float = 0.0
    error_rate_per_minute: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker para evitar chamadas a componentes com falhas recorrentes.

    Estados:
    - CLOSED: Operação normal
    - OPEN: Componente com falhas, não tentar
    - HALF_OPEN: Tentar uma vez para ver se recuperou
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts

        self.failures = 0
        self.last_failure_time = 0.0
        self.half_open_success_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def record_success(self):
        """Registra sucesso."""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.half_open_success_count += 1
                if self.half_open_success_count >= self.half_open_attempts:
                    self.state = "CLOSED"
                    self.failures = 0
                    logger.info("[CIRCUIT] Circuit breaker fechado (recuperado)")
            else:
                self.failures = max(0, self.failures - 1)

    def record_failure(self):
        """Registra falha."""
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                self.half_open_success_count = 0
                logger.warning(f"[CIRCUIT] Circuit breaker aberto ({self.failures} falhas)")

    def can_execute(self) -> bool:
        """Verifica se pode executar operação."""
        with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                # Verificar se passou timeout
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("[CIRCUIT] Circuit breaker em HALF_OPEN")
                    return True
                return False
            elif self.state == "HALF_OPEN":
                return True
            return False

    def get_state(self) -> str:
        """Retorna estado atual."""
        with self.lock:
            return self.state


class ErrorRecoverySystem:
    """
    Sistema de recovery de erros.

    Gerencia:
    - Classificação de erros
    - Seleção de estratégias de recovery
    - Circuit breakers por componente
    - Estatísticas de erros
    - Logging detalhado
    """

    def __init__(
        self,
        enable_auto_recovery: bool = True,
        max_recovery_attempts: int = 3,
        global_circuit_breaker: bool = True
    ):
        self.enable_auto_recovery = enable_auto_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.global_circuit_breaker = global_circuit_breaker

        # Circuit breakers por componente
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Estatísticas
        self.stats = ErrorStats()

        # Histórico de erros
        self.error_history: deque = deque(maxlen=100)

        # Handlers customizados de recovery
        self.recovery_handlers: dict[ErrorType, list[RecoveryAction]] = {}

        # Lock para thread safety
        self.lock = threading.Lock()

        # Configurar handlers padrão
        self._setup_default_handlers()

        logger.info("[ERROR_RECOVERY] Sistema inicializado")

    def _setup_default_handlers(self):
        """Configura handlers de recovery padrão."""

        # Screenshot failure
        self.recovery_handlers[ErrorType.SCREENSHOT_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                description="Retry screenshot after delay",
                max_attempts=2,
                delay_between_attempts=0.5
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Use alternative screenshot method",
                handler=self._fallback_screenshot_method
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                description="Restart screenshot taker",
                handler=self._restart_screenshot_taker
            )
        ]

        # ADB failure - auto-reconnect strategy
        self.recovery_handlers[ErrorType.ADB_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                description="Retry ADB command with exponential backoff",
                max_attempts=3,
                delay_between_attempts=2.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Attempt ADB reconnect",
                handler=self._reconnect_adb
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                description="Restart ADB connection",
                handler=self._restart_adb_connection
            )
        ]

        # State detection failure
        self.recovery_handlers[ErrorType.STATE_DETECTION_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry state detection",
                max_attempts=2
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Use fallback state detector",
                handler=self._fallback_state_detector
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                description="Use simple pixel detection only",
                handler=self._degrade_state_detection
            )
        ]

        # ADB failure
        self.recovery_handlers[ErrorType.ADB_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                description="Retry ADB command after delay",
                max_attempts=3,
                delay_between_attempts=1.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                description="Restart ADB connection",
                handler=self._restart_adb_connection
            )
        ]

        # YOLO failure
        self.recovery_handlers[ErrorType.YOLO_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry YOLO inference",
                max_attempts=2
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                description="Use heuristic detection without YOLO",
                handler=self._degrade_vision
            )
        ]

        # Memory error
        self.recovery_handlers[ErrorType.MEMORY_ERROR] = [
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                description="Reduce memory usage",
                handler=self._reduce_memory_usage
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                description="Skip current operation",
                max_attempts=1
            )
        ]

        # Network error
        self.recovery_handlers[ErrorType.NETWORK_ERROR] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                description="Retry after network delay",
                max_attempts=3,
                delay_between_attempts=2.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                description="Skip network-dependent operation"
            )
        ]

        # Timeout error
        self.recovery_handlers[ErrorType.TIMEOUT_ERROR] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry with longer timeout",
                max_attempts=2
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                description="Skip slow operation"
            )
        ]

        # Unknown error
        self.recovery_handlers[ErrorType.UNKNOWN_ERROR] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry unknown error",
                max_attempts=1
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                description="Skip to avoid cascading failures"
            )
        ]

    def classify_error(self, exception: Exception, component: str = "", operation: str = "") -> ErrorContext:
        """
        Classifica um erro e determina seu tipo e severidade.

        Args:
            exception: A exceção ocorrida
            component: Componente onde ocorreu o erro
            operation: Operação que estava sendo executada

        Returns:
            ErrorContext com classificação do erro
        """

        error_type = ErrorType.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM

        # Classificar por tipo de exceção
        exception_name = type(exception).__name__

        if exception_name in ["MemoryError", "OutOfMemoryError"]:
            error_type = ErrorType.MEMORY_ERROR
            severity = ErrorSeverity.HIGH
        elif exception_name in ["ConnectionError", "TimeoutError", "HTTPError"]:
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        elif exception_name in ["TimeoutException", "asyncio.TimeoutError"]:
            error_type = ErrorType.TIMEOUT_ERROR
            severity = ErrorSeverity.MEDIUM
        elif "screenshot" in str(exception).lower() or "capture" in str(exception).lower():
            error_type = ErrorType.SCREENSHOT_FAILURE
            severity = ErrorSeverity.HIGH
        elif "adb" in str(exception).lower() or "device" in str(exception).lower():
            error_type = ErrorType.ADB_FAILURE
            severity = ErrorSeverity.HIGH
        elif "yolo" in str(exception).lower() or "detection" in str(exception).lower():
            error_type = ErrorType.YOLO_FAILURE
            severity = ErrorSeverity.MEDIUM
        elif "state" in str(exception).lower():
            error_type = ErrorType.STATE_DETECTION_FAILURE
            severity = ErrorSeverity.MEDIUM

        # Criar contexto
        context = ErrorContext(
            error_type=error_type,
            severity=severity,
            exception=exception,
            traceback_str=traceback.format_exc(),
            component=component,
            operation=operation
        )

        logger.error(f"[ERROR_RECOVERY] Erro classificado: {error_type.value} ({severity.value}) - {component}.{operation}")

        return context

    def handle_error(
        self,
        context: ErrorContext,
        wrapper_instance: object | None = None
    ) -> bool:
        """
        Tenta recuperar de um erro automaticamente.

        Args:
            context: Contexto do erro
            wrapper_instance: Instância do wrapper para recovery

        Returns:
            True se recovery foi bem-sucedido, False caso contrário
        """

        if not self.enable_auto_recovery:
            logger.warning("[ERROR_RECOVERY] Auto-recovery desabilitado")
            return False

        # Atualizar estatísticas
        with self.lock:
            self.stats.total_errors += 1
            self.stats.errors_by_type[context.error_type] += 1
            self.stats.errors_by_severity[context.severity] += 1
            self.stats.last_error_time = time.time()
            self.error_history.append(context)

        # Verificar circuit breaker do componente
        if self.global_circuit_breaker and context.component:
            if context.component not in self.circuit_breakers:
                self.circuit_breakers[context.component] = CircuitBreaker()

            cb = self.circuit_breakers[context.component]
            if not cb.can_execute():
                logger.warning(f"[ERROR_RECOVERY] Circuit breaker aberto para {context.component}")
                return False

        # Obter ações de recovery
        actions = self.recovery_handlers.get(context.error_type, [])

        if not actions:
            logger.warning(f"[ERROR_RECOVERY] Nenhuma ação de recovery para {context.error_type.value}")
            return False

        # Tentar cada ação de recovery
        for action in actions:
            logger.info(f"[ERROR_RECOVERY] Tentando recovery: {action.description} ({action.strategy.value})")

            try:
                # Executar handler se existir
                if action.handler:
                    success = action.handler(wrapper_instance, context, action.params)
                else:
                    # Handler padrão
                    success = self._default_recovery_handler(action, wrapper_instance)

                if success:
                    # Registrar sucesso no circuit breaker
                    if context.component and context.component in self.circuit_breakers:
                        self.circuit_breakers[context.component].record_success()

                    with self.lock:
                        self.stats.recovery_success += 1

                    logger.info(f"[ERROR_RECOVERY] Recovery bem-sucedido: {action.description}")
                    return True
                else:
                    logger.debug(f"[ERROR_RECOVERY] Recovery falhou: {action.description}")

            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.error(f"[ERROR_RECOVERY] Erro ao executar recovery: {e}")

        # Todas as tentativas falharam
        if context.component and context.component in self.circuit_breakers:
            self.circuit_breakers[context.component].record_failure()

        with self.lock:
            self.stats.recovery_failure += 1

        logger.error(f"[ERROR_RECOVERY] Todas as tentativas de recovery falharam para {context.error_type.value}")
        return False

    def _default_recovery_handler(
        self,
        action: RecoveryAction,
        wrapper_instance: object | None
    ) -> bool:
        """Handler padrão para ações sem handler específico."""

        if action.strategy == RecoveryStrategy.RETRY:
            # Simplesmente retorna True para tentar novamente
            return True

        elif action.strategy == RecoveryStrategy.RETRY_WITH_DELAY:
            time.sleep(action.delay_between_attempts)
            return True

        elif action.strategy == RecoveryStrategy.SKIP:
            return True  # Skip é sempre "bem-sucedido"

        return False

    # Handlers específicos de recovery

    def _fallback_screenshot_method(self, wrapper, context, params) -> bool:
        """Usa método alternativo de captura de tela."""
        if wrapper and hasattr(wrapper, 'screenshot'):
            try:
                # Tentar método alternativo (ex: ADB screencap em vez de Win32)
                if hasattr(wrapper.screenshot, 'capture_adb'):
                    wrapper.screenshot.capture_adb()
                    logger.info("[ERROR_RECOVERY] Usando ADB screencap como fallback")
                    return True
            except (ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[ERROR_RECOVERY] Fallback screenshot falhou: {e}")
        return False

    def _restart_screenshot_taker(self, wrapper, context, params) -> bool:
        """Reinicia o screenshot taker."""
        if wrapper and hasattr(wrapper, 'screenshot'):
            try:
                # Reinicializar screenshot taker
                from pylaai_real.screenshot_taker import ScreenshotTaker
                wrapper.screenshot = ScreenshotTaker()
                logger.info("[ERROR_RECOVERY] Screenshot taker reiniciado")
                return True
            except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.error(f"[ERROR_RECOVERY] Falha ao reiniciar screenshot taker: {e}")
        return False

    def _fallback_state_detector(self, wrapper, context, params) -> bool:
        """Usa detector de estado alternativo."""
        if wrapper and hasattr(wrapper, 'state_manager'):
            try:
                # Tentar usar detector legado se unified falhou
                if hasattr(wrapper.state_manager, 'state_finder'):
                    logger.info("[ERROR_RECOVERY] Usando StateFinder como fallback")
                    return True
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.error(f"[ERROR_RECOVERY] Fallback state detector falhou: {e}")
        return False

    def _degrade_state_detection(self, wrapper, context, params) -> bool:
        """Degrada detecção de estado para método mais simples."""
        if wrapper and hasattr(wrapper, 'state_manager'):
            try:
                # Usar apenas detecção por pixel
                logger.info("[ERROR_RECOVERY] Degrading para detecção por pixel apenas")
                return True
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.error(f"[ERROR_RECOVERY] Degradation falhou: {e}")
        return False

    def _restart_adb_connection(self, wrapper, context, params) -> bool:
        """Reinicia conexão ADB."""
        if wrapper and hasattr(wrapper, 'emulator_controller'):
            try:
                wrapper.emulator_controller.reconnect()
                logger.info("[ERROR_RECOVERY] Conexão ADB reiniciada")
                return True
            except (ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[ERROR_RECOVERY] Falha ao reiniciar ADB: {e}")
        return False

    def _reconnect_adb(self, wrapper, context, params) -> bool:
        """Attempt ADB reconnect with exponential backoff and ResilientADB support.

        Tries up to 3 reconnection attempts with increasing delays (2s, 4s, 8s).
        Resets circuit breaker on ResilientADB if available, then forces a
        health check to verify the connection is alive.
        """
        max_attempts = params.get("max_attempts", 3) if params else 3
        base_delay = params.get("base_delay", 2.0) if params else 2.0

        for attempt in range(1, max_attempts + 1):
            try:
                if wrapper and hasattr(wrapper, 'emulator_controller'):
                    ec = wrapper.emulator_controller

                    # Reset ResilientADB circuit breaker if available
                    if hasattr(ec, 'adb') and hasattr(ec.adb, '_resilient_adb') and ec.adb._resilient_adb:
                        ec.adb._resilient_adb._close_circuit()
                        ec.adb._resilient_adb.state.last_health_ok = False
                        logger.info(
                            f"[ERROR_RECOVERY] ResilientADB circuit breaker reset (attempt {attempt}/{max_attempts})"
                        )

                    # Attempt reconnect
                    connected = ec.connect()
                    if connected:
                        # Verify with a health check ping
                        if hasattr(ec, 'adb') and hasattr(ec.adb, 'ping'):
                            try:
                                ec.adb.ping()
                            except (ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                                logger.warning("[ERROR_RECOVERY] ADB connected but ping failed, retrying...")
                                if attempt < max_attempts:
                                    delay = base_delay * (2 ** (attempt - 1))
                                    logger.info(f"[ERROR_RECOVERY] Retrying in {delay:.1f}s...")
                                    time.sleep(delay)
                                    continue

                        logger.info(f"[ERROR_RECOVERY] ADB reconnect succeeded on attempt {attempt}")
                        return True
                    else:
                        logger.warning(
                            f"[ERROR_RECOVERY] ADB reconnect attempt {attempt}/{max_attempts} failed"
                        )
                else:
                    logger.warning("[ERROR_RECOVERY] No emulator_controller available for reconnect")
                    return False

            except (ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[ERROR_RECOVERY] ADB reconnect attempt {attempt} exception: {e}")

            # Exponential backoff before next attempt
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"[ERROR_RECOVERY] Retrying ADB reconnect in {delay:.1f}s...")
                time.sleep(delay)

        logger.error(f"[ERROR_RECOVERY] ADB reconnect failed after {max_attempts} attempts")
        return False

    def _degrade_vision(self, wrapper, context, params) -> bool:
        """Degrada sistema de visão para heurísticas."""
        try:
            logger.info("[ERROR_RECOVERY] Degrading vision system para heurísticas")
            return True
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error(f"[ERROR_RECOVERY] Vision degradation falhou: {e}")
        return False

    def _reduce_memory_usage(self, wrapper, context, params) -> bool:
        """Reduz uso de memória."""
        try:
            import gc
            gc.collect()
            logger.info("[ERROR_RECOVERY] Garbage collection executada")
            return True
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.error(f"[ERROR_RECOVERY] Memory reduction falhou: {e}")
        return False

    def get_stats(self) -> dict:
        """Retorna estatísticas de erros."""
        with self.lock:
            return {
                "total_errors": self.stats.total_errors,
                "errors_by_type": {k.value: v for k, v in self.stats.errors_by_type.items()},
                "errors_by_severity": {k.value: v for k, v in self.stats.errors_by_severity.items()},
                "recovery_success": self.stats.recovery_success,
                "recovery_failure": self.stats.recovery_failure,
                "recovery_rate": self.stats.recovery_success / max(1, self.stats.total_errors),
                "last_error_time": self.stats.last_error_time,
                "circuit_breakers": {
                    comp: cb.get_state()
                    for comp, cb in self.circuit_breakers.items()
                }
            }


# Decorator para tratamento automático de erros
def with_error_recovery(
    error_recovery: ErrorRecoverySystem,
    component: str = "",
    operation: str = ""
):
    """
    Decorator para tratamento automático de erros.

    Args:
        error_recovery: Instância do ErrorRecoverySystem
        component: Nome do componente
        operation: Nome da operação
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Tentar executar função
            try:
                return func(*args, **kwargs)

            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                # Classificar erro
                context = error_recovery.classify_error(e, component, operation)

                # Tentar recovery
                recovered = error_recovery.handle_error(context, args[0] if args else None)

                if recovered:
                    # Tentar novamente após recovery
                    try:
                        return func(*args, **kwargs)
                    except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                        # Se falhar novamente, propagar erro
                        logger.error(f"[ERROR_RECOVERY] Função falhou mesmo após recovery: {e}")
                        raise

                # Se não recuperou, propagar erro
                raise

        return wrapper
    return decorator


# Classe de conveniência para integrar com wrapper
class ErrorRecoveryIntegration:
    """Integração do sistema de recovery com o wrapper."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.error_recovery = ErrorRecoverySystem()
        self.enabled = True

    def enable(self):
        """Habilita sistema de recovery."""
        self.enabled = True
        logger.info("[ERROR_RECOVERY] Sistema habilitado")

    def disable(self):
        """Desabilita sistema de recovery."""
        self.enabled = False
        logger.info("[ERROR_RECOVERY] Sistema desabilitado")

    def wrap_method(self, method_name: str, component: str = "", operation: str = ""):
        """Envolve um método do wrapper com tratamento de erro."""

        if not hasattr(self.wrapper, method_name):
            logger.warning(f"[ERROR_RECOVERY] Método {method_name} não encontrado")
            return

        original_method = getattr(self.wrapper, method_name)

        @with_error_recovery(self.error_recovery, component, operation or method_name)
        def wrapped_method(*args, **kwargs):
            return original_method(*args, **kwargs)

        setattr(self.wrapper, method_name, wrapped_method)
        logger.info(f"[ERROR_RECOVERY] Método {method_name} envolvido com tratamento de erro")

    def wrap_main_loop(self):
        """Envolve o loop principal do wrapper com tratamento de erro."""
        self.wrap_method("_main_loop", "wrapper", "main_loop")

    def wrap_screenshot(self):
        """Envolve captura de screenshot com tratamento de erro."""
        if hasattr(self.wrapper, 'screenshot'):
            original_capture = self.wrapper.screenshot.capture

            @with_error_recovery(self.error_recovery, "screenshot", "capture")
            def wrapped_capture(*args, **kwargs):
                return original_capture(*args, **kwargs)

            self.wrapper.screenshot.capture = wrapped_capture
            logger.info("[ERROR_RECOVERY] Screenshot.capture envolvido com tratamento de erro")

    def get_recovery_stats(self) -> dict:
        """Retorna estatísticas de recovery."""
        return self.error_recovery.get_stats()
