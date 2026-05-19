"""
test_error_recovery.py

Testes para o sistema de Error Recovery e integracao com o wrapper.
Simula cenarios de erro para verificar que o recovery funciona.

Executar: py tests/test_error_recovery.py
"""

import sys
import os
import time
import importlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Adicionar projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)

# --- Direct imports to avoid __init__.py chain import issues ---

def _import_module_direct(module_name, file_path):
    """Import a module directly from file path, bypassing __init__.py chain."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_error_recovery_module():
    """Import error_recovery directly, avoiding core/__init__.py."""
    return _import_module_direct(
        "core.error_recovery",
        project_root / "core" / "error_recovery.py"
    )


def _get_state_recovery_module():
    """Import state_recovery directly."""
    return _import_module_direct(
        "pylaai_real.state_recovery",
        project_root / "pylaai_real" / "state_recovery.py"
    )


def _get_auto_calibrator_module():
    """Import auto_calibrator directly."""
    return _import_module_direct(
        "pylaai_real.auto_calibrator",
        project_root / "pylaai_real" / "auto_calibrator.py"
    )


def _get_ocr_detector_module():
    """Import ocr_state_detector directly."""
    return _import_module_direct(
        "pylaai_real.ocr_state_detector",
        project_root / "pylaai_real" / "ocr_state_detector.py"
    )


def _get_unified_detector_module():
    """Import unified_state_detector directly."""
    return _import_module_direct(
        "pylaai_real.unified_state_detector",
        project_root / "pylaai_real" / "unified_state_detector.py"
    )


def _get_debug_visualizer_module():
    """Import debug_visualizer directly."""
    return _import_module_direct(
        "pylaai_real.debug_visualizer",
        project_root / "pylaai_real" / "debug_visualizer.py"
    )


# --- Pre-load modules to avoid chain import ---
_er_mod = None
_sr_mod = None
_ac_mod = None
_ocr_mod = None
_dv_mod = None


def _ensure_error_recovery():
    global _er_mod
    if _er_mod is None:
        _er_mod = _get_error_recovery_module()
    return _er_mod


def _ensure_state_recovery():
    global _sr_mod
    if _sr_mod is None:
        _sr_mod = _get_state_recovery_module()
    return _sr_mod


def _ensure_auto_calibrator():
    global _ac_mod
    if _ac_mod is None:
        _ac_mod = _get_auto_calibrator_module()
    return _ac_mod


def _ensure_ocr_detector():
    global _ocr_mod
    if _ocr_mod is None:
        _ocr_mod = _get_ocr_detector_module()
    return _ocr_mod


def _ensure_debug_visualizer():
    global _dv_mod
    if _dv_mod is None:
        _dv_mod = _get_debug_visualizer_module()
    return _dv_mod


# ============================================================
# Test classes
# ============================================================


class TestErrorRecoverySystem:
    """Testes para o ErrorRecoverySystem."""

    def test_classify_screenshot_error(self):
        """Testa classificacao de erro de screenshot."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem()
        exception = RuntimeError("Failed to capture screenshot")
        context = system.classify_error(exception, "screenshot", "capture")

        assert context.error_type == mod.ErrorType.SCREENSHOT_FAILURE
        assert context.severity == mod.ErrorSeverity.HIGH
        assert context.component == "screenshot"
        assert context.operation == "capture"

    def test_classify_adb_error(self):
        """Testa classificacao de erro ADB."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem()
        exception = ConnectionError("ADB device not found")
        context = system.classify_error(exception, "emulator_controller", "connect")

        assert context.error_type in (mod.ErrorType.ADB_FAILURE, mod.ErrorType.NETWORK_ERROR)

    def test_classify_memory_error(self):
        """Testa classificacao de erro de memoria."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem()
        exception = MemoryError("Out of memory")
        context = system.classify_error(exception, "wrapper", "main_loop")

        assert context.error_type == mod.ErrorType.MEMORY_ERROR
        assert context.severity == mod.ErrorSeverity.HIGH

    def test_classify_unknown_error(self):
        """Testa classificacao de erro desconhecido."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem()
        exception = ValueError("Some random error")
        context = system.classify_error(exception, "unknown_component", "unknown_op")

        assert context.error_type == mod.ErrorType.UNKNOWN_ERROR

    def test_handle_error_with_retry(self):
        """Testa recovery com retry."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem(enable_auto_recovery=True)
        exception = RuntimeError("Temporary failure")

        context = system.classify_error(exception, "test", "retry_test")
        # Forcar tipo de erro que tem retry
        context.error_type = mod.ErrorType.TIMEOUT_ERROR

        recovered = system.handle_error(context, None)
        assert isinstance(recovered, bool)

    def test_handle_error_disabled(self):
        """Testa que recovery nao executa quando desabilitado."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem(enable_auto_recovery=False)
        exception = RuntimeError("Error")

        context = system.classify_error(exception, "test", "disabled_test")
        recovered = system.handle_error(context, None)

        assert recovered is False

    def test_circuit_breaker_basic(self):
        """Testa circuit breaker basico."""
        mod = _ensure_error_recovery()
        cb = mod.CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        # Inicialmente fechado (pode executar)
        assert cb.can_execute() is True
        assert cb.get_state() == "CLOSED"

        # Registrar falhas
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True  # Ainda abaixo do threshold

        cb.record_failure()  # 3 falhas = abrir
        assert cb.can_execute() is False
        assert cb.get_state() == "OPEN"

        # Apos timeout, deve ir para HALF_OPEN
        time.sleep(1.1)
        assert cb.can_execute() is True
        assert cb.get_state() == "HALF_OPEN"

        # Sucesso deve fechar
        cb.record_success()
        cb.record_success()
        assert cb.get_state() == "CLOSED"

    def test_circuit_breaker_reopens_on_failure(self):
        """Testa que circuit breaker reabre apos falha em HALF_OPEN."""
        mod = _ensure_error_recovery()
        cb = mod.CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)

        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == "OPEN"

        time.sleep(0.6)
        assert cb.can_execute() is True  # HALF_OPEN

        cb.record_failure()  # Falha em HALF_OPEN -> OPEN
        assert cb.get_state() == "OPEN"

    def test_error_stats(self):
        """Testa estatisticas de erros."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem(enable_auto_recovery=True)

        for _ in range(3):
            exception = RuntimeError("Test error")
            context = system.classify_error(exception, "test", "stats_test")
            system.handle_error(context, None)

        stats = system.get_stats()
        assert stats["total_errors"] == 3
        assert "errors_by_type" in stats
        assert "errors_by_severity" in stats

    def test_recovery_handler_memory(self):
        """Testa handler de recovery para erro de memoria."""
        mod = _ensure_error_recovery()

        system = mod.ErrorRecoverySystem(enable_auto_recovery=True)

        context = mod.ErrorContext(
            error_type=mod.ErrorType.MEMORY_ERROR,
            severity=mod.ErrorSeverity.HIGH,
            exception=MemoryError("OOM"),
            traceback_str="",
            component="wrapper",
            operation="main_loop"
        )

        recovered = system.handle_error(context, None)
        assert isinstance(recovered, bool)


class TestStateRecoverySystem:
    """Testes para o StateRecoverySystem."""

    def test_detect_loop(self):
        """Testa deteccao de loop de estado."""
        mod = _ensure_state_recovery()
        history = mod.StateHistory()
        for _ in range(5):
            history.add("lobby")

        assert history.detect_loop() is True

    def test_detect_no_loop(self):
        """Testa que estados variados nao sao detectados como loop."""
        mod = _ensure_state_recovery()
        history = mod.StateHistory()
        for state in ["lobby", "loading", "in_game", "lobby", "loading"]:
            history.add(state)

        assert history.detect_loop() is False

    def test_detect_oscillation(self):
        """Testa deteccao de oscilacao entre 2 estados."""
        mod = _ensure_state_recovery()
        history = mod.StateHistory()
        for _ in range(5):
            history.add("lobby")
            history.add("loading")

        assert history.detect_oscillation() is True

    def test_needs_recovery_unknown_state(self):
        """Testa que estado unknown por muito tempo precisa de recovery."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(
            emulator_controller=None,
            max_unknown_duration=0.1
        )

        # Set state to unknown and inject old timestamps into history
        # time_in_current_state() requires len(timestamps) >= 2
        recovery.current_state = "unknown"
        recovery.state_start_time = time.time() - 0.2
        recovery.history.states.append("unknown")
        recovery.history.timestamps.append(time.time() - 0.3)
        recovery.history.states.append("unknown")
        recovery.history.timestamps.append(time.time() - 0.2)

        assert recovery._needs_recovery("unknown", 0.0) is True

    def test_recovery_strategies_exist(self):
        """Testa que estrategias de recovery estao definidas."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(emulator_controller=None)

        assert "unknown" in recovery.recovery_strategies
        assert "stuck" in recovery.recovery_strategies
        assert "loop" in recovery.recovery_strategies
        assert "popup" in recovery.recovery_strategies

    def test_get_recovery_status(self):
        """Testa status de recovery."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(emulator_controller=None)
        status = recovery.get_recovery_status()

        assert "is_recovering" in status
        assert "recovery_attempts" in status
        assert "current_state" in status

    def test_cancel_recovery(self):
        """Testa cancelamento de recovery."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(emulator_controller=None)
        recovery.active_recovery = [1, 2, 3]
        recovery.cancel_recovery()
        assert recovery.active_recovery is None


class TestAutoCalibrator:
    """Testes para o AutoCalibrator."""

    def test_init_without_templates(self):
        """Testa inicializacao sem templates."""
        mod = _ensure_auto_calibrator()
        calibrator = mod.AutoCalibrator(
            templates_dir=Path("nonexistent_dir"),
            enable_cache=False
        )
        assert len(calibrator.templates) == 0

    def test_detect_element_fallback(self):
        """Testa deteccao com fallback para coordenadas fixas."""
        mod = _ensure_auto_calibrator()
        import numpy as np

        calibrator = mod.AutoCalibrator(
            templates_dir=Path("nonexistent_dir"),
            enable_cache=False
        )

        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result = calibrator.detect_element(
            screenshot, "play_button",
            fallback_coords=(1800, 970)
        )

        assert result is not None
        assert result.x == 1800
        assert result.y == 970
        assert result.method == "fallback"

    def test_invalidate_cache(self):
        """Testa invalidacao de cache."""
        mod = _ensure_auto_calibrator()
        calibrator = mod.AutoCalibrator(enable_cache=False)

        calibrator.coords_cache["test"] = mod.CalibratedCoords(
            element_name="test", x=100, y=200,
            confidence=0.9, method="template"
        )

        calibrator.invalidate_cache("test")
        assert "test" not in calibrator.coords_cache

    def test_get_all_cached_coords(self):
        """Testa obtencao de todas as coordenadas cacheadas."""
        mod = _ensure_auto_calibrator()
        calibrator = mod.AutoCalibrator(enable_cache=False)
        calibrator.coords_cache["elem1"] = mod.CalibratedCoords(
            element_name="elem1", x=100, y=200,
            confidence=0.9, method="template"
        )

        coords = calibrator.get_all_cached_coords()
        assert "elem1" in coords


class TestOCRStateDetector:
    """Testes para o OCRStateDetector."""

    def test_init_without_easyocr(self):
        """Testa inicializacao sem EasyOCR instalado."""
        mod = _ensure_ocr_detector()
        detector = mod.OCRStateDetector()
        assert detector._reader is None  # Lazy loading

    def test_text_patterns_defined(self):
        """Testa que padroes de texto estao definidos."""
        mod = _ensure_ocr_detector()
        assert mod.OCRElement.PLAY_BUTTON in mod.OCRStateDetector().text_patterns
        assert mod.OCRElement.VICTORY in mod.OCRStateDetector().text_patterns
        assert mod.OCRElement.DEFEAT in mod.OCRStateDetector().text_patterns

    def test_get_detection_stats(self):
        """Testa estatisticas de deteccao."""
        mod = _ensure_ocr_detector()
        detector = mod.OCRStateDetector()
        stats = detector.get_detection_stats()

        assert "reader_available" in stats
        assert "confidence_threshold" in stats

    def test_parse_structured_hud_text(self):
        """Testa parsing estruturado de timer, score e estado de habilidade."""
        mod = _ensure_ocr_detector()
        detector = mod.OCRStateDetector()

        assert detector.parse_timer_text("1:23") == 83.0
        assert detector.parse_timer_text("O:4S") == 45.0
        assert detector.parse_score_text("2-1") == (2, 1)
        assert detector.parse_score_text("8 / 6") == (8, 6)
        assert detector.parse_ability_state("READY") is True
        assert detector.parse_ability_state("COOLDOWN") is False

    def test_extract_hud_text_best_effort(self):
        """Testa extração HUD com fake reader e múltiplas ROIs."""
        mod = _ensure_ocr_detector()
        detector = mod.OCRStateDetector()

        class _FakeReader:
            def readtext(self, image):
                return [
                    ([(0, 0), (1, 0), (1, 1), (0, 1)], "1:23", 0.91),
                    ([(0, 0), (1, 0), (1, 1), (0, 1)], "2-1", 0.88),
                    ([(0, 0), (1, 0), (1, 1), (0, 1)], "READY", 0.93),
                ]

        detector._reader = _FakeReader()

        import numpy as np

        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        hud = detector.extract_hud_text(screenshot)

        assert hud["match_timer_text"] == "1:23"
        assert hud["match_time_remaining"] == 83.0
        assert hud["score_text"] == "2-1"
        assert hud["match_score"] == (2, 1)
        assert hud["ability_texts"]
        assert hud["ability_states"]


class TestUnifiedStateDetectorOCRFallback:
    """Testes para fallback OCR no UnifiedStateDetector."""

    def test_ocr_fallback_when_pixel_and_template_fail(self):
        """Quando pixel/template não detectam nada, OCR deve resolver o estado."""
        u_mod = _get_unified_detector_module()
        o_mod = _get_ocr_detector_module()

        class _FakeOCR:
            def detect_state_from_text(self, image):
                return "lobby", 0.85

        images_path = project_root / "images"
        detector = u_mod.UnifiedStateDetector(
            images_path=images_path,
            window_w=1920,
            window_h=1080,
            ocr_detector=_FakeOCR(),
        )

        import numpy as np
        black_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Forçar pixel/template a falharem para que OCR seja usado
        detector._detect_by_pixels = lambda img: u_mod.DetectedState("unknown", 0.0, "pixel")
        detector._detect_by_templates = lambda img: u_mod.DetectedState("unknown", 0.0, "template")

        result = detector.detect(black_image)

        assert result.state == "lobby"
        assert result.method == "ocr"
        assert result.confidence > 0.0


class TestDebugVisualizer:
    """Testes para o DebugVisualizer."""

    def test_init(self):
        """Testa inicializacao do visualizer."""
        mod = _ensure_debug_visualizer()
        viz = mod.DebugVisualizer(mode=mod.DebugMode.DETAILED)
        assert viz.mode == mod.DebugMode.DETAILED
        assert viz.is_running is False

    def test_update_overlay(self):
        """Testa atualizacao de overlay."""
        mod = _ensure_debug_visualizer()
        viz = mod.DebugVisualizer(mode=mod.DebugMode.BASIC)
        overlay = mod.DebugOverlay(
            state="lobby",
            state_confidence=0.9,
            fps=30.0
        )
        viz.update_overlay(overlay)
        assert viz.current_overlay.state == "lobby"

    def test_toggle_pause(self):
        """Testa pausa."""
        mod = _ensure_debug_visualizer()
        viz = mod.DebugVisualizer(mode=mod.DebugMode.BASIC)
        assert viz.is_paused is False
        viz.toggle_pause()
        assert viz.is_paused is True

    def test_set_mode(self):
        """Testa mudanca de modo."""
        mod = _ensure_debug_visualizer()
        viz = mod.DebugVisualizer(mode=mod.DebugMode.BASIC)
        viz.set_mode(mod.DebugMode.COMBAT)
        assert viz.mode == mod.DebugMode.COMBAT


class TestErrorScenarios:
    """Testes de cenarios de erro que podem ocorrer em producao."""

    def test_screenshot_failure_recovery(self):
        """Simula falha de screenshot e verifica recovery."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem(enable_auto_recovery=True)

        context = mod.ErrorContext(
            error_type=mod.ErrorType.SCREENSHOT_FAILURE,
            severity=mod.ErrorSeverity.HIGH,
            exception=RuntimeError("Screenshot capture failed"),
            traceback_str="",
            component="screenshot",
            operation="capture"
        )

        recovered = system.handle_error(context, None)
        assert system.get_stats()["total_errors"] >= 1

    def test_yolo_failure_recovery(self):
        """Simula falha do YOLO e verifica graceful degradation."""
        mod = _ensure_error_recovery()
        system = mod.ErrorRecoverySystem(enable_auto_recovery=True)

        context = mod.ErrorContext(
            error_type=mod.ErrorType.YOLO_FAILURE,
            severity=mod.ErrorSeverity.MEDIUM,
            exception=RuntimeError("YOLO inference failed"),
            traceback_str="",
            component="vision",
            operation="detect"
        )

        recovered = system.handle_error(context, None)
        assert isinstance(recovered, bool)

    def test_multiple_errors_circuit_breaker(self):
        """Simula multiplos erros e verifica circuit breaker."""
        mod = _ensure_error_recovery()
        cb = mod.CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        for _ in range(3):
            cb.record_failure()

        assert cb.get_state() == "OPEN"
        assert cb.can_execute() is False

    def test_state_stuck_recovery(self):
        """Simula bot preso em estado e verifica recovery."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(
            emulator_controller=None,
            max_unknown_duration=0.1
        )

        # Set state to unknown with old timestamps (need >= 2 for time_in_current_state)
        recovery.current_state = "unknown"
        recovery.state_start_time = time.time() - 0.2
        recovery.history.states.append("unknown")
        recovery.history.timestamps.append(time.time() - 0.3)
        recovery.history.states.append("unknown")
        recovery.history.timestamps.append(time.time() - 0.2)

        needs = recovery._needs_recovery("unknown", 0.0)
        assert needs is True

    def test_state_loop_recovery(self):
        """Simula loop de estado e verifica recovery."""
        mod = _ensure_state_recovery()
        recovery = mod.StateRecoverySystem(
            emulator_controller=None,
            max_loop_duration=0.1
        )

        # To trigger loop detection, we need 5 identical recent states in history.
        # update_state only adds to history when state CHANGES, so we must
        # alternate states to build up history, then end with 5 "lobby" entries.
        # But since same-state calls don't add entries, we need to manipulate
        # history directly for this test.
        for _ in range(5):
            recovery.history.states.append("lobby")
            recovery.history.timestamps.append(time.time() - 0.2)  # Old timestamps

        # Set current state to lobby
        recovery.current_state = "lobby"

        # Now _needs_recovery should detect loop (5 identical) and time > max_loop_duration
        needs = recovery._needs_recovery("lobby", 0.9)
        assert needs is True


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    """Executa todos os testes manualmente."""
    test_classes = [
        TestErrorRecoverySystem,
        TestStateRecoverySystem,
        TestAutoCalibrator,
        TestOCRStateDetector,
        TestDebugVisualizer,
        TestErrorScenarios,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{'=' * 60}")
        print(f"  {test_class.__name__}")
        print(f"{'=' * 60}")

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            total += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  PASS {method_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {method_name}: {e}")
                failed += 1
                errors.append((test_class.__name__, method_name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailed tests detail:")
        for cls, method, err in errors:
            print(f"  - {cls}.{method}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
