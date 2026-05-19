import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import importlib.util
import pytest

_spec = importlib.util.spec_from_file_location("anti_ban", str(_repo_root / "core" / "anti_ban.py"))
_anti_ban_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_anti_ban_mod)

AntiBanSystem = _anti_ban_mod.AntiBanSystem
AntiBanConfig = _anti_ban_mod.AntiBanConfig
PatternDetector = _anti_ban_mod.PatternDetector
WinRateLimiter = _anti_ban_mod.WinRateLimiter
ScheduleRandomizer = _anti_ban_mod.ScheduleRandomizer
ActionObfuscator = _anti_ban_mod.ActionObfuscator
FingerprintRandomizer = _anti_ban_mod.FingerprintRandomizer

_adv_spec = importlib.util.spec_from_file_location(
    "anti_ban_advanced", str(_repo_root / "pylaai_real" / "anti_ban_advanced.py")
)
_adv_mod = importlib.util.module_from_spec(_adv_spec)
_adv_spec.loader.exec_module(_adv_mod)

AdvancedAntiBanSystem = _adv_mod.AdvancedAntiBanSystem
SessionOrchestrator = _adv_mod.SessionOrchestrator


def test_pattern_detector():
    pd = PatternDetector(window_size=5, threshold=0.85)
    # Repetir a mesma ação várias vezes
    for _ in range(5):
        pd.record_action("click", (100, 200))
    assert pd.detect_repetitive_pattern() is True

    pd2 = PatternDetector(window_size=5, threshold=0.85)
    # Ações variadas
    for i in range(5):
        pd2.record_action(f"action_{i}", (i, i))
    assert pd2.detect_repetitive_pattern() is False


def test_win_rate_limiter():
    wrl = WinRateLimiter(max_rate=0.75, min_rate=0.20)
    for _ in range(10):
        wrl.record_result("win")
    assert wrl.should_throttle() is True
    assert wrl.get_current_win_rate() == 1.0

    wrl2 = WinRateLimiter(max_rate=0.75, min_rate=0.20)
    for _ in range(5):
        wrl2.record_result("loss")
    assert wrl2.should_intensify() is True
    assert wrl2.should_throttle() is False


def test_schedule_randomizer():
    sr = ScheduleRandomizer(variance_minutes=30)
    next_time = sr.get_next_start_time(base_hour=10)
    assert next_time.hour in [9, 10, 11]

    # should_play_now retorna bool
    result = sr.should_play_now()
    assert isinstance(result, bool)


def test_action_obfuscator():
    ao = ActionObfuscator(noise_probability=1.0, max_delay=1.0)
    delay = ao.apply("click", 0.5)
    assert delay >= 0.5
    assert delay <= 1.5

    ao2 = ActionObfuscator(noise_probability=0.0, max_delay=1.0)
    delay2 = ao2.apply("click", 0.5)
    assert delay2 == 0.5


def test_fingerprint_randomizer():
    fr = FingerprintRandomizer(change_interval_hours=24.0)
    fp = fr.get_fingerprint()
    assert "delay_multiplier" in fp
    assert "aggression_bias" in fp


def test_anti_ban_system():
    config = AntiBanConfig(enabled=True, max_matches_per_hour=2, min_break_between_matches_sec=0)
    absys = AntiBanSystem(config)
    # Forçar todos os schedules a permitir jogo
    absys.session_schedule.should_play_now = lambda: True
    absys.schedule_randomizer.should_play_now = lambda: True
    assert absys.should_start_match() is True
    absys.record_match_result("win")
    absys.record_match_result("win")
    # Após 2 partidas, deve bloquear
    assert absys.should_start_match() is False
    status = absys.get_status()
    assert status["enabled"] is True
    assert status["matches_this_hour"] == 2


def test_session_orchestrator_pacing_and_pressure():
    plan = _adv_mod.SessionPlan(
        warmup_matches=2,
        fatigue_start_match=4,
        max_matches_per_hour=10,
        break_interval_min=999,
        break_interval_max=999,
        target_duration_min=999,
        target_duration_max=999,
    )
    orchestrator = SessionOrchestrator(plan)

    base_delay = 0.5
    warmup_delay = orchestrator.recommend_pacing(base_delay, "menu_nav")
    orchestrator.record_match_start()
    orchestrator.record_match_start()
    fatigue_delay = orchestrator.recommend_pacing(base_delay, "menu_nav")

    assert warmup_delay >= base_delay
    assert fatigue_delay >= base_delay
    assert 0.0 <= orchestrator.get_session_pressure() <= 1.0


def test_advanced_antiban_adaptive_pacing_responds_to_risk_and_session():
    system = AdvancedAntiBanSystem({"enabled": True})
    system.session_orchestrator.plan.warmup_matches = 4
    system.session_orchestrator.plan.fatigue_start_match = 4
    system.session_orchestrator.plan.max_matches_per_hour = 2

    base = 0.4
    initial = system.get_adaptive_pacing(base, "menu_nav")

    system.record_action("tap", (100, 200), interval=0.01)
    system.record_action("tap", (100, 200), interval=0.01)
    system.record_action("tap", (100, 200), interval=0.01)
    system.record_action("tap", (100, 200), interval=0.01)
    system.record_action("tap", (100, 200), interval=0.01)
    risk_paced = system.get_adaptive_pacing(base, "menu_nav")

    assert initial > 0
    assert risk_paced >= base * 0.5
    assert system.get_status()["session"]["session_pressure"] >= 0
