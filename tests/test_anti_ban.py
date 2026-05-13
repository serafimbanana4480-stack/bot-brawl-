import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import importlib.util

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
    # Forçar schedule a True para o teste
    import random
    original_random = random.random
    random.random = lambda: 0.01  # should_play_now retorna True em qualquer horário (menor que thresholds)
    try:
        assert absys.should_start_match() is True
        absys.record_match_result("win")
        absys.record_match_result("win")
        # Após 2 partidas, deve bloquear
        assert absys.should_start_match() is False
    finally:
        random.random = original_random
    status = absys.get_status()
    assert status["enabled"] is True
    assert status["matches_this_hour"] == 2
