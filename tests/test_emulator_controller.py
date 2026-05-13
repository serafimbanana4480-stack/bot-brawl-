"""
Testes para emulator_controller.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from emulator_controller import EmulatorConfig, ADBController


class TestEmulatorConfig:
    def test_default_config(self):
        cfg = EmulatorConfig()
        assert cfg.name == "LDPlayer"
        assert cfg.adb_port == 5555
        assert cfg.resolution == (1920, 1080)

    def test_bluestacks_config(self):
        cfg = EmulatorConfig.for_bluestacks()
        assert cfg.name == "BlueStacks"
        assert cfg.window_title == "BlueStacks App Player"

    def test_ldplayer_config(self):
        cfg = EmulatorConfig.for_ldplayer()
        assert cfg.name == "LDPlayer"
        assert cfg.window_title == "LDPlayer"


class TestADBController:
    def test_sanitize_device_id(self):
        clean = ADBController._sanitize_device_id("emulator-5554")
        assert clean == "emulator-5554"

    def test_sanitize_device_id_dangerous(self):
        clean = ADBController._sanitize_device_id("127.0.0.1:5555; rm -rf /")
        assert ";" not in clean
        assert "rm" not in clean
        assert clean.startswith("127.0.0.1:5555")

    def test_init(self):
        cfg = EmulatorConfig(adb_port=5555)
        adb = ADBController(cfg)
        assert adb.device_id == "emulator-5555"
