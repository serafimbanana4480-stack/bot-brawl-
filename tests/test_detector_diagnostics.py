import sys
from pathlib import Path
from unittest.mock import patch

# Ensure repository root is on sys.path so top-level packages like 'brawl_bot' can be imported
repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from brawl_bot import emulator_detector


def test_detection_report_fields():
    with patch.object(emulator_detector, "get_adb_path", return_value="mock_adb"):
        report = emulator_detector.get_detection_report()

    assert isinstance(report, dict), "Report should be a dict"

    # Telemetry fields
    assert 'emulators' in report and isinstance(report['emulators'], list)
    assert 'pywin32_available' in report and isinstance(report['pywin32_available'], bool)
    assert 'psutil_available' in report and isinstance(report['psutil_available'], bool)
    assert 'chosen_adb_path' in report and isinstance(report['chosen_adb_path'], str)

    # chosen_adb_path should be non-empty string
    assert report['chosen_adb_path'] != "", "chosen_adb_path must be a non-empty string"

    # Each emulator dict should have deterministic fields
    for emu in report['emulators']:
        assert isinstance(emu, dict)
        assert 'name' in emu and isinstance(emu['name'], str)
        assert 'type' in emu and isinstance(emu['type'], str)
        assert 'connected' in emu and isinstance(emu['connected'], bool)
        # Optional fields may be None or specific types
        assert 'adb_id' in emu
        assert 'window_title' in emu
        assert 'window_handle' in emu

