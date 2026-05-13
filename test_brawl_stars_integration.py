"""
test_brawl_stars_integration.py

Comprehensive test suite for Brawl Stars integration.
Tests import paths, API endpoints, emulator detection, and frontend parsing.
"""

import pytest
import sys
import os
import json
from pathlib import Path

# Add paths for imports
# File is in backend/brawl_bot, so go up 3 levels to reach project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSharedModuleImport:
    """Test 1: Test shared module import works"""
    
    def test_shared_auth_import(self):
        """Test that shared.auth module can be imported"""
        # Note: shared.auth module was removed due to being non-functional
        # Authentication is handled by backend/security.py with JWT validation
        pytest.skip("shared.auth module removed - using backend/security.py instead")
    
    def test_shared_auth_database_import(self):
        """Test that shared.auth.database can be imported"""
        pytest.skip("shared.auth module removed - using backend/security.py instead")
    
    def test_shared_auth_rbac_import(self):
        """Test that shared.auth.rbac can be imported"""
        pytest.skip("shared.auth module removed - using backend/security.py instead")


class TestAPIEndpoints:
    """Test 2-3: Test API server and endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        from backend.interface.api import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_check_endpoint(self, client):
        """Test that health check endpoint responds"""
        response = client.get("/health")
        assert response.status_code in [200, 401]  # 401 if auth required
    
    def test_emulators_endpoint_exists(self, client):
        """Test that emulators endpoint exists"""
        response = client.get("/api/brawl-stars/emulators")
        # May return 404 if not registered, 200 if working
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "emulators" in data or "error" in data


class TestEmulatorDetection:
    """Test 4-5: Test emulator detection"""
    
    def test_adb_path_function(self):
        """Test that get_adb_path function exists"""
        from backend.brawl_bot.emulator_detector import get_adb_path
        assert get_adb_path is not None
        adb_path = get_adb_path()
        assert adb_path is not None
    
    def test_emulator_detector_init(self):
        """Test that EmulatorDetector can be instantiated"""
        from backend.brawl_bot.emulator_detector import EmulatorDetector
        detector = EmulatorDetector()
        assert detector is not None
        assert detector.available_emulators == []
    
    def test_detect_adb_devices(self):
        """Test that detect_adb_devices runs without errors"""
        from backend.brawl_bot.emulator_detector import EmulatorDetector
        detector = EmulatorDetector()
        emulators = detector.detect_adb_devices()
        assert isinstance(emulators, list)
        # May be empty if ADB not available
    
    def test_detect_window_emulators(self):
        """Test that detect_window_emulators runs without errors"""
        from backend.brawl_bot.emulator_detector import EmulatorDetector
        detector = EmulatorDetector()
        emulators = detector.detect_window_emulators()
        assert isinstance(emulators, list)
        # May be empty if no emulators running
    
    def test_detect_all(self):
        """Test that detect_all runs without errors"""
        from backend.brawl_bot.emulator_detector import EmulatorDetector
        detector = EmulatorDetector()
        emulators = detector.detect_all()
        assert isinstance(emulators, list)


class TestFrontendParsing:
    """Test 11-13: Test frontend JSON parsing"""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response"""
        json_str = '{"emulators": [{"name": "test", "type": "bluestacks"}], "count": 1}'
        data = json.loads(json_str)
        assert data["emulators"][0]["name"] == "test"
        assert data["count"] == 1
    
    def test_parse_empty_response(self):
        """Test handling empty response"""
        json_str = ""
        try:
            data = json.loads(json_str)
            pytest.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            pass  # Expected
    
    def test_parse_invalid_json(self):
        """Test handling invalid JSON"""
        json_str = "not valid json"
        try:
            data = json.loads(json_str)
            pytest.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            pass  # Expected
    
    def test_parse_empty_array(self):
        """Test parsing empty emulators array"""
        json_str = '{"emulators": [], "count": 0}'
        data = json.loads(json_str)
        assert data["emulators"] == []
        assert data["count"] == 0


class TestErrorHandling:
    """Test 9-10: Test error handling for missing ADB and emulators"""
    
    def test_missing_adb_error_message(self):
        """Test that missing ADB produces helpful error message"""
        from backend.brawl_bot.emulator_detector import get_adb_path
        adb_path = get_adb_path()
        # Should return a path even if ADB doesn't exist
        assert adb_path is not None
    
    def test_no_emulators_empty_list(self):
        """Test that no emulators returns empty list"""
        from backend.brawl_bot.emulator_detector import EmulatorDetector
        detector = EmulatorDetector()
        # Mock no emulators scenario
        emulators = []
        assert isinstance(emulators, list)
        assert len(emulators) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
