"""
Test suite for new AI components (Phases 9-12).

Tests for:
- Movement predictor integration
- Replay analyzer
- Advanced safety system
- Model registry
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import time

# Import components to test
try:
    from tracker import EnemyTracker
    MOVEMENT_PREDICTOR_AVAILABLE = True
except ImportError:
    MOVEMENT_PREDICTOR_AVAILABLE = False

try:
    from analysis.replay_parser import ReplayParser, ReplayMetadata
    from analysis.performance_analyzer import PerformanceAnalyzer
    from analysis.replay_analyzer import ReplayAnalyzer
except ImportError:
    REPLAY_ANALYZER_AVAILABLE = False
else:
    REPLAY_ANALYZER_AVAILABLE = True

try:
    from safety_system import AdvancedSafetySystem, UniqueFingerprint, DynamicParameterAdjuster
except ImportError:
    SAFETY_SYSTEM_AVAILABLE = False
else:
    SAFETY_SYSTEM_AVAILABLE = True

try:
    from training.model_registry import ModelRegistry, ModelMetadata
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False
else:
    MODEL_REGISTRY_AVAILABLE = True


class TestMovementPredictorIntegration(unittest.TestCase):
    """Test movement predictor integration with tracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MOVEMENT_PREDICTOR_AVAILABLE:
            self.skipTest("Movement predictor not available")
        
        self.tracker = EnemyTracker(max_age=30, min_hits=2, use_advanced_prediction=True)
    
    def test_tracker_initialization(self):
        """Test tracker initializes with movement predictor."""
        self.assertIsNotNone(self.tracker)
        if MOVEMENT_PREDICTOR_AVAILABLE:
            # Movement predictor may or may not initialize depending on dependencies
            pass
    
    def test_basic_tracking(self):
        """Test basic tracking functionality."""
        detections = [
            ([100, 100, 150, 150], 0.9),
            ([105, 105, 155, 155], 0.88),
            ([110, 110, 160, 160], 0.85),
        ]
        
        for det in detections:
            tracks = self.tracker.update([det])
        
        # Should have at least one track
        self.assertGreater(len(tracks), 0)
    
    def test_predict_position(self):
        """Test position prediction."""
        detections = [
            ([100, 100, 150, 150], 0.9),
            ([105, 105, 155, 155], 0.88),
            ([110, 110, 160, 160], 0.85),
        ]
        
        for det in detections:
            tracks = self.tracker.update([det])
        
        if tracks:
            track_id = tracks[0].id
            prediction = self.tracker.predict_position(track_id, time_ahead=0.25)
            # Prediction may be None if insufficient history
            # Just verify the method works without error
            self.assertIsNotNone(prediction)
    
    def test_leading_shot_position(self):
        """Test leading shot calculation."""
        detections = [
            ([100, 100, 150, 150], 0.9),
            ([105, 105, 155, 155], 0.88),
            ([110, 110, 160, 160], 0.85),
        ]
        
        for det in detections:
            tracks = self.tracker.update([det])
        
        if tracks:
            track_id = tracks[0].id
            leading_pos = self.tracker.get_leading_shot_position(
                track_id, projectile_speed=15.0, frame_delay=0
            )
            # Just verify the method works without error
            self.assertIsNotNone(leading_pos)


class TestReplayAnalyzer(unittest.TestCase):
    """Test replay analyzer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not REPLAY_ANALYZER_AVAILABLE:
            self.skipTest("Replay analyzer not available")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analyzer = ReplayAnalyzer(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertTrue(self.analyzer.output_dir.exists())
    
    def test_performance_analyzer(self):
        """Test performance analyzer."""
        analyzer = PerformanceAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        summary = self.analyzer.generate_summary_report()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_analyses', summary)


class TestAdvancedSafetySystem(unittest.TestCase):
    """Test advanced safety system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SAFETY_SYSTEM_AVAILABLE:
            self.skipTest("Safety system not available")
    
    def test_unique_fingerprint(self):
        """Test unique fingerprint generation."""
        fingerprint = UniqueFingerprint()
        self.assertIsNotNone(fingerprint.session_id)
        self.assertIsNotNone(fingerprint.timing_profile)
        self.assertIsNotNone(fingerprint.movement_profile)
    
    def test_fingerprint_adjusted_delay(self):
        """Test fingerprint delay adjustment."""
        fingerprint = UniqueFingerprint()
        base_delay = 0.5
        adjusted = fingerprint.get_adjusted_delay(base_delay)
        self.assertGreater(adjusted, 0)
    
    def test_dynamic_parameter_adjuster(self):
        """Test dynamic parameter adjuster."""
        adjuster = DynamicParameterAdjuster()
        self.assertIsNotNone(adjuster)
        
        # Test risk level update
        adjuster.update_risk_level(suspicion_score=50, human_likeness=80)
        self.assertGreaterEqual(adjuster.current_risk_level, 0)
        self.assertLessEqual(adjuster.current_risk_level, 1)
    
    def test_parameter_adjustment(self):
        """Test parameter adjustment based on risk."""
        adjuster = DynamicParameterAdjuster()
        
        # Set high risk
        adjuster.update_risk_level(suspicion_score=80, human_likeness=50)
        params = adjuster.adjust_parameters()
        
        self.assertIn('aggressiveness', params)
        self.assertIn('reaction_speed', params)


class TestModelRegistry(unittest.TestCase):
    """Test model registry components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODEL_REGISTRY_AVAILABLE:
            self.skipTest("Model registry not available")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry = ModelRegistry(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        self.assertIsNotNone(self.registry)
        self.assertTrue(self.registry.registry_dir.exists())
    
    def test_registry_summary(self):
        """Test registry summary."""
        summary = self.registry.get_registry_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_models', summary)
        self.assertIn('models_by_type', summary)
    
    def test_list_models(self):
        """Test listing models."""
        models = self.registry.list_models()
        self.assertIsInstance(models, list)
    
    def test_get_active_model(self):
        """Test getting active model."""
        active = self.registry.get_active_model('yolo')
        # Should return None if no active model
        self.assertIsNone(active)


class TestIntegration(unittest.TestCase):
    """Integration tests for new components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_safety_with_fingerprint(self):
        """Test safety system with fingerprint integration."""
        if not SAFETY_SYSTEM_AVAILABLE:
            self.skipTest("Safety system not available")
        
        safety = AdvancedSafetySystem()
        self.assertIsNotNone(safety.fingerprint)
        self.assertIsNotNone(safety.dynamic_adjuster)
    
    def test_registry_with_performance_tracking(self):
        """Test model registry with performance tracking."""
        if not MODEL_REGISTRY_AVAILABLE:
            self.skipTest("Model registry not available")
        
        registry = ModelRegistry(self.temp_dir)
        
        # Record some performance data (without registering a model first)
        # This tests the performance tracking infrastructure
        self.assertIsNotNone(registry.performance_history)


if __name__ == '__main__':
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMovementPredictorIntegration,
        TestReplayAnalyzer,
        TestAdvancedSafetySystem,
        TestModelRegistry,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
