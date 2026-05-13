"""
Auto-retraining system for continuous model improvement.
Monitors performance and triggers retraining when needed.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import logging


@dataclass
class PerformanceMetrics:
    """Metrics for evaluating bot performance."""
    timestamp: float
    
    # Combat metrics
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    
    # Win/Loss
    matches_played: int = 0
    matches_won: int = 0
    
    # Detection quality
    detection_accuracy: float = 0.0
    tracking_consistency: float = 0.0
    false_positive_rate: float = 0.0
    
    # Decision quality
    good_decisions: int = 0
    bad_decisions: int = 0
    
    # Game metrics
    avg_survival_time: float = 0.0
    trophies_gained: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.matches_won / self.matches_played
    
    @property
    def kda_ratio(self) -> float:
        if self.deaths == 0:
            return self.kills
        return self.kills / self.deaths
    
    @property
    def decision_accuracy(self) -> float:
        total = self.good_decisions + self.bad_decisions
        if total == 0:
            return 0.0
        return self.good_decisions / total


@dataclass
class RetrainTrigger:
    """Conditions that trigger retraining."""
    min_matches_before_retrain: int = 10
    win_rate_threshold: float = 0.4
    min_detection_accuracy: float = 0.7
    max_false_positive_rate: float = 0.2
    decision_accuracy_threshold: float = 0.6
    
    # Time-based triggers
    max_days_without_retrain: int = 7
    min_new_samples: int = 500


class PerformanceMonitor:
    """
    Monitors bot performance and collects metrics.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_session: Optional[PerformanceMetrics] = None
        
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger("performance_monitor")
        
    def start_session(self):
        """Start a new monitoring session."""
        with self._lock:
            self.current_session = PerformanceMetrics(
                timestamp=time.time()
            )
    
    def end_session(self) -> PerformanceMetrics:
        """End current session and save metrics."""
        with self._lock:
            if self.current_session:
                self.metrics_history.append(self.current_session)
                self._save_metrics(self.current_session)
                session = self.current_session
                self.current_session = None
                return session
            return None
    
    def record_kill(self):
        """Record a kill."""
        with self._lock:
            if self.current_session:
                self.current_session.kills += 1
    
    def record_death(self):
        """Record a death."""
        with self._lock:
            if self.current_session:
                self.current_session.deaths += 1
    
    def record_damage(self, dealt: float = 0, taken: float = 0):
        """Record damage dealt/taken."""
        with self._lock:
            if self.current_session:
                self.current_session.damage_dealt += dealt
                self.current_session.damage_taken += taken
    
    def record_match_result(self, won: bool, survival_time: float):
        """Record match result."""
        with self._lock:
            if self.current_session:
                self.current_session.matches_played += 1
                if won:
                    self.current_session.matches_won += 1
                
                # Update average survival time
                n = self.current_session.matches_played
                old_avg = self.current_session.avg_survival_time
                self.current_session.avg_survival_time = (
                    (old_avg * (n - 1) + survival_time) / n
                )
    
    def record_decision(self, was_good: bool):
        """Record decision quality."""
        with self._lock:
            if self.current_session:
                if was_good:
                    self.current_session.good_decisions += 1
                else:
                    self.current_session.bad_decisions += 1
    
    def update_detection_metrics(
        self,
        accuracy: float,
        tracking: float,
        false_positive: float
    ):
        """Update detection quality metrics."""
        with self._lock:
            if self.current_session:
                self.current_session.detection_accuracy = accuracy
                self.current_session.tracking_consistency = tracking
                self.current_session.false_positive_rate = false_positive
    
    def _save_metrics(self, metrics: PerformanceMetrics):
        """Save metrics to file."""
        timestamp = datetime.fromtimestamp(metrics.timestamp)
        filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def get_recent_metrics(self, n_sessions: int = 10) -> PerformanceMetrics:
        """Aggregate metrics from recent sessions."""
        recent = self.metrics_history[-n_sessions:]
        
        if not recent:
            return PerformanceMetrics(timestamp=time.time())
        
        # Aggregate
        aggregated = PerformanceMetrics(timestamp=time.time())
        
        aggregated.kills = sum(m.kills for m in recent)
        aggregated.deaths = sum(m.deaths for m in recent)
        aggregated.damage_dealt = sum(m.damage_dealt for m in recent)
        aggregated.damage_taken = sum(m.damage_taken for m in recent)
        aggregated.matches_played = sum(m.matches_played for m in recent)
        aggregated.matches_won = sum(m.matches_won for m in recent)
        aggregated.good_decisions = sum(m.good_decisions for m in recent)
        aggregated.bad_decisions = sum(m.bad_decisions for m in recent)
        
        # Average detection metrics
        aggregated.detection_accuracy = sum(m.detection_accuracy for m in recent) / len(recent)
        aggregated.tracking_consistency = sum(m.tracking_consistency for m in recent) / len(recent)
        aggregated.false_positive_rate = sum(m.false_positive_rate for m in recent) / len(recent)
        
        return aggregated


class RetrainOrchestrator:
    """
    Orchestrates auto-retraining pipeline with replay analyzer integration.
    """
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        trigger_conditions: RetrainTrigger = None,
        dataset_dir: Path = None,
        models_dir: Path = None,
        replay_analyzer = None
    ):
        self.monitor = monitor
        self.triggers = trigger_conditions or RetrainTrigger()
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.replay_analyzer = replay_analyzer
        
        self.last_retrain_time: Optional[float] = None
        self.retrain_count: int = 0
        
        self.logger = logging.getLogger("retrain_orchestrator")
        
        # Callbacks
        self.on_retrain_start: Optional[Callable] = None
        self.on_retrain_complete: Optional[Callable] = None
        self.on_retrain_failed: Optional[Callable] = None
    
    def should_retrain(self) -> tuple[bool, str]:
        """
        Check if retraining should be triggered.
        
        Returns:
            (should_retrain, reason)
        """
        # Get recent performance
        metrics = self.monitor.get_recent_metrics(n_sessions=20)
        
        # Check minimum matches
        if metrics.matches_played < self.triggers.min_matches_before_retrain:
            return False, f"Not enough matches ({metrics.matches_played}/{self.triggers.min_matches_before_retrain})"
        
        # Check win rate
        if metrics.win_rate < self.triggers.win_rate_threshold:
            return True, f"Low win rate: {metrics.win_rate:.1%}"
        
        # Check detection accuracy
        if metrics.detection_accuracy < self.triggers.min_detection_accuracy:
            return True, f"Poor detection: {metrics.detection_accuracy:.1%}"
        
        # Check false positive rate
        if metrics.false_positive_rate > self.triggers.max_false_positive_rate:
            return True, f"High false positives: {metrics.false_positive_rate:.1%}"
        
        # Check decision accuracy
        if metrics.decision_accuracy < self.triggers.decision_accuracy_threshold:
            return True, f"Poor decisions: {metrics.decision_accuracy:.1%}"
        
        # Check time since last retrain
        if self.last_retrain_time:
            days_since = (time.time() - self.last_retrain_time) / 86400
            if days_since > self.triggers.max_days_without_retrain:
                return True, f"Time since retrain: {days_since:.1f} days"
        
        return False, "Performance satisfactory"
    
    def trigger_retrain(self) -> bool:
        """
        Trigger the retraining process.
        
        Returns:
            True if retraining started successfully
        """
        self.logger.info("Starting auto-retraining...")
        
        if self.on_retrain_start:
            self.on_retrain_start()
        
        try:
            # Step 1: Capture new data
            self._step_capture_new_data()
            
            # Step 2: Validate dataset
            if not self._step_validate_dataset():
                raise RuntimeError("Dataset validation failed")
            
            # Step 3: Retrain model
            new_model_path = self._step_retrain_model()
            
            # Step 4: Validate new model
            if not self._step_validate_model(new_model_path):
                raise RuntimeError("New model validation failed")
            
            # Step 5: Deploy if better
            if self._step_compare_and_deploy(new_model_path):
                self.logger.info("Retraining complete, new model deployed")
                self.retrain_count += 1
                self.last_retrain_time = time.time()
                
                if self.on_retrain_complete:
                    self.on_retrain_complete(new_model_path)
                
                return True
            else:
                self.logger.info("New model not better, keeping current")
                return False
                
        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")
            if self.on_retrain_failed:
                self.on_retrain_failed(str(e))
            return False
    
    def _step_capture_new_data(self):
        """Capture new gameplay data."""
        self.logger.info("Capturing new gameplay data...")
        # This would integrate with dataset_pipeline
        pass
    
    def _step_validate_dataset(self) -> bool:
        """Validate dataset quality."""
        self.logger.info("Validating dataset...")
        
        if not self.dataset_dir or not self.dataset_dir.exists():
            return False
        
        # Check minimum samples
        image_count = len(list(self.dataset_dir.glob("*.jpg")))
        if image_count < self.triggers.min_new_samples:
            self.logger.warning(f"Not enough samples: {image_count}")
            return False
        
        return True
    
    def _step_retrain_model(self) -> Path:
        """Execute model retraining."""
        self.logger.info("Retraining model...")
        
        # This would call train_yolo.py
        # For now, placeholder
        timestamp = int(time.time())
        new_model_path = self.models_dir / f"model_retrain_{timestamp}.pt"
        
        return new_model_path
    
    def _step_validate_model(self, model_path: Path) -> bool:
        """Validate new model performance."""
        self.logger.info("Validating new model...")
        
        if not model_path.exists():
            return False
        
        # Run validation
        # Return True if model meets minimum thresholds
        return True
    
    def _step_compare_and_deploy(self, new_model_path: Path) -> bool:
        """Compare new model with current and deploy if better."""
        self.logger.info("Comparing models...")
        
        # Simplified: always deploy new model
        # In real implementation, would run A/B test
        return True
    
    def run_monitoring_loop(self, check_interval: float = 3600):
        """
        Run continuous monitoring and auto-retrain loop.
        
        Args:
            check_interval: Seconds between checks
        """
        self.logger.info(f"Starting monitoring loop (interval: {check_interval}s)")
        
        while True:
            should_retrain, reason = self.should_retrain()
            
            if should_retrain:
                self.logger.info(f"Retraining triggered: {reason}")
                self.trigger_retrain()
            
            time.sleep(check_interval)
    
    def curate_dataset_from_replays(self, replay_dir: Path) -> bool:
        """
        Curate dataset from replay analysis.
        
        Uses replay analyzer to identify high-quality gameplay segments
        and extract them for training.
        
        Args:
            replay_dir: Directory containing replay videos
            
        Returns:
            True if dataset curation successful
        """
        if not self.replay_analyzer:
            self.logger.warning("No replay analyzer available for dataset curation")
            return False
        
        try:
            self.logger.info(f"Curating dataset from replays in {replay_dir}")
            
            # Analyze replays
            results = self.replay_analyzer.analyze_directory(replay_dir)
            
            # Filter high-quality replays (overall_score > 0.7)
            high_quality = [r for r in results if r.performance_report.overall_score > 0.7]
            
            self.logger.info(f"Found {len(high_quality)} high-quality replays out of {len(results)}")
            
            # Extract key frames from high-quality replays
            curated_dir = self.dataset_dir / "curated"
            curated_dir.mkdir(parents=True, exist_ok=True)
            
            for result in high_quality:
                # Extract frames around key events
                key_frames = self.replay_analyzer.replay_parser.extract_key_frames(
                    result.parsed_replay.events,
                    window_seconds=2.0
                )
                
                # Save frames to curated dataset
                for frame_idx, frame in key_frames:
                    frame_path = curated_dir / f"{result.replay_id}_frame_{frame_idx}.jpg"
                    # Save frame (would use cv2.imwrite in real implementation)
                    # cv2.imwrite(str(frame_path), frame)
            
            self.logger.info(f"Curated dataset saved to {curated_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset curation failed: {e}")
            return False
    
    def get_training_recommendations(self) -> List[str]:
        """
        Get training recommendations from replay analysis.
        
        Returns:
            List of training recommendations
        """
        if not self.replay_analyzer:
            return ["No replay analyzer available"]
        
        return self.replay_analyzer.get_training_recommendations()


class ContinuousLearner:
    """
    High-level interface for continuous learning.
    """
    
    def __init__(
        self,
        log_dir: Path,
        dataset_dir: Path,
        models_dir: Path
    ):
        self.monitor = PerformanceMonitor(log_dir)
        self.orchestrator = RetrainOrchestrator(
            monitor=self.monitor,
            dataset_dir=dataset_dir,
            models_dir=models_dir
        )
        
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start continuous learning."""
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self.orchestrator.run_monitoring_loop,
            kwargs={"check_interval": 3600},
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop(self):
        """Stop continuous learning."""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def record_match(self, won: bool, stats: dict):
        """Record a completed match."""
        self.monitor.record_match_result(won, stats.get("survival_time", 0))
        self.monitor.record_damage(
            dealt=stats.get("damage_dealt", 0),
            taken=stats.get("damage_taken", 0)
        )
        
        if stats.get("kills"):
            for _ in range(stats["kills"]):
                self.monitor.record_kill()
        
        if stats.get("deaths"):
            for _ in range(stats["deaths"]):
                self.monitor.record_death()
    
    def check_and_retrain(self) -> bool:
        """Manually check if retraining is needed and trigger."""
        should_retrain, reason = self.orchestrator.should_retrain()
        
        if should_retrain:
            return self.orchestrator.trigger_retrain()
        
        return False


def create_continuous_learner(
    log_dir: str,
    dataset_dir: str,
    models_dir: str
) -> ContinuousLearner:
    """Factory function to create continuous learning system."""
    return ContinuousLearner(
        log_dir=Path(log_dir),
        dataset_dir=Path(dataset_dir),
        models_dir=Path(models_dir)
    )
