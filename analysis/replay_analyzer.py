"""
replay_analyzer.py

Complete replay analysis pipeline.

Integrates replay parsing and performance analysis to provide
comprehensive insights from gameplay recordings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .replay_parser import ReplayParser, ParsedReplay
from .performance_analyzer import PerformanceAnalyzer, PerformanceReport

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    replay_id: str
    parsed_replay: ParsedReplay
    performance_report: PerformanceReport
    analysis_timestamp: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'replay_id': self.replay_id,
            'performance': asdict(self.performance_report),
            'analysis_timestamp': self.analysis_timestamp,
            'recommendations': self.recommendations,
            # Note: parsed_replay is not included (too large)
        }


class ReplayAnalyzer:
    """
    Complete replay analysis pipeline.
    
    Combines replay parsing and performance analysis to provide
    actionable insights from gameplay recordings.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.analysis_history: List[AnalysisResult] = []
    
    def analyze_replay(self, 
                      video_path: Path,
                      action_log_path: Optional[Path] = None,
                      replay_id: Optional[str] = None,
                      save_result: bool = True) -> AnalysisResult:
        """
        Analyze a single replay.
        
        Args:
            video_path: Path to gameplay video
            action_log_path: Optional path to action log
            replay_id: Optional identifier for the replay
            save_result: If True, save analysis result to file
            
        Returns:
            AnalysisResult object
        """
        # Generate replay ID if not provided
        if replay_id is None:
            replay_id = video_path.stem
        
        logger.info(f"Analyzing replay: {replay_id}")
        
        # Parse replay
        parser = ReplayParser(video_path, action_log_path)
        parsed = parser.parse(load_frames=False)
        
        # Analyze performance
        report = self.performance_analyzer.analyze(parsed, replay_id)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(report)
        
        # Create result
        result = AnalysisResult(
            replay_id=replay_id,
            parsed_replay=parsed,
            performance_report=report,
            analysis_timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
        self.analysis_history.append(result)
        
        # Save result if requested
        if save_result:
            self._save_result(result)
        
        logger.info(f"Analysis complete for {replay_id}")
        
        return result
    
    def analyze_directory(self,
                         video_dir: Path,
                         action_log_dir: Optional[Path] = None,
                         pattern: str = "*.mp4") -> List[AnalysisResult]:
        """
        Analyze all replays in a directory.
        
        Args:
            video_dir: Directory containing video files
            action_log_dir: Optional directory containing action logs
            pattern: File pattern to match (e.g., "*.mp4")
            
        Returns:
            List of AnalysisResult objects
        """
        video_dir = Path(video_dir)
        action_log_dir = Path(action_log_dir) if action_log_dir else None
        
        results = []
        video_files = list(video_dir.glob(pattern))
        
        logger.info(f"Found {len(video_files)} video files to analyze")
        
        for video_path in video_files:
            # Try to find corresponding action log
            action_log_path = None
            if action_log_dir:
                log_name = video_path.stem + ".json"
                potential_log = action_log_dir / log_name
                if potential_log.exists():
                    action_log_path = potential_log
            
            try:
                result = self.analyze_replay(
                    video_path=video_path,
                    action_log_path=action_log_path,
                    save_result=True
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {video_path.name}: {e}")
        
        logger.info(f"Analyzed {len(results)} replays successfully")
        
        return results
    
    def _generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate actionable recommendations from performance report."""
        recommendations = []
        
        # Combat recommendations
        if report.combat.accuracy < 0.5:
            recommendations.append(
                "Train aim assist model on more data to improve accuracy"
            )
        if report.combat.kda_ratio < 1.0:
            recommendations.append(
                "Review decision policy - consider more conservative playstyle"
            )
        if report.combat.damage_efficiency < 1.0:
            recommendations.append(
                "Improve dodging behavior and cover usage"
            )
        
        # Movement recommendations
        if report.movement.idle_time > 10.0:
            recommendations.append(
                "Optimize pathfinding to reduce idle time"
            )
        
        # Decision recommendations
        if report.decision.decision_quality < 0.6:
            recommendations.append(
                "Collect more training data for behavior cloning"
            )
            recommendations.append(
                "Consider retraining neural policy with recent gameplay data"
            )
        if report.decision.reaction_time_avg > 0.5:
            recommendations.append(
                "Optimize vision pipeline for faster detection"
            )
        
        # Overall recommendations
        if report.overall_score < 0.5:
            recommendations.append(
                "Overall performance below threshold - comprehensive review recommended"
            )
        elif report.overall_score > 0.8:
            recommendations.append(
                "Performance is excellent - consider pushing to higher trophy ranges"
            )
        
        if not recommendations:
            recommendations.append("Performance is satisfactory - continue monitoring")
        
        return recommendations
    
    def _save_result(self, result: AnalysisResult):
        """Save analysis result to JSON file."""
        try:
            filename = f"analysis_{result.replay_id}.json"
            output_path = self.output_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved analysis result to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
    
    def generate_summary_report(self, n_recent: int = 10) -> Dict[str, Any]:
        """
        Generate summary report of recent analyses.
        
        Args:
            n_recent: Number of recent analyses to include
            
        Returns:
            Summary report dictionary
        """
        if not self.analysis_history:
            return {
                'total_analyses': 0,
                'recent_analyses': 0,
                'average_overall_score': 0.0,
                'average_kda_ratio': 0.0,
                'average_accuracy': 0.0,
                'top_improvement_areas': [],
                'trends': self.performance_analyzer.get_trends(n_recent),
                'average_performance': self.performance_analyzer.get_average_performance(n_recent),
            }
        
        recent = self.analysis_history[-n_recent:]
        
        # Calculate averages
        avg_overall = sum(r.performance_report.overall_score for r in recent) / len(recent)
        avg_kda = sum(r.performance_report.combat.kda_ratio for r in recent) / len(recent)
        avg_accuracy = sum(r.performance_report.combat.accuracy for r in recent) / len(recent)
        
        # Collect common improvement areas
        improvement_counts = {}
        for result in recent:
            for area in result.performance_report.improvement_areas:
                improvement_counts[area] = improvement_counts.get(area, 0) + 1
        
        # Sort by frequency
        top_improvements = sorted(improvement_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            'total_analyses': len(self.analysis_history),
            'recent_analyses': len(recent),
            'average_overall_score': avg_overall,
            'average_kda_ratio': avg_kda,
            'average_accuracy': avg_accuracy,
            'top_improvement_areas': [{'area': k, 'count': v} for k, v in top_improvements],
            'trends': self.performance_analyzer.get_trends(n_recent),
            'average_performance': self.performance_analyzer.get_average_performance(n_recent),
        }
        
        return summary
    
    def save_summary_report(self, n_recent: int = 10):
        """Save summary report to file."""
        try:
            summary = self.generate_summary_report(n_recent)
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = self.output_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary report to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
    
    def get_training_recommendations(self) -> List[str]:
        """
        Get recommendations for model training based on analysis history.
        
        Returns:
            List of training recommendations
        """
        if not self.analysis_history:
            return ["No analysis data available - collect more replays"]
        
        recommendations = []
        
        # Get average performance
        avg_perf = self.performance_analyzer.get_average_performance()
        
        # Check if accuracy is low
        if avg_perf.get('accuracy', 0) < 0.6:
            recommendations.append(
                "Low accuracy detected - consider retraining vision model with more labeled data"
            )
        
        # Check if decision quality is low
        if avg_perf.get('decision_quality', 0) < 0.6:
            recommendations.append(
                "Poor decision quality - collect more gameplay data for behavior cloning"
            )
            recommendations.append(
                "Consider running CQL offline RL on collected replay buffer"
            )
        
        # Check overall trends
        trends = self.performance_analyzer.get_trends()
        if 'overall_scores' in trends and len(trends['overall_scores']) > 5:
            recent_scores = trends['overall_scores'][-5:]
            if all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                recommendations.append(
                    "Performance declining - immediate retraining recommended"
                )
        
        if not recommendations:
            recommendations.append("Performance is stable - continue monitoring")
        
        return recommendations
