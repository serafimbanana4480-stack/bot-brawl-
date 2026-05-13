"""Replay Analyst Agent - Analyzes match replays for insights and learning"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class ReplayEvent:
    timestamp: float
    event_type: str
    details: Dict[str, Any]
    importance: float


@dataclass
class ReplayAnalysis:
    replay_id: str
    duration: float
    events: List[ReplayEvent]
    key_moments: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    strategic_patterns: List[str]


class ReplayAnalystAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.replay_cache: Dict[str, ReplayAnalysis] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.pattern_database: Dict[str, List[str]] = defaultdict(list)
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "analyze")
        
        try:
            if action == "analyze":
                result = await self._analyze_replay(message.content)
            elif action == "extract_patterns":
                result = await self._extract_patterns(message.content)
            elif action == "compare":
                result = await self._compare_replays(message.content)
            elif action == "timeline":
                result = await self._generate_timeline(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.85,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        recent_analyses = context.get("recent_analyses", [])
        
        if not recent_analyses:
            return {
                "status": "ready",
                "cached_replays": len(self.replay_cache),
                "patterns_learned": sum(len(p) for p in self.pattern_database.values()),
                "confidence": 0.7,
            }
        
        common_patterns = self._find_common_patterns(recent_analyses)
        
        return {
            "status": "ready",
            "common_patterns": common_patterns,
            "improvement_areas": self._identify_improvement_areas(recent_analyses),
            "confidence": 0.85,
        }
    
    async def _analyze_replay(self, content: Dict[str, Any]) -> Dict[str, Any]:
        replay_data = content.get("replay_data")
        replay_id = content.get("replay_id", str(int(time.time())))
        
        if not replay_data:
            return {"error": "No replay data provided"}
        
        events = self._extract_events(replay_data)
        
        key_moments = self._identify_key_moments(events)
        
        metrics = self._calculate_performance_metrics(replay_data, events)
        
        patterns = self._identify_strategic_patterns(events)
        
        analysis = ReplayAnalysis(
            replay_id=replay_id,
            duration=replay_data.get("duration", 0),
            events=events,
            key_moments=key_moments,
            performance_metrics=metrics,
            strategic_patterns=patterns,
        )
        
        self.replay_cache[replay_id] = analysis
        self.analysis_history.append({
            "replay_id": replay_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        })
        
        for pattern in patterns:
            self.pattern_database[pattern].append(replay_id)
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.LEARNING_UPDATE,
            data={
                "replay_id": replay_id,
                "metrics": metrics,
                "patterns_found": len(patterns),
            },
        ))
        
        return {
            "replay_id": replay_id,
            "duration": analysis.duration,
            "key_moments": key_moments,
            "performance_metrics": metrics,
            "strategic_patterns": patterns,
            "timeline": [(e.timestamp, e.event_type, e.importance) for e in events],
        }
    
    async def _extract_patterns(self, content: Dict[str, Any]) -> Dict[str, Any]:
        replay_ids = content.get("replay_ids", list(self.replay_cache.keys()))
        
        all_patterns = []
        for replay_id in replay_ids:
            if replay_id in self.replay_cache:
                all_patterns.extend(self.replay_cache[replay_id].strategic_patterns)
        
        pattern_counts = defaultdict(int)
        for pattern in all_patterns:
            pattern_counts[pattern] += 1
        
        common_patterns = sorted(
            [{"pattern": p, "count": c} for p, c in pattern_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
        
        return {
            "total_patterns": len(set(all_patterns)),
            "common_patterns": common_patterns,
            "replay_count": len(replay_ids),
        }
    
    async def _compare_replays(self, content: Dict[str, Any]) -> Dict[str, Any]:
        replay_id_1 = content.get("replay_id_1")
        replay_id_2 = content.get("replay_id_2")
        
        if replay_id_1 not in self.replay_cache or replay_id_2 not in self.replay_cache:
            return {"error": "One or both replays not found"}
        
        analysis_1 = self.replay_cache[replay_id_1]
        analysis_2 = self.replay_cache[replay_id_2]
        
        metric_differences = {}
        for key in analysis_1.performance_metrics:
            if key in analysis_2.performance_metrics:
                metric_differences[key] = (
                    analysis_2.performance_metrics[key] - analysis_1.performance_metrics[key]
                )
        
        pattern_differences = {
            "only_in_first": list(set(analysis_1.strategic_patterns) - set(analysis_2.strategic_patterns)),
            "only_in_second": list(set(analysis_2.strategic_patterns) - set(analysis_1.strategic_patterns)),
            "common": list(set(analysis_1.strategic_patterns) & set(analysis_2.strategic_patterns)),
        }
        
        return {
            "replay_1_metrics": analysis_1.performance_metrics,
            "replay_2_metrics": analysis_2.performance_metrics,
            "metric_differences": metric_differences,
            "pattern_differences": pattern_differences,
            "recommendation": self._generate_comparison_recommendation(metric_differences),
        }
    
    async def _generate_timeline(self, content: Dict[str, Any]) -> Dict[str, Any]:
        replay_id = content.get("replay_id")
        
        if replay_id not in self.replay_cache:
            return {"error": "Replay not found"}
        
        analysis = self.replay_cache[replay_id]
        
        timeline = []
        for event in analysis.events:
            timeline.append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "details": event.details,
                "importance": event.importance,
                "category": self._categorize_event(event),
            })
        
        categorized = defaultdict(list)
        for item in timeline:
            categorized[item["category"]].append(item)
        
        return {
            "replay_id": replay_id,
            "duration": analysis.duration,
            "timeline": timeline,
            "categorized": dict(categorized),
            "key_moments": analysis.key_moments,
        }
    
    def _extract_events(self, replay_data: Dict[str, Any]) -> List[ReplayEvent]:
        events = []
        
        game_events = replay_data.get("events", [])
        for event_data in game_events:
            event = ReplayEvent(
                timestamp=event_data.get("time", 0),
                event_type=event_data.get("type", "generic"),
                details=event_data.get("details", {}),
                importance=event_data.get("importance", 0.5),
            )
            events.append(event)
        
        events.sort(key=lambda e: e.timestamp)
        return events
    
    def _identify_key_moments(self, events: List[ReplayEvent]) -> List[Dict[str, Any]]:
        key_moments = []
        
        high_importance_events = [e for e in events if e.importance > 0.7]
        
        for event in high_importance_events:
            key_moments.append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "description": self._describe_event(event),
                "significance": event.importance,
            })
        
        return key_moments
    
    def _calculate_performance_metrics(self, replay_data: Dict[str, Any],
                                     events: List[ReplayEvent]) -> Dict[str, float]:
        metrics = {
            "kills": replay_data.get("kills", 0),
            "deaths": replay_data.get("deaths", 0),
            "assists": replay_data.get("assists", 0),
            "damage_dealt": replay_data.get("damage_dealt", 0),
            "damage_taken": replay_data.get("damage_taken", 0),
            "healing": replay_data.get("healing", 0),
            "objectives_captured": replay_data.get("objectives_captured", 0),
            "survival_time": replay_data.get("duration", 0),
        }
        
        if metrics["deaths"] > 0:
            metrics["kda_ratio"] = (metrics["kills"] + metrics["assists"]) / metrics["deaths"]
        else:
            metrics["kda_ratio"] = metrics["kills"] + metrics["assists"]
        
        if metrics["damage_dealt"] > 0:
            metrics["damage_efficiency"] = metrics["damage_dealt"] / (metrics["damage_dealt"] + metrics["damage_taken"])
        else:
            metrics["damage_efficiency"] = 0.5
        
        return metrics
    
    def _identify_strategic_patterns(self, events: List[ReplayEvent]) -> List[str]:
        patterns = []
        
        fight_events = [e for e in events if "fight" in e.event_type.lower()]
        if len(fight_events) > 10:
            patterns.append("frequent_engagement")
        
        retreat_events = [e for e in events if "retreat" in e.event_type.lower()]
        if len(retreat_events) > len(fight_events) * 0.5:
            patterns.append("cautious_playstyle")
        
        objective_events = [e for e in events if "objective" in e.event_type.lower()]
        if len(objective_events) > 5:
            patterns.append("objective_focused")
        
        kill_events = [e for e in events if "kill" in e.event_type.lower()]
        death_events = [e for e in events if "death" in e.event_type.lower()]
        
        if kill_events and death_events:
            if len(kill_events) > len(death_events) * 2:
                patterns.append("aggressive_playstyle")
        
        early_deaths = [e for e in death_events if e.timestamp < 30]
        if len(early_deaths) > 2:
            patterns.append("slow_start")
        
        return patterns
    
    def _describe_event(self, event: ReplayEvent) -> str:
        descriptions = {
            "kill": f"Eliminated enemy at {event.timestamp:.1f}s",
            "death": f"Died at {event.timestamp:.1f}s",
            "objective": f"Captured objective at {event.timestamp:.1f}s",
            "fight": f"Combat at {event.timestamp:.1f}s",
            "retreat": f"Retreated at {event.timestamp:.1f}s",
        }
        
        event_key = event.event_type.lower().split("_")[0]
        return descriptions.get(event_key, f"Event at {event.timestamp:.1f}s")
    
    def _categorize_event(self, event: ReplayEvent) -> str:
        if "kill" in event.event_type.lower():
            return "combat"
        elif "objective" in event.event_type.lower():
            return "objective"
        elif "retreat" in event.event_type.lower():
            return "survival"
        elif "ability" in event.event_type.lower():
            return "ability"
        return "other"
    
    def _find_common_patterns(self, analyses: List[Dict[str, Any]]) -> List[str]:
        return ["aggressive_early", "objective_focus"]
    
    def _identify_improvement_areas(self, analyses: List[Dict[str, Any]]) -> List[str]:
        return ["positioning", "ability_usage"]
    
    def _generate_comparison_recommendation(self, differences: Dict[str, float]) -> str:
        if differences.get("kda_ratio", 0) < 0:
            return "Focus on surviving longer in fights"
        if differences.get("damage_efficiency", 0) < 0:
            return "Work on damage mitigation"
        return "Maintain current performance"
