"""Reflection/Critic Agent - Self-evaluation and improvement analysis"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Reflection:
    id: str
    timestamp: float
    context: Dict[str, Any]
    actions_taken: List[str]
    outcomes: Dict[str, Any]
    analysis: Dict[str, Any]
    improvements: List[str]


class ReflectionAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.reflection_history: List[Reflection] = []
        self.critic_scores: Dict[str, float] = {}
        self.improvement_suggestions: List[Dict[str, Any]] = []
        
        self.evaluation_criteria = {
            "decision_quality": 1.0,
            "execution_efficiency": 1.0,
            "adaptability": 1.0,
            "resource_usage": 1.0,
            "goal_alignment": 1.0,
        }
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "reflect")
        
        try:
            if action == "reflect":
                result = await self._perform_reflection(message.content)
            elif action == "critique":
                result = await self._critique_action(message.content)
            elif action == "improve":
                result = await self._generate_improvements(message.content)
            elif action == "evaluate":
                result = await self._evaluate_performance(message.content)
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
        recent_reflections = context.get("recent_reflections", [])
        performance_data = context.get("performance_data", {})
        
        avg_quality = self._calculate_average_quality(recent_reflections)
        
        identified_issues = self._identify_issues(recent_reflections)
        
        return {
            "reflection_count": len(self.reflection_history),
            "average_quality": avg_quality,
            "identified_issues": identified_issues,
            "pending_improvements": len(self.improvement_suggestions),
            "confidence": 0.85,
        }
    
    async def _perform_reflection(self, content: Dict[str, Any]) -> Dict[str, Any]:
        context = content.get("context", {})
        actions = content.get("actions", [])
        outcomes = content.get("outcomes", {})
        
        analysis = self._analyze_situation(context, actions, outcomes)
        
        improvements = self._generate_improvements_from_analysis(analysis)
        
        reflection = Reflection(
            id=f"reflection_{len(self.reflection_history)}",
            timestamp=time.time(),
            context=context,
            actions_taken=actions,
            outcomes=outcomes,
            analysis=analysis,
            improvements=improvements,
        )
        
        self.reflection_history.append(reflection)
        
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.LEARNING_UPDATE,
            data={
                "reflection_id": reflection.id,
                "quality_score": analysis.get("overall_quality", 0.5),
                "improvements_count": len(improvements),
            },
        ))
        
        return {
            "reflection_id": reflection.id,
            "analysis": analysis,
            "improvements": improvements,
            "quality_score": analysis.get("overall_quality", 0.5),
        }
    
    async def _critique_action(self, content: Dict[str, Any]) -> Dict[str, Any]:
        action = content.get("action", {})
        expected_outcome = content.get("expected_outcome", {})
        actual_outcome = content.get("actual_outcome", {})
        
        critique = self._detailed_critique(action, expected_outcome, actual_outcome)
        
        return {
            "action_id": action.get("id"),
            "critique": critique,
            "scores": critique.get("scores", {}),
            "recommendations": critique.get("recommendations", []),
        }
    
    async def _generate_improvements(self, content: Dict[str, Any]) -> Dict[str, Any]:
        focus_area = content.get("focus_area", "all")
        num_improvements = content.get("count", 5)
        
        if focus_area == "all":
            relevant_reflections = self.reflection_history
        else:
            relevant_reflections = [
                r for r in self.reflection_history
                if focus_area in str(r.analysis)
            ]
        
        improvements = []
        for reflection in relevant_reflections[-10:]:
            for improvement in reflection.improvements:
                if len(improvements) >= num_improvements:
                    break
                improvements.append({
                    "text": improvement,
                    "source_reflection": reflection.id,
                    "priority": reflection.analysis.get("overall_quality", 0.5),
                })
        
        improvements.sort(key=lambda x: x["priority"], reverse=True)
        
        return {
            "focus_area": focus_area,
            "improvements": improvements[:num_improvements],
            "total_available": len(improvements),
        }
    
    async def _evaluate_performance(self, content: Dict[str, Any]) -> Dict[str, Any]:
        time_period = content.get("period", "recent")
        metrics = content.get("metrics", {})
        
        if time_period == "recent":
            reflections = self.reflection_history[-20:]
        elif time_period == "all":
            reflections = self.reflection_history
        else:
            reflections = self.reflection_history[-10:]
        
        if not reflections:
            return {
                "status": "no_data",
                "message": "No reflections available for evaluation",
            }
        
        avg_scores = {
            "decision_quality": 0.0,
            "execution_efficiency": 0.0,
            "adaptability": 0.0,
            "resource_usage": 0.0,
            "goal_alignment": 0.0,
        }
        
        for reflection in reflections:
            for key in avg_scores:
                if key in reflection.analysis:
                    avg_scores[key] += reflection.analysis[key]
        
        count = len(reflections)
        for key in avg_scores:
            avg_scores[key] /= count
        
        overall_score = sum(avg_scores.values()) / len(avg_scores)
        
        trend = "stable"
        if len(reflections) >= 5:
            recent_avg = sum(r.analysis.get("overall_quality", 0.5) for r in reflections[-5:]) / 5
            older_avg = sum(r.analysis.get("overall_quality", 0.5) for r in reflections[-10:-5]) / 5 if len(reflections) > 5 else recent_avg
            if recent_avg > older_avg + 0.05:
                trend = "improving"
            elif recent_avg < older_avg - 0.05:
                trend = "declining"
        
        return {
            "period": time_period,
            "reflections_analyzed": len(reflections),
            "average_scores": avg_scores,
            "overall_score": overall_score,
            "trend": trend,
            "evaluation_criteria": self.evaluation_criteria,
        }
    
    def _analyze_situation(self, context: Dict[str, Any],
                          actions: List[str],
                          outcomes: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {}
        
        expected_outcome = context.get("expected_outcome", {})
        actual_outcome = outcomes
        
        outcome_match = self._calculate_outcome_match(expected_outcome, actual_outcome)
        analysis["outcome_match"] = outcome_match
        
        decision_quality = self._evaluate_decision_quality(context, actions)
        analysis["decision_quality"] = decision_quality
        
        execution_efficiency = self._evaluate_execution_efficiency(actions, outcomes)
        analysis["execution_efficiency"] = execution_efficiency
        
        adaptability = self._evaluate_adaptability(context, outcomes)
        analysis["adaptability"] = adaptability
        
        resource_usage = self._evaluate_resource_usage(context, outcomes)
        analysis["resource_usage"] = resource_usage
        
        goal_alignment = self._evaluate_goal_alignment(context, actions, outcomes)
        analysis["goal_alignment"] = goal_alignment
        
        analysis["overall_quality"] = (
            decision_quality * 0.3 +
            execution_efficiency * 0.2 +
            adaptability * 0.2 +
            resource_usage * 0.15 +
            goal_alignment * 0.15
        )
        
        return analysis
    
    def _generate_improvements_from_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        improvements = []
        
        if analysis.get("decision_quality", 1.0) < 0.7:
            improvements.append("Consider gathering more information before decisions")
            improvements.append("Review similar past situations for reference")
        
        if analysis.get("execution_efficiency", 1.0) < 0.7:
            improvements.append("Optimize action sequencing")
            improvements.append("Reduce unnecessary delays in execution")
        
        if analysis.get("adaptability", 1.0) < 0.7:
            improvements.append("Develop contingency plans for unexpected changes")
            improvements.append("Improve response time to environmental changes")
        
        if analysis.get("resource_usage", 1.0) < 0.7:
            improvements.append("Optimize resource allocation")
            improvements.append("Avoid wasteful resource expenditure")
        
        if analysis.get("goal_alignment", 1.0) < 0.7:
            improvements.append("Ensure actions directly support objectives")
            improvements.append("Re-evaluate priority of current goals")
        
        return improvements
    
    def _detailed_critique(self, action: Dict[str, Any],
                         expected_outcome: Dict[str, Any],
                         actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        critique = {
            "scores": {},
            "recommendations": [],
            "strengths": [],
            "weaknesses": [],
        }
        
        match_score = self._calculate_outcome_match(expected_outcome, actual_outcome)
        critique["scores"]["outcome_match"] = match_score
        
        if match_score > 0.8:
            critique["strengths"].append("Actions aligned well with expectations")
        elif match_score < 0.5:
            critique["weaknesses"].append("Significant deviation from expected outcome")
            critique["recommendations"].append("Review decision-making process")
        
        return critique
    
    def _calculate_outcome_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
        if not expected:
            return 0.5
        
        matches = 0
        total = len(expected)
        
        for key, value in expected.items():
            if key in actual:
                if actual[key] == value:
                    matches += 1
                elif isinstance(value, (int, float)) and isinstance(actual[key], (int, float)):
                    if abs(actual[key] - value) / max(value, 1) < 0.2:
                        matches += 0.5
        
        return matches / total if total > 0 else 0.5
    
    def _evaluate_decision_quality(self, context: Dict, actions: List) -> float:
        if not actions:
            return 0.3
        
        info_quality = context.get("information_quality", 0.7)
        options_considered = context.get("options_considered", 1)
        
        options_factor = min(1.0, options_considered / 3.0)
        
        return (info_quality * 0.6 + options_factor * 0.4)
    
    def _evaluate_execution_efficiency(self, actions: List, outcomes: Dict) -> float:
        if not actions:
            return 0.3
        
        time_taken = outcomes.get("time_taken", 100)
        expected_time = outcomes.get("expected_time", 100)
        
        if expected_time > 0:
            efficiency = min(1.0, expected_time / time_taken)
        else:
            efficiency = 0.5
        
        return efficiency
    
    def _evaluate_adaptability(self, context: Dict, outcomes: Dict) -> float:
        changes_encountered = context.get("changes_encountered", 0)
        successful_adaptations = outcomes.get("successful_adaptations", 0)
        
        if changes_encountered == 0:
            return 0.8
        
        return min(1.0, successful_adaptations / changes_encountered)
    
    def _evaluate_resource_usage(self, context: Dict, outcomes: Dict) -> float:
        resources_used = outcomes.get("resources_used", 100)
        resources_available = context.get("resources_available", 100)
        
        if resources_available == 0:
            return 0.5
        
        usage_ratio = resources_used / resources_available
        
        if usage_ratio < 0.5:
            return 0.6
        elif usage_ratio < 0.8:
            return 1.0
        elif usage_ratio < 1.0:
            return 0.8
        else:
            return 0.4
    
    def _evaluate_goal_alignment(self, context: Dict, actions: List, outcomes: Dict) -> float:
        goal = context.get("goal", {})
        goal_achieved = outcomes.get("goal_achieved", False)
        
        if goal_achieved:
            return 0.9
        
        partial_achievement = outcomes.get("partial_achievement", 0.0)
        return partial_achievement * 0.8
    
    def _calculate_average_quality(self, reflections: List[Reflection]) -> float:
        if not reflections:
            return 0.0
        return sum(r.analysis.get("overall_quality", 0.5) for r in reflections) / len(reflections)
    
    def _identify_issues(self, reflections: List[Reflection]) -> List[str]:
        issues = []
        
        low_quality_reflections = [r for r in reflections if r.analysis.get("overall_quality", 1.0) < 0.6]
        if len(low_quality_reflections) > len(reflections) * 0.3:
            issues.append("High frequency of low-quality decisions")
        
        return issues
