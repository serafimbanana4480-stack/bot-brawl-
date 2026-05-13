"""Navigation Agent - Pathfinding, movement planning and spatial awareness"""

import asyncio
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import heapq

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Waypoint:
    position: Tuple[float, float]
    arrived: bool = False
    wait_time: float = 0.0


@dataclass
class Path:
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time: float
    obstacles: List[Tuple[float, float]]


class NavigationAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.current_path: Optional[Path] = None
        self.known_obstacles: List[Tuple[float, float]] = []
        self.safety_zones: List[Tuple[float, float, float]] = []
        self.last_position: Optional[Tuple[float, float]] = None
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "navigate")
        
        try:
            if action == "navigate":
                result = await self._navigate_to_target(message.content)
            elif action == "pathfind":
                result = await self._calculate_path(message.content)
            elif action == "avoid":
                result = await self._avoid_obstacle(message.content)
            elif action == "patrol":
                result = await self._patrol_area(message.content)
            elif action == "retreat":
                result = await self._calculate_retreat_path(message.content)
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
        current_pos = context.get("current_position", (0, 0))
        target_pos = context.get("target_position")
        obstacles = context.get("obstacles", [])
        
        if not target_pos:
            return {
                "navigation_status": "idle",
                "path_available": False,
                "confidence": 0.0,
            }
        
        distance = self._calculate_distance(current_pos, target_pos)
        path_clear = self._is_path_clear(current_pos, target_pos, obstacles)
        
        return {
            "navigation_status": "active" if self.current_path else "idle",
            "distance_to_target": distance,
            "path_clear": path_clear,
            "estimated_time": distance / 200.0,
            "confidence": 0.8 if self.current_path else 0.3,
        }
    
    async def _navigate_to_target(self, content: Dict[str, Any]) -> Dict[str, Any]:
        target = content.get("target")
        obstacles = content.get("obstacles", [])
        preferred_route = content.get("preferred_route")
        
        if not target:
            return {"error": "No target specified"}
        
        current_pos = content.get("current_position", self.last_position or (0, 0))
        self.last_position = current_pos
        
        path = self._a_star_pathfind(current_pos, target, obstacles)
        
        if path:
            self.current_path = path
            next_waypoint = path.waypoints[0] if path.waypoints else None
            
            return {
                "status": "navigating",
                "path": {
                    "waypoints": [(w.position[0], w.position[1]) for w in path.waypoints],
                    "total_distance": path.total_distance,
                    "estimated_time": path.estimated_time,
                },
                "next_waypoint": (next_waypoint.position[0], next_waypoint.position[1]) if next_waypoint else None,
                "waypoint_count": len(path.waypoints),
            }
        
        return {
            "status": "no_path_found",
            "obstacles_encountered": len(obstacles),
        }
    
    async def _calculate_path(self, content: Dict[str, Any]) -> Dict[str, Any]:
        start = content.get("start", self.last_position or (0, 0))
        end = content.get("end")
        obstacles = content.get("obstacles", [])
        
        if not end:
            return {"error": "No destination specified"}
        
        path = self._a_star_pathfind(start, end, obstacles)
        
        if path:
            return {
                "path_found": True,
                "waypoints": [(w.position[0], w.position[1]) for w in path.waypoints],
                "total_distance": path.total_distance,
                "estimated_time": path.estimated_time,
                "obstacles_to_avoid": path.obstacles,
            }
        
        return {
            "path_found": False,
            "alternative_routes": self._find_alternative_routes(start, end, obstacles),
        }
    
    async def _avoid_obstacle(self, content: Dict[str, Any]) -> Dict[str, Any]:
        obstacle_position = content.get("obstacle_position")
        current_position = content.get("current_position", self.last_position or (0, 0))
        target = content.get("target")
        
        self.known_obstacles.append(obstacle_position)
        
        if target:
            avoidance_point = self._calculate_avoidance_point(
                current_position, obstacle_position, target
            )
            
            path = self._a_star_pathfind(
                current_position, target, self.known_obstacles
            )
            
            return {
                "status": "avoiding",
                "obstacle_position": obstacle_position,
                "avoidance_point": avoidance_point,
                "path": path.waypoints if path else None,
            }
        
        return {
            "status": "obstacle_acknowledged",
            "obstacle_position": obstacle_position,
        }
    
    async def _patrol_area(self, content: Dict[str, Any]) -> Dict[str, Any]:
        patrol_points = content.get("patrol_points", [])
        current_position = content.get("current_position", self.last_position or (0, 0))
        
        if not patrol_points:
            patrol_points = self._generate_default_patrol(current_position)
        
        patrol_path = []
        for point in patrol_points:
            waypoints = self._a_star_pathfind(
                patrol_path[-1] if patrol_path else current_position,
                point,
                self.known_obstacles
            )
            if waypoints:
                patrol_path.extend(waypoints.waypoints)
        
        return {
            "status": "patrolling",
            "patrol_points": patrol_points,
            "patrol_path": [(w.position[0], w.position[1]) for w in patrol_path],
            "estimated_duration": len(patrol_path) * 2.0,
        }
    
    async def _calculate_retreat_path(self, content: Dict[str, Any]) -> Dict[str, Any]:
        threat_position = content.get("threat_position")
        current_position = content.get("current_position", self.last_position or (0, 0))
        
        safe_positions = self._find_safe_positions(current_position, threat_position)
        
        if safe_positions:
            best_safety = max(safe_positions, key=lambda p: p[2])
            path = self._a_star_pathfind(
                current_position, (best_safety[0], best_safety[1]), self.known_obstacles
            )
            
            return {
                "status": "retreating",
                "destination": (best_safety[0], best_safety[1]),
                "safety_score": best_safety[2],
                "path": [(w.position[0], w.position[1]) for w in path.waypoints] if path else None,
            }
        
        away_direction = (
            current_position[0] - (threat_position[0] if threat_position else 0),
            current_position[1] - (threat_position[1] if threat_position else 0),
        )
        
        return {
            "status": "emergency_retreat",
            "direction": away_direction,
            "fallback": True,
        }
    
    def _a_star_pathfind(self, start: Tuple[float, float], 
                        goal: Tuple[float, float],
                        obstacles: List[Tuple[float, float]]) -> Optional[Path]:
        if self._calculate_distance(start, goal) < 10:
            return Path(
                waypoints=[Waypoint(position=goal)],
                total_distance=0,
                estimated_time=0,
                obstacles=[],
            )
        
        waypoints = [
            Waypoint(position=start),
            Waypoint(position=goal),
        ]
        
        direct_distance = self._calculate_distance(start, goal)
        
        return Path(
            waypoints=waypoints,
            total_distance=direct_distance,
            estimated_time=direct_distance / 200.0,
            obstacles=[],
        )
    
    def _find_alternative_routes(self, start: Tuple[float, float],
                               goal: Tuple[float, float],
                               obstacles: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        return [
            [start, (start[0] + 100, start[1]), goal],
            [start, (start[0], start[1] + 100), goal],
        ]
    
    def _calculate_avoidance_point(self, current: Tuple[float, float],
                                  obstacle: Tuple[float, float],
                                  target: Tuple[float, float]) -> Tuple[float, float]:
        obstacle_vector = (
            obstacle[0] - current[0],
            obstacle[1] - current[1],
        )
        
        length = math.sqrt(obstacle_vector[0]**2 + obstacle_vector[1]**2)
        if length > 0:
            normalized = (obstacle_vector[0] / length, obstacle_vector[1] / length)
            perpendicular = (-normalized[1], normalized[0])
            
            avoidance_distance = 50.0
            avoidance = (
                current[0] + perpendicular[0] * avoidance_distance,
                current[1] + perpendicular[1] * avoidance_distance,
            )
            
            return avoidance
        
        return (current[0] + 50, current[1])
    
    def _generate_default_patrol(self, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        return [
            (center[0] + 100, center[1]),
            (center[0] + 100, center[1] + 100),
            (center[0], center[1] + 100),
            (center[0] - 100, center[1]),
        ]
    
    def _find_safe_positions(self, current: Tuple[float, float],
                           threat: Optional[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        if not threat:
            return [(current[0], current[1], 1.0)]
        
        safe_positions = []
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in angles:
            rad = math.radians(angle)
            x = current[0] + math.cos(rad) * 150
            y = current[1] + math.sin(rad) * 150
            
            distance_from_threat = self._calculate_distance((x, y), threat)
            safety_score = min(1.0, distance_from_threat / 300.0)
            
            safe_positions.append((x, y, safety_score))
        
        return safe_positions
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _is_path_clear(self, start: Tuple[float, float],
                     end: Tuple[float, float],
                     obstacles: List[Tuple[float, float]]) -> bool:
        return True
