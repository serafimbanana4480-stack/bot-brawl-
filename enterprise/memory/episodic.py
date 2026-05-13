"""Episodic Memory - Event-based memory storage"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Episode:
    id: str
    events: List[Dict[str, Any]]
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
    importance: float = 0.5


class EpisodicMemory:
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: Optional[Episode] = None
        self.episode_counter = 0
    
    def start_episode(self, context: Optional[Dict] = None) -> str:
        self.episode_counter += 1
        episode_id = f"episode_{self.episode_counter}"
        
        self.current_episode = Episode(
            id=episode_id,
            events=[],
            start_time=time.time(),
            context=context,
        )
        
        return episode_id
    
    def add_event(self, event: Dict[str, Any]):
        if self.current_episode:
            self.current_episode.events.append({
                "timestamp": time.time(),
                **event,
            })
    
    def end_episode(self, outcome: str, key_insights: Optional[List[str]] = None):
        if self.current_episode:
            self.current_episode.end_time = time.time()
            self.current_episode.outcome = outcome
            if key_insights:
                self.current_episode.key_insights = key_insights
            
            self.episodes.append(self.current_episode)
            self.current_episode = None
    
    def get_recent(self, count: int = 10) -> List[Episode]:
        episodes = list(self.episodes)
        return episodes[-count:]
    
    def get_by_outcome(self, outcome: str) -> List[Episode]:
        return [e for e in self.episodes if e.outcome == outcome]
    
    def search(self, query: str) -> List[Episode]:
        results = []
        query_lower = query.lower()
        
        for episode in self.episodes:
            if query_lower in str(episode.events).lower():
                results.append(episode)
            elif any(query_lower in insight.lower() for insight in episode.key_insights):
                results.append(episode)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        episodes_list = list(self.episodes)
        
        outcomes = {}
        for ep in episodes_list:
            outcome = ep.outcome or "unknown"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        total_events = sum(len(ep.events) for ep in episodes_list)
        
        return {
            "total_episodes": len(episodes_list),
            "outcomes": outcomes,
            "total_events": total_events,
            "avg_events_per_episode": total_events / len(episodes_list) if episodes_list else 0,
            "current_episode_active": self.current_episode is not None,
        }
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = None
