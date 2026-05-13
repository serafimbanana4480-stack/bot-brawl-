"""Memory Agent - Hybrid memory system management"""

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType, ConfidenceScore
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class MemoryEntry:
    id: str
    content: Dict[str, Any]
    memory_type: str
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    linked_entries: List[str] = field(default_factory=list)


@dataclass
class EpisodicMemory:
    episode_id: str
    events: List[Dict[str, Any]]
    start_time: float
    end_time: Optional[float]
    outcome: Optional[str]
    key_insights: List[str]


class MemoryAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.short_term_memory: deque = deque(maxlen=100)
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.long_term_memory: Dict[str, MemoryEntry] = {}
        
        self.episodic_buffer: List[EpisodicMemory] = []
        self.current_episode: Optional[EpisodicMemory] = None
        
        self.memory_config = {
            "consolidation_threshold": 50,
            "forgetting_threshold": 0.1,
            "importance_threshold": 0.7,
            "max_short_term": 100,
            "max_long_term": 10000,
        }
        
        self.access_patterns: Dict[str, int] = {}
        
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "store")
        
        try:
            if action == "store":
                result = await self._store_memory(message.content)
            elif action == "retrieve":
                result = await self._retrieve_memory(message.content)
            elif action == "consolidate":
                result = await self._consolidate_memory(message.content)
            elif action == "forget":
                result = await self._forget_memory(message.content)
            elif action == "search":
                result = await self._semantic_search(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.9,
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
        current_task = context.get("current_task")
        
        relevant_memories = []
        if current_task:
            relevant_memories = await self._retrieve_relevant_for_task(current_task)
        
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "current_episode_active": self.current_episode is not None,
            "relevant_memories": len(relevant_memories),
            "confidence": 0.9,
        }
    
    async def _store_memory(self, content: Dict[str, Any]) -> Dict[str, Any]:
        memory_type = content.get("type", "short_term")
        data = content.get("data", {})
        importance = content.get("importance", 0.5)
        tags = content.get("tags", [])
        
        memory_id = self._generate_memory_id(data)
        
        entry = MemoryEntry(
            id=memory_id,
            content=data,
            memory_type=memory_type,
            importance=importance,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            tags=tags,
        )
        
        if memory_type == "short_term":
            self.short_term_memory.append(entry)
            self.working_memory[memory_id] = entry
        elif memory_type == "long_term":
            self.long_term_memory[memory_id] = entry
        else:
            self.working_memory[memory_id] = entry
        
        if self.current_episode:
            self.current_episode.events.append({
                "type": "memory_store",
                "memory_id": memory_id,
                "timestamp": time.time(),
            })
        
        if len(self.short_term_memory) >= self.memory_config["consolidation_threshold"]:
            await self._trigger_consolidation()
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.MEMORY_STORED,
            data={"memory_id": memory_id, "type": memory_type},
        ))
        
        return {
            "memory_id": memory_id,
            "stored_in": memory_type,
            "total_memories": len(self.working_memory) + len(self.long_term_memory),
        }
    
    async def _retrieve_memory(self, content: Dict[str, Any]) -> Dict[str, Any]:
        memory_id = content.get("memory_id")
        memory_type = content.get("type")
        
        entry = None
        
        if memory_id:
            if memory_id in self.working_memory:
                entry = self.working_memory[memory_id]
            elif memory_id in self.long_term_memory:
                entry = self.long_term_memory[memory_id]
        elif memory_type:
            entry = self._find_latest_by_type(memory_type)
        
        if entry:
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.access_patterns[memory_id] = self.access_patterns.get(memory_id, 0) + 1
            
            return {
                "found": True,
                "memory": entry.content,
                "memory_id": entry.id,
                "metadata": {
                    "type": entry.memory_type,
                    "importance": entry.importance,
                    "created_at": entry.created_at,
                    "access_count": entry.access_count,
                },
            }
        
        return {"found": False, "error": "Memory not found"}
    
    async def _consolidate_memory(self, content: Dict[str, Any]) -> Dict[str, Any]:
        consolidated_count = 0
        forgotten_count = 0
        
        entries_to_consolidate = []
        
        for entry in list(self.short_term_memory):
            if entry.importance >= self.memory_config["importance_threshold"]:
                entries_to_consolidate.append(entry)
            elif entry.access_count < 3:
                forgotten_count += 1
            else:
                entries_to_consolidate.append(entry)
        
        for entry in entries_to_consolidate:
            self.long_term_memory[entry.id] = entry
            if entry.id in self.working_memory:
                del self.working_memory[entry.id]
            consolidated_count += 1
        
        self.short_term_memory.clear()
        
        if len(self.long_term_memory) > self.memory_config["max_long_term"]:
            await self._prune_long_term_memory()
        
        return {
            "consolidated": consolidated_count,
            "forgotten": forgotten_count,
            "long_term_size": len(self.long_term_memory),
        }
    
    async def _forget_memory(self, content: Dict[str, Any]) -> Dict[str, Any]:
        memory_id = content.get("memory_id")
        reason = content.get("reason", "manual")
        
        forgotten_from = None
        
        if memory_id in self.working_memory:
            del self.working_memory[memory_id]
            forgotten_from = "working"
        elif memory_id in self.long_term_memory:
            del self.long_term_memory[memory_id]
            forgotten_from = "long_term"
        
        self.short_term_memory = deque(
            [e for e in self.short_term_memory if e.id != memory_id],
            maxlen=self.memory_config["max_short_term"]
        )
        
        return {
            "forgotten": forgotten_from is not None,
            "forgotten_from": forgotten_from,
            "reason": reason,
        }
    
    async def _semantic_search(self, content: Dict[str, Any]) -> Dict[str, Any]:
        query = content.get("query", "")
        limit = content.get("limit", 10)
        memory_types = content.get("types", ["short_term", "long_term", "working"])
        
        all_entries = []
        if "short_term" in memory_types or "working" in memory_types:
            all_entries.extend(list(self.working_memory.values()))
        if "long_term" in memory_types:
            all_entries.extend(list(self.long_term_memory.values()))
        
        scored_entries = []
        for entry in all_entries:
            relevance = self._calculate_relevance(entry, query)
            if relevance > 0.3:
                scored_entries.append({
                    "entry": entry,
                    "relevance": relevance,
                })
        
        scored_entries.sort(key=lambda x: (x["relevance"], x["entry"].access_count), reverse=True)
        
        results = []
        for item in scored_entries[:limit]:
            entry = item["entry"]
            results.append({
                "memory_id": entry.id,
                "content": entry.content,
                "relevance": item["relevance"],
                "type": entry.memory_type,
                "importance": entry.importance,
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(scored_entries),
        }
    
    async def start_episode(self, episode_id: str) -> Dict[str, Any]:
        self.current_episode = EpisodicMemory(
            episode_id=episode_id,
            events=[],
            start_time=time.time(),
            end_time=None,
            outcome=None,
            key_insights=[],
        )
        
        return {"episode_id": episode_id, "status": "started"}
    
    async def end_episode(self, outcome: str) -> Dict[str, Any]:
        if not self.current_episode:
            return {"error": "No active episode"}
        
        self.current_episode.end_time = time.time()
        self.current_episode.outcome = outcome
        
        self.episodic_buffer.append(self.current_episode)
        
        if len(self.episodic_buffer) > 100:
            self.episodic_buffer = self.episodic_buffer[-100:]
        
        episode_summary = {
            "episode_id": self.current_episode.episode_id,
            "duration": self.current_episode.end_time - self.current_episode.start_time,
            "event_count": len(self.current_episode.events),
            "outcome": outcome,
        }
        
        self.current_episode = None
        
        return episode_summary
    
    async def retrieve_episode(self, episode_id: str) -> Optional[EpisodicMemory]:
        for episode in self.episodic_buffer:
            if episode.episode_id == episode_id:
                return episode
        return None
    
    async def _trigger_consolidation(self):
        await self._consolidate_memory({})
    
    async def _prune_long_term_memory(self):
        entries_by_importance = sorted(
            self.long_term_memory.items(),
            key=lambda x: (x[1].importance, x[1].access_count),
        )
        
        prune_count = len(self.long_term_memory) - self.memory_config["max_long_term"]
        for entry_id, _ in entries_by_importance[:prune_count]:
            del self.long_term_memory[entry_id]
    
    def _generate_memory_id(self, data: Dict[str, Any]) -> str:
        content_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _find_latest_by_type(self, memory_type: str) -> Optional[MemoryEntry]:
        candidates = [
            e for e in self.short_term_memory 
            if e.memory_type == memory_type
        ]
        if candidates:
            return max(candidates, key=lambda e: e.created_at)
        return None
    
    def _calculate_relevance(self, entry: MemoryEntry, query: str) -> float:
        query_lower = query.lower()
        
        if query_lower in str(entry.content).lower():
            return 0.9
        
        for tag in entry.tags:
            if query_lower in tag.lower():
                return 0.7
        
        return 0.3
    
    async def _retrieve_relevant_for_task(self, task: str) -> List[MemoryEntry]:
        results = await self._semantic_search({"query": task, "limit": 5})
        return results.get("results", [])
