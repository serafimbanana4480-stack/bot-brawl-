"""Hybrid Memory System - Combines short-term, long-term, vector, and episodic memory"""

import asyncio
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np


class VectorMemory:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    def add(self, key: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)
        self.vectors[key] = vector
        self.metadata[key] = metadata or {}
        
    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if len(query) != self.dimension:
            query = self._pad_or_truncate(query)
            
        query = query / (np.linalg.norm(query) + 1e-8)
        
        similarities = []
        for key, vector in self.vectors.items():
            normed = vector / (np.linalg.norm(vector) + 1e-8)
            sim = np.dot(query, normed)
            similarities.append((key, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _pad_or_truncate(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) < self.dimension:
            return np.pad(vector, (0, self.dimension - len(vector)))
        return vector[:self.dimension]


class EpisodicMemory:
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: List[Dict[str, Any]] = []
        
    def add_episode(self, episode: Dict[str, Any]):
        episode["timestamp"] = time.time()
        self.episodes.append(episode)
        
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        return self.episodes[-count:]
    
    def search_episodes(self, query: str) -> List[Dict[str, Any]]:
        results = []
        for episode in self.episodes:
            if query.lower() in str(episode).lower():
                results.append(episode)
        return results


class SemanticMemory:
    def __init__(self):
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.concepts: Dict[str, List[str]] = {}
        
    def add_fact(self, subject: str, predicate: str, object: Any, confidence: float = 1.0):
        fact_id = f"{subject}_{predicate}"
        self.facts[fact_id] = {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        
        if subject not in self.concepts:
            self.concepts[subject] = []
        self.concepts[subject].append(fact_id)
    
    def get_facts_about(self, subject: str) -> List[Dict[str, Any]]:
        fact_ids = self.concepts.get(subject, [])
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]
    
    def query(self, predicate: str) -> List[Dict[str, Any]]:
        return [f for f in self.facts.values() if f["predicate"] == predicate]


@dataclass
class MemoryBlock:
    id: str
    content: Any
    memory_type: str
    importance: float
    created_at: float
    access_count: int
    access_history: List[float] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)


class HybridMemorySystem:
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        self.vector_memory = VectorMemory(dimension=config.get("vector_dim", 384))
        self.episodic_memory = EpisodicMemory(max_episodes=config.get("max_episodes", 1000))
        self.semantic_memory = SemanticMemory()
        
        self.short_term: deque = deque(maxlen=config.get("short_term_size", 100))
        self.working_memory: Dict[str, MemoryBlock] = {}
        self.long_term: Dict[str, MemoryBlock] = {}
        
        self.consolidation_threshold = config.get("consolidation_threshold", 50)
        self.importance_threshold = config.get("importance_threshold", 0.7)
        
    async def store(self, content: Any, memory_type: str = "short_term",
                   importance: float = 0.5, tags: List[str] = None,
                   embedding: np.ndarray = None) -> str:
        memory_id = self._generate_id(content)
        
        block = MemoryBlock(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=time.time(),
            access_count=1,
            embeddings=embedding,
            tags=tags or [],
        )
        
        if memory_type == "short_term":
            self.short_term.append(block)
            self.working_memory[memory_id] = block
            
        elif memory_type == "long_term":
            self.long_term[memory_id] = block
            
        elif memory_type == "episodic":
            self.episodic_memory.add_episode({
                "id": memory_id,
                "content": content,
                "importance": importance,
            })
            
        elif memory_type == "semantic" and isinstance(content, dict):
            if "subject" in content and "predicate" in content:
                self.semantic_memory.add_fact(
                    content["subject"],
                    content["predicate"],
                    content.get("object"),
                    importance,
                )
        
        if embedding is not None:
            self.vector_memory.add(memory_id, embedding, {"importance": importance})
        
        if len(self.short_term) >= self.consolidation_threshold:
            await self._consolidate()
        
        return memory_id
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryBlock]:
        if memory_id in self.working_memory:
            block = self.working_memory[memory_id]
            block.access_count += 1
            block.access_history.append(time.time())
            return block
            
        if memory_id in self.long_term:
            block = self.long_term[memory_id]
            block.access_count += 1
            block.access_history.append(time.time())
            return block
            
        return None
    
    async def search(self, query: Any, memory_types: List[str] = None,
                    top_k: int = 10) -> List[MemoryBlock]:
        results = []
        memory_types = memory_types or ["short_term", "working", "long_term"]
        
        if isinstance(query, np.ndarray):
            vector_results = self.vector_memory.search(query, top_k)
            for memory_id, score in vector_results:
                block = await self.retrieve(memory_id)
                if block:
                    block.metadata["relevance_score"] = score
                    results.append(block)
                    
        elif isinstance(query, str):
            for block in list(self.working_memory.values()) + list(self.long_term.values()):
                if query.lower() in str(block.content).lower():
                    results.append(block)
                    
        elif isinstance(query, dict):
            for block in list(self.working_memory.values()) + list(self.long_term.values()):
                if all(k in block.content and block.content[k] == v for k, v in query.items()):
                    results.append(block)
        
        results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return results[:top_k]
    
    async def recall(self, query: str) -> List[Dict[str, Any]]:
        episodic_results = self.episodic_memory.search_episodes(query)
        
        semantic_results = []
        for fact in self.semantic_memory.facts.values():
            if query.lower() in str(fact).lower():
                semantic_results.append(fact)
        
        return {
            "episodes": episodic_results,
            "semantic": semantic_results,
        }
    
    async def _consolidate(self):
        to_consolidate = []
        
        for block in self.short_term:
            if block.importance >= self.importance_threshold:
                to_consolidate.append(block)
        
        for block in to_consolidate:
            memory_id = block.id
            self.long_term[memory_id] = block
            if memory_id in self.working_memory:
                del self.working_memory[memory_id]
        
        self.short_term.clear()
        
        if len(self.long_term) > 10000:
            await self._prune_long_term()
    
    async def _prune_long_term(self):
        items = list(self.long_term.items())
        items.sort(key=lambda x: (x[1].importance, x[1].access_count))
        
        prune_count = len(items) - 8000
        for memory_id, _ in items[:prune_count]:
            del self.long_term[memory_id]
    
    def _generate_id(self, content: Any) -> str:
        content_str = str(content)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "short_term_size": len(self.short_term),
            "working_memory_size": len(self.working_memory),
            "long_term_size": len(self.long_term),
            "vector_memory_size": len(self.vector_memory.vectors),
            "episodic_memory_size": len(self.episodic_memory.episodes),
            "semantic_facts_size": len(self.semantic_memory.facts),
        }
