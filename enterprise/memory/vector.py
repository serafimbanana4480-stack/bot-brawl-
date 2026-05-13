"""Vector Memory - Embedding-based memory storage and retrieval"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import hashlib
import json


@dataclass
class MemoryVector:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float
    access_count: int


class VectorMemory:
    def __init__(self, dimension: int = 384, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, MemoryVector] = {}
        
    def add(self, content: Any, vector: np.ndarray, 
            metadata: Optional[Dict] = None) -> str:
        vector_id = self._generate_id(content)
        
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)
        
        memory = MemoryVector(
            id=vector_id,
            vector=vector / (np.linalg.norm(vector) + 1e-8),
            metadata=metadata or {},
            created_at=0.0,
            access_count=1,
        )
        
        self.vectors[vector_id] = memory
        return vector_id
    
    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if len(query) != self.dimension:
            query = self._pad_or_truncate(query)
        
        query = query / (np.linalg.norm(query) + 1e-8)
        
        similarities = []
        for vid, memory in self.vectors.items():
            if self.metric == "cosine":
                sim = np.dot(query, memory.vector)
            else:
                sim = -np.linalg.norm(query - memory.vector)
            
            similarities.append((vid, float(sim), memory.metadata))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get(self, vector_id: str) -> Optional[MemoryVector]:
        if vector_id in self.vectors:
            memory = self.vectors[vector_id]
            memory.access_count += 1
            return memory
        return None
    
    def delete(self, vector_id: str) -> bool:
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False
    
    def clear(self):
        self.vectors.clear()
    
    def __len__(self):
        return len(self.vectors)
    
    def _generate_id(self, content: Any) -> str:
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _pad_or_truncate(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) < self.dimension:
            return np.pad(vector, (0, self.dimension - len(vector)))
        return vector[:self.dimension]
