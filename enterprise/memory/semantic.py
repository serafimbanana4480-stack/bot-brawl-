"""Semantic Memory - Fact-based knowledge storage"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import time
import hashlib


@dataclass
class Fact:
    subject: str
    predicate: str
    object: Any
    confidence: float
    timestamp: float
    source: Optional[str] = None


class SemanticMemory:
    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.subjects: Dict[str, Set[str]] = {}
        self.predicates: Dict[str, Set[str]] = {}
        self.concepts: Dict[str, List[str]] = {}
    
    def add_fact(self, subject: str, predicate: str, obj: Any,
                  confidence: float = 1.0, source: Optional[str] = None) -> str:
        fact_id = self._generate_fact_id(subject, predicate)
        
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            timestamp=time.time(),
            source=source,
        )
        
        self.facts[fact_id] = fact
        
        if subject not in self.subjects:
            self.subjects[subject] = set()
        self.subjects[subject].add(fact_id)
        
        if predicate not in self.predicates:
            self.predicates[predicate] = set()
        self.predicates[predicate].add(fact_id)
        
        concept_key = subject.split("_")[0] if "_" in subject else subject
        if concept_key not in self.concepts:
            self.concepts[concept_key] = []
        self.concepts[concept_key].append(fact_id)
        
        return fact_id
    
    def get_facts_about(self, subject: str) -> List[Fact]:
        fact_ids = self.subjects.get(subject, set())
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]
    
    def query_predicate(self, predicate: str) -> List[Fact]:
        fact_ids = self.predicates.get(predicate, set())
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]
    
    def find_relationships(self, subject: str, depth: int = 1) -> Dict[str, List]:
        relationships = {
            "directly_related": [],
            "inferred": [],
        }
        
        direct_facts = self.get_facts_about(subject)
        for fact in direct_facts:
            relationships["directly_related"].append({
                "predicate": fact.predicate,
                "object": fact.object,
            })
        
        if depth > 1:
            for fact in direct_facts:
                if isinstance(fact.object, str):
                    related = self.get_facts_about(fact.object)
                    for rel in related:
                        relationships["inferred"].append({
                            "via": fact.predicate,
                            "predicate": rel.predicate,
                            "object": rel.object,
                        })
        
        return relationships
    
    def search(self, query: str) -> List[Fact]:
        results = []
        query_lower = query.lower()
        
        for fact in self.facts.values():
            if (query_lower in fact.subject.lower() or
                query_lower in fact.predicate.lower() or
                query_lower in str(fact.object).lower()):
                results.append(fact)
        
        return results
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        nodes = []
        edges = []
        
        for fact in self.facts.values():
            if fact.subject not in [n["id"] for n in nodes]:
                nodes.append({"id": fact.subject, "type": "subject"})
            
            if fact.predicate not in [n["id"] for n in nodes]:
                nodes.append({"id": fact.predicate, "type": "predicate"})
            
            edges.append({
                "from": fact.subject,
                "to": fact.predicate,
                "label": str(fact.object)[:50],
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def __len__(self):
        return len(self.facts)
    
    def _generate_fact_id(self, subject: str, predicate: str) -> str:
        key = f"{subject}_{predicate}_{time.time()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
