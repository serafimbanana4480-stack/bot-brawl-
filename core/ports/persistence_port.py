"""
core/ports/persistence_port.py

Persistence Port — abstract interface for saving/loading bot state.

Adapters:
    - FilePersistenceAdapter (state_persistence.py)
    - DatabasePersistenceAdapter (future)
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class PersistencePort(abc.ABC):
    """Abstract persistence interface for bot state checkpointing."""

    @abc.abstractmethod
    def save_state(self, state: Dict[str, Any], label: str = "checkpoint") -> bool:
        """Save current bot state."""
        ...

    @abc.abstractmethod
    def load_state(self, label: str = "checkpoint") -> Optional[Dict[str, Any]]:
        """Load saved bot state."""
        ...

    @abc.abstractmethod
    def list_checkpoints(self) -> Dict[str, Any]:
        """List available checkpoints with metadata."""
        ...
