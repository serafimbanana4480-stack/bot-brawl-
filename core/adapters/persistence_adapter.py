"""
core/adapters/persistence_adapter.py

Adapter: StatePersistence -> PersistencePort
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.ports.persistence_port import PersistencePort

logger = logging.getLogger(__name__)


class PersistenceAdapter(PersistencePort):
    """Wraps StatePersistence to satisfy PersistencePort."""

    def __init__(self, persistence=None):
        self._persist = persistence

    def save_state(self, state: Dict[str, Any], label: str = "checkpoint") -> bool:
        if self._persist is None:
            return False
        try:
            if hasattr(self._persist, "save_checkpoint"):
                self._persist.save_checkpoint(state, label)
                return True
        except Exception as e:
            logger.warning(f"[PERSISTENCE_ADAPTER] Save failed: {e}")
        return False

    def load_state(self, label: str = "checkpoint") -> Optional[Dict[str, Any]]:
        if self._persist is None:
            return None
        try:
            if hasattr(self._persist, "load_checkpoint"):
                return self._persist.load_checkpoint(label)
        except Exception as e:
            logger.warning(f"[PERSISTENCE_ADAPTER] Load failed: {e}")
        return None

    def list_checkpoints(self) -> Dict[str, Any]:
        if self._persist is None:
            return {}
        try:
            if hasattr(self._persist, "list_checkpoints"):
                return self._persist.list_checkpoints()
        except Exception:
            pass
        return {}
