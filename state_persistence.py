"""
state_persistence.py

Sistema de persistência de estado para recuperação de falhas.
Guarda o estado atual do bot para poder restaurá-lo após falhas.

Funcionalidades:
- Guardar estado completo periodicamente
- Restaurar estado após restart
- Checkpoints de recovery
- Histórico de estados
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class StateCheckpoint:
    """Checkpoint de estado guardado."""
    timestamp: float
    session_id: str
    current_state: str
    match_state: Dict[str, Any]
    brawler_queue: Dict[str, Any]
    statistics: Dict[str, Any]
    rl_state: Optional[Dict] = None
    version: str = "1.0"


class StatePersistence:
    """Sistema de persistência de estado."""

    def __init__(self, save_dir: Path = Path("pylaai_workspace/state")):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.save_dir / "checkpoint.json"
        self.history_file = self.save_dir / "state_history.json"
        self._lock = threading.Lock()
        self._last_save_time = 0
        self._save_interval = 30.0  # Guardar a cada 30 segundos
        self._current_checkpoint: Optional[StateCheckpoint] = None
        self._session_id = f"{int(time.time())}"

    def save_checkpoint(
        self,
        current_state: str,
        match_state: Dict[str, Any],
        brawler_queue: Dict[str, Any],
        statistics: Dict[str, Any],
        rl_state: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        """
        Guarda um checkpoint do estado atual.

        Args:
            current_state: Estado atual do jogo
            match_state: Informação sobre a partida atual
            brawler_queue: Estado da fila de brawlers
            statistics: Estatísticas da sessão
            rl_state: Estado do RL engine (opcional)
            force: Forçar gravação mesmo se intervalo não passou

        Returns:
            True se guardou com sucesso
        """
        with self._lock:
            now = time.time()

            # Verificar intervalo mínimo
            if not force and (now - self._last_save_time) < self._save_interval:
                return False

            try:
                checkpoint = StateCheckpoint(
                    timestamp=now,
                    session_id=self._session_id,
                    current_state=current_state,
                    match_state=match_state,
                    brawler_queue=brawler_queue,
                    statistics=statistics,
                    rl_state=rl_state,
                )

                # Guardar checkpoint principal
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(checkpoint), f, indent=2)

                # Atualizar histórico
                self._append_to_history(checkpoint)

                self._current_checkpoint = checkpoint
                self._last_save_time = now
                logger.debug(f"[STATE_PERSIST] Checkpoint guardado: {current_state}")
                return True

            except Exception as e:
                logger.error(f"[STATE_PERSIST] Erro ao guardar checkpoint: {e}")
                return False

    def _append_to_history(self, checkpoint: StateCheckpoint):
        """Adiciona checkpoint ao histórico."""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            # Manter apenas últimos 100 checkpoints
            history.append(asdict(checkpoint))
            if len(history) > 100:
                history = history[-100:]

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.warning(f"[STATE_PERSIST] Erro ao guardar histórico: {e}")

    def load_checkpoint(self) -> Optional[StateCheckpoint]:
        """Carrega o último checkpoint guardado."""
        with self._lock:
            if not self.checkpoint_file.exists():
                logger.debug("[STATE_PERSIST] Nenhum checkpoint encontrado")
                return None

            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                checkpoint = StateCheckpoint(**data)
                self._current_checkpoint = checkpoint
                logger.info(f"[STATE_PERSIST] Checkpoint carregado: {checkpoint.current_state}")
                return checkpoint

            except Exception as e:
                logger.error(f"[STATE_PERSIST] Erro ao carregar checkpoint: {e}")
                return None

    def get_last_state(self) -> Optional[str]:
        """Retorna o último estado guardado."""
        checkpoint = self.load_checkpoint()
        return checkpoint.current_state if checkpoint else None

    def get_state_age(self) -> float:
        """Retorna há quanto tempo (segundos) o último checkpoint foi guardado."""
        if not self.checkpoint_file.exists():
            return float('inf')

        try:
            stat = self.checkpoint_file.stat()
            return time.time() - stat.st_mtime
        except Exception:
            return float('inf')

    def is_recent(self, max_age_seconds: float = 60.0) -> bool:
        """Verifica se existe um checkpoint recente."""
        return self.get_state_age() < max_age_seconds

    def clear(self) -> bool:
        """Apaga todos os dados de persistência."""
        with self._lock:
            try:
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                if self.history_file.exists():
                    self.history_file.unlink()
                self._current_checkpoint = None
                logger.info("[STATE_PERSIST] Dados de persistência limpos")
                return True
            except Exception as e:
                logger.error(f"[STATE_PERSIST] Erro ao limpar persistência: {e}")
                return False

    def get_history(self, limit: int = 10) -> list:
        """Retorna o histórico de estados."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-limit:]
        except Exception as e:
            logger.error(f"[STATE_PERSIST] Erro ao carregar histórico: {e}")
            return []

    def start_new_session(self) -> str:
        """Inicia uma nova sessão (novo session_id)."""
        with self._lock:
            self._session_id = f"{int(time.time())}"
            logger.info(f"[STATE_PERSIST] Nova sessão: {self._session_id}")
            return self._session_id

    def get_current_session_id(self) -> str:
        """Retorna o ID da sessão atual."""
        return self._session_id