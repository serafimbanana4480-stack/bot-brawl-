"""
neural/distributed_orchestrator.py

Distributed RL: Multi-Bot Coordination.

Um bot é fracassável. Múltiplos bots aprendem mais rápido compartilhando
experiências e um modelo central.

Arquitetura:
- Cada bot executa localmente (coleta experiências)
- Experiências são enviadas para buffer compartilhado (Redis/memória)
- Treinador central consolida e treina modelo global
- Modelo global é sincronizado de volta para todos os bots

Benefício: convergência 5-10x mais rápida.
"""

import json
import pickle
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Uma experiência de um bot para o buffer compartilhado."""
    bot_id: str
    state: List[float]  # Serializado
    action: int
    reward: float
    next_state: List[float]
    done: bool
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Experience":
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in cls.__dataclass_fields__.values()}})


class LocalExperienceBuffer:
    """Buffer local de experiências antes de enviar para central."""

    def __init__(self, max_size: int = 10000, flush_interval: int = 100):
        self.buffer: deque = deque(maxlen=max_size)
        self.flush_interval = flush_interval
        self._lock = threading.Lock()

    def add(self, experience: Experience):
        with self._lock:
            self.buffer.append(experience)

    def should_flush(self) -> bool:
        with self._lock:
            return len(self.buffer) >= self.flush_interval

    def flush(self) -> List[Experience]:
        with self._lock:
            batch = list(self.buffer)
            self.buffer.clear()
            return batch

    def size(self) -> int:
        with self._lock:
            return len(self.buffer)


class DistributedLearningOrchestrator:
    """
    Orquestra múltiplos bots e centraliza aprendizado.

    Modo standalone (sem Redis): usa buffer em memória compartilhada
    via arquivo JSON. Para produção com múltiplas máquinas, substituir
    por Redis ou fila de mensagens.
    """

    def __init__(
        self,
        shared_buffer_path: Path = Path("data/distributed_experiences.jsonl"),
        model_sync_path: Path = Path("models/distributed/latest_model.pt"),
        max_shared_experiences: int = 100000,
    ):
        self.shared_buffer_path = Path(shared_buffer_path)
        self.shared_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_sync_path = Path(model_sync_path)
        self.model_sync_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_shared_experiences = max_shared_experiences

        self._local_buffers: Dict[str, LocalExperienceBuffer] = {}
        self._central_model_state: Optional[bytes] = None
        self._lock = threading.RLock()

        # Métricas
        self._experiences_received = 0
        self._syncs_completed = 0
        self._bots_registered: set = set()

        logger.info("[DISTRIBUTED] Orquestrador inicializado")

    # ------------------------------------------------------------------
    # Registro de bots
    # ------------------------------------------------------------------

    def register_bot(self, bot_id: str, buffer_size: int = 10000):
        """Registra um novo bot no sistema distribuído."""
        with self._lock:
            if bot_id not in self._local_buffers:
                self._local_buffers[bot_id] = LocalExperienceBuffer(max_size=buffer_size)
                self._bots_registered.add(bot_id)
                logger.info("[DISTRIBUTED] Bot registrado: %s", bot_id)

    def unregister_bot(self, bot_id: str):
        """Remove um bot."""
        with self._lock:
            self._local_buffers.pop(bot_id, None)
            self._bots_registered.discard(bot_id)

    # ------------------------------------------------------------------
    # Coleta de experiências
    # ------------------------------------------------------------------

    def push_experience(self, bot_id: str, experience: Experience):
        """
        Bot envia experiência para buffer local.
        """
        buffer = self._local_buffers.get(bot_id)
        if not buffer:
            logger.warning("[DISTRIBUTED] Bot %s não registrado", bot_id)
            return

        buffer.add(experience)

        # Flush automático se buffer cheio
        if buffer.should_flush():
            self._flush_bot_experiences(bot_id)

    def _flush_bot_experiences(self, bot_id: str):
        """Envia experiências do buffer local para o compartilhado."""
        buffer = self._local_buffers.get(bot_id)
        if not buffer:
            return

        batch = buffer.flush()
        if not batch:
            return

        with self._lock:
            try:
                with open(self.shared_buffer_path, "a", encoding="utf-8") as f:
                    for exp in batch:
                        f.write(json.dumps(exp.to_dict(), default=str) + "\n")
                self._experiences_received += len(batch)
                logger.debug("[DISTRIBUTED] %d experiências de %s persistidas", len(batch), bot_id)
            except Exception as e:
                logger.error("[DISTRIBUTED] Erro ao persistir experiências: %s", e)

    # ------------------------------------------------------------------
    # Agregação e treinamento central
    # ------------------------------------------------------------------

    def aggregate_and_train(self, model_trainer: Any, batch_size: int = 10000) -> Optional[Any]:
        """
        Consolida experiências de todos os bots e treina modelo central.

        Args:
            model_trainer: Objeto com método train_on_batch(experiences)
            batch_size: Máximo de experiências por treinamento

        Returns:
            Modelo treinado ou None
        """
        experiences = self._read_shared_experiences(batch_size)
        if not experiences:
            logger.debug("[DISTRIBUTED] Nenhuma experiência para treinar")
            return None

        logger.info("[DISTRIBUTED] Treinando modelo central com %d experiências", len(experiences))

        try:
            # Treinar
            model_trainer.train_on_batch(experiences)

            # Salvar estado do modelo
            model_state = model_trainer.state_dict()
            self._save_central_model(model_state)
            self._syncs_completed += 1

            logger.info("[DISTRIBUTED] Modelo central treinado e salvo")
            return model_trainer

        except Exception as e:
            logger.error("[DISTRIBUTED] Erro no treinamento central: %s", e)
            return None

    def _read_shared_experiences(self, limit: int) -> List[Experience]:
        """Lê experiências do buffer compartilhado."""
        if not self.shared_buffer_path.exists():
            return []

        experiences = []
        try:
            with open(self.shared_buffer_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        experiences.append(Experience.from_dict(data))
                        if len(experiences) >= limit:
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("[DISTRIBUTED] Erro ao ler buffer: %s", e)

        return experiences

    def _save_central_model(self, state_dict: Dict):
        """Salva estado do modelo central."""
        try:
            with open(self.model_sync_path, "wb") as f:
                pickle.dump(state_dict, f)
            self._central_model_state = pickle.dumps(state_dict)
        except Exception as e:
            logger.error("[DISTRIBUTED] Erro ao salvar modelo central: %s", e)

    # ------------------------------------------------------------------
    # Sincronização para bots
    # ------------------------------------------------------------------

    def get_latest_model_state(self) -> Optional[bytes]:
        """
        Retorna estado do modelo central para sincronização.
        Bots chamam periodicamente para atualizar seus modelos locais.
        """
        if self._central_model_state:
            return self._central_model_state

        if self.model_sync_path.exists():
            try:
                with open(self.model_sync_path, "rb") as f:
                    self._central_model_state = f.read()
                return self._central_model_state
            except Exception as e:
                logger.warning("[DISTRIBUTED] Erro ao ler modelo central: %s", e)
        return None

    def sync_bot_model(self, bot_id: str, local_model: Any) -> bool:
        """
        Sincroniza modelo de um bot com o modelo central.
        Retorna True se sincronizou.
        """
        state_data = self.get_latest_model_state()
        if not state_data:
            return False

        try:
            state_dict = pickle.loads(state_data)
            local_model.load_state_dict(state_dict)
            logger.info("[DISTRIBUTED] Bot %s sincronizado com modelo central", bot_id)
            return True
        except Exception as e:
            logger.warning("[DISTRIBUTED] Falha na sincronização de %s: %s", bot_id, e)
            return False

    # ------------------------------------------------------------------
    # Manutenção
    # ------------------------------------------------------------------

    def compact_experiences(self, max_age_hours: float = 24.0):
        """
        Remove experiências antigas do buffer compartilhado.
        """
        if not self.shared_buffer_path.exists():
            return

        cutoff = time.time() - (max_age_hours * 3600)
        kept = 0
        removed = 0

        temp_path = self.shared_buffer_path.with_suffix(".tmp")
        try:
            with open(self.shared_buffer_path, "r", encoding="utf-8") as f_in, \
                 open(temp_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        if data.get("timestamp", 0) > cutoff:
                            f_out.write(line)
                            kept += 1
                        else:
                            removed += 1
                    except Exception:
                        f_out.write(line)
                        kept += 1

            temp_path.replace(self.shared_buffer_path)
            logger.info("[DISTRIBUTED] Compactação: %d mantidas, %d removidas", kept, removed)
        except Exception as e:
            logger.error("[DISTRIBUTED] Erro na compactação: %s", e)
            temp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Status do sistema distribuído."""
        total_local = sum(b.size() for b in self._local_buffers.values())
        return {
            "bots_registered": len(self._bots_registered),
            "bot_ids": list(self._bots_registered),
            "experiences_in_local_buffers": total_local,
            "experiences_received_total": self._experiences_received,
            "syncs_completed": self._syncs_completed,
            "model_available": self._central_model_state is not None,
            "shared_buffer_size_mb": round(self.shared_buffer_path.stat().st_size / (1024 * 1024), 2) if self.shared_buffer_path.exists() else 0,
        }
