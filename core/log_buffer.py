"""
core/log_buffer.py

Buffer circular thread-safe para logs em tempo real.
Alimentado por um logging.Handler custom para capturar todos os logs do bot.
"""

import logging
import threading
import time
from collections import deque
from typing import List, Dict, Optional


class LogBuffer:
    """Buffer circular thread-safe para logs."""

    def __init__(self, max_lines: int = 500):
        self._lock = threading.RLock()
        self._buffer: deque = deque(maxlen=max_lines)
        self._listeners: List[threading.Event] = []
        self._listeners_lock = threading.Lock()

    def append(self, record: logging.LogRecord):
        """Adiciona um log record ao buffer."""
        entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        with self._lock:
            self._buffer.append(entry)
        # Notificar listeners (SSE)
        with self._listeners_lock:
            for ev in list(self._listeners):
                try:
                    ev.set()
                except Exception:
                    pass

    def get_lines(self, limit: int = 100, level: Optional[str] = None,
                  component: Optional[str] = None, search: Optional[str] = None) -> List[Dict]:
        """Retorna linhas do buffer com filtros opcionais."""
        with self._lock:
            lines = list(self._buffer)

        # Aplicar filtros
        if level and level != "ALL":
            level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
            min_level = level_order.get(level, 0)
            lines = [l for l in lines if level_order.get(l["level"], 0) >= min_level]
        if component and component != "ALL":
            lines = [l for l in lines if component.lower() in l["logger"].lower() or component.lower() in l["module"].lower()]
        if search:
            lines = [l for l in lines if search.lower() in l["message"].lower()]

        return lines[-limit:]

    def clear(self):
        """Limpa o buffer."""
        with self._lock:
            self._buffer.clear()

    def add_listener(self, event: threading.Event):
        """Adiciona um listener para notificacoes de novo log (SSE)."""
        with self._listeners_lock:
            self._listeners.append(event)

    def remove_listener(self, event: threading.Event):
        """Remove um listener."""
        with self._listeners_lock:
            if event in self._listeners:
                self._listeners.remove(event)

    def get_stats(self) -> Dict:
        """Retorna estatisticas do buffer."""
        with self._lock:
            total = len(self._buffer)
            if total == 0:
                return {"total": 0, "by_level": {}}
            by_level = {}
            for line in self._buffer:
                by_level[line["level"]] = by_level.get(line["level"], 0) + 1
            return {"total": total, "by_level": by_level}


class DashboardLogHandler(logging.Handler):
    """Logging handler que envia logs para o LogBuffer."""

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord):
        try:
            self.buffer.append(record)
        except Exception:
            self.handleError(record)


# Singleton global para o projeto
_global_log_buffer: Optional[LogBuffer] = None
_global_handler: Optional[DashboardLogHandler] = None


def get_log_buffer(max_lines: int = 500) -> LogBuffer:
    """Retorna (ou cria) o LogBuffer global."""
    global _global_log_buffer
    if _global_log_buffer is None:
        _global_log_buffer = LogBuffer(max_lines=max_lines)
    return _global_log_buffer


def install_log_buffer(root_logger_name: str = "", max_lines: int = 500) -> LogBuffer:
    """Instala o LogBuffer no root logger (ou logger especificado)."""
    global _global_log_buffer, _global_handler
    buf = get_log_buffer(max_lines)
    if _global_handler is None:
        _global_handler = DashboardLogHandler(buf)
        _global_handler.setLevel(logging.DEBUG)
        if root_logger_name:
            logger = logging.getLogger(root_logger_name)
        else:
            logger = logging.getLogger()
        logger.addHandler(_global_handler)
    return buf


def uninstall_log_buffer():
    """Remove o handler do logger."""
    global _global_handler
    if _global_handler is not None:
        logging.getLogger().removeHandler(_global_handler)
        _global_handler = None
