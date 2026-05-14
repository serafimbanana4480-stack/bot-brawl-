"""
core/log_buffer.py

Buffer circular thread-safe para logs em tempo real.
Alimentado por um logging.Handler custom para capturar todos os logs do bot.

Features:
- In-memory circular buffer for real-time dashboard consumption
- Rotating file writer for persistent log storage with size-based rotation
- SSE listener notifications for live updates
"""

import gzip
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional


class RotatingLogWriter:
    """Thread-safe rotating file writer for log entries.
    
    Writes log entries to a file, rotating when the file exceeds
    max_file_size_mb. Keeps up to max_backup_files compressed backups.
    """

    def __init__(
        self,
        log_dir: Path,
        base_name: str = "bot.log",
        max_file_size_mb: float = 10.0,
        max_backup_files: int = 5,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_backup_files = max_backup_files
        self._lock = threading.Lock()
        self._current_file = None
        self._current_size = 0
        self._open_file()

    def _open_file(self):
        """Open the current log file for appending."""
        filepath = self.log_dir / self.base_name
        # Check if rotation needed on open
        if filepath.exists() and filepath.stat().st_size >= self.max_file_size_bytes:
            self._rotate()
        self._current_file = open(filepath, "a", encoding="utf-8")
        self._current_size = filepath.stat().st_size if filepath.exists() else 0

    def _rotate(self):
        """Rotate log files: compress current to .gz, shift backups, delete oldest."""
        # Close current file
        if self._current_file:
            try:
                self._current_file.close()
            except Exception:
                pass

        filepath = self.log_dir / self.base_name
        if not filepath.exists():
            return

        # Shift existing backups
        for i in range(self.max_backup_files - 1, 0, -1):
            src = self.log_dir / f"{self.base_name}.{i}.gz"
            dst = self.log_dir / f"{self.base_name}.{i + 1}.gz"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

        # Compress current file to .1.gz
        backup_path = self.log_dir / f"{self.base_name}.1.gz"
        try:
            with open(filepath, "rb") as f_in:
                with gzip.open(backup_path, "wb") as f_out:
                    f_out.writelines(f_in)
            filepath.unlink()
        except Exception:
            pass  # If compression fails, just truncate

    def write_entry(self, entry: Dict):
        """Write a log entry to the file.
        
        Args:
            entry: Dict with log entry data (timestamp, level, logger, message, etc.)
        """
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            try:
                if self._current_size >= self.max_file_size_bytes:
                    self._rotate()
                    self._open_file()
                self._current_file.write(line)
                self._current_file.flush()
                self._current_size += len(line.encode("utf-8"))
            except Exception:
                pass  # Never let file writing crash the bot

    def close(self):
        """Close the current log file."""
        with self._lock:
            if self._current_file:
                try:
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None

    def get_stats(self) -> Dict:
        """Return writer statistics."""
        filepath = self.log_dir / self.base_name
        return {
            "log_dir": str(self.log_dir),
            "current_file_size_mb": round(self._current_size / (1024 * 1024), 2),
            "max_file_size_mb": round(self.max_file_size_bytes / (1024 * 1024), 2),
            "backup_files": len(list(self.log_dir.glob(f"{self.base_name}.*.gz"))),
        }


class LogBuffer:
    """Buffer circular thread-safe para logs com suporte a rotacao em ficheiro."""

    def __init__(
        self,
        max_lines: int = 500,
        log_dir: Optional[Path] = None,
        max_file_size_mb: float = 10.0,
        max_backup_files: int = 5,
        enable_file_writer: bool = False,
    ):
        self._lock = threading.RLock()
        self._buffer: deque = deque(maxlen=max_lines)
        self._listeners: List[threading.Event] = []
        self._listeners_lock = threading.Lock()

        # Rotating file writer (optional)
        self._file_writer: Optional[RotatingLogWriter] = None
        if enable_file_writer and log_dir:
            self._file_writer = RotatingLogWriter(
                log_dir=log_dir,
                max_file_size_mb=max_file_size_mb,
                max_backup_files=max_backup_files,
            )

    def append(self, record: logging.LogRecord):
        """Adiciona um log record ao buffer e ao ficheiro de rotacao."""
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

        # Write to rotating file if enabled
        if self._file_writer:
            self._file_writer.write_entry(entry)

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
        """Retorna estatisticas do buffer e do file writer."""
        with self._lock:
            total = len(self._buffer)
            if total == 0:
                stats = {"total": 0, "by_level": {}}
            else:
                by_level = {}
                for line in self._buffer:
                    by_level[line["level"]] = by_level.get(line["level"], 0) + 1
                stats = {"total": total, "by_level": by_level}

        # Add file writer stats if available
        if self._file_writer:
            stats["file_writer"] = self._file_writer.get_stats()

        return stats

    def close(self):
        """Close the file writer if active."""
        if self._file_writer:
            self._file_writer.close()


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


def install_log_buffer(
    root_logger_name: str = "",
    max_lines: int = 500,
    log_dir: Optional[Path] = None,
    max_file_size_mb: float = 10.0,
    max_backup_files: int = 5,
    enable_file_writer: bool = False,
) -> LogBuffer:
    """Instala o LogBuffer no root logger (ou logger especificado).
    
    Args:
        root_logger_name: Logger name to attach to (empty = root logger)
        max_lines: Maximum lines in the in-memory buffer
        log_dir: Directory for rotating log files
        max_file_size_mb: Max size per log file before rotation
        max_backup_files: Number of compressed backup files to keep
        enable_file_writer: Whether to enable persistent file logging
    """
    global _global_log_buffer, _global_handler
    if _global_log_buffer is None:
        _global_log_buffer = LogBuffer(
            max_lines=max_lines,
            log_dir=log_dir,
            max_file_size_mb=max_file_size_mb,
            max_backup_files=max_backup_files,
            enable_file_writer=enable_file_writer,
        )
    if _global_handler is None:
        _global_handler = DashboardLogHandler(_global_log_buffer)
        _global_handler.setLevel(logging.DEBUG)
        if root_logger_name:
            logger = logging.getLogger(root_logger_name)
        else:
            logger = logging.getLogger()
        logger.addHandler(_global_handler)
    return _global_log_buffer


def uninstall_log_buffer():
    """Remove o handler do logger e fecha o file writer."""
    global _global_handler, _global_log_buffer
    if _global_handler is not None:
        logging.getLogger().removeHandler(_global_handler)
        _global_handler = None
    if _global_log_buffer is not None:
        _global_log_buffer.close()
