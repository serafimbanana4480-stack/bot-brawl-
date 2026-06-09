"""
core/logging_config.py

Structured JSON logging with correlation IDs, ISO8601 timestamps,
log rotation, and configurable levels.
"""

import json
import logging
import logging.handlers
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()
DEFAULT_LOG_FILE = os.getenv("LOG_FILE", "logs/brawl_bot.log")
DEFAULT_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10_000_000").replace("_", ""))
DEFAULT_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Standard library JSON formatter for non-structlog fallback."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "source": f"{record.pathname}:{record.lineno}",
        }
        if hasattr(record, "correlation_id"):
            log_dict["correlation_id"] = record.correlation_id
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_dict, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Correlation ID filter
# ---------------------------------------------------------------------------
class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        cid = getattr(record, "correlation_id", None) or _get_correlation_id()
        record.correlation_id = cid
        return True


_CORRELATION_ID: str = ""


def _get_correlation_id() -> str:
    global _CORRELATION_ID
    if not _CORRELATION_ID:
        _CORRELATION_ID = str(uuid.uuid4())[:8]
    return _CORRELATION_ID


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set (or generate) the global correlation ID for the current session."""
    global _CORRELATION_ID
    _CORRELATION_ID = cid or str(uuid.uuid4())[:8]
    return _CORRELATION_ID


def get_correlation_id() -> str:
    """Return the current correlation ID."""
    return _get_correlation_id()


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def setup_logging(
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure structured logging (structlog + stdlib)."""
    lvl = (level or DEFAULT_LOG_LEVEL).upper()
    format_type = (fmt or DEFAULT_LOG_FORMAT).lower()
    file_path = log_file or DEFAULT_LOG_FILE

    _ensure_dir(file_path)

    # --- structlog processors ---
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.ExtraAdder(),
    ]

    if format_type == "json":
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=sys.platform != "win32")
        )

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # --- stdlib root logger ---
    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(lvl)
    console.addFilter(CorrelationIdFilter())

    if format_type == "json":
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s"
            )
        )
    root.addHandler(console)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(lvl)
    file_handler.addFilter(CorrelationIdFilter())
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # Suppress noisy 3rd-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog-wrapped logger."""
    return structlog.get_logger(name)
