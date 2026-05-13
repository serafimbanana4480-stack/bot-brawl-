"""
realtime_logs.py

Sistema de logs em tempo real com WebSocket para o dashboard.
"""

import asyncio
import json
import logging
from typing import Set, Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import threading


@dataclass
class LogEntry:
    """Entrada de log estruturada"""
    timestamp: str
    level: str
    message: str
    category: str  # "system", "vision", "match", "safety", "control", "lobby", "combat", "state", "auto_tuning", "brawler", "humanization"
    data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "category": self.category,
            "data": self.data
        }


class WebSocketLogHandler(logging.Handler):
    """
    Handler de logging que envia para WebSocket.
    Permite ver logs em tempo real no dashboard.
    """
    
    def __init__(self, log_manager: 'RealtimeLogManager'):
        super().__init__()
        self.log_manager = log_manager
        self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Envia log para o manager"""
        try:
            # Determinar categoria baseado no nome do logger
            category = self._get_category(record.name)
            
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=record.levelname,
                message=self.format(record),
                category=category
            )
            
            self.log_manager.add_log(entry)
        except Exception:
            self.handleError(record)
    
    def _get_category(self, logger_name: str) -> str:
        """Determina categoria do log"""
        logger_name_lower = logger_name.lower()
        
        if "vision" in logger_name_lower or "tracker" in logger_name_lower:
            return "vision"
        elif "match" in logger_name_lower:
            return "match"
        elif "safety" in logger_name_lower:
            return "safety"
        elif "emulator" in logger_name_lower:
            return "control"
        elif "lobby" in logger_name_lower:
            return "lobby"
        elif "play" in logger_name_lower or "combat" in logger_name_lower:
            return "combat"
        elif "state" in logger_name_lower:
            return "state"
        elif "auto_tuner" in logger_name_lower or "tuning" in logger_name_lower:
            return "auto_tuning"
        elif "brawler" in logger_name_lower:
            return "brawler"
        elif "human" in logger_name_lower:
            return "humanization"
        return "system"


class RealtimeLogManager:
    """
    Manager de logs em tempo real.
    Mantém histórico e envia para WebSocket connections.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.logs: List[LogEntry] = []
        self.clients: Set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        
        # Estatísticas
        self.stats = {
            "total_logs": 0,
            "by_category": {},
            "by_level": {}
        }
    
    def add_log(self, entry: LogEntry) -> None:
        """Adiciona log ao histórico"""
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_history:
                self.logs = self.logs[-self.max_history:]
            
            # Atualizar estatísticas
            self.stats["total_logs"] += 1
            self.stats["by_category"][entry.category] = \
                self.stats["by_category"].get(entry.category, 0) + 1
            self.stats["by_level"][entry.level] = \
                self.stats["by_level"].get(entry.level, 0) + 1
        
        # Notificar clientes WebSocket (Thread-safe)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast(entry)))
        except Exception:
            pass # Silencioso se não houver loop ativo
    
    def log(self, message: str, level: str = "INFO", 
            category: str = "system", data: Optional[Dict] = None) -> None:
        """Método conveniente para adicionar log"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            category=category,
            data=data
        )
        self.add_log(entry)
    
    async def _broadcast(self, entry: LogEntry) -> None:
        """Envia log para todos os clientes conectados"""
        if not self.clients:
            return
        
        message = json.dumps(entry.to_dict())
        dead_clients = set()
        
        for queue in self.clients:
            try:
                await queue.put(message)
            except Exception:
                dead_clients.add(queue)
        
        # Remover clientes mortos
        self.clients -= dead_clients
    
    def subscribe(self) -> asyncio.Queue:
        """Subscreve para receber logs em tempo real"""
        queue = asyncio.Queue()
        self.clients.add(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Cancela subscrição"""
        self.clients.discard(queue)
    
    def get_recent_logs(self, n: int = 100, 
                       category: Optional[str] = None,
                       level: Optional[str] = None) -> List[Dict]:
        """Retorna logs recentes com filtros opcionais"""
        logs = self.logs
        
        if category:
            logs = [l for l in logs if l.category == category]
        if level:
            logs = [l for l in logs if l.level == level]
        
        return [l.to_dict() for l in logs[-n:]]
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de logs"""
        return self.stats.copy()
    
    def clear(self) -> None:
        """Limpa todos os logs"""
        with self._lock:
            self.logs.clear()
            self.stats = {"total_logs": 0, "by_category": {}, "by_level": {}}


# Singleton instance
_log_manager: Optional[RealtimeLogManager] = None


def get_log_manager() -> RealtimeLogManager:
    """Retorna instância singleton do log manager"""
    global _log_manager
    if _log_manager is None:
        _log_manager = RealtimeLogManager()
    return _log_manager


def setup_logging() -> None:
    """Configura logging para usar WebSocket"""
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remover handlers existentes
    root_logger.handlers.clear()
    
    # Adicionar WebSocket handler
    ws_handler = WebSocketLogHandler(get_log_manager())
    ws_handler.setLevel(logging.INFO)
    root_logger.addHandler(ws_handler)
    
    # Adicionar console handler também
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
