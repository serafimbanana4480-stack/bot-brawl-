"""Structured Logging Service - Enterprise-grade logging with context"""

import asyncio
import json
import logging
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    logger: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "logger": self.logger,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "context": self.context,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class StructuredLogging:
    def __init__(self, service_name: str = "enterprise-ai", log_level: int = logging.INFO):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.log_buffer: List[LogEntry] = []
        self.max_buffer_size = 10000
        
        self._context: Dict[str, Any] = {}
        
    def set_context(self, **kwargs):
        self._context.update(kwargs)
    
    def clear_context(self):
        self._context = {}
    
    def _create_entry(self, level: str, message: str, 
                     error: Exception = None, **kwargs) -> LogEntry:
        context = {**self._context, **kwargs}
        
        trace_id = context.pop("trace_id", None)
        span_id = context.pop("span_id", None)
        agent_id = context.pop("agent_id", None)
        workflow_id = context.pop("workflow_id", None)
        
        error_info = None
        if error:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": str(error.__traceback__) if hasattr(error, '__traceback__') else None,
            }
        
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level,
            message=message,
            logger=self.service_name,
            trace_id=trace_id,
            span_id=span_id,
            agent_id=agent_id,
            workflow_id=workflow_id,
            context=context,
            error=error_info,
        )
        
        self.log_buffer.append(entry)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]
        
        return entry
    
    def debug(self, message: str, **kwargs):
        entry = self._create_entry("DEBUG", message, **kwargs)
        self.logger.debug(entry.to_json())
    
    def info(self, message: str, **kwargs):
        entry = self._create_entry("INFO", message, **kwargs)
        self.logger.info(entry.to_json())
    
    def warning(self, message: str, **kwargs):
        entry = self._create_entry("WARNING", message, **kwargs)
        self.logger.warning(entry.to_json())
    
    def error(self, message: str, error: Exception = None, **kwargs):
        entry = self._create_entry("ERROR", message, error=error, **kwargs)
        self.logger.error(entry.to_json())
    
    def critical(self, message: str, error: Exception = None, **kwargs):
        entry = self._create_entry("CRITICAL", message, error=error, **kwargs)
        self.logger.critical(entry.to_json())
    
    def log_decision(self, decision: str, agent_id: str, confidence: float,
                    reasoning: str, **kwargs):
        self.info(
            f"Decision made: {decision}",
            agent_id=agent_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            event_type="decision",
            **kwargs
        )
    
    def log_agent_action(self, agent_id: str, action: str, target: str = None,
                       result: str = None, **kwargs):
        self.info(
            f"Agent action: {action}",
            agent_id=agent_id,
            action=action,
            target=target,
            result=result,
            event_type="agent_action",
            **kwargs
        )
    
    def log_workflow_event(self, workflow_id: str, event: str, step: str = None,
                          status: str = None, **kwargs):
        self.info(
            f"Workflow event: {event}",
            workflow_id=workflow_id,
            event=event,
            step=step,
            status=status,
            event_type="workflow",
            **kwargs
        )
    
    def log_learning_update(self, agent_id: str, update_type: str,
                          metrics: Dict[str, float], **kwargs):
        self.info(
            f"Learning update: {update_type}",
            agent_id=agent_id,
            update_type=update_type,
            metrics=metrics,
            event_type="learning",
            **kwargs
        )
    
    def get_logs(self, level: str = None, agent_id: str = None,
                workflow_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        logs = self.log_buffer
        
        if level:
            logs = [l for l in logs if l.level == level]
        if agent_id:
            logs = [l for l in logs if l.agent_id == agent_id]
        if workflow_id:
            logs = [l for l in logs if l.workflow_id == workflow_id]
        
        return [l.to_dict() for l in logs[-limit:]]
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.log_buffer:
            return {
                "total_logs": 0,
                "by_level": {},
            }
        
        by_level = {}
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            by_level[level.lower()] = sum(1 for l in self.log_buffer if l.level == level)
        
        return {
            "total_logs": len(self.log_buffer),
            "by_level": by_level,
            "with_errors": sum(1 for l in self.log_buffer if l.error),
            "with_trace": sum(1 for l in self.log_buffer if l.trace_id),
        }
