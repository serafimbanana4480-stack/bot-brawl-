"""
Bot monitoring automation.
Provides triggers and actions for monitoring bot crashes, auto-restart, and notifications.
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from pathlib import Path

from .rule_engine import Trigger, Action, TriggerContext, ActionResult

logger = logging.getLogger(__name__)


class BotCrashTrigger(Trigger):
    """Trigger that detects bot crashes."""
    
    def __init__(self, bot_pid: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__("bot_crash_trigger", config)
        self.bot_pid = bot_pid
        self.last_status = True  # True = running
        self.check_interval = config.get('check_interval', 5) if config else 5
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if bot process has crashed."""
        if self.bot_pid:
            try:
                process = psutil.Process(self.bot_pid)
                current_status = process.is_running()
                
                if current_status and not self.last_status:
                    # Bot restarted
                    self.last_status = True
                    context.data['event'] = 'bot_restarted'
                    context.data['pid'] = self.bot_pid
                    return True
                elif not current_status and self.last_status:
                    # Bot crashed
                    self.last_status = False
                    context.data['event'] = 'bot_crashed'
                    context.data['pid'] = self.bot_pid
                    logger.error(f"Bot crash detected: PID {self.bot_pid}")
                    return True
            except psutil.NoSuchProcess:
                if self.last_status:
                    self.last_status = False
                    context.data['event'] = 'bot_crashed'
                    context.data['pid'] = self.bot_pid
                    logger.error(f"Bot crash detected: PID {self.bot_pid} not found")
                    return True
        
        return False
    
    async def check(self) -> Optional[TriggerContext]:
        """Check trigger and return context if triggered."""
        context = TriggerContext()
        if await self.evaluate(context):
            return context
        return None


class BotMemoryHighTrigger(Trigger):
    """Trigger that detects high memory usage."""
    
    def __init__(self, bot_pid: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__("bot_memory_high_trigger", config)
        self.bot_pid = bot_pid
        self.threshold_mb = config.get('memory_threshold_mb', 2000) if config else 2000
        self.check_interval = config.get('check_interval', 10) if config else 10
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if bot memory usage is high."""
        if self.bot_pid:
            try:
                process = psutil.Process(self.bot_pid)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                if memory_mb > self.threshold_mb:
                    context.data['event'] = 'memory_high'
                    context.data['pid'] = self.bot_pid
                    context.data['memory_mb'] = memory_mb
                    context.data['threshold_mb'] = self.threshold_mb
                    logger.warning(f"High memory usage detected: {memory_mb:.1f}MB > {self.threshold_mb}MB")
                    return True
            except psutil.NoSuchProcess:
                pass
        
        return False


class AutoRestartAction(Action):
    """Action to automatically restart the bot."""
    
    def __init__(self, restart_command: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("auto_restart_action", config)
        self.restart_command = restart_command
        self.max_restarts = config.get('max_restarts', 3) if config else 3
        self.restart_count = 0
        self.restart_window = timedelta(minutes=config.get('restart_window_minutes', 10)) if config else timedelta(minutes=10)
        self.last_restart: Optional[datetime] = None
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Execute bot restart."""
        # Check restart limit
        now = datetime.utcnow()
        if self.last_restart and (now - self.last_restart) < self.restart_window:
            if self.restart_count >= self.max_restarts:
                return ActionResult(
                    success=False,
                    message=f"Restart limit reached ({self.max_restarts} in {self.restart_window})"
                )
        
        try:
            logger.info(f"Executing restart command: {self.restart_command}")
            process = await asyncio.create_subprocess_shell(
                self.restart_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.restart_count += 1
            self.last_restart = now
            
            # Reset count after window
            if self.last_restart and (now - self.last_restart) > self.restart_window:
                self.restart_count = 0
            
            return ActionResult(
                success=True,
                message=f"Bot restart initiated (attempt {self.restart_count})",
                data={'pid': process.pid}
            )
        except Exception as e:
            logger.error(f"Failed to restart bot: {e}")
            return ActionResult(
                success=False,
                message=f"Restart failed: {str(e)}"
            )


class NotificationAction(Action):
    """Action to send notifications (logging placeholder)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("notification_action", config)
        self.notification_channel = config.get('channel', 'log') if config else 'log'
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Send notification."""
        event = context.data.get('event', 'unknown')
        pid = context.data.get('pid', 'unknown')
        
        message = f"Bot monitoring event: {event} (PID: {pid})"
        
        if self.notification_channel == 'log':
            logger.warning(f"NOTIFICATION: {message}")
            return ActionResult(
                success=True,
                message=f"Notification logged: {message}"
            )
        else:
            # Placeholder for other notification channels (email, Slack, etc.)
            logger.info(f"Notification would be sent to {self.notification_channel}: {message}")
            return ActionResult(
                success=True,
                message=f"Notification queued for {self.notification_channel}"
            )


class BotMonitoringAutomation:
    """Bot monitoring automation setup."""
    
    def __init__(self, bot_pid: Optional[int] = None, restart_command: Optional[str] = None):
        self.bot_pid = bot_pid
        self.restart_command = restart_command
        self.rules = []
    
    def setup_crash_detection(self, max_restarts: int = 3, restart_window_minutes: int = 10):
        """Setup crash detection and auto-restart."""
        from .rule_engine import Rule
        
        if not self.restart_command:
            logger.warning("No restart command provided, auto-restart will not work")
            return None
        
        trigger = BotCrashTrigger(
            bot_pid=self.bot_pid,
            config={'check_interval': 5}
        )
        
        actions = [
            AutoRestartAction(
                restart_command=self.restart_command,
                config={'max_restarts': max_restarts, 'restart_window_minutes': restart_window_minutes}
            ),
            NotificationAction(config={'channel': 'log'})
        ]
        
        rule = Rule(
            name="bot_crash_auto_restart",
            trigger=trigger,
            actions=actions,
            cooldown_seconds=30
        )
        
        self.rules.append(rule)
        logger.info("Setup bot crash detection and auto-restart")
        return rule
    
    def setup_memory_monitoring(self, memory_threshold_mb: int = 2000):
        """Setup high memory monitoring."""
        from .rule_engine import Rule
        
        trigger = BotMemoryHighTrigger(
            bot_pid=self.bot_pid,
            config={'memory_threshold_mb': memory_threshold_mb, 'check_interval': 10}
        )
        
        actions = [
            NotificationAction(config={'channel': 'log'})
        ]
        
        rule = Rule(
            name="bot_memory_high_alert",
            trigger=trigger,
            actions=actions,
            cooldown_seconds=60
        )
        
        self.rules.append(rule)
        logger.info(f"Setup bot memory monitoring (threshold: {memory_threshold_mb}MB)")
        return rule
    
    def get_rules(self):
        """Get all configured rules."""
        return self.rules
