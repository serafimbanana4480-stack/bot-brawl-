"""
Emulator monitoring automation.
Provides triggers and actions for ADB reconnection, error logging, and emulator health monitoring.
"""

import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from pathlib import Path

from .rule_engine import Trigger, Action, TriggerContext, ActionResult

logger = logging.getLogger(__name__)


class ADBDisconnectionTrigger(Trigger):
    """Trigger that detects ADB disconnection from emulator."""
    
    def __init__(self, device_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("adb_disconnection_trigger", config)
        self.device_id = device_id
        self.last_status = True  # True = connected
        self.check_interval = config.get('check_interval', 10) if config else 10
        self.adb_path = config.get('adb_path', 'adb') if config else 'adb'
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if ADB device is disconnected."""
        try:
            # Run adb devices command
            process = await asyncio.create_subprocess_shell(
                f"{self.adb_path} devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode()
                # Check if device is in output
                current_status = self.device_id in output
                
                if not current_status and self.last_status:
                    # Device disconnected
                    self.last_status = False
                    context.data['event'] = 'adb_disconnected'
                    context.data['device_id'] = self.device_id
                    logger.error(f"ADB disconnection detected: {self.device_id}")
                    return True
                elif current_status and not self.last_status:
                    # Device reconnected
                    self.last_status = True
                    context.data['event'] = 'adb_reconnected'
                    context.data['device_id'] = self.device_id
                    logger.info(f"ADB reconnection detected: {self.device_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to check ADB status: {e}")
        
        return False


class EmulatorProcessCrashTrigger(Trigger):
    """Trigger that detects emulator process crashes."""
    
    def __init__(self, emulator_process_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("emulator_process_crash_trigger", config)
        self.emulator_process_name = emulator_process_name
        self.last_status = True
        self.check_interval = config.get('check_interval', 5) if config else 5
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if emulator process is running."""
        try:
            # Check if process is running
            process = await asyncio.create_subprocess_shell(
                f"tasklist /FI \"IMAGENAME eq {self.emulator_process_name}\"",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode()
                current_status = self.emulator_process_name.lower() in output.lower()
                
                if not current_status and self.last_status:
                    # Process crashed
                    self.last_status = False
                    context.data['event'] = 'emulator_crashed'
                    context.data['process_name'] = self.emulator_process_name
                    logger.error(f"Emulator process crash detected: {self.emulator_process_name}")
                    return True
                elif current_status and not self.last_status:
                    # Process restarted
                    self.last_status = True
                    context.data['event'] = 'emulator_restarted'
                    context.data['process_name'] = self.emulator_process_name
                    logger.info(f"Emulator process restarted: {self.emulator_process_name}")
                    return True
        except Exception as e:
            logger.error(f"Failed to check emulator process: {e}")
        
        return False


class ADBReconnectAction(Action):
    """Action to attempt ADB reconnection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("adb_reconnect_action", config)
        self.adb_path = config.get('adb_path', 'adb') if config else 'adb'
        self.max_attempts = config.get('max_attempts', 3) if config else 3
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Attempt to reconnect ADB."""
        device_id = context.data.get('device_id')
        
        if not device_id:
            return ActionResult(
                success=False,
                message="No device ID in context"
            )
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"ADB reconnection attempt {attempt}/{self.max_attempts} for {device_id}")
                
                # Try to reconnect
                process = await asyncio.create_subprocess_shell(
                    f"{self.adb_path} connect {device_id}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"ADB reconnection successful: {device_id}")
                    return ActionResult(
                        success=True,
                        message=f"ADB reconnected on attempt {attempt}",
                        data={'device_id': device_id, 'attempt': attempt}
                    )
                else:
                    logger.warning(f"ADB reconnection attempt {attempt} failed")
                    await asyncio.sleep(2)  # Wait before retry
                
            except Exception as e:
                logger.error(f"ADB reconnection error on attempt {attempt}: {e}")
        
        return ActionResult(
            success=False,
            message=f"ADB reconnection failed after {self.max_attempts} attempts",
            data={'device_id': device_id, 'attempts': self.max_attempts}
        )


class EmulatorRestartAction(Action):
    """Action to restart the emulator."""
    
    def __init__(self, emulator_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("emulator_restart_action", config)
        self.emulator_path = emulator_path
        self.max_restarts = config.get('max_restarts', 3) if config else 3
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Restart the emulator."""
        process_name = context.data.get('process_name')
        
        try:
            logger.info(f"Restarting emulator: {self.emulator_path}")
            
            process = await asyncio.create_subprocess_shell(
                f'"{self.emulator_path}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            return ActionResult(
                success=True,
                message=f"Emulator restart initiated (PID: {process.pid})",
                data={'pid': process.pid, 'path': self.emulator_path}
            )
        except Exception as e:
            logger.error(f"Failed to restart emulator: {e}")
            return ActionResult(
                success=False,
                message=f"Emulator restart failed: {str(e)}"
            )


class ErrorLoggingAction(Action):
    """Action to log emulator errors with detailed context."""
    
    def __init__(self, log_file: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__("error_logging_action", config)
        self.log_file = Path(log_file) if log_file else None
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Log error with detailed context."""
        event = context.data.get('event', 'unknown')
        timestamp = context.timestamp.isoformat()
        
        log_entry = f"[{timestamp}] EVENT: {event}\n"
        for key, value in context.data.items():
            log_entry += f"  {key}: {value}\n"
        
        logger.error(f"Emulator error logged: {log_entry}")
        
        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, 'a') as f:
                    f.write(log_entry + "\n")
                return ActionResult(
                    success=True,
                    message=f"Error logged to {self.log_file}",
                    data={'log_file': str(self.log_file)}
                )
            except Exception as e:
                logger.error(f"Failed to write to log file: {e}")
        
        return ActionResult(
            success=True,
            message="Error logged",
            data={'log_entry': log_entry}
        )


class EmulatorMonitoringAutomation:
    """Emulator monitoring automation setup."""
    
    def __init__(self, device_id: Optional[str] = None, emulator_process_name: Optional[str] = None):
        self.device_id = device_id
        self.emulator_process_name = emulator_process_name
        self.rules = []
    
    def setup_adb_monitoring(self, adb_path: str = 'adb'):
        """Setup ADB disconnection monitoring and auto-reconnect."""
        from .rule_engine import Rule
        
        if not self.device_id:
            logger.warning("No device ID provided, ADB monitoring will not work")
            return None
        
        trigger = ADBDisconnectionTrigger(
            device_id=self.device_id,
            config={'adb_path': adb_path}
        )
        
        actions = [
            ADBReconnectAction(config={'adb_path': adb_path}),
            ErrorLoggingAction()
        ]
        
        rule = Rule(
            name="emulator_adb_reconnect",
            trigger=trigger,
            actions=actions,
            cooldown_seconds=30
        )
        
        self.rules.append(rule)
        logger.info(f"Setup ADB monitoring for device: {self.device_id}")
        return rule
    
    def setup_process_monitoring(self, emulator_path: Optional[str] = None):
        """Setup emulator process crash monitoring."""
        from .rule_engine import Rule
        
        if not self.emulator_process_name:
            logger.warning("No emulator process name provided, process monitoring will not work")
            return None
        
        trigger = EmulatorProcessCrashTrigger(
            emulator_process_name=self.emulator_process_name
        )
        
        actions = []
        if emulator_path:
            actions.append(EmulatorRestartAction(emulator_path=emulator_path))
        actions.append(ErrorLoggingAction())
        
        rule = Rule(
            name="emulator_process_monitor",
            trigger=trigger,
            actions=actions,
            cooldown_seconds=60
        )
        
        self.rules.append(rule)
        logger.info(f"Setup emulator process monitoring: {self.emulator_process_name}")
        return rule
    
    def get_rules(self):
        """Get all configured rules."""
        return self.rules
