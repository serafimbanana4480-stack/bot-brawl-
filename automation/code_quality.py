"""
Code quality automation.
Provides triggers and actions for linting, formatting, and testing on file changes.
"""

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from .rule_engine import Trigger, Action, TriggerContext, ActionResult

logger = logging.getLogger(__name__)


class FileChangeTrigger(Trigger):
    """Trigger that detects file changes in specified directories."""
    
    def __init__(self, watch_paths: List[str], config: Optional[Dict[str, Any]] = None):
        super().__init__("file_change_trigger", config)
        self.watch_paths = [Path(p) for p in watch_paths]
        self.file_patterns = config.get('file_patterns', ['*.py']) if config else ['*.py']
        self.observer: Optional[Observer] = None
        self.last_change: Optional[datetime] = None
        self.changed_files: List[str] = []
    
    async def start(self):
        """Start watching for file changes."""
        await super().start()
        
        class FileChangeHandler(FileSystemEventHandler):
            def __init__(self, parent_trigger):
                self.parent = parent_trigger
            
            def on_modified(self, event):
                if not event.is_directory:
                    file_path = Path(event.src_path)
                    # Check if file matches patterns
                    for pattern in self.parent.file_patterns:
                        if file_path.match(pattern):
                            self.parent.changed_files.append(str(file_path))
                            self.parent.last_change = datetime.utcnow()
                            logger.debug(f"File changed: {file_path}")
                            break
        
        self.observer = Observer()
        for watch_path in self.watch_paths:
            if watch_path.exists():
                handler = FileChangeHandler(self)
                self.observer.schedule(handler, str(watch_path), recursive=True)
                logger.info(f"Watching for changes in: {watch_path}")
        
        self.observer.start()
    
    async def stop(self):
        """Stop watching for file changes."""
        await super().stop()
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Check if any files have changed."""
        if self.changed_files and self.last_change:
            context.data['event'] = 'file_changed'
            context.data['changed_files'] = self.changed_files.copy()
            context.data['change_time'] = self.last_change
            self.changed_files.clear()
            return True
        return False


class LinterAction(Action):
    """Action to run linter (ruff)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("linter_action", config)
        self.linter_command = config.get('linter_command', 'ruff check') if config else 'ruff check'
        self.target_paths = config.get('target_paths', ['backend']) if config else ['backend']
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Run linter on changed files or target paths."""
        changed_files = context.data.get('changed_files', [])
        
        if changed_files:
            # Run linter on changed files only
            target = ' '.join(changed_files)
        else:
            # Run on target paths
            target = ' '.join(self.target_paths)
        
        command = f"{self.linter_command} {target}"
        logger.info(f"Running linter: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            message = "Linter passed" if success else f"Linter found issues (exit code {process.returncode})"
            
            return ActionResult(
                success=success,
                message=message,
                data={
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'exit_code': process.returncode
                }
            )
        except Exception as e:
            logger.error(f"Failed to run linter: {e}")
            return ActionResult(
                success=False,
                message=f"Linter failed: {str(e)}"
            )


class FormatterAction(Action):
    """Action to run formatter (black)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("formatter_action", config)
        self.formatter_command = config.get('formatter_command', 'black --check') if config else 'black --check'
        self.target_paths = config.get('target_paths', ['backend']) if config else ['backend']
        self.auto_fix = config.get('auto_fix', False) if config else False
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Run formatter on changed files or target paths."""
        changed_files = context.data.get('changed_files', [])
        
        if changed_files:
            # Run formatter on changed files only
            target = ' '.join(changed_files)
        else:
            # Run on target paths
            target = ' '.join(self.target_paths)
        
        # Use black (without --check) if auto_fix is enabled
        command = self.formatter_command
        if self.auto_fix and '--check' in command:
            command = command.replace('--check', '')
        
        logger.info(f"Running formatter: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                f"{command} {target}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            message = "Formatter check passed" if success else f"Formatter found issues (exit code {process.returncode})"
            
            return ActionResult(
                success=success,
                message=message,
                data={
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'exit_code': process.returncode
                }
            )
        except Exception as e:
            logger.error(f"Failed to run formatter: {e}")
            return ActionResult(
                success=False,
                message=f"Formatter failed: {str(e)}"
            )


class TestRunnerAction(Action):
    """Action to run tests."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("test_runner_action", config)
        self.test_command = config.get('test_command', 'pytest') if config else 'pytest'
        self.test_paths = config.get('test_paths', ['tests']) if config else ['tests']
        self.fail_on_error = config.get('fail_on_error', False) if config else False
    
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Run tests."""
        target = ' '.join(self.test_paths)
        command = f"{self.test_command} {target}"
        logger.info(f"Running tests: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            message = "Tests passed" if success else f"Tests failed (exit code {process.returncode})"
            
            return ActionResult(
                success=success,
                message=message,
                data={
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'exit_code': process.returncode
                }
            )
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return ActionResult(
                success=False,
                message=f"Test runner failed: {str(e)}"
            )


class CodeQualityAutomation:
    """Code quality automation setup."""
    
    def __init__(self, watch_paths: List[str]):
        self.watch_paths = watch_paths
        self.rules = []
    
    def setup_linting(self, target_paths: Optional[List[str]] = None):
        """Setup automatic linting on file changes."""
        from .rule_engine import Rule
        
        trigger = FileChangeTrigger(
            watch_paths=self.watch_paths,
            config={'file_patterns': ['*.py']}
        )
        
        action = LinterAction(config={'target_paths': target_paths or self.watch_paths})
        
        rule = Rule(
            name="code_quality_linting",
            trigger=trigger,
            actions=[action],
            cooldown_seconds=5
        )
        
        self.rules.append(rule)
        logger.info("Setup code quality linting automation")
        return rule
    
    def setup_formatting(self, target_paths: Optional[List[str]] = None, auto_fix: bool = False):
        """Setup automatic formatting check on file changes."""
        from .rule_engine import Rule
        
        trigger = FileChangeTrigger(
            watch_paths=self.watch_paths,
            config={'file_patterns': ['*.py']}
        )
        
        action = FormatterAction(
            config={
                'target_paths': target_paths or self.watch_paths,
                'auto_fix': auto_fix
            }
        )
        
        rule = Rule(
            name="code_quality_formatting",
            trigger=trigger,
            actions=[action],
            cooldown_seconds=5
        )
        
        self.rules.append(rule)
        logger.info(f"Setup code quality formatting automation (auto_fix: {auto_fix})")
        return rule
    
    def setup_testing(self, test_paths: Optional[List[str]] = None):
        """Setup automatic testing on file changes."""
        from .rule_engine import Rule
        
        trigger = FileChangeTrigger(
            watch_paths=self.watch_paths,
            config={'file_patterns': ['*.py']}
        )
        
        action = TestRunnerAction(config={'test_paths': test_paths or ['tests']})
        
        rule = Rule(
            name="code_quality_testing",
            trigger=trigger,
            actions=[action],
            cooldown_seconds=10
        )
        
        self.rules.append(rule)
        logger.info("Setup code quality testing automation")
        return rule
    
    def get_rules(self):
        """Get all configured rules."""
        return self.rules
