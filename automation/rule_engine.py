"""
Rule engine for automation system.
Provides infrastructure for triggers, actions, and rules.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TriggerContext:
    """Context information for trigger evaluation."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class Trigger(ABC):
    """Base class for triggers."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._callbacks: List[Callable] = []
        self._running = False
    
    @abstractmethod
    async def evaluate(self, context: TriggerContext) -> bool:
        """Evaluate if trigger condition is met."""
        pass
    
    async def check(self) -> Optional[TriggerContext]:
        """Check trigger and return context if triggered."""
        context = TriggerContext()
        if await self.evaluate(context):
            return context
        return None
    
    async def start(self):
        """Start monitoring for trigger conditions."""
        self._running = True
        logger.info(f"Trigger '{self.name}' started")
    
    async def stop(self):
        """Stop monitoring for trigger conditions."""
        self._running = False
        logger.info(f"Trigger '{self.name}' stopped")


class Action(ABC):
    """Base class for actions."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, context: TriggerContext) -> ActionResult:
        """Execute the action."""
        pass


@dataclass
class Rule:
    """Represents an automation rule."""
    name: str
    trigger: Trigger
    actions: List[Action]
    enabled: bool = True
    cooldown_seconds: int = 0
    last_triggered: Optional[datetime] = None
    
    async def evaluate(self, context: TriggerContext) -> bool:
        """Evaluate if rule should trigger."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.cooldown_seconds > 0 and self.last_triggered:
            elapsed = (datetime.utcnow() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.debug(f"Rule '{self.name}' in cooldown ({elapsed:.1f}s < {self.cooldown_seconds}s)")
                return False
        
        # Evaluate trigger
        return await self.trigger.evaluate(context)
    
    async def execute(self, context: TriggerContext) -> List[ActionResult]:
        """Execute all actions in the rule."""
        results = []
        logger.info(f"Executing rule '{self.name}' with {len(self.actions)} actions")
        
        for action in self.actions:
            try:
                result = await action.execute(context)
                results.append(result)
                logger.info(f"Action '{action.name}' completed: {result.success} - {result.message}")
            except Exception as e:
                logger.error(f"Action '{action.name}' failed: {e}")
                results.append(ActionResult(
                    success=False,
                    message=f"Exception: {str(e)}"
                ))
        
        self.last_triggered = datetime.utcnow()
        return results


class RuleEngine:
    """Rule engine for managing and executing automation rules."""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_rule(self, rule: Rule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        logger.info(f"Added rule '{rule.name}' to engine")
    
    def remove_rule(self, rule_name: str):
        """Remove a rule from the engine."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed rule '{rule_name}' from engine")
    
    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def enable_rule(self, rule_name: str):
        """Enable a rule."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
            logger.info(f"Enabled rule '{rule_name}'")
    
    def disable_rule(self, rule_name: str):
        """Disable a rule."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
            logger.info(f"Disabled rule '{rule_name}'")
    
    async def start(self):
        """Start the rule engine."""
        if self._running:
            logger.warning("Rule engine already running")
            return
        
        self._running = True
        logger.info("Starting rule engine")
        
        # Start all triggers
        for rule in self.rules:
            await rule.trigger.start()
        
        # Start evaluation loop
        self._task = asyncio.create_task(self._evaluation_loop())
    
    async def stop(self):
        """Stop the rule engine."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping rule engine")
        
        # Cancel evaluation loop
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Stop all triggers
        for rule in self.rules:
            await rule.trigger.stop()
    
    async def _evaluation_loop(self):
        """Main evaluation loop for rules."""
        while self._running:
            try:
                for rule in self.rules:
                    if not rule.enabled:
                        continue
                    
                    context = TriggerContext()
                    if await rule.evaluate(context):
                        logger.info(f"Rule '{rule.name}' triggered")
                        await rule.execute(context)
                
                # Sleep before next evaluation
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(5)
    
    async def evaluate_once(self):
        """Evaluate all rules once (for testing)."""
        results = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            context = TriggerContext()
            if await rule.evaluate(context):
                logger.info(f"Rule '{rule.name}' triggered")
                action_results = await rule.execute(context)
                results.append({
                    'rule': rule.name,
                    'actions': action_results
                })
        
        return results
