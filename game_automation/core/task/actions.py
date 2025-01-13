"""Advanced task action system."""

from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from enum import Enum, auto
import logging

from .task_types import Task, TaskStatus
from ..control.control_manager import ControlManager, ControlAction
from ..recognition.state_analyzer import GameState
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ActionCondition:
    """Base class for action conditions"""
    
    async def check(self, context: Dict[str, Any]) -> bool:
        """Check if condition is met
        
        Args:
            context: Action context
            
        Returns:
            bool: True if condition is met
        """
        raise NotImplementedError()

class StateCondition(ActionCondition):
    """Game state condition"""
    
    def __init__(self, state: str):
        """Initialize state condition
        
        Args:
            state: Required game state
        """
        self.state = state
        
    async def check(self, context: Dict[str, Any]) -> bool:
        """Check if game state matches
        
        Args:
            context: Action context
            
        Returns:
            bool: True if state matches
        """
        current_state = context.get('current_state')
        return current_state == self.state

class RegionCondition(ActionCondition):
    """Screen region condition"""
    
    def __init__(
        self,
        region_id: str,
        min_confidence: float = 0.8
    ):
        """Initialize region condition
        
        Args:
            region_id: Region identifier
            min_confidence: Minimum detection confidence
        """
        self.region_id = region_id
        self.min_confidence = min_confidence
        
    async def check(self, context: Dict[str, Any]) -> bool:
        """Check if region is detected
        
        Args:
            context: Action context
            
        Returns:
            bool: True if region is detected
        """
        regions = context.get('detected_regions', {})
        if self.region_id in regions:
            return regions[self.region_id]['confidence'] >= self.min_confidence
        return False

class CompositeAction:
    """Base class for composite actions"""
    
    def __init__(
        self,
        name: str,
        conditions: List[ActionCondition] = None,
        timeout: float = None,
        retry_count: int = 0,
        retry_delay: float = 1.0
    ):
        """Initialize composite action
        
        Args:
            name: Action name
            conditions: Required conditions
            timeout: Action timeout in seconds
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.conditions = conditions or []
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.attempts = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute composite action
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if action succeeded
        """
        raise NotImplementedError()
        
    async def check_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if all conditions are met
        
        Args:
            context: Action context
            
        Returns:
            bool: True if all conditions are met
        """
        for condition in self.conditions:
            if not await condition.check(context):
                return False
        return True

class ParallelAction(CompositeAction):
    """Execute multiple actions in parallel"""
    
    def __init__(
        self,
        name: str,
        actions: List[CompositeAction],
        **kwargs
    ):
        """Initialize parallel action
        
        Args:
            name: Action name
            actions: Actions to execute in parallel
            **kwargs: Additional arguments
        """
        super().__init__(name, **kwargs)
        self.actions = actions
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute actions in parallel
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if all actions succeeded
        """
        if not await self.check_conditions(context):
            return False
            
        # Create tasks for each action
        tasks = [
            asyncio.create_task(
                action.execute(control_manager, context)
            )
            for action in self.actions
        ]
        
        try:
            # Wait for all actions to complete
            results = await asyncio.gather(*tasks)
            return all(results)
        except Exception as e:
            self.error = str(e)
            return False

class SequentialAction(CompositeAction):
    """Execute multiple actions in sequence"""
    
    def __init__(
        self,
        name: str,
        actions: List[CompositeAction],
        **kwargs
    ):
        """Initialize sequential action
        
        Args:
            name: Action name
            actions: Actions to execute in sequence
            **kwargs: Additional arguments
        """
        super().__init__(name, **kwargs)
        self.actions = actions
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute actions in sequence
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if all actions succeeded
        """
        if not await self.check_conditions(context):
            return False
            
        # Execute each action in sequence
        for action in self.actions:
            if not await action.execute(control_manager, context):
                return False
        return True

class ConditionalAction(CompositeAction):
    """Execute action based on condition"""
    
    def __init__(
        self,
        name: str,
        condition: ActionCondition,
        if_action: CompositeAction,
        else_action: Optional[CompositeAction] = None,
        **kwargs
    ):
        """Initialize conditional action
        
        Args:
            name: Action name
            condition: Condition to check
            if_action: Action to execute if condition is met
            else_action: Action to execute if condition is not met
            **kwargs: Additional arguments
        """
        super().__init__(name, **kwargs)
        self.condition = condition
        self.if_action = if_action
        self.else_action = else_action
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute conditional action
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if action succeeded
        """
        if not await self.check_conditions(context):
            return False
            
        # Check condition
        if await self.condition.check(context):
            return await self.if_action.execute(control_manager, context)
        elif self.else_action:
            return await self.else_action.execute(control_manager, context)
        return True

class RepeatAction(CompositeAction):
    """Repeat action multiple times"""
    
    def __init__(
        self,
        name: str,
        action: CompositeAction,
        count: int = None,
        until: ActionCondition = None,
        **kwargs
    ):
        """Initialize repeat action
        
        Args:
            name: Action name
            action: Action to repeat
            count: Number of repetitions (None for infinite)
            until: Repeat until condition is met
            **kwargs: Additional arguments
        """
        super().__init__(name, **kwargs)
        self.action = action
        self.count = count
        self.until = until
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute repeated action
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if action succeeded
        """
        if not await self.check_conditions(context):
            return False
            
        iteration = 0
        while True:
            # Check count limit
            if self.count is not None and iteration >= self.count:
                break
                
            # Execute action
            if not await self.action.execute(control_manager, context):
                return False
                
            # Check until condition
            if self.until and await self.until.check(context):
                break
                
            iteration += 1
            
        return True

class DelayAction(CompositeAction):
    """Add delay between actions"""
    
    def __init__(
        self,
        name: str,
        delay: float,
        **kwargs
    ):
        """Initialize delay action
        
        Args:
            name: Action name
            delay: Delay in seconds
            **kwargs: Additional arguments
        """
        super().__init__(name, **kwargs)
        self.delay = delay
        
    async def execute(
        self,
        control_manager: ControlManager,
        context: Dict[str, Any]
    ) -> bool:
        """Execute delay
        
        Args:
            control_manager: Control manager instance
            context: Action context
            
        Returns:
            bool: True if action succeeded
        """
        if not await self.check_conditions(context):
            return False
            
        await asyncio.sleep(self.delay)
        return True
