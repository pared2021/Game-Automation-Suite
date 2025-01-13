"""Task action handler for control system integration."""
from typing import Dict, List, Optional, Type, Any, Callable, Union
from datetime import datetime
import asyncio
import logging
from enum import Enum, auto

from .error.error_manager import GameAutomationError
from .control.control_manager import ControlManager, ControlAction
from .task_types import Task, TaskStatus
from .recognition.state_analyzer import GameState
from .monitor.task_monitor import TaskMonitor
from .actions import (
    CompositeAction,
    ParallelAction,
    SequentialAction,
    ConditionalAction,
    RepeatAction,
    DelayAction,
    ActionCondition,
    StateCondition,
    RegionCondition
)

logger = logging.getLogger(__name__)

class TaskActionError(GameAutomationError):
    """Task action related errors"""
    pass

class ActionType(Enum):
    """Action types for game control"""
    CLICK = auto()
    DRAG = auto()
    KEY_PRESS = auto()
    KEY_HOLD = auto()
    WAIT = auto()
    SEQUENCE = auto()
    PARALLEL = auto()
    CONDITIONAL = auto()
    REPEAT = auto()
    DELAY = auto()

class TaskAction:
    """Task action definition"""
    
    def __init__(
        self,
        action_type: ActionType,
        params: Dict = None,
        timeout: float = None,
        retry_count: int = 0,
        retry_delay: float = 1.0
    ):
        """Initialize task action.
        
        Args:
            action_type: Action type
            params: Action parameters
            timeout: Action timeout in seconds
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.action_type = action_type
        self.params = params or {}
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.attempts: int = 0
        self.metrics: Dict[str, Any] = {}

class ActionSequence:
    """Sequence of task actions"""
    
    def __init__(self, name: str, actions: List[TaskAction]):
        """Initialize action sequence.
        
        Args:
            name: Sequence name
            actions: List of actions
        """
        self.name = name
        self.actions = actions
        self.current_index = 0
        
    @property
    def current_action(self) -> Optional[TaskAction]:
        """Get current action.
        
        Returns:
            Optional[TaskAction]: Current action or None if complete
        """
        if self.current_index >= len(self.actions):
            return None
        return self.actions[self.current_index]
        
    def next_action(self) -> Optional[TaskAction]:
        """Get next action.
        
        Returns:
            Optional[TaskAction]: Next action or None if complete
        """
        self.current_index += 1
        return self.current_action

class TaskActionHandler:
    """Task action handler"""
    
    def __init__(
        self,
        control_manager: ControlManager,
        task_monitor: Optional[TaskMonitor] = None
    ):
        """Initialize action handler.
        
        Args:
            control_manager: Control manager instance
            task_monitor: Optional task monitor instance
        """
        self.control_manager = control_manager
        self._action_handlers: Dict[ActionType, Callable] = {}
        self._sequences: Dict[str, Union[ActionSequence, CompositeAction]] = {}
        self._running_sequences: Dict[str, asyncio.Task] = {}
        self._task_monitor = task_monitor
        self._logger = logging.getLogger(__name__)
        
        # Register default handlers
        self._register_default_handlers()
        
    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable
    ):
        """Register action handler.
        
        Args:
            action_type: Action type
            handler: Handler function
        """
        self._action_handlers[action_type] = handler
        
    def register_sequence(
        self,
        sequence: ActionSequence
    ):
        """Register action sequence.
        
        Args:
            sequence: Action sequence
        """
        self._sequences[sequence.name] = sequence
        
    def register_composite_action(
        self,
        name: str,
        action: CompositeAction
    ):
        """Register composite action
        
        Args:
            name: Action name
            action: Composite action
        """
        self._sequences[name] = action
        logger.info(f"Registered composite action: {name}")
        
    async def execute_action(
        self,
        action: TaskAction,
        task: Task
    ) -> bool:
        """Execute task action.
        
        Args:
            action: Task action
            task: Associated task
            
        Returns:
            bool: True if action succeeded
        """
        # Start action monitoring
        action_id = None
        if self._task_monitor:
            action_id = self._task_monitor.start_action_monitoring(
                task.task_id,
                action
            )
            
        try:
            # Get handler
            handler = self._action_handlers.get(action.action_type)
            if not handler:
                raise TaskActionError(
                    f"No handler for action type {action.action_type.name}"
                )
                
            # Start action
            action.start_time = datetime.now()
            action.attempts += 1
            
            # Execute with timeout
            if action.timeout:
                try:
                    await asyncio.wait_for(
                        handler(action, task),
                        timeout=action.timeout
                    )
                except asyncio.TimeoutError:
                    raise TaskActionError("Action timeout")
            else:
                await handler(action, task)
                
            # Complete action
            action.end_time = datetime.now()
            
            # End monitoring with success
            if self._task_monitor and action_id:
                self._task_monitor.end_action_monitoring(
                    task.task_id,
                    action_id,
                    success=True
                )
                
            return True
            
        except Exception as e:
            # Handle failure
            action.error = str(e)
            action.end_time = datetime.now()
            
            # End monitoring with failure
            if self._task_monitor and action_id:
                self._task_monitor.end_action_monitoring(
                    task.task_id,
                    action_id,
                    success=False,
                    error=str(e)
                )
            
            # Retry if attempts remain
            if action.attempts <= action.retry_count:
                self._logger.warning(
                    f"Action failed, retrying ({action.attempts}/{action.retry_count})"
                )
                if self._task_monitor and action_id:
                    self._task_monitor.update_action_retry(
                        task.task_id,
                        action_id
                    )
                await asyncio.sleep(action.retry_delay)
                return await self.execute_action(action, task)
                
            self._logger.error(f"Action failed: {e}")
            return False
            
    async def execute_sequence(
        self,
        sequence_name: str,
        task: Task
    ) -> bool:
        """Execute action sequence.
        
        Args:
            sequence_name: Sequence name
            task: Associated task
            
        Returns:
            bool: True if sequence succeeded
        """
        # Get sequence
        sequence = self._sequences.get(sequence_name)
        if not sequence:
            raise TaskActionError(f"Sequence {sequence_name} not found")
            
        # Reset sequence
        if isinstance(sequence, ActionSequence):
            sequence.current_index = 0
        
        # Execute actions
        if isinstance(sequence, ActionSequence):
            while action := sequence.current_action:
                if not await self.execute_action(action, task):
                    return False
                sequence.next_action()
        else:
            return await self.execute_composite_action(sequence, task)
            
        return True
        
    async def execute_composite_action(
        self,
        action: CompositeAction,
        task: Task
    ) -> bool:
        """Execute composite action
        
        Args:
            action: Composite action
            task: Associated task
            
        Returns:
            bool: True if action succeeded
        """
        # Start action monitoring
        action_id = None
        if self._task_monitor:
            action_id = self._task_monitor.start_action_monitoring(
                task.task_id,
                action
            )
            
        try:
            # Prepare context
            context = {
                'task': task,
                'current_state': task.current_state,
                'detected_regions': task.detected_regions
            }
            
            # Execute action
            action.start_time = datetime.now()
            action.attempts += 1
            
            success = await action.execute(self.control_manager, context)
            
            # Complete action
            action.end_time = datetime.now()
            
            # End monitoring
            if self._task_monitor and action_id:
                self._task_monitor.end_action_monitoring(
                    task.task_id,
                    action_id,
                    success=success
                )
                
            return success
            
        except Exception as e:
            # Handle failure
            action.error = str(e)
            action.end_time = datetime.now()
            
            # End monitoring
            if self._task_monitor and action_id:
                self._task_monitor.end_action_monitoring(
                    task.task_id,
                    action_id,
                    success=False,
                    error=str(e)
                )
                
            # Retry if attempts remain
            if action.attempts <= action.retry_count:
                logger.warning(
                    f"Action failed, retrying ({action.attempts}/{action.retry_count})"
                )
                if self._task_monitor and action_id:
                    self._task_monitor.update_action_retry(
                        task.task_id,
                        action_id
                    )
                await asyncio.sleep(action.retry_delay)
                return await self.execute_composite_action(action, task)
                
            logger.error(f"Action failed: {e}")
            return False
            
    async def _handle_click(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle click action.
        
        Args:
            action: Task action
            task: Associated task
        """
        position = action.params.get('position')
        if not position:
            raise TaskActionError("Click position not specified")
            
        await self.control_manager.click(
            x=position[0],
            y=position[1]
        )
        
    async def _handle_drag(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle drag action.
        
        Args:
            action: Task action
            task: Associated task
        """
        start = action.params.get('start')
        end = action.params.get('end')
        if not start or not end:
            raise TaskActionError("Drag start/end not specified")
            
        await self.control_manager.drag(
            start_x=start[0],
            start_y=start[1],
            end_x=end[0],
            end_y=end[1]
        )
        
    async def _handle_key_press(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle key press action.
        
        Args:
            action: Task action
            task: Associated task
        """
        key = action.params.get('key')
        if not key:
            raise TaskActionError("Key not specified")
            
        await self.control_manager.key_press(key)
        
    async def _handle_key_hold(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle key hold action.
        
        Args:
            action: Task action
            task: Associated task
        """
        key = action.params.get('key')
        duration = action.params.get('duration', 1.0)
        if not key:
            raise TaskActionError("Key not specified")
            
        await self.control_manager.key_hold(
            key,
            duration=duration
        )
        
    async def _handle_wait(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle wait action.
        
        Args:
            action: Task action
            task: Associated task
        """
        duration = action.params.get('duration', 1.0)
        await asyncio.sleep(duration)
        
    async def _handle_sequence(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle sequence action.
        
        Args:
            action: Task action
            task: Associated task
        """
        sequence_name = action.params.get('sequence')
        if not sequence_name:
            raise TaskActionError("Sequence name not specified")
            
        if not await self.execute_sequence(sequence_name, task):
            raise TaskActionError(f"Sequence {sequence_name} failed")
            
    async def _handle_parallel(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle parallel action
        
        Args:
            action: Task action
            task: Associated task
        """
        actions = action.params.get('actions', [])
        if not actions:
            raise TaskActionError("No actions specified for parallel execution")
            
        parallel = ParallelAction(
            name=f"parallel_{task.task_id}",
            actions=[
                self._sequences[a] if isinstance(a, str) else a
                for a in actions
            ]
        )
        return await self.execute_composite_action(parallel, task)
        
    async def _handle_conditional(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle conditional action
        
        Args:
            action: Task action
            task: Associated task
        """
        params = action.params
        if not all(k in params for k in ['condition', 'if_action']):
            raise TaskActionError("Missing required parameters for conditional action")
            
        condition = params['condition']
        if isinstance(condition, dict):
            condition = StateCondition(condition['state'])
            
        if_action = params['if_action']
        if isinstance(if_action, str):
            if_action = self._sequences[if_action]
            
        else_action = params.get('else_action')
        if isinstance(else_action, str):
            else_action = self._sequences[else_action]
            
        conditional = ConditionalAction(
            name=f"conditional_{task.task_id}",
            condition=condition,
            if_action=if_action,
            else_action=else_action
        )
        return await self.execute_composite_action(conditional, task)
        
    async def _handle_repeat(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle repeat action
        
        Args:
            action: Task action
            task: Associated task
        """
        params = action.params
        if 'action' not in params:
            raise TaskActionError("Missing action parameter for repeat")
            
        repeat_action = params['action']
        if isinstance(repeat_action, str):
            repeat_action = self._sequences[repeat_action]
            
        until = params.get('until')
        if isinstance(until, dict):
            until = StateCondition(until['state'])
            
        repeat = RepeatAction(
            name=f"repeat_{task.task_id}",
            action=repeat_action,
            count=params.get('count'),
            until=until
        )
        return await self.execute_composite_action(repeat, task)
        
    async def _handle_delay(
        self,
        action: TaskAction,
        task: Task
    ):
        """Handle delay action
        
        Args:
            action: Task action
            task: Associated task
        """
        delay = action.params.get('delay', 1.0)
        delay_action = DelayAction(
            name=f"delay_{task.task_id}",
            delay=delay
        )
        return await self.execute_composite_action(delay_action, task)
        
    def _register_default_handlers(self):
        """Register default action handlers"""
        # Basic actions
        self._action_handlers[ActionType.CLICK] = self._handle_click
        self._action_handlers[ActionType.DRAG] = self._handle_drag
        self._action_handlers[ActionType.KEY_PRESS] = self._handle_key_press
        self._action_handlers[ActionType.KEY_HOLD] = self._handle_key_hold
        self._action_handlers[ActionType.WAIT] = self._handle_wait
        self._action_handlers[ActionType.SEQUENCE] = self._handle_sequence
        
        # Advanced actions
        self._action_handlers[ActionType.PARALLEL] = self._handle_parallel
        self._action_handlers[ActionType.CONDITIONAL] = self._handle_conditional
        self._action_handlers[ActionType.REPEAT] = self._handle_repeat
        self._action_handlers[ActionType.DELAY] = self._handle_delay
        
    async def cleanup(self):
        """Clean up resources."""
        # Cancel running sequences
        for sequence_task in self._running_sequences.values():
            sequence_task.cancel()
            try:
                await sequence_task
            except asyncio.CancelledError:
                pass
                
        self._running_sequences.clear()
