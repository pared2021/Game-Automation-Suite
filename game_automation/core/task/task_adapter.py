"""Task adapter for game state integration."""
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
import asyncio
import logging

from .recognition.state_analyzer import StateAnalyzer, GameState
from .task_types import Task, TaskStatus, TaskType
from .task_executor import TaskExecutor
from .error.error_manager import GameAutomationError

class TaskAdapterError(GameAutomationError):
    """Task adapter related errors"""
    pass

class StateTaskMapping:
    """Mapping between game states and tasks."""
    
    def __init__(self):
        """Initialize state task mapping."""
        self.state_to_tasks: Dict[Type[GameState], List[Type[Task]]] = {}
        self.task_requirements: Dict[Type[Task], List[Type[GameState]]] = {}
        
    def register_mapping(
        self,
        state_type: Type[GameState],
        task_type: Type[Task]
    ):
        """Register state to task mapping.
        
        Args:
            state_type: Game state type
            task_type: Task type
        """
        if state_type not in self.state_to_tasks:
            self.state_to_tasks[state_type] = []
        self.state_to_tasks[state_type].append(task_type)
        
        if task_type not in self.task_requirements:
            self.task_requirements[task_type] = []
        self.task_requirements[task_type].append(state_type)
        
    def get_tasks_for_state(
        self,
        state_type: Type[GameState]
    ) -> List[Type[Task]]:
        """Get tasks for game state.
        
        Args:
            state_type: Game state type
            
        Returns:
            List[Type[Task]]: List of task types
        """
        return self.state_to_tasks.get(state_type, [])
        
    def get_required_states(
        self,
        task_type: Type[Task]
    ) -> List[Type[GameState]]:
        """Get required states for task.
        
        Args:
            task_type: Task type
            
        Returns:
            List[Type[GameState]]: List of required state types
        """
        return self.task_requirements.get(task_type, [])

class TaskAdapter:
    """Task adapter for game state integration."""
    
    def __init__(
        self,
        state_analyzer: StateAnalyzer,
        task_executor: TaskExecutor
    ):
        """Initialize task adapter.
        
        Args:
            state_analyzer: State analyzer instance
            task_executor: Task executor instance
        """
        self.state_analyzer = state_analyzer
        self.task_executor = task_executor
        self.state_mapping = StateTaskMapping()
        self._logger = logging.getLogger(__name__)
        
    def register_state_task(
        self,
        state_type: Type[GameState],
        task_type: Type[Task]
    ):
        """Register state to task mapping.
        
        Args:
            state_type: Game state type
            task_type: Task type
        """
        self.state_mapping.register_mapping(state_type, task_type)
        
    async def handle_state_change(self, new_state: GameState):
        """Handle game state change.
        
        Args:
            new_state: New game state
        """
        try:
            # Get tasks for state
            state_type = type(new_state)
            task_types = self.state_mapping.get_tasks_for_state(state_type)
            
            for task_type in task_types:
                # Create and add task
                task_name = f"{task_type.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.task_executor.add_task(
                    name=task_name,
                    task_type=TaskType.CUSTOM,  # Use custom type for state-triggered tasks
                    priority=task_type.default_priority,
                    params={
                        'state_type': state_type.__name__,
                        'state_data': new_state.__dict__
                    }
                )
                
            self._logger.debug(
                f"Created {len(task_types)} tasks for state {state_type.__name__}"
            )
            
        except Exception as e:
            self._logger.error(f"Error handling state change: {e}")
            raise TaskAdapterError(f"Failed to handle state change: {e}")
            
    async def validate_task_state(
        self,
        task: Task,
        current_state: GameState
    ) -> bool:
        """Validate task state requirements.
        
        Args:
            task: Task instance
            current_state: Current game state
            
        Returns:
            bool: True if task can run in current state
        """
        try:
            # Get required states
            required_states = self.state_mapping.get_required_states(type(task))
            
            # Check if current state matches requirements
            return any(
                isinstance(current_state, state_type)
                for state_type in required_states
            )
            
        except Exception as e:
            self._logger.error(f"Error validating task state: {e}")
            return False
