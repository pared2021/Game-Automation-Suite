"""Task priority management."""
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import math

from game_automation.core.error.error_manager import (
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)
from .task_types import Task, TaskStatus

class PriorityLevel(Enum):
    """Priority level enumeration"""
    LOWEST = 0
    VERY_LOW = 1
    LOW = 2
    BELOW_NORMAL = 3
    NORMAL = 4
    ABOVE_NORMAL = 5
    HIGH = 6
    VERY_HIGH = 7
    HIGHEST = 8
    CRITICAL = 9
    
    @classmethod
    def from_int(cls, value: int) -> "PriorityLevel":
        """Convert integer to priority level.
        
        Args:
            value: Integer value (0-9)
            
        Returns:
            PriorityLevel: Priority level
        """
        value = max(0, min(value, 9))
        return list(cls)[value]
        
    @classmethod
    def to_event_priority(cls, value: "PriorityLevel") -> int:
        """Convert to event priority.
        
        Args:
            value: Priority level
            
        Returns:
            int: Event priority (0-9)
        """
        return value.value
        
    @classmethod
    def to_task_priority(cls, value: "PriorityLevel") -> int:
        """Convert to task priority.
        
        Args:
            value: Priority level
            
        Returns:
            int: Task priority (0-3)
        """
        if value.value <= 2:
            return 0  # LOW
        elif value.value <= 5:
            return 1  # NORMAL
        elif value.value <= 7:
            return 2  # HIGH
        else:
            return 3  # URGENT

class PriorityError(GameAutomationError):
    """Priority related errors"""
    pass

class PriorityManager:
    """Task priority manager."""
    
    def __init__(self):
        """Initialize priority manager."""
        # Base priorities
        self._base_priorities: Dict[str, PriorityLevel] = {}
        
        # Dynamic priorities
        self._dynamic_priorities: Dict[str, float] = {}
        
        # Priority inheritance
        self._inherited_priorities: Dict[str, PriorityLevel] = {}
        self._inheritance_graph: Dict[str, Set[str]] = {}
        
        # Priority inversion prevention
        self._resource_holders: Dict[str, Set[str]] = {}  # resource -> tasks
        self._task_resources: Dict[str, Set[str]] = {}    # task -> resources
        
        # Priority history
        self._priority_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Priority boost tracking
        self._boost_start_times: Dict[str, datetime] = {}
        self._boost_durations: Dict[str, float] = {}
        
        # Priority aging
        self._creation_times: Dict[str, datetime] = {}
        self._aging_factor = 0.1  # Priority boost per minute
        
        # Priority ceiling
        self._resource_ceilings: Dict[str, PriorityLevel] = {}
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
    def register_task(
        self,
        task: Task,
        base_priority: Optional[PriorityLevel] = None
    ) -> None:
        """Register task with priority manager.
        
        Args:
            task: Task instance
            base_priority: Optional base priority level
        """
        task_id = task.task_id
        
        # Set base priority
        if base_priority is None:
            base_priority = PriorityLevel.NORMAL
        self._base_priorities[task_id] = base_priority
        
        # Initialize dynamic priority
        self._dynamic_priorities[task_id] = 0.0
        
        # Initialize inheritance
        self._inheritance_graph[task_id] = set()
        
        # Initialize resource tracking
        self._task_resources[task_id] = set()
        
        # Record creation time
        self._creation_times[task_id] = datetime.now()
        
        # Initialize history
        self._priority_history[task_id] = [{
            'timestamp': datetime.now(),
            'base_priority': base_priority,
            'dynamic_priority': 0.0,
            'inherited_priority': None,
            'effective_priority': base_priority,
            'reason': 'Task registered'
        }]
        
        self._logger.info(
            f"Registered task {task.name} ({task_id}) with priority {base_priority.name}"
        )
        
    def unregister_task(self, task_id: str) -> None:
        """Unregister task from priority manager.
        
        Args:
            task_id: Task ID
        """
        # Remove from base priorities
        if task_id in self._base_priorities:
            del self._base_priorities[task_id]
            
        # Remove from dynamic priorities
        if task_id in self._dynamic_priorities:
            del self._dynamic_priorities[task_id]
            
        # Remove from inheritance
        if task_id in self._inherited_priorities:
            del self._inherited_priorities[task_id]
        if task_id in self._inheritance_graph:
            del self._inheritance_graph[task_id]
            
        # Remove from resource tracking
        if task_id in self._task_resources:
            resources = self._task_resources[task_id]
            for resource in resources:
                if resource in self._resource_holders:
                    self._resource_holders[resource].discard(task_id)
            del self._task_resources[task_id]
            
        # Remove from boost tracking
        if task_id in self._boost_start_times:
            del self._boost_start_times[task_id]
        if task_id in self._boost_durations:
            del self._boost_durations[task_id]
            
        # Remove from creation times
        if task_id in self._creation_times:
            del self._creation_times[task_id]
            
        self._logger.info(f"Unregistered task {task_id}")
        
    def get_priority(self, task_id: str) -> PriorityLevel:
        """Get effective task priority.
        
        Args:
            task_id: Task ID
            
        Returns:
            PriorityLevel: Effective priority level
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
            
        # Get base priority
        base_priority = self._base_priorities[task_id]
        base_value = base_priority.value
        
        # Add dynamic priority
        dynamic_value = self._dynamic_priorities.get(task_id, 0.0)
        
        # Add inherited priority
        inherited_priority = self._inherited_priorities.get(task_id)
        if inherited_priority:
            inherited_value = inherited_priority.value
            effective_value = max(base_value + dynamic_value, inherited_value)
        else:
            effective_value = base_value + dynamic_value
            
        # Apply aging
        creation_time = self._creation_times[task_id]
        age_minutes = (datetime.now() - creation_time).total_seconds() / 60
        aging_boost = age_minutes * self._aging_factor
        effective_value += aging_boost
        
        # Clamp to valid range
        effective_value = max(0, min(9, effective_value))
        
        return PriorityLevel.from_int(int(effective_value))
        
    def set_base_priority(
        self,
        task_id: str,
        priority: PriorityLevel
    ) -> None:
        """Set task base priority.
        
        Args:
            task_id: Task ID
            priority: Priority level
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
            
        old_priority = self._base_priorities[task_id]
        self._base_priorities[task_id] = priority
        
        self._record_priority_change(
            task_id,
            'Base priority changed',
            old_priority=old_priority,
            new_priority=priority
        )
        
    def boost_priority(
        self,
        task_id: str,
        boost: float,
        duration: Optional[float] = None
    ) -> None:
        """Temporarily boost task priority.
        
        Args:
            task_id: Task ID
            boost: Priority boost value
            duration: Optional boost duration in seconds
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
            
        old_boost = self._dynamic_priorities.get(task_id, 0.0)
        self._dynamic_priorities[task_id] = boost
        
        if duration:
            self._boost_start_times[task_id] = datetime.now()
            self._boost_durations[task_id] = duration
            
        self._record_priority_change(
            task_id,
            'Priority boosted',
            old_boost=old_boost,
            new_boost=boost,
            duration=duration
        )
        
    def inherit_priority(
        self,
        task_id: str,
        from_task_id: str
    ) -> None:
        """Inherit priority from another task.
        
        Args:
            task_id: Task ID
            from_task_id: Task to inherit from
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
        if from_task_id not in self._base_priorities:
            raise PriorityError(f"Task {from_task_id} not registered")
            
        # Add inheritance edge
        self._inheritance_graph[from_task_id].add(task_id)
        
        # Update inherited priority
        inherited_priority = self.get_priority(from_task_id)
        old_priority = self._inherited_priorities.get(task_id)
        self._inherited_priorities[task_id] = inherited_priority
        
        self._record_priority_change(
            task_id,
            'Priority inherited',
            from_task=from_task_id,
            old_priority=old_priority,
            new_priority=inherited_priority
        )
        
    def acquire_resource(
        self,
        task_id: str,
        resource_id: str
    ) -> None:
        """Acquire resource and handle priority inheritance.
        
        Args:
            task_id: Task ID
            resource_id: Resource ID
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
            
        # Add to resource tracking
        if resource_id not in self._resource_holders:
            self._resource_holders[resource_id] = set()
        self._resource_holders[resource_id].add(task_id)
        self._task_resources[task_id].add(resource_id)
        
        # Check for priority inversion
        ceiling = self._resource_ceilings.get(resource_id)
        if ceiling:
            current_priority = self.get_priority(task_id)
            if current_priority.value < ceiling.value:
                # Boost priority to ceiling
                self.boost_priority(
                    task_id,
                    ceiling.value - current_priority.value
                )
                self._record_priority_change(
                    task_id,
                    'Priority ceiling protocol',
                    resource=resource_id,
                    ceiling=ceiling
                )
                
    def release_resource(
        self,
        task_id: str,
        resource_id: str
    ) -> None:
        """Release resource and handle priority inheritance.
        
        Args:
            task_id: Task ID
            resource_id: Resource ID
        """
        if task_id not in self._base_priorities:
            raise PriorityError(f"Task {task_id} not registered")
            
        # Remove from resource tracking
        if resource_id in self._resource_holders:
            self._resource_holders[resource_id].discard(task_id)
        if task_id in self._task_resources:
            self._task_resources[task_id].discard(resource_id)
            
        # Remove priority boost if it was due to ceiling protocol
        ceiling = self._resource_ceilings.get(resource_id)
        if ceiling:
            current_priority = self.get_priority(task_id)
            if current_priority == ceiling:
                self._dynamic_priorities[task_id] = 0.0
                self._record_priority_change(
                    task_id,
                    'Priority ceiling removed',
                    resource=resource_id
                )
                
    def set_resource_ceiling(
        self,
        resource_id: str,
        ceiling: PriorityLevel
    ) -> None:
        """Set resource priority ceiling.
        
        Args:
            resource_id: Resource ID
            ceiling: Priority ceiling
        """
        self._resource_ceilings[resource_id] = ceiling
        
        # Update any current holders
        if resource_id in self._resource_holders:
            for task_id in self._resource_holders[resource_id]:
                current_priority = self.get_priority(task_id)
                if current_priority.value < ceiling.value:
                    self.boost_priority(
                        task_id,
                        ceiling.value - current_priority.value
                    )
                    
    def get_priority_history(
        self,
        task_id: str
    ) -> List[Dict[str, Any]]:
        """Get task priority history.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[Dict[str, Any]]: Priority history
        """
        return self._priority_history.get(task_id, [])
        
    def _record_priority_change(
        self,
        task_id: str,
        reason: str,
        **kwargs
    ) -> None:
        """Record priority change in history.
        
        Args:
            task_id: Task ID
            reason: Change reason
            **kwargs: Additional data to record
        """
        if task_id not in self._priority_history:
            self._priority_history[task_id] = []
            
        entry = {
            'timestamp': datetime.now(),
            'base_priority': self._base_priorities.get(task_id),
            'dynamic_priority': self._dynamic_priorities.get(task_id),
            'inherited_priority': self._inherited_priorities.get(task_id),
            'effective_priority': self.get_priority(task_id),
            'reason': reason,
            **kwargs
        }
        
        self._priority_history[task_id].append(entry)
        
    def update(self) -> None:
        """Update priority manager state."""
        current_time = datetime.now()
        
        # Update temporary boosts
        for task_id in list(self._boost_start_times.keys()):
            start_time = self._boost_start_times[task_id]
            duration = self._boost_durations[task_id]
            if (current_time - start_time).total_seconds() >= duration:
                # Remove boost
                self._dynamic_priorities[task_id] = 0.0
                del self._boost_start_times[task_id]
                del self._boost_durations[task_id]
                
                self._record_priority_change(
                    task_id,
                    'Priority boost expired'
                )
                
    def get_task_resources(self, task_id: str) -> Set[str]:
        """Get resources held by task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Set[str]: Set of resource IDs
        """
        return self._task_resources.get(task_id, set())
        
    def get_resource_holders(self, resource_id: str) -> Set[str]:
        """Get tasks holding resource.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            Set[str]: Set of task IDs
        """
        return self._resource_holders.get(resource_id, set())
