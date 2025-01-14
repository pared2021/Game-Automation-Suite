"""Task lifecycle management."""

import asyncio
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from utils.logger import detailed_logger
from utils.metrics import MetricsCollector
from .task_types import Task, TaskStatus
from .task_priority import PriorityLevel

class TaskState(Enum):
    """Task state enumeration"""
    CREATED = auto()      # Task just created
    VALIDATING = auto()   # Validating task parameters
    PENDING = auto()      # Waiting to be scheduled
    READY = auto()        # Ready to execute
    PREPARING = auto()    # Preparing for execution
    RUNNING = auto()      # Currently executing
    PAUSED = auto()       # Execution paused
    RESUMING = auto()     # Resuming from pause
    BLOCKED = auto()      # Blocked by dependencies
    WAITING = auto()      # Waiting for resources
    COMPLETING = auto()   # Completing execution
    COMPLETED = auto()    # Execution completed
    FAILED = auto()       # Execution failed
    CANCELLED = auto()    # Execution cancelled
    TIMEOUT = auto()      # Execution timeout
    RETRYING = auto()     # Retrying after failure
    ABORTING = auto()     # Aborting execution
    SUSPENDED = auto()    # Long-term suspended
    RECOVERING = auto()   # Recovering from failure
    UNKNOWN = auto()      # Unknown state

@dataclass
class StateMetrics:
    """State transition metrics."""
    entry_count: int = 0
    exit_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_entry: Optional[datetime] = None
    last_exit: Optional[datetime] = None
    error_count: int = 0
    timeout_count: int = 0

@dataclass
class TransitionMetrics:
    """Transition metrics."""
    attempt_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_attempt: Optional[datetime] = None
    error_types: Dict[str, int] = field(default_factory=dict)

class TaskEvent:
    """Task lifecycle event."""
    
    def __init__(self, task: Task, event_type: str, 
                 old_state: Optional[TaskState] = None,
                 new_state: Optional[TaskState] = None,
                 data: Optional[Dict[str, Any]] = None):
        """Initialize task event.
        
        Args:
            task: Task instance
            event_type: Event type
            old_state: Optional old state
            new_state: Optional new state
            data: Optional event data
        """
        self.task = task
        self.event_type = event_type
        self.old_state = old_state
        self.new_state = new_state
        self.data = data or {}
        self.timestamp = datetime.now()
        self.priority = PriorityLevel.NORMAL

class StateTransition:
    """Task state transition."""
    
    def __init__(self, 
                 from_state: TaskState,
                 to_state: TaskState,
                 conditions: Optional[List[Callable[[Task], bool]]] = None,
                 pre_actions: Optional[List[Callable[[Task], None]]] = None,
                 post_actions: Optional[List[Callable[[Task], None]]] = None,
                 timeout: Optional[float] = None,
                 rollback: Optional[Callable[[Task], None]] = None):
        """Initialize state transition.
        
        Args:
            from_state: Source state
            to_state: Target state
            conditions: Optional transition conditions
            pre_actions: Optional pre-transition actions
            post_actions: Optional post-transition actions
            timeout: Optional transition timeout
            rollback: Optional rollback action
        """
        self.from_state = from_state
        self.to_state = to_state
        self.conditions = conditions or []
        self.pre_actions = pre_actions or []
        self.post_actions = post_actions or []
        self.timeout = timeout
        self.rollback = rollback
        self.metrics = TransitionMetrics()

class TaskLifecycleManager:
    """Task lifecycle manager."""
    
    def __init__(self):
        """Initialize task lifecycle manager."""
        # State machine
        self._transitions: Dict[TaskState, List[StateTransition]] = {}
        self._setup_transitions()
        
        # Task tracking
        self._task_states: Dict[str, TaskState] = {}
        self._task_data: Dict[str, Dict[str, Any]] = {}
        
        # Event handling
        self._event_handlers: Dict[str, List[Callable[[TaskEvent], None]]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._state_metrics: Dict[TaskState, StateMetrics] = {
            state: StateMetrics() for state in TaskState
        }
        self._collector = MetricsCollector()
        
        # Health check
        self._health_checks: List[Callable[[], bool]] = []
        self._last_health_check = datetime.now()
        self._health_check_interval = 60  # seconds
        
        # Logging
        self._logger = logging.getLogger(__name__)
        
    def _setup_transitions(self):
        """Setup state transitions."""
        # Define valid transitions
        transitions = [
            # Initialization path
            StateTransition(
                TaskState.CREATED,
                TaskState.VALIDATING,
                pre_actions=[self._validate_task]
            ),
            StateTransition(
                TaskState.VALIDATING,
                TaskState.PENDING,
                conditions=[self._is_valid]
            ),
            StateTransition(
                TaskState.PENDING,
                TaskState.READY,
                conditions=[self._check_dependencies]
            ),
            
            # Execution path
            StateTransition(
                TaskState.READY,
                TaskState.PREPARING,
                pre_actions=[self._prepare_execution]
            ),
            StateTransition(
                TaskState.PREPARING,
                TaskState.RUNNING,
                conditions=[self._check_resources]
            ),
            StateTransition(
                TaskState.RUNNING,
                TaskState.COMPLETING,
                conditions=[self._check_completion],
                post_actions=[self._cleanup_resources]
            ),
            StateTransition(
                TaskState.COMPLETING,
                TaskState.COMPLETED,
                post_actions=[self._finalize_task]
            ),
            
            # Pause/Resume path
            StateTransition(
                TaskState.RUNNING,
                TaskState.PAUSED,
                pre_actions=[self._pause_execution]
            ),
            StateTransition(
                TaskState.PAUSED,
                TaskState.RESUMING,
                conditions=[self._can_resume]
            ),
            StateTransition(
                TaskState.RESUMING,
                TaskState.RUNNING,
                post_actions=[self._resume_execution]
            ),
            
            # Error handling path
            StateTransition(
                TaskState.RUNNING,
                TaskState.FAILED,
                conditions=[self._check_failure],
                post_actions=[self._handle_failure]
            ),
            StateTransition(
                TaskState.FAILED,
                TaskState.RETRYING,
                conditions=[self._can_retry],
                pre_actions=[self._prepare_retry]
            ),
            StateTransition(
                TaskState.RETRYING,
                TaskState.READY,
                post_actions=[self._reset_task]
            ),
            
            # Recovery path
            StateTransition(
                TaskState.FAILED,
                TaskState.RECOVERING,
                conditions=[self._can_recover],
                pre_actions=[self._prepare_recovery]
            ),
            StateTransition(
                TaskState.RECOVERING,
                TaskState.READY,
                conditions=[self._recovery_complete]
            ),
            
            # Blocking path
            StateTransition(
                TaskState.PENDING,
                TaskState.BLOCKED,
                conditions=[self._check_blocked]
            ),
            StateTransition(
                TaskState.BLOCKED,
                TaskState.READY,
                conditions=[self._check_dependencies]
            ),
            
            # Resource waiting path
            StateTransition(
                TaskState.READY,
                TaskState.WAITING,
                conditions=[self._check_resource_constraints]
            ),
            StateTransition(
                TaskState.WAITING,
                TaskState.READY,
                conditions=[self._check_resources]
            ),
            
            # Cancellation path
            StateTransition(
                TaskState.RUNNING,
                TaskState.ABORTING,
                pre_actions=[self._prepare_abort]
            ),
            StateTransition(
                TaskState.ABORTING,
                TaskState.CANCELLED,
                post_actions=[self._cleanup_resources]
            ),
            
            # Timeout path
            StateTransition(
                TaskState.RUNNING,
                TaskState.TIMEOUT,
                conditions=[self._check_timeout],
                post_actions=[self._handle_timeout]
            ),
            
            # Suspension path
            StateTransition(
                TaskState.RUNNING,
                TaskState.SUSPENDED,
                pre_actions=[self._prepare_suspension]
            ),
            StateTransition(
                TaskState.SUSPENDED,
                TaskState.RESUMING,
                conditions=[self._can_resume_from_suspension]
            )
        ]
        
        # Build transition map
        for transition in transitions:
            if transition.from_state not in self._transitions:
                self._transitions[transition.from_state] = []
            self._transitions[transition.from_state].append(transition)
            
    async def _process_events(self):
        """Process task events."""
        while True:
            try:
                event = await self._event_queue.get()
                
                # Execute event handlers
                handlers = (
                    self._event_handlers.get(event.event_type, []) +
                    self._event_handlers.get('*', [])
                )
                
                for handler in handlers:
                    try:
                        await asyncio.create_task(handler(event))
                    except Exception as e:
                        self._logger.error(
                            f"Event handler failed: {event.event_type} - {str(e)}"
                        )
                        
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Event processing failed: {str(e)}")
                await asyncio.sleep(1.0)
                
    def start(self):
        """Start lifecycle manager."""
        if not self._event_task:
            self._event_task = asyncio.create_task(self._process_events())
            
    def stop(self):
        """Stop lifecycle manager."""
        if self._event_task:
            self._event_task.cancel()
            self._event_task = None
            
    def register_task(self, task: Task, max_retries: int = 3) -> None:
        """Register task with lifecycle manager.
        
        Args:
            task: Task to register
            max_retries: Maximum retry attempts
        """
        self._task_states[task.task_id] = TaskState.CREATED
        self._task_data[task.task_id] = {
            'retry_count': 0,
            'max_retries': max_retries,
            'state_history': [],
            'resources': set(),
            'metrics': {},
            'recovery_attempts': 0,
            'suspension_time': None
        }
        
        # Add task hooks
        task.on_complete(lambda t: self.transition_state(t, TaskState.COMPLETED))
        task.on_fail(lambda t: self.transition_state(t, TaskState.FAILED))
        
        # Emit event
        self._emit_event(TaskEvent(
            task,
            'task_registered',
            new_state=TaskState.CREATED
        ))
        
        # Initial transition
        self.transition_state(task, TaskState.VALIDATING)
        
    def transition_state(self, task: Task, target_state: TaskState) -> bool:
        """Transition task to target state.
        
        Args:
            task: Task to transition
            target_state: Target state
            
        Returns:
            bool: True if transition successful
        """
        current_state = self._task_states.get(task.task_id)
        if not current_state:
            return False
            
        # Find valid transition
        valid_transition = None
        for transition in self._transitions.get(current_state, []):
            if transition.to_state == target_state:
                # Check all conditions
                conditions_met = all(
                    condition(task) for condition in transition.conditions
                )
                if conditions_met:
                    valid_transition = transition
                    break
                    
        if not valid_transition:
            self._logger.warning(
                f"Invalid state transition: {task.task_id} "
                f"from {current_state} to {target_state}"
            )
            return False
            
        # Update metrics
        transition.metrics.attempt_count += 1
        transition.metrics.last_attempt = datetime.now()
        
        try:
            # Execute pre-actions
            for action in valid_transition.pre_actions:
                action(task)
                
            # Update state
            old_state = self._task_states[task.task_id]
            self._task_states[task.task_id] = target_state
            
            # Update state metrics
            if old_state != target_state:
                old_metrics = self._state_metrics[old_state]
                new_metrics = self._state_metrics[target_state]
                
                # Exit old state
                old_metrics.exit_count += 1
                if old_metrics.last_entry:
                    duration = (datetime.now() - old_metrics.last_entry).total_seconds()
                    old_metrics.total_duration += duration
                    old_metrics.min_duration = min(old_metrics.min_duration, duration)
                    old_metrics.max_duration = max(old_metrics.max_duration, duration)
                old_metrics.last_exit = datetime.now()
                
                # Enter new state
                new_metrics.entry_count += 1
                new_metrics.last_entry = datetime.now()
            
            # Record history
            self._task_data[task.task_id]['state_history'].append({
                'from_state': old_state,
                'to_state': target_state,
                'timestamp': datetime.now(),
                'duration': (datetime.now() - transition.metrics.last_attempt).total_seconds()
            })
            
            # Execute post-actions
            for action in valid_transition.post_actions:
                action(task)
                
            # Update transition metrics
            transition.metrics.success_count += 1
            duration = (datetime.now() - transition.metrics.last_attempt).total_seconds()
            transition.metrics.total_duration += duration
            transition.metrics.min_duration = min(transition.metrics.min_duration, duration)
            transition.metrics.max_duration = max(transition.metrics.max_duration, duration)
            
            # Emit event
            self._emit_event(TaskEvent(
                task,
                'state_changed',
                old_state=old_state,
                new_state=target_state
            ))
            
            self._logger.info(
                f"Task state transition: {task.task_id} "
                f"from {old_state} to {target_state}"
            )
            return True
            
        except Exception as e:
            # Update error metrics
            transition.metrics.failure_count += 1
            error_type = type(e).__name__
            transition.metrics.error_types[error_type] = (
                transition.metrics.error_types.get(error_type, 0) + 1
            )
            
            # Execute rollback if available
            if valid_transition.rollback:
                try:
                    valid_transition.rollback(task)
                except Exception as rollback_error:
                    self._logger.error(
                        f"Rollback failed: {task.task_id} - {str(rollback_error)}"
                    )
                    
            self._logger.error(f"Transition failed: {task.task_id} - {str(e)}")
            return False
            
    def add_event_handler(self, event_type: str,
                         handler: Callable[[TaskEvent], None]) -> None:
        """Add event handler.
        
        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    def _emit_event(self, event: TaskEvent) -> None:
        """Emit task event.
        
        Args:
            event: Event to emit
        """
        asyncio.create_task(self._event_queue.put(event))
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get lifecycle metrics.
        
        Returns:
            Dict[str, Any]: Metrics data
        """
        metrics = {
            'states': {
                state.name: {
                    'entry_count': self._state_metrics[state].entry_count,
                    'exit_count': self._state_metrics[state].exit_count,
                    'total_duration': self._state_metrics[state].total_duration,
                    'min_duration': self._state_metrics[state].min_duration,
                    'max_duration': self._state_metrics[state].max_duration,
                    'error_count': self._state_metrics[state].error_count,
                    'timeout_count': self._state_metrics[state].timeout_count
                }
                for state in TaskState
            },
            'transitions': {
                f"{t.from_state.name}->{t.to_state.name}": {
                    'attempt_count': t.metrics.attempt_count,
                    'success_count': t.metrics.success_count,
                    'failure_count': t.metrics.failure_count,
                    'total_duration': t.metrics.total_duration,
                    'min_duration': t.metrics.min_duration,
                    'max_duration': t.metrics.max_duration,
                    'error_types': t.metrics.error_types
                }
                for transitions in self._transitions.values()
                for t in transitions
            }
        }
        
        return metrics
        
    def add_health_check(self, check: Callable[[], bool]) -> None:
        """Add health check function.
        
        Args:
            check: Health check function
        """
        self._health_checks.append(check)
        
    def check_health(self) -> bool:
        """Check lifecycle manager health.
        
        Returns:
            bool: True if healthy
        """
        now = datetime.now()
        if (now - self._last_health_check).total_seconds() < self._health_check_interval:
            return True
            
        self._last_health_check = now
        
        try:
            return all(check() for check in self._health_checks)
        except Exception as e:
            self._logger.error(f"Health check failed: {str(e)}")
            return False
            
    def get_task_resources(self, task_id: str) -> Set[str]:
        """Get resources held by task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Set[str]: Set of resource IDs
        """
        return self._task_data.get(task_id, {}).get('resources', set())
        
    def get_task_metrics(self, task_id: str) -> Dict[str, Any]:
        """Get task metrics.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Task metrics
        """
        return self._task_data.get(task_id, {}).get('metrics', {})
