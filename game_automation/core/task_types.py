from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from utils.logger import detailed_logger

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = auto()    # Waiting to execute
    RUNNING = auto()    # Currently executing
    COMPLETED = auto()  # Execution completed
    FAILED = auto()     # Execution failed
    PAUSED = auto()     # Paused

class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class Task:
    """Base task class"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 dependencies: List[str] = None,
                 timeout: float = None):
        """Initialize task
        
        Args:
            task_id: Task ID
            name: Task name
            priority: Task priority
            dependencies: List of dependent task IDs
            timeout: Task timeout in seconds
        """
        self.task_id = task_id
        self.name = name
        self.priority = priority
        self.dependencies = dependencies or []
        self.timeout = timeout
        
        self.status = TaskStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.progress: float = 0.0
        self.performance_metrics: Dict[str, Any] = {}
        
        self._on_complete_callbacks: List[Callable] = []
        self._on_fail_callbacks: List[Callable] = []
        self._on_progress_callbacks: List[Callable] = []

    def execute(self) -> bool:
        """Execute task
        
        Returns:
            bool: Whether execution was successful
        """
        try:
            self.start_time = datetime.now()
            self.status = TaskStatus.RUNNING
            detailed_logger.info(f"Starting task: {self.name} ({self.task_id})")
            
            success = self._execute()
            
            self.end_time = datetime.now()
            if success:
                self.status = TaskStatus.COMPLETED
                self.progress = 1.0
                detailed_logger.info(f"Task completed: {self.name}")
                self._trigger_complete_callbacks()
            else:
                self.status = TaskStatus.FAILED
                detailed_logger.error(f"Task failed: {self.name}")
                self._trigger_fail_callbacks()
            
            return success
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error_message = str(e)
            self.end_time = datetime.now()
            detailed_logger.error(f"Task execution error: {self.name} - {str(e)}")
            self._trigger_fail_callbacks()
            return False

    def _execute(self) -> bool:
        """Actual task execution logic, subclasses must implement this
        
        Returns:
            bool: Whether execution was successful
        """
        raise NotImplementedError("Task._execute() must be implemented by subclass")

    def update_progress(self, progress: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update task progress
        
        Args:
            progress: Progress value (0-1)
            metrics: Performance metrics
        """
        self.progress = max(0.0, min(1.0, progress))
        if metrics:
            self.performance_metrics.update(metrics)
        self._trigger_progress_callbacks()

    def pause(self) -> None:
        """Pause task"""
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.PAUSED
            detailed_logger.info(f"Task paused: {self.name}")

    def resume(self) -> None:
        """Resume task"""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            detailed_logger.info(f"Task resumed: {self.name}")

    def on_complete(self, callback: Callable) -> None:
        """Add completion callback
        
        Args:
            callback: Callback function
        """
        self._on_complete_callbacks.append(callback)

    def on_fail(self, callback: Callable) -> None:
        """Add failure callback
        
        Args:
            callback: Callback function
        """
        self._on_fail_callbacks.append(callback)

    def on_progress(self, callback: Callable) -> None:
        """Add progress callback
        
        Args:
            callback: Callback function
        """
        self._on_progress_callbacks.append(callback)

    def _trigger_complete_callbacks(self) -> None:
        """Trigger completion callbacks"""
        for callback in self._on_complete_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"Complete callback error: {str(e)}")

    def _trigger_fail_callbacks(self) -> None:
        """Trigger failure callbacks"""
        for callback in self._on_fail_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"Fail callback error: {str(e)}")

    def _trigger_progress_callbacks(self) -> None:
        """Trigger progress callbacks"""
        for callback in self._on_progress_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"Progress callback error: {str(e)}")

    def to_dict(self) -> Dict:
        """Convert to dictionary format
        
        Returns:
            Dict: Task information dictionary
        """
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority.name,
            'status': self.status.name,
            'dependencies': self.dependencies,
            'progress': self.progress,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }
