"""Task execution and management."""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
from enum import Enum, auto
import uuid
import logging

from game_automation.core.error.error_manager import (
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)
# Temporarily comment out recognition imports for testing
# from game_automation.core.recognition.state_analyzer import GameState
from game_automation.core.task.task_adapter import TaskAdapter
from game_automation.core.monitor import TaskMonitor

# Temporary GameState mock
class GameState:
    """Temporary mock for GameState"""
    pass

class TaskExecutorError(GameAutomationError):
    """Task executor related errors"""
    pass

class TaskType(Enum):
    """任务类型"""
    DAILY = auto()      # 日常任务
    WEEKLY = auto()     # 周常任务
    EVENT = auto()      # 活动任务
    STORY = auto()      # 剧情任务
    BATTLE = auto()     # 战斗任务
    RESOURCE = auto()   # 资源任务
    CUSTOM = auto()     # 自定义任务

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()    # 等待中
    RUNNING = auto()    # 运行中
    COMPLETED = auto()  # 已完成
    FAILED = auto()     # 失败
    CANCELLED = auto()  # 已取消

class Task:
    """任务"""
    def __init__(
        self,
        task_id: str,
        name: str,
        task_type: TaskType,
        priority: TaskPriority,
        params: Dict = None
    ):
        self.task_id = task_id
        self.name = name
        self.task_type = task_type
        self.priority = priority
        self.params = params or {}
        
        self.status = TaskStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        
        # State validation
        self.required_states: List[type] = []
        self.current_state: Optional[GameState] = None
        
    @property
    def state_valid(self) -> bool:
        """Check if current state is valid for task.
        
        Returns:
            bool: True if state is valid
        """
        if not self.required_states:
            return True
            
        if not self.current_state:
            return False
            
        return any(
            isinstance(self.current_state, state_type)
            for state_type in self.required_states
        )

class TaskExecutor:
    """任务执行器"""
    
    def __init__(
        self,
        task_adapter: TaskAdapter = None,
        task_monitor: Optional[TaskMonitor] = None
    ):
        """Initialize task executor.
        
        Args:
            task_adapter: Optional task adapter instance
            task_monitor: Optional task monitor instance
        """
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_handlers: Dict[TaskType, Callable] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._task_adapter = task_adapter
        self._task_monitor = task_monitor
        self._logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize executor."""
        # Register default handlers
        self._register_default_handlers()
        
    async def cleanup(self):
        """Clean up resources."""
        # Cancel all running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        self._running_tasks.clear()
        self._tasks.clear()
        
    def _register_default_handlers(self):
        """Register default task handlers."""
        self._task_handlers[TaskType.CUSTOM] = self._new_task_handler
        
    def register_task_handler(
        self,
        task_type: TaskType,
        handler: Callable
    ):
        """Register task handler.
        
        Args:
            task_type: Task type
            handler: Handler function
        """
        self._task_handlers[task_type] = handler
        
    async def add_task(
        self,
        name: str,
        task_type: TaskType,
        priority: TaskPriority,
        params: Dict = None
    ) -> str:
        """Add task.
        
        Args:
            name: Task name
            task_type: Task type
            priority: Task priority
            params: Task parameters
            
        Returns:
            str: Task ID
        """
        # Create task
        task_id = str(uuid.uuid4())
        task = Task(task_id, name, task_type, priority, params)
        
        # Add to tasks
        self._tasks[task_id] = task
        
        # Add to queue
        await self._task_queue.put((priority.value, task_id))
        
        # Start monitoring
        if self._task_monitor:
            self._task_monitor.start_task_monitoring(task)
        
        self._logger.debug(f"Added task {name} ({task_id})")
        return task_id
        
    async def cancel_task(self, task_id: str):
        """Cancel task.
        
        Args:
            task_id: Task ID
        """
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            try:
                await self._running_tasks[task_id]
            except asyncio.CancelledError:
                pass
                
            del self._running_tasks[task_id]
            
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
        self._logger.debug(f"Cancelled task {task_id}")
        
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task info.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict]: Task info
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
            
        return {
            'task_id': task.task_id,
            'name': task.name,
            'type': task.task_type.name,
            'priority': task.priority.name,
            'status': task.status.name,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'error': task.error,
            'params': task.params
        }
        
    def get_tasks(
        self,
        task_type: TaskType = None,
        status: TaskStatus = None
    ) -> List[Dict]:
        """Get task list.
        
        Args:
            task_type: Filter by task type
            status: Filter by status
            
        Returns:
            List[Dict]: Task list
        """
        tasks = []
        for task in self._tasks.values():
            if task_type and task.task_type != task_type:
                continue
                
            if status and task.status != status:
                continue
                
            tasks.append(self.get_task(task.task_id))
            
        return tasks
        
    async def start(self):
        """Start executor."""
        if self._running:
            return
            
        self._running = True
        asyncio.create_task(self._process_tasks())
        
    async def stop(self):
        """Stop executor."""
        if not self._running:
            return
            
        self._running = False
        await self.cleanup()
        
    async def _process_tasks(self):
        """Process task queue."""
        try:
            while self._running:
                try:
                    # Get next task
                    _, task_id = await self._task_queue.get()
                    task = self._tasks.get(task_id)
                    if not task:
                        continue
                        
                    # Validate state if adapter exists
                    if (
                        self._task_adapter and
                        not await self._task_adapter.validate_task_state(
                            task,
                            task.current_state
                        )
                    ):
                        self._logger.warning(
                            f"Task {task_id} state validation failed"
                        )
                        if self._task_monitor:
                            self._task_monitor.update_state_changes(task_id)
                        continue
                        
                    # Get handler
                    handler = self._task_handlers.get(task.task_type)
                    if not handler:
                        self._logger.error(
                            f"No handler for task type {task.task_type.name}"
                        )
                        continue
                        
                    # Start task
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    
                    # Run handler
                    self._running_tasks[task_id] = asyncio.create_task(
                        handler(task)
                    )
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._logger.error(f"Error processing tasks: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            pass
            
    async def _new_task_handler(self, task: Task):
        """New task type handler.
        
        Args:
            task: Task instance
        """
        try:
            # Process task
            self._logger.info(f"Processing task {task.name}")
            
            # Monitor resource usage
            if self._task_monitor:
                # This is just an example, you should implement actual resource monitoring
                self._task_monitor.update_resource_usage(
                    task.task_id,
                    memory_usage=50.0,  # Example value
                    cpu_usage=30.0      # Example value
                )
            
            await asyncio.sleep(1.0)  # Simulate work
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # End monitoring
            if self._task_monitor:
                self._task_monitor.end_task_monitoring(task)
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            if self._task_monitor:
                self._task_monitor.end_task_monitoring(task)
            raise
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            self._logger.error(f"Task {task.name} failed: {e}")
            if self._task_monitor:
                self._task_monitor.end_task_monitoring(task)
            
        finally:
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
