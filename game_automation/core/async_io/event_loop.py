"""Event Loop Management for Game Automation Suite.

This module provides a custom event loop implementation that extends asyncio's event loop
with game-specific functionality and performance optimizations.
"""

import asyncio
import logging
import time
import heapq
import psutil
import traceback
from enum import Enum
from typing import Optional, Dict, Set, List, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Task states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priorities."""
    LOW = 0
    NORMAL = 5
    HIGH = 9
    
    @classmethod
    def from_int(cls, value: int) -> "TaskPriority":
        """Convert integer to TaskPriority."""
        if value <= 3:
            return cls.LOW
        elif value <= 6:
            return cls.NORMAL
        else:
            return cls.HIGH


@dataclass
class TaskInfo:
    """Information about a task."""
    name: str
    priority: TaskPriority
    state: TaskState
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error: Optional[Exception] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class EventLoopStats:
    """Statistics for event loop performance monitoring."""
    start_time: datetime = field(default_factory=datetime.now)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    avg_task_duration: float = 0.0
    total_task_duration: float = 0.0
    peak_memory_usage: int = 0
    current_memory_usage: int = 0
    cpu_usage: float = 0.0
    current_tasks: Dict[str, TaskInfo] = field(default_factory=dict)
    task_history: List[TaskInfo] = field(default_factory=list)
    error_count: int = 0
    last_error_time: Optional[datetime] = None
    last_error: Optional[str] = None


class Timer:
    """Timer for scheduling tasks."""
    def __init__(self):
        self._tasks: List[Tuple[datetime, int, asyncio.Task]] = []
        self._counter = 0
        
    def schedule(self, when: datetime, task: asyncio.Task) -> int:
        """Schedule a task to run at a specific time.
        
        Args:
            when: When to run the task
            task: Task to run
            
        Returns:
            Timer ID
        """
        self._counter += 1
        heapq.heappush(self._tasks, (when, self._counter, task))
        return self._counter
        
    def get_ready(self) -> List[asyncio.Task]:
        """Get tasks that are ready to run."""
        now = datetime.now()
        ready = []
        
        while self._tasks and self._tasks[0][0] <= now:
            _, _, task = heapq.heappop(self._tasks)
            ready.append(task)
            
        return ready
        
    def cancel(self, timer_id: int) -> bool:
        """Cancel a scheduled task.
        
        Args:
            timer_id: Timer ID to cancel
            
        Returns:
            Whether the task was cancelled
        """
        for i, (_, tid, task) in enumerate(self._tasks):
            if tid == timer_id:
                self._tasks.pop(i)
                task.cancel()
                heapq.heapify(self._tasks)
                return True
        return False


class EventLoop:
    """Custom event loop implementation for game automation.
    
    Features:
    - Performance monitoring
    - Task prioritization
    - Error recovery
    - Resource management
    - Timer scheduling
    """
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize the event loop.
        
        Args:
            loop: Optional event loop to use. If None, a new loop will be created.
        """
        self._loop = loop
        self._stats = EventLoopStats()
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._timer = Timer()
        self._task_queues: Dict[TaskPriority, List[asyncio.Task]] = {
            p: [] for p in TaskPriority
        }
        self._process = psutil.Process()
        
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the underlying asyncio event loop."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    @property
    def stats(self) -> EventLoopStats:
        """Get current event loop statistics."""
        return self._stats
        
    @contextmanager
    async def run_context(self):
        """Context manager for running the event loop."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    async def start(self):
        """Start the event loop."""
        if self._running:
            logger.warning("Event loop is already running")
            return
            
        self._running = True
        self._stats = EventLoopStats()
        
        try:
            logger.info("Starting event loop")
            await self._run_loop()
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            self._stats.error_count += 1
            self._stats.last_error_time = datetime.now()
            self._stats.last_error = str(e)
            raise
        finally:
            self._running = False
            
    async def stop(self):
        """Stop the event loop."""
        if not self._running:
            logger.warning("Event loop is not running")
            return
            
        logger.info("Stopping event loop")
        self._running = False
        
        # Cancel all running tasks
        for task_info in self._stats.current_tasks.values():
            task_info.state = TaskState.CANCELLED
            task_info.end_time = datetime.now()
            self._stats.cancelled_tasks += 1
            
            # Find and cancel the task
            for task in asyncio.all_tasks(self.loop):
                if task.get_name() == task_info.name:
                    task.cancel()
                    
        # Wait for tasks to finish
        await asyncio.gather(*asyncio.all_tasks(self.loop), 
                           return_exceptions=True)
        
    async def _run_loop(self):
        """Main event loop execution."""
        while self._running:
            try:
                # Process timer tasks
                ready_tasks = self._timer.get_ready()
                for task in ready_tasks:
                    await self._run_task(task)
                
                # Process queued tasks by priority
                for priority in TaskPriority:
                    queue = self._task_queues[priority]
                    while queue:
                        task = queue.pop(0)
                        await self._run_task(task)
                
                # Update stats
                self._update_stats()
                
                # Prevent CPU hogging
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event loop iteration: {e}")
                logger.error(traceback.format_exc())
                self._stats.error_count += 1
                self._stats.last_error_time = datetime.now()
                self._stats.last_error = str(e)
                # Implement recovery mechanism
                await self._handle_error(e)
                
    async def _run_task(self, task: asyncio.Task):
        """Run a task and update its statistics."""
        name = task.get_name()
        if name not in self._stats.current_tasks:
            return
            
        task_info = self._stats.current_tasks[name]
        task_info.state = TaskState.RUNNING
        task_info.start_time = datetime.now()
        
        try:
            await task
            task_info.state = TaskState.COMPLETED
            self._stats.completed_tasks += 1
        except Exception as e:
            task_info.state = TaskState.FAILED
            task_info.error = e
            self._stats.failed_tasks += 1
            
            # Retry if possible
            if task_info.retries < task_info.max_retries:
                task_info.retries += 1
                new_task = self.loop.create_task(task.get_coro(), name=name)
                self._task_queues[task_info.priority].append(new_task)
            else:
                logger.error(f"Task {name} failed after {task_info.retries} retries: {e}")
        finally:
            task_info.end_time = datetime.now()
            task_info.duration = (task_info.end_time - task_info.start_time).total_seconds()
            self._stats.task_history.append(task_info)
            del self._stats.current_tasks[name]
                
    async def _handle_error(self, error: Exception):
        """Handle event loop errors."""
        # Log the error
        logger.error(f"Event loop error: {error}")
        logger.error(traceback.format_exc())
        
        # Implement recovery actions
        await asyncio.sleep(1.0)  # Wait before retrying
        
        # TODO: Implement more sophisticated recovery mechanisms
        # - State recovery
        # - Resource cleanup
        # - System health check
                
    def _update_stats(self):
        """Update event loop statistics."""
        # Update system stats
        self._stats.current_memory_usage = self._process.memory_info().rss
        self._stats.peak_memory_usage = max(
            self._stats.peak_memory_usage,
            self._stats.current_memory_usage
        )
        self._stats.cpu_usage = self._process.cpu_percent()
        
        # Update task stats
        if self._stats.completed_tasks > 0:
            self._stats.avg_task_duration = (
                self._stats.total_task_duration / self._stats.completed_tasks
            )
            
    async def create_task(
        self,
        coro: Any,
        name: Optional[str] = None,
        priority: int = TaskPriority.NORMAL.value,
        schedule_at: Optional[datetime] = None
    ) -> asyncio.Task:
        """Create a new task in the event loop.
        
        Args:
            coro: Coroutine to run
            name: Task name
            priority: Task priority (0-9, higher is more important)
            schedule_at: When to schedule the task
            
        Returns:
            asyncio.Task: Created task
        """
        if not self._running:
            raise RuntimeError("Event loop is not running")
            
        # Create task
        task = self.loop.create_task(coro, name=name)
        task_priority = TaskPriority.from_int(priority)
        
        # Create task info
        task_info = TaskInfo(
            name=task.get_name(),
            priority=task_priority,
            state=TaskState.PENDING
        )
        self._stats.current_tasks[task.get_name()] = task_info
        self._stats.total_tasks += 1
        
        # Schedule or queue task
        if schedule_at is not None:
            self._timer.schedule(schedule_at, task)
        else:
            self._task_queues[task_priority].append(task)
        
        return task
        
    def schedule_task(
        self,
        coro: Any,
        when: datetime,
        name: Optional[str] = None,
        priority: int = TaskPriority.NORMAL.value
    ) -> Tuple[asyncio.Task, int]:
        """Schedule a task to run at a specific time.
        
        Args:
            coro: Coroutine to run
            when: When to run the task
            name: Task name
            priority: Task priority
            
        Returns:
            Tuple of (Task, Timer ID)
        """
        task = self.create_task(coro, name, priority, schedule_at=when)
        timer_id = self._timer.schedule(when, task)
        return task, timer_id
        
    def cancel_scheduled_task(self, timer_id: int) -> bool:
        """Cancel a scheduled task.
        
        Args:
            timer_id: Timer ID to cancel
            
        Returns:
            Whether the task was cancelled
        """
        return self._timer.cancel(timer_id)
