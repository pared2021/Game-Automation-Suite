"""Task manager for game automation."""
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
import json
import os
import asyncio
from enum import Enum

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .task_types import Task, TaskStatus, TaskPriority
from .task_rule import TaskRuleManager
from .task_history import TaskHistory, TaskHistoryEntry
from .task_scheduler import TaskScheduler, SchedulingStrategy
from ..monitor.task_monitor import TaskMonitor, MetricType
from .task_lifecycle import TaskLifecycleManager, TaskState
from .task_dependency import TaskDependencyManager, DependencyType

class TaskError(GameAutomationError):
    """Task related errors"""
    pass

class TaskExecutionMode(Enum):
    """Task execution modes"""
    SEQUENTIAL = "sequential"  # Execute tasks one by one
    PARALLEL = "parallel"    # Execute multiple tasks in parallel
    HYBRID = "hybrid"       # Mix of sequential and parallel

class TaskManager:
    """Task manager"""
    
    def __init__(
        self,
        state_dir: str = "data/task_state",
        max_parallel_tasks: int = 3,
        execution_mode: TaskExecutionMode = TaskExecutionMode.SEQUENTIAL,
        monitor_dir: str = "data/monitor"
    ):
        """Initialize task manager
        
        Args:
            state_dir: State save directory
            max_parallel_tasks: Maximum number of parallel tasks
            execution_mode: Task execution mode
            monitor_dir: Monitor data directory
        """
        self.state_dir = state_dir
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.running_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Execution settings
        self.max_parallel_tasks = max_parallel_tasks
        self.execution_mode = execution_mode
        
        # Task groups
        self.task_groups: Dict[str, Set[str]] = {}
        
        # Rule manager
        self.rule_manager = TaskRuleManager()
        
        # History manager
        self.history = TaskHistory()
        
        # Task monitor
        self.monitor = TaskMonitor(monitor_dir)
        
        # Task lifecycle manager
        self.lifecycle = TaskLifecycleManager()
        
        # Task dependency manager
        self.dependency = TaskDependencyManager(self.lifecycle)
        
        # Task scheduler
        self.scheduler = TaskScheduler(
            dependency_manager=self.dependency,
            task_monitor=self.monitor,
            max_parallel_tasks=max_parallel_tasks
        )
        
        # Monitoring config
        self.auto_save_interval = 300  # 5 minutes auto save
        self.last_save_time = datetime.now()
        
        # Task hooks
        self.pre_execute_hooks: List[Callable[[Task], None]] = []
        self.post_execute_hooks: List[Callable[[Task], None]] = []
        
        # Add default monitoring hooks
        self._add_monitoring_hooks()
        
        # Add lifecycle hooks
        self._add_lifecycle_hooks()

    def _add_monitoring_hooks(self):
        """Add default monitoring hooks"""
        def pre_execute_monitor(task: Task):
            # Start monitoring
            self.monitor.start_monitoring(task)
            
            # Record start time
            start_time = datetime.now()
            task.execution_start_time = start_time
            
            # Add event
            self.monitor.add_event(
                task.task_id,
                "execution_start",
                f"Task {task.name} started execution",
                {
                    "priority": task.priority.value,
                    "dependencies": [t.task_id for t in task.dependencies]
                }
            )
            
        def post_execute_monitor(task: Task):
            # Calculate execution time
            if hasattr(task, 'execution_start_time'):
                duration = datetime.now() - task.execution_start_time
                self.monitor.add_metric(
                    task.task_id,
                    "execution_time",
                    MetricType.DURATION,
                    duration,
                    {"unit": "seconds"}
                )
                
            # Add completion event
            status = "success" if task.status == TaskStatus.COMPLETED else "failed"
            self.monitor.add_event(
                task.task_id,
                f"execution_{status}",
                f"Task {task.name} {status}",
                {"error": str(task.error) if task.error else None}
            )
            
            # Save monitoring data
            self.monitor.save_metrics(task.task_id)
            self.monitor.save_diagnostics(task.task_id)
            
            # Stop monitoring
            self.monitor.stop_monitoring(task.task_id)
            
        self.add_pre_execute_hook(pre_execute_monitor)
        self.add_post_execute_hook(post_execute_monitor)

    def _add_lifecycle_hooks(self):
        """Add lifecycle management hooks"""
        # State hooks
        def on_ready(task: Task):
            # Add to execution queue when ready
            if task not in self.task_queue:
                self.task_queue.append(task)
                self._sort_queue()
                
        def on_running(task: Task):
            # Add to running tasks list
            if task not in self.running_tasks:
                self.running_tasks.append(task)
                
        def on_completed(task: Task):
            # Move to completed list
            if task in self.running_tasks:
                self.running_tasks.remove(task)
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
                
        def on_failed(task: Task):
            # Move to failed list
            if task in self.running_tasks:
                self.running_tasks.remove(task)
            if task not in self.failed_tasks:
                self.failed_tasks.append(task)
                
        # Add state hooks
        self.lifecycle.add_state_hook(TaskState.READY, on_ready)
        self.lifecycle.add_state_hook(TaskState.RUNNING, on_running)
        self.lifecycle.add_state_hook(TaskState.COMPLETED, on_completed)
        self.lifecycle.add_state_hook(TaskState.FAILED, on_failed)
        
        # Add transition hook for monitoring
        def on_transition(task: Task, from_state: TaskState, to_state: TaskState):
            self.monitor.add_event(
                task.task_id,
                "state_transition",
                f"Task state changed from {from_state} to {to_state}",
                {
                    "from_state": from_state.name,
                    "to_state": to_state.name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        self.lifecycle.add_transition_hook(on_transition)

    def add_task_group(self, group_name: str, task_ids: List[str]) -> None:
        """Add task group
        
        Args:
            group_name: Group name
            task_ids: List of task IDs in group
        """
        self.task_groups[group_name] = set(task_ids)
        detailed_logger.info(f"Added task group: {group_name} with {len(task_ids)} tasks")

    def get_group_tasks(self, group_name: str) -> List[Task]:
        """Get tasks in group
        
        Args:
            group_name: Group name
            
        Returns:
            List[Task]: List of tasks in group
        """
        if group_name not in self.task_groups:
            raise TaskError(f"Task group not found: {group_name}")
            
        return [
            self.tasks[task_id]
            for task_id in self.task_groups[group_name]
            if task_id in self.tasks
        ]

    def add_pre_execute_hook(self, hook: Callable[[Task], None]) -> None:
        """Add pre-execute hook
        
        Args:
            hook: Hook function
        """
        self.pre_execute_hooks.append(hook)

    def add_post_execute_hook(self, hook: Callable[[Task], None]) -> None:
        """Add post-execute hook
        
        Args:
            hook: Hook function
        """
        self.post_execute_hooks.append(hook)

    async def initialize(self):
        """Initialize manager."""
        # Initialize scheduler
        await self.scheduler.initialize()
        
    async def cleanup(self):
        """Clean up resources."""
        # Clean up scheduler
        await self.scheduler.cleanup()
        
        # Clean up other resources
        self.tasks.clear()
        self.task_queue.clear()
        self.running_tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        
    async def add_task(self, task: Task, max_retries: int = 3) -> None:
        """Add task
        
        Args:
            task: Task instance
            max_retries: Maximum retry attempts
        """
        self.tasks[task.task_id] = task
        
        # Register with lifecycle manager
        self.lifecycle.register_task(task, max_retries)
        
        # Add to scheduler
        await self.scheduler.add_task(task)
        
        # Add callbacks
        task.on_complete(self._on_task_complete)
        task.on_fail(self._on_task_fail)
        task.on_progress(self._on_task_progress)
        
        detailed_logger.info(f"Added task: {task.name} ({task.task_id})")
        
    async def cancel_task(self, task_id: str):
        """Cancel task.
        
        Args:
            task_id: Task ID
        """
        # Remove from scheduler
        self.scheduler.remove_task(task_id)
        
        # Cancel task
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
        detailed_logger.info(f"Cancelled task {task_id}")
        
    def get_task_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task statistics.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task statistics
        """
        return self.scheduler.get_task_stats(task_id)
        
    def set_scheduling_strategy(self, strategy: SchedulingStrategy) -> None:
        """Set scheduling strategy.
        
        Args:
            strategy: Scheduling strategy
        """
        self.scheduler.strategy = strategy

    async def execute_task(self, task: Task) -> bool:
        """Execute task
        
        Args:
            task: Task to execute
            
        Returns:
            bool: Whether execution succeeded
        """
        try:
            # Check if task is ready
            task_state = self.lifecycle.get_task_state(task.task_id)
            if task_state != TaskState.READY:
                detailed_logger.warning(
                    f"Task not ready for execution: {task.task_id} "
                    f"(current state: {task_state})"
                )
                return False
            
            # Transition to running state
            if not self.lifecycle.transition_state(task, TaskState.RUNNING):
                return False
            
            # Run pre-execute hooks
            for hook in self.pre_execute_hooks:
                try:
                    hook(task)
                except Exception as e:
                    detailed_logger.error(f"Pre-execute hook failed: {str(e)}")
                    self.monitor.add_error(
                        task.task_id,
                        "Hook execution failed",
                        {"hook": "pre_execute", "error": str(e)}
                    )
            
            # Execute task
            success = await task.execute()
            
            # Run post-execute hooks
            for hook in self.post_execute_hooks:
                try:
                    hook(task)
                except Exception as e:
                    detailed_logger.error(f"Post-execute hook failed: {str(e)}")
                    self.monitor.add_error(
                        task.task_id,
                        "Hook execution failed",
                        {"hook": "post_execute", "error": str(e)}
                    )
            
            # Add history entry
            self._add_history_entry(task)
            
            # Check auto save
            self._check_auto_save()
            
            return success
            
        except Exception as e:
            detailed_logger.error(f"Task execution error: {task.task_id} - {str(e)}")
            self.monitor.add_error(
                task.task_id,
                "Execution error",
                {"error": str(e)}
            )
            return False

    def add_task_dependency(self,
                          task_id: str,
                          dependency_id: str,
                          dep_type: DependencyType = DependencyType.HARD) -> None:
        """Add task dependency
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
            dep_type: Dependency type
        """
        self.dependency.add_dependency(task_id, dependency_id, dep_type)
        
    def remove_task_dependency(self, task_id: str, dependency_id: str) -> None:
        """Remove task dependency
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
        """
        self.dependency.remove_dependency(task_id, dependency_id)
        
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies
        
        Args:
            task_id: Task ID
            
        Returns:
            List[str]: List of dependency task IDs
        """
        return self.dependency.get_dependencies(task_id)
        
    def get_dependent_tasks(self, task_id: str) -> List[str]:
        """Get tasks that depend on this task
        
        Args:
            task_id: Task ID
            
        Returns:
            List[str]: List of dependent task IDs
        """
        return self.dependency.get_dependent_tasks(task_id)
        
    def get_critical_path(self, task_id: str) -> Optional[List[str]]:
        """Get critical dependency path
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[List[str]]: Critical path or None if no dependencies
        """
        return self.dependency.get_critical_path(task_id)
        
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied
        
        Args:
            task: Task to check
            
        Returns:
            bool: True if dependencies satisfied
        """
        return self.dependency.are_dependencies_satisfied(task.task_id)

    def _on_task_complete(self, task: Task) -> None:
        """Task completion callback
        
        Args:
            task: Completed task
        """
        # Add history entry
        self._add_history_entry(task)
        
        # Check if auto save needed
        self._check_auto_save()

    def _on_task_fail(self, task: Task) -> None:
        """Task failure callback
        
        Args:
            task: Failed task
        """
        # Add history entry
        self._add_history_entry(task)
        
        # Check if auto save needed
        self._check_auto_save()

    def _on_task_progress(self, task: Task) -> None:
        """Task progress update callback
        
        Args:
            task: Updated task
        """
        # Check timeout
        if task.timeout and task.start_time:
            elapsed = datetime.now() - task.start_time
            if elapsed.total_seconds() > task.timeout:
                task.status = TaskStatus.FAILED
                task.error_message = "Task execution timeout"
                detailed_logger.warning(f"Task timeout: {task.name}")
                self._on_task_fail(task)

    def _add_history_entry(self, task: Task) -> None:
        """Add task history entry
        
        Args:
            task: Task instance
        """
        if task.start_time:
            duration = None
            if task.end_time:
                duration = task.end_time - task.start_time
                
            entry = TaskHistoryEntry(
                task_id=task.task_id,
                task_name=task.name,
                status=task.status.name,
                start_time=task.start_time,
                end_time=task.end_time,
                duration=duration,
                error_message=task.error_message,
                performance_metrics=task.performance_metrics
            )
            
            self.history.add_entry(entry)

    def _check_auto_save(self) -> None:
        """Check if auto save needed"""
        current_time = datetime.now()
        if (current_time - self.last_save_time).total_seconds() >= self.auto_save_interval:
            self._auto_save_state()
            self.last_save_time = current_time

    def _auto_save_state(self) -> None:
        """Auto save state"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.state_dir, f"task_state_{timestamp}.json")
            self.save_state(filepath)
            
            # Clean up old auto save files
            self._cleanup_old_states()
            
        except Exception as e:
            detailed_logger.error(f"Auto save state failed: {str(e)}")

    def _cleanup_old_states(self, keep_days: int = 7) -> None:
        """Clean up old state files
        
        Args:
            keep_days: Days to keep files
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            for filename in os.listdir(self.state_dir):
                if filename.startswith("task_state_"):
                    filepath = os.path.join(self.state_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        detailed_logger.info(f"Cleaned up old state file: {filename}")
                        
        except Exception as e:
            detailed_logger.error(f"Clean up old states failed: {str(e)}")

    @log_exception
    def remove_task(self, task_id: str) -> None:
        """Remove task
        
        Args:
            task_id: Task ID
        """
        if task_id not in self.tasks:
            raise TaskError(f"Task does not exist: {task_id}")
            
        task = self.tasks[task_id]
        
        # Remove from lists
        if task in self.task_queue:
            self.task_queue.remove(task)
        if task in self.running_tasks:
            self.running_tasks.remove(task)
        if task in self.completed_tasks:
            self.completed_tasks.remove(task)
        if task in self.failed_tasks:
            self.failed_tasks.remove(task)
            
        # Remove task rule
        self.rule_manager.remove_rule(task_id)
        
        del self.tasks[task_id]
        detailed_logger.info(f"Removed task: {task.name} ({task_id})")

    @log_exception
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Task]: Task instance or None if not found
        """
        return self.tasks.get(task_id)

    @log_exception
    def execute_next_task(self) -> bool:
        """Execute next task
        
        Returns:
            bool: Whether successfully started task execution
        """
        if not self.task_queue:
            return False
            
        # Get next executable task
        next_task = self._get_next_executable_task()
        if not next_task:
            return False
            
        # Move from queue to running list
        self.task_queue.remove(next_task)
        self.running_tasks.append(next_task)
        
        # Execute task
        success = next_task.execute()
        
        # Update task status
        self.running_tasks.remove(next_task)
        if success:
            self.completed_tasks.append(next_task)
        else:
            self.failed_tasks.append(next_task)
        
        return True

    @log_exception
    def execute_all_tasks(self) -> None:
        """Execute all tasks"""
        while self.task_queue:
            self.execute_next_task()

    def _sort_queue(self) -> None:
        """Sort task queue"""
        # Group tasks by priority
        priority_groups: Dict[TaskPriority, List[Task]] = {}
        for task in self.task_queue:
            if task.priority not in priority_groups:
                priority_groups[task.priority] = []
            priority_groups[task.priority].append(task)
            
        # Sort each priority group
        sorted_queue = []
        for priority in sorted(priority_groups.keys(), reverse=True):
            group = priority_groups[priority]
            
            # Sort by dependencies and execution time
            group.sort(key=lambda x: (
                len(x.dependencies),  # Less dependencies first
                x.estimated_duration or float('inf'),  # Shorter tasks first
                x.task_id  # Sort by ID if equal
            ))
            
            sorted_queue.extend(group)
            
        self.task_queue = sorted_queue

    def _get_next_executable_task(self) -> Optional[Task]:
        """Get next executable task
        
        Returns:
            Optional[Task]: Next executable task or None if none available
        """
        for task in self.task_queue:
            # Check if dependencies met
            dependencies_met = self._check_dependencies(task)
            
            # Check if task rule satisfied
            rule = self.rule_manager.get_rule(task.task_id)
            if rule and not rule.evaluate({'task_manager': self}):
                continue
                
            if dependencies_met:
                return task
        return None

    @log_exception
    def save_state(self, filepath: str) -> None:
        """Save task state to file
        
        Args:
            filepath: Save path
        """
        state = {
            'tasks': [task.to_dict() for task in self.tasks.values()],
            'queue': [task.task_id for task in self.task_queue],
            'running': [task.task_id for task in self.running_tasks],
            'completed': [task.task_id for task in self.completed_tasks],
            'failed': [task.task_id for task in self.failed_tasks],
            'last_save_time': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            detailed_logger.info(f"Task state saved: {filepath}")
        except Exception as e:
            raise TaskError(f"Save task state failed: {str(e)}")

    @log_exception
    def load_state(self, filepath: str) -> None:
        """Load task state from file
        
        Args:
            filepath: State file path
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Clear current state
            self.tasks.clear()
            self.task_queue.clear()
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            
            # Rebuild tasks
            for task_data in state['tasks']:
                task = self._create_task_from_dict(task_data)
                self.tasks[task.task_id] = task
                
                # Recreate task rule
                self.rule_manager.create_rule(task)
            
            # Restore queue
            self.task_queue = [self.tasks[task_id] for task_id in state['queue']]
            self.running_tasks = [self.tasks[task_id] for task_id in state['running']]
            self.completed_tasks = [self.tasks[task_id] for task_id in state['completed']]
            self.failed_tasks = [self.tasks[task_id] for task_id in state['failed']]
            
            # Update last save time
            if 'last_save_time' in state:
                self.last_save_time = datetime.fromisoformat(state['last_save_time'])
            
            detailed_logger.info(f"Task state loaded: {filepath}")
            
        except Exception as e:
            raise TaskError(f"Load task state failed: {str(e)}")

    def _create_task_from_dict(self, data: Dict) -> Task:
        """Create task instance from dictionary
        
        Args:
            data: Task data dictionary
            
        Returns:
            Task: Task instance
        """
        task = Task(
            task_id=data['task_id'],
            name=data['name'],
            priority=TaskPriority[data['priority']],
            dependencies=data['dependencies']
        )
        
        task.status = TaskStatus[data['status']]
        task.progress = data['progress']
        
        if data['start_time']:
            task.start_time = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            task.end_time = datetime.fromisoformat(data['end_time'])
            
        task.error_message = data['error_message']
        task.performance_metrics = data.get('performance_metrics', {})
        
        return task

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[TaskStatus]: Task status or None if not found
        """
        task = self.get_task(task_id)
        return task.status if task else None

    def get_task_progress(self, task_id: str) -> Optional[float]:
        """Get task progress
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[float]: Task progress or None if not found
        """
        task = self.get_task(task_id)
        return task.progress if task else None

    def get_statistics(self) -> Dict:
        """Get task statistics
        
        Returns:
            Dict: Statistics information
        """
        stats = {
            'total': len(self.tasks),
            'pending': len(self.task_queue),
            'running': len(self.running_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }
        
        # Add rule statistics
        stats['rules'] = self.rule_manager.get_statistics()
        
        # Add history statistics
        stats['history'] = self.history.get_statistics()
        
        return stats
