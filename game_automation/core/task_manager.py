from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import os

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .task_types import Task, TaskStatus, TaskPriority
from .task_rule import TaskRuleManager
from .task_history import TaskHistory, TaskHistoryEntry

class TaskError(GameAutomationError):
    """Task related errors"""
    pass

class TaskManager:
    """Task manager"""
    
    def __init__(self, state_dir: str = "data/task_state"):
        """Initialize task manager
        
        Args:
            state_dir: State save directory
        """
        self.state_dir = state_dir
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.running_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Rule manager
        self.rule_manager = TaskRuleManager()
        
        # History manager
        self.history = TaskHistory()
        
        # Monitoring config
        self.auto_save_interval = 300  # 5 minutes auto save
        self.last_save_time = datetime.now()

    @log_exception
    def add_task(self, task: Task) -> None:
        """Add task
        
        Args:
            task: Task instance
        """
        if task.task_id in self.tasks:
            raise TaskError(f"Task ID already exists: {task.task_id}")
            
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        self._sort_queue()
        
        # Create task rule
        self.rule_manager.create_rule(task)
        
        # Add task monitoring callbacks
        task.on_complete(self._on_task_complete)
        task.on_fail(self._on_task_fail)
        task.on_progress(self._on_task_progress)
        
        detailed_logger.info(f"Added task: {task.name} ({task.task_id})")

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
        self.task_queue.sort(key=lambda x: (
            -x.priority.value,  # Higher priority first
            len(x.dependencies),  # Less dependencies first
            x.task_id  # Sort by ID if equal
        ))

    def _get_next_executable_task(self) -> Optional[Task]:
        """Get next executable task
        
        Returns:
            Optional[Task]: Next executable task or None if none available
        """
        for task in self.task_queue:
            # Check if dependencies met
            dependencies_met = all(
                self.get_task(dep_id) in self.completed_tasks
                for dep_id in task.dependencies
            )
            
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
