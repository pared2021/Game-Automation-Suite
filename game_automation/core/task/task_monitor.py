"""Task performance monitoring system."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

from .task_types import Task, TaskStatus, TaskType
from .task_action_handler import TaskAction, ActionType

@dataclass
class TaskMetrics:
    """Task performance metrics."""
    task_id: str
    task_type: TaskType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    execution_time: float = 0.0
    action_count: int = 0
    action_success_rate: float = 0.0
    retry_count: int = 0
    error_count: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    state_changes: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary.
        
        Returns:
            Dict: Metrics as dictionary
        """
        data = asdict(self)
        # Convert datetime to ISO format
        data['start_time'] = data['start_time'].isoformat()
        if data['end_time']:
            data['end_time'] = data['end_time'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskMetrics':
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            TaskMetrics: Metrics instance
        """
        # Convert ISO format to datetime
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)

@dataclass
class ActionMetrics:
    """Action performance metrics."""
    action_id: str
    action_type: ActionType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    retry_count: int = 0
    error_message: Optional[str] = None
    resource_usage: Dict[str, float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary.
        
        Returns:
            Dict: Metrics as dictionary
        """
        data = asdict(self)
        data['start_time'] = data['start_time'].isoformat()
        if data['end_time']:
            data['end_time'] = data['end_time'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ActionMetrics':
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            ActionMetrics: Metrics instance
        """
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)

class TaskMonitor:
    """Task performance monitor."""
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        """Initialize task monitor.
        
        Args:
            metrics_dir: Metrics storage directory
        """
        self.metrics_dir = metrics_dir
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        self._task_metrics: Dict[str, TaskMetrics] = {}
        self._action_metrics: Dict[str, List[ActionMetrics]] = defaultdict(list)
        self._logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.execution_time_threshold = 30.0  # seconds
        self.retry_threshold = 3
        self.error_threshold = 5
        self.resource_usage_threshold = 80.0  # percent
        
    def start_task_monitoring(self, task: Task) -> None:
        """Start monitoring task.
        
        Args:
            task: Task instance
        """
        metrics = TaskMetrics(
            task_id=task.task_id,
            task_type=task.task_type,
            start_time=datetime.now()
        )
        self._task_metrics[task.task_id] = metrics
        
    def end_task_monitoring(self, task: Task) -> None:
        """End task monitoring.
        
        Args:
            task: Task instance
        """
        if task.task_id not in self._task_metrics:
            return
            
        metrics = self._task_metrics[task.task_id]
        metrics.end_time = datetime.now()
        metrics.status = task.status
        metrics.execution_time = (metrics.end_time - metrics.start_time).total_seconds()
        
        # Calculate action metrics
        action_metrics = self._action_metrics[task.task_id]
        if action_metrics:
            metrics.action_count = len(action_metrics)
            metrics.action_success_rate = sum(
                1 for m in action_metrics if m.success
            ) / len(action_metrics)
            metrics.retry_count = sum(m.retry_count for m in action_metrics)
            metrics.error_count = sum(
                1 for m in action_metrics if m.error_message
            )
            
        # Check thresholds and log warnings
        self._check_performance_thresholds(metrics)
        
        # Save metrics
        self._save_metrics(metrics)
        
    def start_action_monitoring(
        self,
        task_id: str,
        action: TaskAction
    ) -> str:
        """Start monitoring action.
        
        Args:
            task_id: Task ID
            action: Action instance
            
        Returns:
            str: Action monitoring ID
        """
        action_id = f"{task_id}_{len(self._action_metrics[task_id])}"
        metrics = ActionMetrics(
            action_id=action_id,
            action_type=action.action_type,
            start_time=datetime.now()
        )
        self._action_metrics[task_id].append(metrics)
        return action_id
        
    def end_action_monitoring(
        self,
        task_id: str,
        action_id: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """End action monitoring.
        
        Args:
            task_id: Task ID
            action_id: Action monitoring ID
            success: Whether action succeeded
            error: Optional error message
        """
        metrics = self._find_action_metrics(task_id, action_id)
        if not metrics:
            return
            
        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.success = success
        metrics.error_message = error
        
    def update_action_retry(
        self,
        task_id: str,
        action_id: str
    ) -> None:
        """Update action retry count.
        
        Args:
            task_id: Task ID
            action_id: Action monitoring ID
        """
        metrics = self._find_action_metrics(task_id, action_id)
        if metrics:
            metrics.retry_count += 1
            
    def update_resource_usage(
        self,
        task_id: str,
        memory_usage: float,
        cpu_usage: float
    ) -> None:
        """Update resource usage metrics.
        
        Args:
            task_id: Task ID
            memory_usage: Memory usage percentage
            cpu_usage: CPU usage percentage
        """
        if task_id in self._task_metrics:
            metrics = self._task_metrics[task_id]
            metrics.memory_usage = memory_usage
            metrics.cpu_usage = cpu_usage
            
    def update_state_changes(self, task_id: str) -> None:
        """Update state change count.
        
        Args:
            task_id: Task ID
        """
        if task_id in self._task_metrics:
            self._task_metrics[task_id].state_changes += 1
            
    def get_task_metrics(self, task_id: str) -> Optional[Dict]:
        """Get task metrics.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict]: Task metrics
        """
        metrics = self._task_metrics.get(task_id)
        return metrics.to_dict() if metrics else None
        
    def get_action_metrics(
        self,
        task_id: str
    ) -> List[Dict]:
        """Get action metrics.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[Dict]: Action metrics
        """
        return [
            metrics.to_dict()
            for metrics in self._action_metrics.get(task_id, [])
        ]
        
    def get_performance_summary(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        task_type: TaskType = None
    ) -> Dict[str, Any]:
        """Get performance summary.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            task_type: Filter by task type
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        metrics = self._filter_metrics(start_time, end_time, task_type)
        
        if not metrics:
            return {}
            
        total_tasks = len(metrics)
        completed_tasks = sum(
            1 for m in metrics if m.status == TaskStatus.COMPLETED
        )
        failed_tasks = sum(
            1 for m in metrics if m.status == TaskStatus.FAILED
        )
        
        avg_execution_time = sum(
            m.execution_time for m in metrics
        ) / total_tasks
        
        avg_action_success = sum(
            m.action_success_rate for m in metrics
        ) / total_tasks
        
        total_errors = sum(m.error_count for m in metrics)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / total_tasks,
            'avg_execution_time': avg_execution_time,
            'avg_action_success': avg_action_success,
            'total_errors': total_errors,
            'avg_memory_usage': sum(m.memory_usage for m in metrics) / total_tasks,
            'avg_cpu_usage': sum(m.cpu_usage for m in metrics) / total_tasks,
            'total_state_changes': sum(m.state_changes for m in metrics)
        }
        
    def _find_action_metrics(
        self,
        task_id: str,
        action_id: str
    ) -> Optional[ActionMetrics]:
        """Find action metrics.
        
        Args:
            task_id: Task ID
            action_id: Action monitoring ID
            
        Returns:
            Optional[ActionMetrics]: Action metrics
        """
        for metrics in self._action_metrics[task_id]:
            if metrics.action_id == action_id:
                return metrics
        return None
        
    def _filter_metrics(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        task_type: TaskType = None
    ) -> List[TaskMetrics]:
        """Filter metrics by criteria.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            task_type: Filter by task type
            
        Returns:
            List[TaskMetrics]: Filtered metrics
        """
        metrics = list(self._task_metrics.values())
        
        if start_time:
            metrics = [
                m for m in metrics
                if m.start_time >= start_time
            ]
            
        if end_time:
            metrics = [
                m for m in metrics
                if m.end_time and m.end_time <= end_time
            ]
            
        if task_type:
            metrics = [
                m for m in metrics
                if m.task_type == task_type
            ]
            
        return metrics
        
    def _check_performance_thresholds(self, metrics: TaskMetrics) -> None:
        """Check performance thresholds and log warnings.
        
        Args:
            metrics: Task metrics
        """
        if metrics.execution_time > self.execution_time_threshold:
            self._logger.warning(
                f"Task {metrics.task_id} exceeded execution time threshold: "
                f"{metrics.execution_time:.2f}s > {self.execution_time_threshold}s"
            )
            
        if metrics.retry_count > self.retry_threshold:
            self._logger.warning(
                f"Task {metrics.task_id} exceeded retry threshold: "
                f"{metrics.retry_count} > {self.retry_threshold}"
            )
            
        if metrics.error_count > self.error_threshold:
            self._logger.warning(
                f"Task {metrics.task_id} exceeded error threshold: "
                f"{metrics.error_count} > {self.error_threshold}"
            )
            
        if metrics.memory_usage > self.resource_usage_threshold:
            self._logger.warning(
                f"Task {metrics.task_id} exceeded memory usage threshold: "
                f"{metrics.memory_usage:.1f}% > {self.resource_usage_threshold}%"
            )
            
        if metrics.cpu_usage > self.resource_usage_threshold:
            self._logger.warning(
                f"Task {metrics.task_id} exceeded CPU usage threshold: "
                f"{metrics.cpu_usage:.1f}% > {self.resource_usage_threshold}%"
            )
            
    def _save_metrics(self, metrics: TaskMetrics) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Task metrics
        """
        try:
            # Create metrics file path
            file_path = os.path.join(
                self.metrics_dir,
                f"task_{metrics.task_id}.json"
            )
            
            # Save task metrics
            data = {
                'task': metrics.to_dict(),
                'actions': [
                    m.to_dict()
                    for m in self._action_metrics[metrics.task_id]
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self._logger.error(f"Failed to save metrics: {e}")
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self._task_metrics.clear()
        self._action_metrics.clear()
