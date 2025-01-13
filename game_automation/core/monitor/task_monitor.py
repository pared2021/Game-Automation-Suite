"""Task monitoring system."""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import os
import psutil
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..task.task_types import Task, TaskStatus
from ...utils.logger import get_logger

logger = get_logger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"    # Incrementing value
    GAUGE = "gauge"       # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    DURATION = "duration"   # Time duration

@dataclass
class MetricValue:
    """Metric value"""
    type: MetricType
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)

class TaskMetrics:
    """Task performance metrics"""
    
    def __init__(self):
        """Initialize metrics"""
        self.metrics: Dict[str, List[MetricValue]] = {}
        
    def add_metric(
        self,
        name: str,
        type: MetricType,
        value: Any,
        labels: Dict[str, str] = None
    ):
        """Add metric value
        
        Args:
            name: Metric name
            type: Metric type
            value: Metric value
            labels: Metric labels
        """
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append(
            MetricValue(
                type=type,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )
        )
        
    def get_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Get metric values
        
        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List[MetricValue]: Metric values
        """
        values = self.metrics.get(name, [])
        
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]
            
        return values
        
    def get_latest(self, name: str) -> Optional[MetricValue]:
        """Get latest metric value
        
        Args:
            name: Metric name
            
        Returns:
            Optional[MetricValue]: Latest value or None
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else None
        
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()

class ResourceMetrics:
    """System resource metrics"""
    
    def __init__(self):
        """Initialize resource metrics"""
        self.metrics = TaskMetrics()
        
    def collect(self):
        """Collect resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.add_metric(
            "cpu_usage",
            MetricType.GAUGE,
            cpu_percent,
            {"unit": "percent"}
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.add_metric(
            "memory_usage",
            MetricType.GAUGE,
            memory.percent,
            {"unit": "percent"}
        )
        self.metrics.add_metric(
            "memory_available",
            MetricType.GAUGE,
            memory.available,
            {"unit": "bytes"}
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics.add_metric(
            "disk_usage",
            MetricType.GAUGE,
            disk.percent,
            {"unit": "percent"}
        )
        self.metrics.add_metric(
            "disk_free",
            MetricType.GAUGE,
            disk.free,
            {"unit": "bytes"}
        )
        
        # Network IO
        net_io = psutil.net_io_counters()
        self.metrics.add_metric(
            "network_bytes_sent",
            MetricType.COUNTER,
            net_io.bytes_sent,
            {"unit": "bytes"}
        )
        self.metrics.add_metric(
            "network_bytes_recv",
            MetricType.COUNTER,
            net_io.bytes_recv,
            {"unit": "bytes"}
        )

class TaskDiagnostics:
    """Task diagnostics information"""
    
    def __init__(self):
        """Initialize diagnostics"""
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
    def add_event(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add diagnostic event
        
        Args:
            event_type: Event type
            message: Event message
            details: Additional details
        """
        self.events.append({
            'type': event_type,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
        
    def add_warning(
        self,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add warning
        
        Args:
            message: Warning message
            details: Additional details
        """
        self.warnings.append({
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
        
    def add_error(
        self,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add error
        
        Args:
            message: Error message
            details: Additional details
        """
        self.errors.append({
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
        
    def clear(self):
        """Clear all diagnostics"""
        self.events.clear()
        self.warnings.clear()
        self.errors.clear()

class TaskMonitor:
    """Task monitoring system"""
    
    def __init__(self, save_dir: str = "data/monitor"):
        """Initialize task monitor
        
        Args:
            save_dir: Directory to save monitoring data
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.resource_metrics = ResourceMetrics()
        self.diagnostics: Dict[str, TaskDiagnostics] = {}
        
        # Active monitoring
        self.active_tasks: Set[str] = set()
        self.monitoring_interval = 60  # seconds
        
    def start_monitoring(self, task: Task):
        """Start monitoring task
        
        Args:
            task: Task to monitor
        """
        task_id = task.task_id
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = TaskMetrics()
        if task_id not in self.diagnostics:
            self.diagnostics[task_id] = TaskDiagnostics()
            
        self.active_tasks.add(task_id)
        logger.info(f"Started monitoring task: {task.name} ({task_id})")
        
    def stop_monitoring(self, task_id: str):
        """Stop monitoring task
        
        Args:
            task_id: Task ID
        """
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
            logger.info(f"Stopped monitoring task: {task_id}")
            
    def add_metric(
        self,
        task_id: str,
        name: str,
        type: MetricType,
        value: Any,
        labels: Dict[str, str] = None
    ):
        """Add task metric
        
        Args:
            task_id: Task ID
            name: Metric name
            type: Metric type
            value: Metric value
            labels: Metric labels
        """
        if task_id not in self.task_metrics:
            self.task_metrics[task_id] = TaskMetrics()
            
        self.task_metrics[task_id].add_metric(name, type, value, labels)
        
    def add_event(
        self,
        task_id: str,
        event_type: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add diagnostic event
        
        Args:
            task_id: Task ID
            event_type: Event type
            message: Event message
            details: Additional details
        """
        if task_id not in self.diagnostics:
            self.diagnostics[task_id] = TaskDiagnostics()
            
        self.diagnostics[task_id].add_event(event_type, message, details)
        
    def add_warning(
        self,
        task_id: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add warning
        
        Args:
            task_id: Task ID
            message: Warning message
            details: Additional details
        """
        if task_id not in self.diagnostics:
            self.diagnostics[task_id] = TaskDiagnostics()
            
        self.diagnostics[task_id].add_warning(message, details)
        
    def add_error(
        self,
        task_id: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Add error
        
        Args:
            task_id: Task ID
            message: Error message
            details: Additional details
        """
        if task_id not in self.diagnostics:
            self.diagnostics[task_id] = TaskDiagnostics()
            
        self.diagnostics[task_id].add_error(message, details)
        
    def get_task_metrics(
        self,
        task_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[TaskMetrics]:
        """Get task metrics
        
        Args:
            task_id: Task ID
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Optional[TaskMetrics]: Task metrics or None
        """
        metrics = self.task_metrics.get(task_id)
        if not metrics:
            return None
            
        filtered = TaskMetrics()
        for name, values in metrics.metrics.items():
            filtered.metrics[name] = [
                v for v in values
                if (not start_time or v.timestamp >= start_time) and
                   (not end_time or v.timestamp <= end_time)
            ]
            
        return filtered
        
    def get_resource_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> TaskMetrics:
        """Get resource metrics
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            TaskMetrics: Resource metrics
        """
        filtered = TaskMetrics()
        for name, values in self.resource_metrics.metrics.metrics.items():
            filtered.metrics[name] = [
                v for v in values
                if (not start_time or v.timestamp >= start_time) and
                   (not end_time or v.timestamp <= end_time)
            ]
            
        return filtered
        
    def get_diagnostics(
        self,
        task_id: str,
        include_events: bool = True,
        include_warnings: bool = True,
        include_errors: bool = True
    ) -> Optional[TaskDiagnostics]:
        """Get task diagnostics
        
        Args:
            task_id: Task ID
            include_events: Include events
            include_warnings: Include warnings
            include_errors: Include errors
            
        Returns:
            Optional[TaskDiagnostics]: Task diagnostics or None
        """
        diag = self.diagnostics.get(task_id)
        if not diag:
            return None
            
        filtered = TaskDiagnostics()
        if include_events:
            filtered.events = diag.events.copy()
        if include_warnings:
            filtered.warnings = diag.warnings.copy()
        if include_errors:
            filtered.errors = diag.errors.copy()
            
        return filtered
        
    def save_metrics(self, task_id: str):
        """Save task metrics
        
        Args:
            task_id: Task ID
        """
        metrics = self.task_metrics.get(task_id)
        if not metrics:
            return
            
        # Convert metrics to JSON
        data = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                name: [
                    {
                        'type': v.type.value,
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat(),
                        'labels': v.labels
                    }
                    for v in values
                ]
                for name, values in metrics.metrics.items()
            }
        }
        
        # Save to file
        filename = f"metrics_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metrics for task {task_id}: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics for task {task_id}: {str(e)}")
            
    def save_diagnostics(self, task_id: str):
        """Save task diagnostics
        
        Args:
            task_id: Task ID
        """
        diag = self.diagnostics.get(task_id)
        if not diag:
            return
            
        # Convert diagnostics to JSON
        data = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'events': diag.events,
            'warnings': diag.warnings,
            'errors': diag.errors
        }
        
        # Save to file
        filename = f"diagnostics_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved diagnostics for task {task_id}: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save diagnostics for task {task_id}: {str(e)}")
            
    def cleanup(self, keep_days: int = 7):
        """Clean up old monitoring data
        
        Args:
            keep_days: Days to keep files
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            for filename in os.listdir(self.save_dir):
                if filename.startswith(("metrics_", "diagnostics_")):
                    filepath = os.path.join(self.save_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old monitoring file: {filename}")
                        
        except Exception as e:
            logger.error(f"Failed to clean up monitoring data: {str(e)}")
            
    def generate_report(
        self,
        task_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate monitoring report
        
        Args:
            task_id: Task ID
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dict[str, Any]: Report data
        """
        # Get metrics
        metrics = self.get_task_metrics(task_id, start_time, end_time)
        resource_metrics = self.get_resource_metrics(start_time, end_time)
        diagnostics = self.get_diagnostics(task_id)
        
        # Calculate statistics
        stats = {}
        if metrics:
            for name, values in metrics.metrics.items():
                if values:
                    metric_type = values[0].type
                    if metric_type in (MetricType.GAUGE, MetricType.COUNTER):
                        values_list = [v.value for v in values]
                        stats[name] = {
                            'min': min(values_list),
                            'max': max(values_list),
                            'avg': sum(values_list) / len(values_list)
                        }
                    elif metric_type == MetricType.DURATION:
                        durations = [v.value.total_seconds() for v in values]
                        stats[name] = {
                            'min': min(durations),
                            'max': max(durations),
                            'avg': sum(durations) / len(durations)
                        }
                        
        # Generate report
        report = {
            'task_id': task_id,
            'period': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'metrics': {
                name: [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat(),
                        'labels': v.labels
                    }
                    for v in values
                ]
                for name, values in (metrics.metrics if metrics else {}).items()
            },
            'resource_metrics': {
                name: [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat(),
                        'labels': v.labels
                    }
                    for v in values
                ]
                for name, values in resource_metrics.metrics.items()
            },
            'statistics': stats,
            'diagnostics': {
                'events': diagnostics.events if diagnostics else [],
                'warnings': diagnostics.warnings if diagnostics else [],
                'errors': diagnostics.errors if diagnostics else []
            }
        }
        
        return report
