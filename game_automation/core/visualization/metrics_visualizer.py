"""Metrics visualization module."""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ..task.task_types import TaskType, TaskStatus
from ..monitor import TaskMetrics, ActionMetrics

class MetricsVisualizer:
    """Metrics visualization."""
    
    def __init__(self, metrics_dir: str = "data/metrics",
                 output_dir: str = "data/visualizations"):
        """Initialize metrics visualizer.
        
        Args:
            metrics_dir: Metrics data directory
            output_dir: Visualization output directory
        """
        self.metrics_dir = metrics_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def load_metrics(self, task_id: str) -> Tuple[TaskMetrics, List[ActionMetrics]]:
        """Load metrics from file.
        
        Args:
            task_id: Task ID
            
        Returns:
            Tuple[TaskMetrics, List[ActionMetrics]]: Task and action metrics
        """
        file_path = os.path.join(self.metrics_dir, f"task_{task_id}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        task_metrics = TaskMetrics.from_dict(data['task'])
        action_metrics = [
            ActionMetrics.from_dict(m)
            for m in data['actions']
        ]
        return task_metrics, action_metrics
        
    def create_task_timeline(self, task_id: str) -> None:
        """Create task timeline visualization.
        
        Args:
            task_id: Task ID
        """
        task_metrics, action_metrics = self.load_metrics(task_id)
        
        # Create figure
        fig = go.Figure()
        
        # Add task timeline
        fig.add_trace(go.Bar(
            x=[task_metrics.start_time],
            y=['Task'],
            width=[(task_metrics.end_time - task_metrics.start_time).total_seconds()],
            base=[task_metrics.start_time],
            name='Task',
            marker_color='rgb(55, 83, 109)'
        ))
        
        # Add action timelines
        for i, metrics in enumerate(action_metrics):
            fig.add_trace(go.Bar(
                x=[metrics.start_time],
                y=[f'Action {i+1}'],
                width=[(metrics.end_time - metrics.start_time).total_seconds()],
                base=[metrics.start_time],
                name=f'Action {metrics.action_type.name}',
                marker_color='rgb(26, 118, 255)' if metrics.success else 'rgb(255, 65, 54)'
            ))
            
        # Update layout
        fig.update_layout(
            title=f'Task Timeline (ID: {task_id})',
            xaxis_title='Time',
            yaxis_title='Components',
            barmode='overlay',
            showlegend=True
        )
        
        # Save figure
        fig.write_html(os.path.join(
            self.output_dir,
            f'task_{task_id}_timeline.html'
        ))
        
    def create_performance_dashboard(self, task_id: str) -> None:
        """Create performance dashboard visualization.
        
        Args:
            task_id: Task ID
        """
        task_metrics, action_metrics = self.load_metrics(task_id)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Resource Usage',
                'Action Success Rate',
                'Error Distribution',
                'State Changes'
            )
        )
        
        # Resource usage
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=task_metrics.memory_usage,
                title={'text': 'Memory Usage (%)'},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Action success rate
        labels = ['Success', 'Failed']
        values = [
            sum(1 for m in action_metrics if m.success),
            sum(1 for m in action_metrics if not m.success)
        ]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                title='Action Results'
            ),
            row=1, col=2
        )
        
        # Error distribution
        error_types = {}
        for metrics in action_metrics:
            if metrics.error_message:
                error_type = metrics.error_message.split(':')[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
        fig.add_trace(
            go.Bar(
                x=list(error_types.keys()),
                y=list(error_types.values()),
                name='Errors'
            ),
            row=2, col=1
        )
        
        # State changes over time
        fig.add_trace(
            go.Scatter(
                x=[m.start_time for m in action_metrics],
                y=list(range(len(action_metrics))),
                mode='lines+markers',
                name='State Changes'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Task Performance Dashboard (ID: {task_id})',
            showlegend=True,
            height=800
        )
        
        # Save figure
        fig.write_html(os.path.join(
            self.output_dir,
            f'task_{task_id}_dashboard.html'
        ))
        
    def create_performance_summary(self, start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 task_type: Optional[TaskType] = None) -> None:
        """Create performance summary visualization.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            task_type: Filter by task type
        """
        # Load all metrics
        metrics_data = []
        for file_name in os.listdir(self.metrics_dir):
            if not file_name.startswith('task_') or not file_name.endswith('.json'):
                continue
                
            with open(os.path.join(self.metrics_dir, file_name), 'r') as f:
                data = json.load(f)
                task_metrics = TaskMetrics.from_dict(data['task'])
                
                # Apply filters
                if start_time and task_metrics.start_time < start_time:
                    continue
                if end_time and task_metrics.end_time > end_time:
                    continue
                if task_type and task_metrics.task_type != task_type:
                    continue
                    
                metrics_data.append(task_metrics)
                
        if not metrics_data:
            return
            
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Task Status Distribution',
                'Average Execution Time by Type',
                'Resource Usage Distribution',
                'Error Rate by Type'
            )
        )
        
        # Task status distribution
        status_counts = {}
        for metrics in metrics_data:
            status_counts[metrics.status.name] = status_counts.get(metrics.status.name, 0) + 1
            
        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                title='Task Status'
            ),
            row=1, col=1
        )
        
        # Average execution time by type
        df = pd.DataFrame([
            {
                'type': m.task_type.name,
                'execution_time': m.execution_time
            }
            for m in metrics_data
        ])
        avg_time = df.groupby('type')['execution_time'].mean()
        
        fig.add_trace(
            go.Bar(
                x=avg_time.index,
                y=avg_time.values,
                name='Avg Execution Time'
            ),
            row=1, col=2
        )
        
        # Resource usage distribution
        fig.add_trace(
            go.Box(
                y=[m.memory_usage for m in metrics_data],
                name='Memory Usage',
                boxpoints='all'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(
                y=[m.cpu_usage for m in metrics_data],
                name='CPU Usage',
                boxpoints='all'
            ),
            row=2, col=1
        )
        
        # Error rate by type
        df = pd.DataFrame([
            {
                'type': m.task_type.name,
                'error_rate': m.error_count / m.action_count if m.action_count > 0 else 0
            }
            for m in metrics_data
        ])
        avg_error = df.groupby('type')['error_rate'].mean()
        
        fig.add_trace(
            go.Bar(
                x=avg_error.index,
                y=avg_error.values,
                name='Error Rate'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Task Performance Summary',
            showlegend=True,
            height=800
        )
        
        # Save figure
        file_name = 'performance_summary'
        if start_time:
            file_name += f'_{start_time.strftime("%Y%m%d")}'
        if end_time:
            file_name += f'_to_{end_time.strftime("%Y%m%d")}'
        if task_type:
            file_name += f'_{task_type.name}'
            
        fig.write_html(os.path.join(
            self.output_dir,
            f'{file_name}.html'
        ))

# Create global metrics visualizer instance
metrics_visualizer = MetricsVisualizer()
