"""Task monitoring window."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from ...core.monitor.task_monitor import TaskMonitor, MetricType
from ...core.task.task_types import TaskStatus
from ...utils.logger import get_logger

logger = get_logger(__name__)

class TaskMonitorWindow(QMainWindow):
    """Task monitoring window"""
    
    def __init__(
        self,
        monitor: TaskMonitor,
        parent: Optional[QWidget] = None
    ):
        """Initialize window
        
        Args:
            monitor: Task monitor instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.monitor = monitor
        
        self.setWindowTitle("Task Monitor")
        self.resize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Add tabs
        tab_widget.addTab(self._create_overview_tab(), "Overview")
        tab_widget.addTab(self._create_metrics_tab(), "Metrics")
        tab_widget.addTab(self._create_diagnostics_tab(), "Diagnostics")
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(1000)  # Update every second
        
    def _create_overview_tab(self) -> QWidget:
        """Create overview tab
        
        Returns:
            QWidget: Tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Task status section
        status_group = QWidget()
        status_layout = QHBoxLayout(status_group)
        
        # Active tasks
        active_widget = QWidget()
        active_layout = QVBoxLayout(active_widget)
        active_layout.addWidget(QLabel("Active Tasks"))
        self.active_table = QTableWidget(0, 3)
        self.active_table.setHorizontalHeaderLabels(["Task", "Status", "Duration"])
        self.active_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        active_layout.addWidget(self.active_table)
        status_layout.addWidget(active_widget)
        
        # Completed tasks
        completed_widget = QWidget()
        completed_layout = QVBoxLayout(completed_widget)
        completed_layout.addWidget(QLabel("Completed Tasks"))
        self.completed_table = QTableWidget(0, 4)
        self.completed_table.setHorizontalHeaderLabels(["Task", "Status", "Duration", "Result"])
        self.completed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        completed_layout.addWidget(self.completed_table)
        status_layout.addWidget(completed_widget)
        
        layout.addWidget(status_group)
        
        # Resource usage section
        resource_group = QWidget()
        resource_layout = QHBoxLayout(resource_group)
        
        # CPU usage
        cpu_widget = QWidget()
        cpu_layout = QVBoxLayout(cpu_widget)
        cpu_layout.addWidget(QLabel("CPU Usage"))
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_progress)
        resource_layout.addWidget(cpu_widget)
        
        # Memory usage
        memory_widget = QWidget()
        memory_layout = QVBoxLayout(memory_widget)
        memory_layout.addWidget(QLabel("Memory Usage"))
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        memory_layout.addWidget(self.memory_progress)
        resource_layout.addWidget(memory_widget)
        
        layout.addWidget(resource_group)
        
        return widget
        
    def _create_metrics_tab(self) -> QWidget:
        """Create metrics tab
        
        Returns:
            QWidget: Tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Task selection
        selection_group = QWidget()
        selection_layout = QHBoxLayout(selection_group)
        
        selection_layout.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.currentTextChanged.connect(self._update_metrics_charts)
        selection_layout.addWidget(self.task_combo)
        
        selection_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.currentTextChanged.connect(self._update_metrics_charts)
        selection_layout.addWidget(self.metric_combo)
        
        selection_layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1h", "6h", "24h", "7d"])
        self.period_combo.currentTextChanged.connect(self._update_metrics_charts)
        selection_layout.addWidget(self.period_combo)
        
        layout.addWidget(selection_group)
        
        # Charts
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view)
        
        return widget
        
    def _create_diagnostics_tab(self) -> QWidget:
        """Create diagnostics tab
        
        Returns:
            QWidget: Tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Task selection
        selection_group = QWidget()
        selection_layout = QHBoxLayout(selection_group)
        
        selection_layout.addWidget(QLabel("Task:"))
        self.diag_task_combo = QComboBox()
        self.diag_task_combo.currentTextChanged.connect(self._update_diagnostics)
        selection_layout.addWidget(self.diag_task_combo)
        
        layout.addWidget(selection_group)
        
        # Diagnostic tables
        tables_group = QWidget()
        tables_layout = QVBoxLayout(tables_group)
        
        # Events table
        tables_layout.addWidget(QLabel("Events"))
        self.events_table = QTableWidget(0, 3)
        self.events_table.setHorizontalHeaderLabels(["Time", "Type", "Message"])
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tables_layout.addWidget(self.events_table)
        
        # Warnings table
        tables_layout.addWidget(QLabel("Warnings"))
        self.warnings_table = QTableWidget(0, 2)
        self.warnings_table.setHorizontalHeaderLabels(["Time", "Message"])
        self.warnings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tables_layout.addWidget(self.warnings_table)
        
        # Errors table
        tables_layout.addWidget(QLabel("Errors"))
        self.errors_table = QTableWidget(0, 2)
        self.errors_table.setHorizontalHeaderLabels(["Time", "Message"])
        self.errors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tables_layout.addWidget(self.errors_table)
        
        layout.addWidget(tables_group)
        
        return widget
        
    def _update_data(self):
        """Update displayed data"""
        try:
            # Update task lists
            active_tasks = [
                task_id for task_id in self.monitor.task_metrics.keys()
                if task_id in self.monitor.active_tasks
            ]
            
            # Update active tasks table
            self.active_table.setRowCount(len(active_tasks))
            for i, task_id in enumerate(active_tasks):
                metrics = self.monitor.get_task_metrics(task_id)
                if not metrics:
                    continue
                    
                # Get latest duration metric
                duration = "N/A"
                if "execution_time" in metrics.metrics:
                    values = metrics.metrics["execution_time"]
                    if values:
                        duration = str(values[-1].value)
                        
                self.active_table.setItem(i, 0, QTableWidgetItem(task_id))
                self.active_table.setItem(i, 1, QTableWidgetItem("Running"))
                self.active_table.setItem(i, 2, QTableWidgetItem(duration))
                
            # Update completed tasks table
            completed_tasks = [
                task_id for task_id in self.monitor.task_metrics.keys()
                if task_id not in self.monitor.active_tasks
            ]
            
            self.completed_table.setRowCount(len(completed_tasks))
            for i, task_id in enumerate(completed_tasks):
                metrics = self.monitor.get_task_metrics(task_id)
                if not metrics:
                    continue
                    
                # Get latest duration metric
                duration = "N/A"
                if "execution_time" in metrics.metrics:
                    values = metrics.metrics["execution_time"]
                    if values:
                        duration = str(values[-1].value)
                        
                # Get task status
                status = "Unknown"
                result = "N/A"
                diag = self.monitor.get_diagnostics(task_id)
                if diag:
                    for event in reversed(diag.events):
                        if event["type"] in ("execution_success", "execution_failed"):
                            status = "Success" if event["type"] == "execution_success" else "Failed"
                            result = event["details"].get("error", "Success") if event["details"] else "N/A"
                            break
                            
                self.completed_table.setItem(i, 0, QTableWidgetItem(task_id))
                self.completed_table.setItem(i, 1, QTableWidgetItem(status))
                self.completed_table.setItem(i, 2, QTableWidgetItem(duration))
                self.completed_table.setItem(i, 3, QTableWidgetItem(result))
                
            # Update resource usage
            resource_metrics = self.monitor.get_resource_metrics()
            
            # CPU usage
            cpu_usage = 0
            if "cpu_usage" in resource_metrics.metrics:
                values = resource_metrics.metrics["cpu_usage"]
                if values:
                    cpu_usage = values[-1].value
            self.cpu_progress.setValue(int(cpu_usage))
            
            # Memory usage
            memory_usage = 0
            if "memory_usage" in resource_metrics.metrics:
                values = resource_metrics.metrics["memory_usage"]
                if values:
                    memory_usage = values[-1].value
            self.memory_progress.setValue(int(memory_usage))
            
            # Update task selection combos
            all_tasks = list(self.monitor.task_metrics.keys())
            current_task = self.task_combo.currentText()
            current_diag_task = self.diag_task_combo.currentText()
            
            self.task_combo.clear()
            self.task_combo.addItems(all_tasks)
            if current_task in all_tasks:
                self.task_combo.setCurrentText(current_task)
                
            self.diag_task_combo.clear()
            self.diag_task_combo.addItems(all_tasks)
            if current_diag_task in all_tasks:
                self.diag_task_combo.setCurrentText(current_diag_task)
                
            # Update metrics combo
            if current_task:
                metrics = self.monitor.get_task_metrics(current_task)
                if metrics:
                    current_metric = self.metric_combo.currentText()
                    self.metric_combo.clear()
                    self.metric_combo.addItems(metrics.metrics.keys())
                    if current_metric in metrics.metrics:
                        self.metric_combo.setCurrentText(current_metric)
                        
            # Update charts and diagnostics if needed
            self._update_metrics_charts()
            self._update_diagnostics()
            
        except Exception as e:
            logger.error(f"Failed to update monitor data: {str(e)}")
            
    def _update_metrics_charts(self):
        """Update metrics charts"""
        try:
            task_id = self.task_combo.currentText()
            metric_name = self.metric_combo.currentText()
            period_text = self.period_combo.currentText()
            
            if not task_id or not metric_name:
                return
                
            # Calculate time range
            end_time = datetime.now()
            if period_text == "1h":
                start_time = end_time - timedelta(hours=1)
            elif period_text == "6h":
                start_time = end_time - timedelta(hours=6)
            elif period_text == "24h":
                start_time = end_time - timedelta(days=1)
            else:  # 7d
                start_time = end_time - timedelta(days=7)
                
            # Get metric data
            metrics = self.monitor.get_task_metrics(task_id, start_time, end_time)
            if not metrics or metric_name not in metrics.metrics:
                return
                
            values = metrics.metrics[metric_name]
            if not values:
                return
                
            # Create chart
            chart = QChart()
            chart.setTitle(f"{metric_name} - {task_id}")
            
            # Create series
            series = QLineSeries()
            
            # Add data points
            min_value = float('inf')
            max_value = float('-inf')
            for value in values:
                if isinstance(value.value, (int, float)):
                    point_value = float(value.value)
                elif hasattr(value.value, 'total_seconds'):
                    point_value = value.value.total_seconds()
                else:
                    continue
                    
                timestamp = value.timestamp.timestamp() * 1000  # Convert to milliseconds
                series.append(timestamp, point_value)
                
                min_value = min(min_value, point_value)
                max_value = max(max_value, point_value)
                
            chart.addSeries(series)
            
            # Set up axes
            axis_x = QValueAxis()
            axis_x.setTickCount(10)
            axis_x.setFormat("mm:ss")
            axis_x.setTitleText("Time")
            chart.addAxis(axis_x, Qt.AlignBottom)
            series.attachAxis(axis_x)
            
            axis_y = QValueAxis()
            axis_y.setRange(min_value * 0.9, max_value * 1.1)
            axis_y.setTitleText(metric_name)
            chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_y)
            
            # Update chart view
            self.chart_view.setChart(chart)
            
        except Exception as e:
            logger.error(f"Failed to update metrics charts: {str(e)}")
            
    def _update_diagnostics(self):
        """Update diagnostics tables"""
        try:
            task_id = self.diag_task_combo.currentText()
            if not task_id:
                return
                
            diagnostics = self.monitor.get_diagnostics(task_id)
            if not diagnostics:
                return
                
            # Update events table
            self.events_table.setRowCount(len(diagnostics.events))
            for i, event in enumerate(diagnostics.events):
                self.events_table.setItem(i, 0, QTableWidgetItem(event["timestamp"]))
                self.events_table.setItem(i, 1, QTableWidgetItem(event["type"]))
                self.events_table.setItem(i, 2, QTableWidgetItem(event["message"]))
                
            # Update warnings table
            self.warnings_table.setRowCount(len(diagnostics.warnings))
            for i, warning in enumerate(diagnostics.warnings):
                self.warnings_table.setItem(i, 0, QTableWidgetItem(warning["timestamp"]))
                self.warnings_table.setItem(i, 1, QTableWidgetItem(warning["message"]))
                
            # Update errors table
            self.errors_table.setRowCount(len(diagnostics.errors))
            for i, error in enumerate(diagnostics.errors):
                self.errors_table.setItem(i, 0, QTableWidgetItem(error["timestamp"]))
                self.errors_table.setItem(i, 1, QTableWidgetItem(error["message"]))
                
        except Exception as e:
            logger.error(f"Failed to update diagnostics: {str(e)}")
