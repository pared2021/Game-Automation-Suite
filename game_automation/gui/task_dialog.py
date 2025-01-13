"""
Task dialog for creating and editing tasks
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QSpinBox, QDateTimeEdit,
    QCheckBox, QTextEdit, QPushButton, QLabel,
    QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from ..core.events.event_manager import EventManager, Event, EventType
from ..core.task_types import TaskType, TaskPriority
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TaskDialog(QDialog):
    """Dialog for creating and editing tasks"""
    
    def __init__(
        self,
        event_manager: EventManager,
        parent=None,
        task_data: Optional[Dict] = None
    ):
        """Initialize task dialog
        
        Args:
            event_manager: Event manager instance
            parent: Parent widget
            task_data: Task data for editing existing task
        """
        super().__init__(parent)
        
        self.event_manager = event_manager
        self.task_data = task_data
        
        self._setup_ui()
        
        if task_data:
            self._load_task_data(task_data)
            
    def _setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("任务配置")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Basic info
        basic_group = QGroupBox("基本信息")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        basic_layout.addRow("任务名称:", self.name_edit)
        
        self.type_combo = QComboBox()
        for task_type in TaskType:
            self.type_combo.addItem(task_type.name)
        basic_layout.addRow("任务类型:", self.type_combo)
        
        self.priority_combo = QComboBox()
        for priority in TaskPriority:
            self.priority_combo.addItem(priority.name)
        basic_layout.addRow("优先级:", self.priority_combo)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        basic_layout.addRow("任务描述:", self.desc_edit)
        
        layout.addWidget(basic_group)
        
        # Schedule
        schedule_group = QGroupBox("调度设置")
        schedule_layout = QFormLayout(schedule_group)
        
        self.scheduled_check = QCheckBox("启用调度")
        self.scheduled_check.stateChanged.connect(self._on_scheduled_changed)
        schedule_layout.addRow(self.scheduled_check)
        
        self.start_time = QDateTimeEdit()
        self.start_time.setDateTime(datetime.now())
        self.start_time.setEnabled(False)
        schedule_layout.addRow("开始时间:", self.start_time)
        
        self.end_time = QDateTimeEdit()
        self.end_time.setDateTime(datetime.now() + timedelta(hours=1))
        self.end_time.setEnabled(False)
        schedule_layout.addRow("结束时间:", self.end_time)
        
        layout.addWidget(schedule_group)
        
        # Repeat
        repeat_group = QGroupBox("重复设置")
        repeat_layout = QFormLayout(repeat_group)
        
        self.repeat_check = QCheckBox("启用重复")
        self.repeat_check.stateChanged.connect(self._on_repeat_changed)
        repeat_layout.addRow(self.repeat_check)
        
        self.repeat_count = QSpinBox()
        self.repeat_count.setMinimum(-1)
        self.repeat_count.setValue(-1)
        self.repeat_count.setEnabled(False)
        repeat_layout.addRow("重复次数(-1为无限):", self.repeat_count)
        
        self.interval = QSpinBox()
        self.interval.setMinimum(1)
        self.interval.setMaximum(3600)
        self.interval.setValue(60)
        self.interval.setEnabled(False)
        repeat_layout.addRow("间隔时间(秒):", self.interval)
        
        layout.addWidget(repeat_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def _on_scheduled_changed(self, state):
        """Handle scheduled checkbox state change"""
        enabled = state == Qt.Checked
        self.start_time.setEnabled(enabled)
        self.end_time.setEnabled(enabled)
        
    def _on_repeat_changed(self, state):
        """Handle repeat checkbox state change"""
        enabled = state == Qt.Checked
        self.repeat_count.setEnabled(enabled)
        self.interval.setEnabled(enabled)
        
    def _load_task_data(self, task_data: Dict):
        """Load task data into UI
        
        Args:
            task_data: Task data to load
        """
        self.name_edit.setText(task_data['name'])
        self.type_combo.setCurrentText(task_data['task_type'])
        self.priority_combo.setCurrentText(task_data['priority'])
        self.desc_edit.setText(task_data.get('description', ''))
        
        if task_data.get('is_scheduled'):
            self.scheduled_check.setChecked(True)
            self.start_time.setDateTime(task_data['start_time'])
            self.end_time.setDateTime(task_data['end_time'])
            
        if task_data.get('is_repeating'):
            self.repeat_check.setChecked(True)
            self.repeat_count.setValue(task_data['repeat_count'])
            self.interval.setValue(task_data['interval'])
            
    def get_task_data(self) -> Dict:
        """Get task data from UI
        
        Returns:
            Dict: Task data
        """
        task_data = {
            'name': self.name_edit.text(),
            'task_type': self.type_combo.currentText(),
            'priority': self.priority_combo.currentText(),
            'description': self.desc_edit.toPlainText(),
            'is_scheduled': self.scheduled_check.isChecked(),
            'is_repeating': self.repeat_check.isChecked()
        }
        
        if task_data['is_scheduled']:
            task_data.update({
                'start_time': self.start_time.dateTime().toPython(),
                'end_time': self.end_time.dateTime().toPython()
            })
            
        if task_data['is_repeating']:
            task_data.update({
                'repeat_count': self.repeat_count.value(),
                'interval': self.interval.value()
            })
            
        return task_data
        
    def accept(self):
        """Handle dialog accept"""
        # Validate input
        if not self.name_edit.text():
            QMessageBox.warning(self, "错误", "请输入任务名称")
            return
            
        if self.scheduled_check.isChecked():
            if self.start_time.dateTime() >= self.end_time.dateTime():
                QMessageBox.warning(self, "错误", "结束时间必须大于开始时间")
                return
                
        super().accept()
