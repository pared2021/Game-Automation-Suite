from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QComboBox, QSpinBox, QPushButton,
                            QFormLayout, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt

class TaskDialog(QDialog):
    def __init__(self, parent=None, task=None):
        super().__init__(parent)
        self.task = task
        self.setWindowTitle("任务配置")
        self.setMinimumWidth(400)
        self.init_ui()
        
        if task:
            self.load_task(task)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本信息组
        basic_group = QGroupBox("基本信息")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["点击", "拖拽", "按键", "等待", "自定义"])
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 10)
        self.priority_spin.setValue(5)
        
        basic_layout.addRow("任务名称:", self.name_edit)
        basic_layout.addRow("任务类型:", self.type_combo)
        basic_layout.addRow("优先级:", self.priority_spin)
        
        layout.addWidget(basic_group)
        
        # 参数设置组
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout(params_group)
        
        self.x_spin = QSpinBox()
        self.x_spin.setRange(-9999, 9999)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(-9999, 9999)
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 10000)
        self.delay_spin.setValue(1000)
        self.delay_spin.setSuffix(" ms")
        
        params_layout.addRow("X坐标:", self.x_spin)
        params_layout.addRow("Y坐标:", self.y_spin)
        params_layout.addRow("延迟:", self.delay_spin)
        
        layout.addWidget(params_group)
        
        # 高级设置组
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QFormLayout(advanced_group)
        
        self.repeat_spin = QSpinBox()
        self.repeat_spin.setRange(1, 999)
        self.repeat_spin.setValue(1)
        self.condition_edit = QLineEdit()
        self.script_edit = QTextEdit()
        self.script_edit.setPlaceholderText("在此输入自定义Python脚本...")
        
        advanced_layout.addRow("重复次数:", self.repeat_spin)
        advanced_layout.addRow("触发条件:", self.condition_edit)
        advanced_layout.addRow("自定义脚本:", self.script_edit)
        
        layout.addWidget(advanced_group)
        
        # 按钮组
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)

    def load_task(self, task):
        """加载任务数据"""
        self.name_edit.setText(task.get('name', ''))
        self.type_combo.setCurrentText(task.get('type', '点击'))
        self.priority_spin.setValue(task.get('priority', 5))
        self.x_spin.setValue(task.get('x', 0))
        self.y_spin.setValue(task.get('y', 0))
        self.delay_spin.setValue(task.get('delay', 1000))
        self.repeat_spin.setValue(task.get('repeat', 1))
        self.condition_edit.setText(task.get('condition', ''))
        self.script_edit.setText(task.get('script', ''))

    def get_task_data(self):
        """获取任务数据"""
        return {
            'name': self.name_edit.text(),
            'type': self.type_combo.currentText(),
            'priority': self.priority_spin.value(),
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'delay': self.delay_spin.value(),
            'repeat': self.repeat_spin.value(),
            'condition': self.condition_edit.text(),
            'script': self.script_edit.toPlainText()
        }
