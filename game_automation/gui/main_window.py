from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QTabWidget, QTextEdit, 
                            QListWidget, QProgressBar, QMessageBox, QFrame, QGroupBox,
                            QListWidgetItem)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor, QPainter
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
import psutil
import os
import json
from utils.config_manager import config_manager
from utils.performance_manager import PerformanceMonitor
from game_automation.core.engine.game_engine import GameEngine
from .config_editor import ConfigEditor
from .task_dialog import TaskDialog

class MainWindow(QMainWindow):
    def __init__(self, game_engine=None):
        super().__init__()
        self.game_engine = game_engine
        self.setWindowTitle("游戏自动化控制面板")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # 初始化性能监控数据
        self.cpu_data = []
        self.memory_data = []
        self.max_data_points = 60  # 1分钟的数据
        
        # 初始化任务列表
        self.tasks = []
        self.load_tasks()  # 加载保存的任务
        
        self.init_ui()
        self.apply_theme()
        self.connect_signals()

    def init_ui(self):
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.init_dashboard_tab()
        self.init_tasks_tab()
        self.init_settings_tab()
        self.init_logs_tab()
        self.init_config_editor_tab()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(1000)  # 每秒更新一次

    def init_dashboard_tab(self):
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        layout.setSpacing(20)

        # 状态信息组
        status_group = QGroupBox("系统状态")
        status_layout = QHBoxLayout(status_group)
        
        # CPU状态
        cpu_frame = QFrame()
        cpu_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        cpu_layout = QVBoxLayout(cpu_frame)
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_progress)
        
        # 内存状态
        memory_frame = QFrame()
        memory_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        memory_layout = QVBoxLayout(memory_frame)
        self.memory_label = QLabel("内存: 0%")
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        memory_layout.addWidget(self.memory_label)
        memory_layout.addWidget(self.memory_progress)
        
        # 任务状态
        task_frame = QFrame()
        task_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        task_layout = QVBoxLayout(task_frame)
        self.tasks_label = QLabel("任务数: 0")
        self.task_progress = QProgressBar()
        self.task_progress.setRange(0, 100)
        task_layout.addWidget(self.tasks_label)
        task_layout.addWidget(self.task_progress)
        
        status_layout.addWidget(cpu_frame)
        status_layout.addWidget(memory_frame)
        status_layout.addWidget(task_frame)
        
        layout.addWidget(status_group)

        # 性能监控图表
        chart_group = QGroupBox("性能监控")
        chart_layout = QVBoxLayout(chart_group)
        
        # 创建图表
        self.performance_chart = QChart()
        self.performance_chart.setTitle("系统性能")
        self.performance_chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        self.performance_chart.legend().setVisible(True)
        self.performance_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        # 创建X轴（时间）
        axis_x = QValueAxis()
        axis_x.setRange(0, self.max_data_points)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("时间 (秒)")
        self.performance_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        
        # 创建Y轴（百分比）
        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setLabelFormat("%d%%")
        axis_y.setTitleText("使用率")
        self.performance_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        # CPU数据系列
        self.cpu_series = QLineSeries()
        self.cpu_series.setName("CPU使用率")
        self.performance_chart.addSeries(self.cpu_series)
        self.cpu_series.attachAxis(axis_x)
        self.cpu_series.attachAxis(axis_y)
        
        # 内存数据系列
        self.memory_series = QLineSeries()
        self.memory_series.setName("内存使用率")
        self.performance_chart.addSeries(self.memory_series)
        self.memory_series.attachAxis(axis_x)
        self.memory_series.attachAxis(axis_y)
        
        # 创建图表视图
        chart_view = QChartView(self.performance_chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_layout.addWidget(chart_view)
        
        layout.addWidget(chart_group)

        # 控制按钮组
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout(control_group)
        
        self.start_button = QPushButton("开始")
        self.stop_button = QPushButton("停止")
        self.pause_button = QPushButton("暂停")
        
        for btn in [self.start_button, self.stop_button, self.pause_button]:
            btn.setMinimumWidth(120)
            btn.setMinimumHeight(40)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addStretch()
        
        layout.addWidget(control_group)
        layout.addStretch()

        self.tab_widget.addTab(dashboard, "仪表盘")

    def init_tasks_tab(self):
        tasks = QWidget()
        layout = QVBoxLayout(tasks)
        layout.setSpacing(10)

        # 任务列表组
        list_group = QGroupBox("任务列表")
        list_layout = QVBoxLayout(list_group)
        
        self.task_list = QListWidget()
        self.task_list.setAlternatingRowColors(True)
        list_layout.addWidget(self.task_list)
        
        layout.addWidget(list_group)

        # 任务控制组
        control_group = QGroupBox("任务控制")
        control_layout = QHBoxLayout(control_group)
        
        self.add_button = QPushButton("添加任务")
        self.remove_button = QPushButton("删除任务")
        self.edit_button = QPushButton("编辑任务")
        
        for btn in [self.add_button, self.remove_button, self.edit_button]:
            btn.setMinimumWidth(120)
            btn.setMinimumHeight(40)
        
        control_layout.addWidget(self.add_button)
        control_layout.addWidget(self.remove_button)
        control_layout.addWidget(self.edit_button)
        control_layout.addStretch()
        
        layout.addWidget(control_group)

        self.tab_widget.addTab(tasks, "任务")
        self.update_task_list()  # 更新任务列表显示

    def init_settings_tab(self):
        settings = QWidget()
        layout = QVBoxLayout(settings)
        layout.setSpacing(20)

        # 基本设置组
        basic_group = QGroupBox("基本设置")
        basic_layout = QVBoxLayout(basic_group)
        
        # 语言设置
        language_layout = QHBoxLayout()
        language_label = QLabel("语言")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["简体中文", "English", "日本語"])
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        
        basic_layout.addLayout(language_layout)
        layout.addWidget(basic_group)

        # 高级设置组
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # 性能模式设置
        performance_layout = QHBoxLayout()
        performance_label = QLabel("性能模式")
        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["平衡", "性能优先", "节能"])
        performance_layout.addWidget(performance_label)
        performance_layout.addWidget(self.performance_combo)
        performance_layout.addStretch()
        
        advanced_layout.addLayout(performance_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()

        self.tab_widget.addTab(settings, "设置")

    def init_logs_tab(self):
        logs = QWidget()
        layout = QVBoxLayout(logs)

        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        self.tab_widget.addTab(logs, "日志")

    def init_config_editor_tab(self):
        self.config_editor = ConfigEditor()
        self.tab_widget.addTab(self.config_editor, "配置编辑器")

    def update_ui(self):
        """更新UI显示的数据"""
        # 获取性能数据
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # 更新标签和进度条
        self.cpu_label.setText(f"CPU: {cpu_percent}%")
        self.cpu_progress.setValue(int(cpu_percent))
        
        self.memory_label.setText(f"内存: {memory_percent}%")
        self.memory_progress.setValue(int(memory_percent))
        
        task_count = self.get_task_count()
        self.tasks_label.setText(f"任务数: {task_count}")
        self.task_progress.setValue(int(task_count))
        
        # 更新性能图表数据
        self.update_performance_chart(cpu_percent, memory_percent)

    def update_performance_chart(self, cpu_percent, memory_percent):
        """更新性能监控图表"""
        # 更新数据列表
        self.cpu_data.append(cpu_percent)
        self.memory_data.append(memory_percent)
        
        # 保持固定数量的数据点
        if len(self.cpu_data) > self.max_data_points:
            self.cpu_data.pop(0)
            self.memory_data.pop(0)
        
        # 更新图表数据
        self.cpu_series.clear()
        self.memory_series.clear()
        
        for i, (cpu, mem) in enumerate(zip(self.cpu_data, self.memory_data)):
            self.cpu_series.append(i, cpu)
            self.memory_series.append(i, mem)

    def get_task_count(self):
        """获取当前任务数"""
        return len(self.tasks)

    def update_task_list(self):
        """更新任务列表显示"""
        self.task_list.clear()
        for task in self.tasks:
            item = QListWidgetItem()
            item.setText(f"{task['name']} - {task['type']} (优先级: {task['priority']})")
            self.task_list.addItem(item)

    def load_tasks(self):
        """从文件加载任务数据"""
        try:
            tasks_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'tasks.json')
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    self.tasks = json.load(f)
        except Exception as e:
            self.log_text.append(f"加载任务数据失败: {str(e)}")
            self.tasks = []

    def save_tasks(self):
        """保存任务数据到文件"""
        try:
            tasks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            os.makedirs(tasks_dir, exist_ok=True)
            tasks_file = os.path.join(tasks_dir, 'tasks.json')
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_text.append(f"保存任务数据失败: {str(e)}")

    def apply_theme(self):
        """应用主题样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 14px;
            }
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #5d5d5d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4d4d4d;
            }
            QTabBar::tab:hover {
                background-color: #5d5d5d;
            }
            QTextEdit, QListWidget {
                background-color: #1d1d1d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel {
                padding: 4px;
                font-size: 14px;
            }
            QComboBox {
                background-color: #3d3d3d;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                min-width: 150px;
            }
            QComboBox:hover {
                background-color: #4d4d4d;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(""" + os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources/icons/down-arrow.svg").replace("\\", "/") + """);
                width: 12px;
                height: 12px;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #1d1d1d;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3d8ec9;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #1d1d1d;
                color: #ffffff;
            }
            QFrame {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 4px;
            }
        """)

    def connect_signals(self):
        """连接信号和槽"""
        # 控制按钮
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        self.pause_button.clicked.connect(self.on_pause)
        
        # 任务按钮
        self.add_button.clicked.connect(self.on_add_task)
        self.remove_button.clicked.connect(self.on_remove_task)
        self.edit_button.clicked.connect(self.on_edit_task)
        
        # 设置选项
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)
        self.performance_combo.currentIndexChanged.connect(self.on_performance_changed)

    def on_start(self):
        """开始按钮点击事件"""
        if self.game_engine:
            self.log_text.append("正在启动游戏引擎...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.game_engine.start()

    def on_stop(self):
        """停止按钮点击事件"""
        if self.game_engine:
            self.log_text.append("正在停止游戏引擎...")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.game_engine.stop()

    def on_pause(self):
        """暂停按钮点击事件"""
        if self.game_engine:
            is_pausing = self.pause_button.text() == "暂停"
            self.log_text.append("已暂停游戏引擎" if is_pausing else "继续游戏引擎")
            self.pause_button.setText("继续" if is_pausing else "暂停")
            if is_pausing:
                self.game_engine.pause()
            else:
                self.game_engine.resume()

    def on_add_task(self):
        """添加任务按钮点击事件"""
        dialog = TaskDialog(self)
        if dialog.exec_():
            task_data = dialog.get_task_data()
            self.tasks.append(task_data)
            self.update_task_list()
            self.log_text.append(f"已添加新任务: {task_data['name']}")
            self.save_tasks()

    def on_remove_task(self):
        """删除任务按钮点击事件"""
        selected_items = self.task_list.selectedItems()
        if selected_items:
            for item in selected_items:
                task_index = self.task_list.row(item)
                self.tasks.pop(task_index)
                self.task_list.takeItem(task_index)
            self.log_text.append("已删除选中的任务")
            self.save_tasks()

    def on_edit_task(self):
        """编辑任务按钮点击事件"""
        selected_items = self.task_list.selectedItems()
        if selected_items:
            task_index = self.task_list.row(selected_items[0])
            task = self.tasks[task_index]
            
            dialog = TaskDialog(self, task)
            if dialog.exec_():
                task_data = dialog.get_task_data()
                self.tasks[task_index] = task_data
                self.update_task_list()
                self.log_text.append(f"已更新任务: {task_data['name']}")
                self.save_tasks()

    def on_language_changed(self, index):
        """语言选择改变事件"""
        languages = ["简体中文", "English", "日本語"]
        self.log_text.append(f"切换语言到: {languages[index]}")
        # TODO: 实现语言切换的逻辑

    def on_performance_changed(self, index):
        """性能模式改变事件"""
        modes = ["平衡", "性能优先", "节能"]
        self.log_text.append(f"切换性能模式到: {modes[index]}")
        if self.game_engine:
            # TODO: 实现性能模式切换的逻辑
            pass
