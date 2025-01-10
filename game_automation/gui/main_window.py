import sys
from typing import Dict, Optional
from datetime import datetime
import asyncio
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QComboBox,
    QSpinBox,
    QTabWidget,
    QGroupBox,
    QScrollArea,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor

from ..core.engine.game_engine import GameEngine, GameState, GameEvent
from ..core.task_executor import TaskExecutor, TaskType, TaskPriority
from ..core.error.error_manager import ErrorCategory, ErrorSeverity

class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(
        self,
        game_engine: GameEngine,
        task_executor: TaskExecutor
    ):
        super().__init__()
        
        self.game_engine = game_engine
        self.task_executor = task_executor
        
        # 窗口设置
        self.setWindowTitle("游戏自动化工具")
        self.setMinimumSize(800, 600)
        
        # 创建界面
        self._create_ui()
        
        # 定时器
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_status)
        self._update_timer.start(1000)  # 每秒更新一次
        
        # 注册事件处理器
        self._register_event_handlers()

    def _create_ui(self):
        """创建界面"""
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 右侧信息面板
        info_panel = self._create_info_panel()
        main_layout.addWidget(info_panel, 2)

    def _create_control_panel(self) -> QWidget:
        """创建控制面板
        
        Returns:
            QWidget: 控制面板组件
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 状态控制
        status_group = QGroupBox("状态控制")
        status_layout = QVBoxLayout(status_group)
        
        self._start_button = QPushButton("启动")
        self._start_button.clicked.connect(self._start_engine)
        status_layout.addWidget(self._start_button)
        
        self._pause_button = QPushButton("暂停")
        self._pause_button.clicked.connect(self._pause_engine)
        self._pause_button.setEnabled(False)
        status_layout.addWidget(self._pause_button)
        
        self._stop_button = QPushButton("停止")
        self._stop_button.clicked.connect(self._stop_engine)
        self._stop_button.setEnabled(False)
        status_layout.addWidget(self._stop_button)
        
        layout.addWidget(status_group)
        
        # 任务控制
        task_group = QGroupBox("任务控制")
        task_layout = QVBoxLayout(task_group)
        
        # 任务类型选择
        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel("任务类型:"))
        self._task_type_combo = QComboBox()
        for task_type in TaskType:
            self._task_type_combo.addItem(task_type.name)
        task_type_layout.addWidget(self._task_type_combo)
        task_layout.addLayout(task_type_layout)
        
        # 任务优先级选择
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("优先级:"))
        self._priority_combo = QComboBox()
        for priority in TaskPriority:
            self._priority_combo.addItem(priority.name)
        priority_layout.addWidget(self._priority_combo)
        task_layout.addLayout(priority_layout)
        
        # 添加任务按钮
        add_task_button = QPushButton("添加任务")
        add_task_button.clicked.connect(self._add_task)
        task_layout.addWidget(add_task_button)
        
        layout.addWidget(task_group)
        
        # 设置组
        settings_group = QGroupBox("设置")
        settings_layout = QVBoxLayout(settings_group)
        
        # FPS设置
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(30)
        self._fps_spin.valueChanged.connect(self._update_fps)
        fps_layout.addWidget(self._fps_spin)
        settings_layout.addLayout(fps_layout)
        
        layout.addWidget(settings_group)
        
        # 添加伸展
        layout.addStretch()
        
        return panel

    def _create_info_panel(self) -> QWidget:
        """创建信息面板
        
        Returns:
            QWidget: 信息面板组件
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标签页
        tab_widget = QTabWidget()
        
        # 状态页
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        
        # 游戏状态
        game_status_group = QGroupBox("游戏状态")
        game_status_layout = QVBoxLayout(game_status_group)
        self._state_label = QLabel()
        game_status_layout.addWidget(self._state_label)
        status_layout.addWidget(game_status_group)
        
        # 场景信息
        scene_group = QGroupBox("场景信息")
        scene_layout = QVBoxLayout(scene_group)
        self._scene_text = QTextEdit()
        self._scene_text.setReadOnly(True)
        scene_layout.addWidget(self._scene_text)
        status_layout.addWidget(scene_group)
        
        tab_widget.addTab(status_tab, "状态")
        
        # 任务页
        task_tab = QWidget()
        task_layout = QVBoxLayout(task_tab)
        
        self._task_text = QTextEdit()
        self._task_text.setReadOnly(True)
        task_layout.addWidget(self._task_text)
        
        tab_widget.addTab(task_tab, "任务")
        
        # 日志页
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        log_layout.addWidget(self._log_text)
        
        tab_widget.addTab(log_tab, "日志")
        
        layout.addWidget(tab_widget)
        
        return panel

    def _register_event_handlers(self):
        """注册事件处理器"""
        self.game_engine.add_event_handler(
            GameEvent.STATE_CHANGED,
            self._handle_state_changed
        )
        
        self.game_engine.add_event_handler(
            GameEvent.SCENE_CHANGED,
            self._handle_scene_changed
        )
        
        self.game_engine.add_event_handler(
            GameEvent.ERROR_OCCURRED,
            self._handle_error
        )

    async def _start_engine(self):
        """启动引擎"""
        try:
            await self.game_engine.start()
            
            self._start_button.setEnabled(False)
            self._pause_button.setEnabled(True)
            self._stop_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"启动失败: {str(e)}"
            )

    async def _pause_engine(self):
        """暂停引擎"""
        try:
            if self.game_engine.get_state() == GameState.RUNNING:
                await self.game_engine.pause()
                self._pause_button.setText("继续")
            else:
                await self.game_engine.resume()
                self._pause_button.setText("暂停")
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"操作失败: {str(e)}"
            )

    async def _stop_engine(self):
        """停止引擎"""
        try:
            await self.game_engine.stop()
            
            self._start_button.setEnabled(True)
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            self._pause_button.setText("暂停")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"停止失败: {str(e)}"
            )

    async def _add_task(self):
        """添加任务"""
        try:
            task_type = TaskType[self._task_type_combo.currentText()]
            priority = TaskPriority[self._priority_combo.currentText()]
            
            task_id = await self.task_executor.add_task(
                name="测试任务",
                task_type=task_type,
                priority=priority
            )
            
            self._log_message(f"添加任务: {task_id}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"添加任务失败: {str(e)}"
            )

    def _update_fps(self, fps: int):
        """更新FPS
        
        Args:
            fps: 帧率
        """
        self.game_engine._frame_time = 1.0 / fps

    def _update_status(self):
        """更新状态显示"""
        # 更新游戏状态
        state = self.game_engine.get_state()
        self._state_label.setText(f"状态: {state.name}")
        
        # TODO: 更新其他状态信息

    async def _handle_state_changed(
        self,
        event: GameEvent,
        data: Dict
    ):
        """处理状态改变事件
        
        Args:
            event: 事件类型
            data: 事件数据
        """
        self._log_message(
            f"状态改变: {data['old_state']} -> {data['new_state']}"
        )

    async def _handle_scene_changed(
        self,
        event: GameEvent,
        data: Dict
    ):
        """处理场景改变事件
        
        Args:
            event: 事件类型
            data: 事件数据
        """
        self._scene_text.setText(
            f"类型: {data['type']}\n"
            f"置信度: {data['confidence']}\n"
            f"元素数量: {len(data['elements'])}"
        )

    async def _handle_error(
        self,
        event: GameEvent,
        data: Dict
    ):
        """处理错误事件
        
        Args:
            event: 事件类型
            data: 事件数据
        """
        self._log_message(
            f"错误: [{data['error_code']}] {data['message']}",
            error=True
        )

    def _log_message(
        self,
        message: str,
        error: bool = False
    ):
        """记录日志消息
        
        Args:
            message: 消息内容
            error: 是否为错误消息
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = "ERROR" if error else "INFO"
        
        self._log_text.moveCursor(QTextCursor.MoveOperation.End)
        self._log_text.insertPlainText(
            f"[{timestamp}] [{level}] {message}\n"
        )
        self._log_text.moveCursor(QTextCursor.MoveOperation.End)

def run_gui(
    game_engine: GameEngine,
    task_executor: TaskExecutor
):
    """运行GUI
    
    Args:
        game_engine: 游戏引擎
        task_executor: 任务执行器
    """
    app = QApplication(sys.argv)
    window = MainWindow(game_engine, task_executor)
    window.show()
    sys.exit(app.exec())
