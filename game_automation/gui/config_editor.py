"""
Configuration editor window
"""

from typing import Dict, Any, Optional, Union
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QPushButton, QMessageBox, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QTabWidget, QWidget
)
from PySide6.QtCore import Qt, Signal, Slot

from ..core.events.event_manager import EventManager, Event, EventType
from ..core.config_manager import ConfigManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ConfigEditor(QDialog):
    """Configuration editor dialog"""
    
    config_changed = Signal(dict)  # Configuration change signal
    
    def __init__(self, event_manager: EventManager, parent=None):
        """Initialize editor
        
        Args:
            event_manager: Event manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.event_manager = event_manager
        self.config_manager = ConfigManager()
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("配置编辑器")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Game settings tab
        game_tab = self._create_game_tab()
        tabs.addTab(game_tab, "游戏设置")
        
        # Recognition settings tab
        recog_tab = self._create_recognition_tab()
        tabs.addTab(recog_tab, "识别设置")
        
        # Task settings tab
        task_tab = self._create_task_tab()
        tabs.addTab(task_tab, "任务设置")
        
        # Debug settings tab
        debug_tab = self._create_debug_tab()
        tabs.addTab(debug_tab, "调试设置")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._reset_config)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self._save_config)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def _create_game_tab(self) -> QWidget:
        """Create game settings tab
        
        Returns:
            QWidget: Game settings widget
        """
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Window settings
        self.window_title = QLineEdit()
        layout.addRow("窗口标题:", self.window_title)
        
        self.process_name = QLineEdit()
        layout.addRow("进程名称:", self.process_name)
        
        # Resolution
        res_group = QGroupBox("分辨率")
        res_layout = QFormLayout(res_group)
        
        self.width = QSpinBox()
        self.width.setRange(640, 3840)
        res_layout.addRow("宽度:", self.width)
        
        self.height = QSpinBox()
        self.height.setRange(480, 2160)
        res_layout.addRow("高度:", self.height)
        
        layout.addRow(res_group)
        
        # Performance
        self.fps_limit = QSpinBox()
        self.fps_limit.setRange(30, 240)
        layout.addRow("FPS限制:", self.fps_limit)
        
        self.input_delay = QSpinBox()
        self.input_delay.setRange(0, 1000)
        self.input_delay.setSuffix(" ms")
        layout.addRow("输入延迟:", self.input_delay)
        
        return widget
        
    def _create_recognition_tab(self) -> QWidget:
        """Create recognition settings tab
        
        Returns:
            QWidget: Recognition settings widget
        """
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.1, 1.0)
        self.confidence.setSingleStep(0.1)
        layout.addRow("置信度阈值:", self.confidence)
        
        self.match_method = QComboBox()
        self.match_method.addItems(["template", "feature", "color"])
        layout.addRow("匹配方法:", self.match_method)
        
        self.max_matches = QSpinBox()
        self.max_matches.setRange(1, 100)
        layout.addRow("最大匹配数:", self.max_matches)
        
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.1, 2.0)
        self.scale_factor.setSingleStep(0.1)
        layout.addRow("缩放因子:", self.scale_factor)
        
        self.use_grayscale = QCheckBox()
        layout.addRow("使用灰度图:", self.use_grayscale)
        
        return widget
        
    def _create_task_tab(self) -> QWidget:
        """Create task settings tab
        
        Returns:
            QWidget: Task settings widget
        """
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.auto_retry = QCheckBox()
        layout.addRow("自动重试:", self.auto_retry)
        
        self.max_retries = QSpinBox()
        self.max_retries.setRange(0, 10)
        layout.addRow("最大重试次数:", self.max_retries)
        
        self.retry_delay = QSpinBox()
        self.retry_delay.setRange(1, 60)
        self.retry_delay.setSuffix(" 秒")
        layout.addRow("重试延迟:", self.retry_delay)
        
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 3600)
        self.timeout.setSuffix(" 秒")
        layout.addRow("超时时间:", self.timeout)
        
        return widget
        
    def _create_debug_tab(self) -> QWidget:
        """Create debug settings tab
        
        Returns:
            QWidget: Debug settings widget
        """
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.save_screenshots = QCheckBox()
        layout.addRow("保存截图:", self.save_screenshots)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        layout.addRow("日志级别:", self.log_level)
        
        self.show_matches = QCheckBox()
        layout.addRow("显示匹配:", self.show_matches)
        
        self.record_video = QCheckBox()
        layout.addRow("录制视频:", self.record_video)
        
        return widget
        
    def _load_config(self):
        """Load configuration"""
        try:
            self.config_manager.load_config()
            config = self.config_manager.get_config()
            
            # Game settings
            game_config = config.get("game", {})
            self.window_title.setText(game_config.get("window_title", ""))
            self.process_name.setText(game_config.get("process_name", ""))
            resolution = game_config.get("resolution", {})
            self.width.setValue(resolution.get("width", 1280))
            self.height.setValue(resolution.get("height", 720))
            self.fps_limit.setValue(game_config.get("fps_limit", 60))
            self.input_delay.setValue(game_config.get("input_delay", 50))
            
            # Recognition settings
            recog_config = config.get("recognition", {})
            self.confidence.setValue(recog_config.get("confidence_threshold", 0.8))
            self.match_method.setCurrentText(recog_config.get("match_method", "template"))
            self.max_matches.setValue(recog_config.get("max_matches", 5))
            self.scale_factor.setValue(recog_config.get("scale_factor", 1.0))
            self.use_grayscale.setChecked(recog_config.get("use_grayscale", True))
            
            # Task settings
            task_config = config.get("task", {})
            self.auto_retry.setChecked(task_config.get("auto_retry", True))
            self.max_retries.setValue(task_config.get("max_retries", 3))
            self.retry_delay.setValue(task_config.get("retry_delay", 5))
            self.timeout.setValue(task_config.get("timeout", 300))
            
            # Debug settings
            debug_config = config.get("debug", {})
            self.save_screenshots.setChecked(debug_config.get("save_screenshots", False))
            self.log_level.setCurrentText(debug_config.get("log_level", "INFO"))
            self.show_matches.setChecked(debug_config.get("show_matches", False))
            self.record_video.setChecked(debug_config.get("record_video", False))
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            QMessageBox.warning(self, "警告", f"加载配置失败: {str(e)}")
            
    def _save_config(self):
        """Save configuration"""
        try:
            # Game settings
            game_config = {
                "window_title": self.window_title.text(),
                "process_name": self.process_name.text(),
                "resolution": {
                    "width": self.width.value(),
                    "height": self.height.value()
                },
                "fps_limit": self.fps_limit.value(),
                "input_delay": self.input_delay.value()
            }
            self.config_manager.set_config(game_config, "game")
            
            # Recognition settings
            recog_config = {
                "confidence_threshold": self.confidence.value(),
                "match_method": self.match_method.currentText(),
                "max_matches": self.max_matches.value(),
                "scale_factor": self.scale_factor.value(),
                "use_grayscale": self.use_grayscale.isChecked()
            }
            self.config_manager.set_config(recog_config, "recognition")
            
            # Task settings
            task_config = {
                "auto_retry": self.auto_retry.isChecked(),
                "max_retries": self.max_retries.value(),
                "retry_delay": self.retry_delay.value(),
                "timeout": self.timeout.value()
            }
            self.config_manager.set_config(task_config, "task")
            
            # Debug settings
            debug_config = {
                "save_screenshots": self.save_screenshots.isChecked(),
                "log_level": self.log_level.currentText(),
                "show_matches": self.show_matches.isChecked(),
                "record_video": self.record_video.isChecked()
            }
            self.config_manager.set_config(debug_config, "debug")
            
            # Save to file
            self.config_manager.save_config()
            
            # Emit event
            self.event_manager.emit(Event(
                EventType.CONFIG_CHANGED,
                self.config_manager.get_config()
            ))
            
            QMessageBox.information(self, "成功", "配置已保存")
            self.accept()
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            
    def _reset_config(self):
        """Reset configuration"""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有配置吗？这将丢失所有修改。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_config()
            self._load_config()
