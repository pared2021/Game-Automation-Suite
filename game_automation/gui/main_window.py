"""
Main window of the application
"""

from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QMessageBox,
    QSplitter, QTreeWidget, QTreeWidgetItem, QApplication,
    QDockWidget, QStackedWidget
)
from PySide6.QtCore import Qt, Slot
import sys

from ..core.events.event_manager import EventManager, Event, EventType
from ..core.task.task_executor import TaskExecutor
from ..core.task.task_manager import TaskManager
from ..core.task.task_adapter import TaskAdapter
from ..core.config.config_manager import ConfigManager
from ..core.engine.game_engine import GameEngine
from .task_dialog import TaskDialog
from .config_editor import ConfigEditor
from .advanced_debug_interface import AdvancedDebugInterface
from .monitor import TaskMonitorWindow
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MainWindow(QMainWindow):
    """Main window of the application"""
    
    def __init__(
        self,
        game_engine: GameEngine,
        task_executor: TaskExecutor,
        event_manager: EventManager,
        task_manager: TaskManager
    ):
        """Initialize main window
        
        Args:
            game_engine: Game engine instance
            task_executor: Task executor instance
            event_manager: Event manager instance
            task_manager: Task manager instance
        """
        super().__init__()
        
        # Store core components
        self.game_engine = game_engine
        self.event_manager = event_manager
        self.task_executor = task_executor
        self.task_manager = task_manager
        self.config_manager = ConfigManager()
        
        # Initialize adapter
        self.task_adapter = TaskAdapter(
            self.event_manager,
            self.task_executor,
            self.task_manager
        )
        
        # Setup UI
        self._setup_ui()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Initialize components
        self._initialize_components()
        
    def _setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("游戏自动化套件")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Task list
        task_widget = QWidget()
        task_layout = QVBoxLayout(task_widget)
        
        # Task list
        self.task_tree = QTreeWidget()
        self.task_tree.setHeaderLabels(["任务", "状态", "进度"])
        self.task_tree.itemSelectionChanged.connect(self._task_selected)
        task_layout.addWidget(self.task_tree)
        
        # Task buttons
        task_button_layout = QHBoxLayout()
        
        self.add_task_btn = QPushButton("添加任务")
        self.add_task_btn.clicked.connect(self._add_task)
        task_button_layout.addWidget(self.add_task_btn)
        
        self.remove_task_btn = QPushButton("删除任务")
        self.remove_task_btn.clicked.connect(self._remove_task)
        self.remove_task_btn.setEnabled(False)
        task_button_layout.addWidget(self.remove_task_btn)
        
        task_layout.addLayout(task_button_layout)
        splitter.addWidget(task_widget)
        
        # Right panel - Stacked widget
        self.stacked_widget = QStackedWidget()
        
        # Task details widget
        self.task_details = QWidget()
        self.stacked_widget.addWidget(self.task_details)
        
        # Config widget
        self.config_editor = ConfigEditor(self.event_manager)
        self.stacked_widget.addWidget(self.config_editor)
        
        # Debug widget
        self.debug_interface = AdvancedDebugInterface(self.event_manager)
        self.stacked_widget.addWidget(self.debug_interface)
        
        # Monitor widget
        self.monitor_window = TaskMonitorWindow(self.task_manager.monitor)
        self.stacked_widget.addWidget(self.monitor_window)
        
        splitter.addWidget(self.stacked_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([400, 800])
        
        # Create toolbar
        toolbar = self.addToolBar("工具栏")
        toolbar.setMovable(False)
        
        # Add toolbar buttons
        details_action = toolbar.addAction("任务详情")
        details_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.task_details))
        
        config_action = toolbar.addAction("配置")
        config_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.config_editor))
        
        debug_action = toolbar.addAction("调试")
        debug_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.debug_interface))
        
        monitor_action = toolbar.addAction("监控")
        monitor_action.triggered.connect(lambda: self.stacked_widget.setCurrentWidget(self.monitor_window))
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel()
        self.status_bar.addWidget(self.status_label)
        
    def _register_event_handlers(self):
        """Register event handlers"""
        self.event_manager.subscribe(EventType.TASK_ADDED, self._on_task_added)
        self.event_manager.subscribe(EventType.TASK_REMOVED, self._on_task_removed)
        self.event_manager.subscribe(EventType.TASK_UPDATED, self._on_task_updated)
        self.event_manager.subscribe(EventType.CONFIG_CHANGED, self._on_config_changed)
        
    async def _initialize_components(self):
        """Initialize components"""
        try:
            # Initialize adapter
            await self.task_adapter.initialize()
            
            # Load tasks
            self._load_tasks()
            
            # Update status
            self._update_status()
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")
            
    def _load_tasks(self):
        """Load tasks"""
        self.task_tree.clear()
        
        for task in self.task_manager.tasks.values():
            self._add_task_item(task)
            
    def _add_task_item(self, task: Dict):
        """Add task item to tree
        
        Args:
            task: Task data
        """
        item = QTreeWidgetItem()
        item.setText(0, task['name'])
        item.setText(1, task['status'])
        item.setText(2, f"{task['progress']:.1f}%")
        item.setData(0, Qt.UserRole, task['task_id'])
        
        self.task_tree.addTopLevelItem(item)
        
    def _update_task_item(self, task: Dict):
        """Update task item in tree
        
        Args:
            task: Task data
        """
        task_id = task['task_id']
        
        for i in range(self.task_tree.topLevelItemCount()):
            item = self.task_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == task_id:
                item.setText(0, task['name'])
                item.setText(1, task['status'])
                item.setText(2, f"{task['progress']:.1f}%")
                break
                
    def _remove_task_item(self, task_id: str):
        """Remove task item from tree
        
        Args:
            task_id: Task ID
        """
        for i in range(self.task_tree.topLevelItemCount()):
            item = self.task_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == task_id:
                self.task_tree.takeTopLevelItem(i)
                break
                
    def _update_status(self):
        """Update status bar"""
        stats = self.task_manager.get_statistics()
        
        status_text = (
            f"总任务: {stats['total']} | "
            f"等待中: {stats['pending']} | "
            f"运行中: {stats['running']} | "
            f"已完成: {stats['completed']} | "
            f"失败: {stats['failed']}"
        )
        
        self.status_label.setText(status_text)
        
    @Slot()
    def _task_selected(self):
        """Handle task selection"""
        items = self.task_tree.selectedItems()
        self.remove_task_btn.setEnabled(bool(items))
        
        if items:
            task_id = items[0].data(0, Qt.UserRole)
            task = self.task_manager.get_task(task_id)
            if task:
                self._show_task_details(task)
                
    def _show_task_details(self, task: Dict):
        """Show task details
        
        Args:
            task: Task data
        """
        # TODO: Implement task details view
        pass
        
    @Slot()
    def _add_task(self):
        """Add new task"""
        dialog = TaskDialog(self.event_manager, self)
        if dialog.exec():
            task_data = dialog.get_task_data()
            self.task_manager.add_task(task_data)
            
    @Slot()
    def _remove_task(self):
        """Remove selected task"""
        items = self.task_tree.selectedItems()
        if not items:
            return
            
        task_id = items[0].data(0, Qt.UserRole)
        
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除选中的任务吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.task_manager.remove_task(task_id)
            
    def _on_task_added(self, event: Event):
        """Handle task added event
        
        Args:
            event: Event instance
        """
        self._add_task_item(event.data)
        self._update_status()
        
    def _on_task_removed(self, event: Event):
        """Handle task removed event
        
        Args:
            event: Event instance
        """
        self._remove_task_item(event.data['task_id'])
        self._update_status()
        
    def _on_task_updated(self, event: Event):
        """Handle task updated event
        
        Args:
            event: Event instance
        """
        self._update_task_item(event.data)
        self._update_status()
        
    def _on_config_changed(self, event: Event):
        """Handle config changed event
        
        Args:
            event: Event instance
        """
        # Reload config
        self.config_manager.load_config()
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        self.task_adapter.cleanup()
        event.accept()

def run_gui(
        game_engine: GameEngine,
        task_executor: TaskExecutor,
        event_manager: EventManager,
        task_manager: TaskManager
    ):
    """Run the GUI application
    
    Args:
        game_engine: Game engine instance
        task_executor: Task executor instance
        event_manager: Event manager instance
        task_manager: Task manager instance
    """
    app = QApplication(sys.argv)
    window = MainWindow(game_engine, task_executor, event_manager, task_manager)
    window.show()
    sys.exit(app.exec_())
