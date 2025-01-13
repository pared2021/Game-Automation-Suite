"""
Advanced debug interface for game automation
"""

import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget,
    QDockWidget, QMenuBar, QStatusBar, QMessageBox,
    QComboBox, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..core.events.event_manager import Event, EventType, EventManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GameStateVisualizer(QWidget):
    """Game state visualization widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Add subplot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("游戏状态可视化")
        
    def update_visualization(self, game_state: Dict[str, Any]):
        """Update visualization with new game state
        
        Args:
            game_state: Game state data
        """
        # Clear previous plot
        self.ax.clear()
        
        # TODO: Implement visualization based on game state
        # This is just a placeholder
        if 'data' in game_state:
            data = game_state['data']
            if isinstance(data, np.ndarray):
                self.ax.imshow(data)
                
        self.canvas.draw()

class GameStateInspector(QWidget):
    """Game state inspection widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["属性", "值"])
        layout.addWidget(self.tree)
        
    def update_state(self, game_state: Dict[str, Any]):
        """Update displayed game state
        
        Args:
            game_state: Game state data
        """
        self.tree.clear()
        
        for key, value in game_state.items():
            item = QTreeWidgetItem([str(key)])
            self.add_items(item, value)
            self.tree.addTopLevelItem(item)
            
        self.tree.expandAll()
        
    def add_items(self, parent: QTreeWidgetItem, value: Any):
        """Add items to tree recursively
        
        Args:
            parent: Parent tree item
            value: Value to add
        """
        if isinstance(value, dict):
            for key, val in value.items():
                item = QTreeWidgetItem([str(key)])
                parent.addChild(item)
                self.add_items(item, val)
        elif isinstance(value, list):
            for i, val in enumerate(value):
                item = QTreeWidgetItem([f"[{i}]"])
                parent.addChild(item)
                self.add_items(item, val)
        else:
            parent.setText(1, str(value))

class ControlPanel(QWidget):
    """Debug control panel"""
    
    def __init__(self, event_manager: EventManager, parent=None):
        """Initialize panel
        
        Args:
            event_manager: Event manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.event_manager = event_manager
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self._on_pause)
        btn_layout.addWidget(self.pause_btn)
        
        self.step_btn = QPushButton("单步")
        self.step_btn.clicked.connect(self._on_step)
        self.step_btn.setEnabled(False)
        btn_layout.addWidget(self.step_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
        
        # Update interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("更新间隔:"))
        
        self.interval = QSpinBox()
        self.interval.setRange(100, 5000)
        self.interval.setSingleStep(100)
        self.interval.setValue(1000)
        self.interval.setSuffix(" ms")
        self.interval.valueChanged.connect(self._on_interval_changed)
        interval_layout.addWidget(self.interval)
        
        layout.addLayout(interval_layout)
        
        # Add stretch
        layout.addStretch()
        
    async def _on_pause(self, checked: bool):
        """Handle pause button click
        
        Args:
            checked: Button checked state
        """
        self.step_btn.setEnabled(checked)
        await self.event_manager.emit(Event(
            EventType.GUI_ACTION,
            {
                'action': 'pause',
                'paused': checked
            }
        ))
        
    async def _on_step(self):
        """Handle step button click"""
        await self.event_manager.emit(Event(
            EventType.GUI_ACTION,
            {
                'action': 'step'
            }
        ))
        
    async def _on_reset(self):
        """Handle reset button click"""
        await self.event_manager.emit(Event(
            EventType.GUI_ACTION,
            {
                'action': 'reset'
            }
        ))
        
    async def _on_interval_changed(self, value: int):
        """Handle interval change
        
        Args:
            value: New interval value
        """
        await self.event_manager.emit(Event(
            EventType.GUI_ACTION,
            {
                'action': 'set_interval',
                'interval': value
            }
        ))

class LogViewer(QWidget):
    """Log viewing widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
    def append_log(self, message: str):
        """Append log message
        
        Args:
            message: Message to append
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def clear_log(self):
        """Clear log messages"""
        self.log_text.clear()

class AdvancedDebugInterface(QMainWindow):
    """Advanced debug interface main window"""
    
    def __init__(self, event_manager: EventManager, parent=None):
        """Initialize interface
        
        Args:
            event_manager: Event manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.event_manager = event_manager
        
        # Create components
        self.visualizer = GameStateVisualizer()
        self.inspector = GameStateInspector()
        self.control_panel = ControlPanel(event_manager)
        self.log_viewer = LogViewer()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update)
        self.update_timer.start(1000)  # Default 1s interval
        
        # Setup UI
        self._setup_ui()
        
        # Subscribe to events
        self._subscribe_events()
        
    def _setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("高级调试界面")
        self.setMinimumSize(1280, 720)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Control
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.control_panel)
        splitter.addWidget(left_widget)
        
        # Center panel - Visualization and Inspection
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        center_splitter = QSplitter(Qt.Vertical)
        center_layout.addWidget(center_splitter)
        
        center_splitter.addWidget(self.visualizer)
        center_splitter.addWidget(self.inspector)
        
        splitter.addWidget(center_widget)
        
        # Right panel - Log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.log_viewer)
        splitter.addWidget(right_widget)
        
        # Set stretch factors
        splitter.setStretchFactor(0, 1)  # Control panel
        splitter.setStretchFactor(1, 2)  # Visualization
        splitter.setStretchFactor(2, 1)  # Log viewer
        
    def _subscribe_events(self):
        """Subscribe to events"""
        self.event_manager.subscribe(
            EventType.GAME_STATE_CHANGED,
            self._on_game_state_changed
        )
        self.event_manager.subscribe(
            EventType.GAME_ERROR,
            self._on_game_error
        )
        
    async def _on_game_state_changed(self, event: Event):
        """Handle game state change event
        
        Args:
            event: Event data
        """
        game_state = event.data
        self.visualizer.update_visualization(game_state)
        self.inspector.update_state(game_state)
        
    async def _on_game_error(self, event: Event):
        """Handle game error event
        
        Args:
            event: Event data
        """
        error = event.data.get('error', 'Unknown error')
        self.log_viewer.append_log(f"错误: {error}")
        
    def _update(self):
        """Update interface periodically"""
        # This is now handled by events
        pass
        
    def closeEvent(self, event):
        """Handle window close event
        
        Args:
            event: Close event
        """
        self.update_timer.stop()
        event.accept()

advanced_debug_interface = AdvancedDebugInterface(EventManager())
advanced_debug_interface.show()