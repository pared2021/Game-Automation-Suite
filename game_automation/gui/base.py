"""
Base GUI components and utilities
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QTabWidget,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
    QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QIcon, QAction

class BaseWindow(QMainWindow):
    """Base window class with common functionality"""
    
    def __init__(self, title="Game Automation", size=(1000, 700)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(*size)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup basic UI components"""
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def show_message(self, message, title="提示", level="info"):
        """Show message dialog"""
        if level == "info":
            QMessageBox.information(self, title, message)
        elif level == "warning":
            QMessageBox.warning(self, title, message)
        elif level == "error":
            QMessageBox.critical(self, title, message)
            
    def show_status(self, message, timeout=5000):
        """Show status bar message"""
        self.status_bar.showMessage(message, timeout)


class WorkerThread(QThread):
    """Base worker thread for background tasks"""
    
    finished = Signal()
    error = Signal(str)
    progress = Signal(int)
    log = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._is_running = True
        
    def stop(self):
        """Stop the worker thread"""
        self._is_running = False
        self.wait()
