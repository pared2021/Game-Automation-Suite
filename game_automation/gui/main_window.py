from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTabWidget, QTextEdit, QListWidget, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor
from utils.config_manager import config_manager
from utils.performance_monitor import performance_monitor
from game_automation.game_engine import game_engine
from .config_editor import ConfigEditor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("游戏自动化控制面板")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.init_ui()
        self.apply_theme()

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

    # ... (other methods remain the same)

    def init_config_editor_tab(self):
        self.config_editor = ConfigEditor()
        self.tab_widget.addTab(self.config_editor, "配置编辑器")

    # ... (rest of the methods remain the same)

def run_gui():
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())