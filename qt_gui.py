import sys
import os
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QTabWidget,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
    QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QAction
from game_automation.game_engine import GameEngine
from utils.logger import detailed_logger
from utils.data_handler import DataHandler

class GameAutomationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("GameAutomationWindow")
        self.setWindowTitle("游戏自动化控制面板")
        self.resize(1000, 700)
        
        # 初始化变量
        self.logger = detailed_logger
        self.game_engine = None
        self.data_handler = DataHandler()
        
        # 加载样式表
        self.load_stylesheets()
        
        # 设置主窗口布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建界面元素
        self.create_navigation()
        self.create_main_content()
        self.create_menu_bar()
        self.create_status_bar()
        
        # 初始化游戏引擎
        self.init_game_engine()

    def load_stylesheets(self):
        style_dir = "frontend/src/assets/ui/qss/dark"
        style_files = [
            "app_window.qss",
            "game_button.qss",
            "game_card.qss",
            "navigation_interface.qss"
        ]
        
        combined_style = ""
        for file in style_files:
            try:
                with open(os.path.join(style_dir, file), 'r') as f:
                    combined_style += f.read() + "\n"
            except Exception as e:
                self.logger.error(f"加载样式表失败 {file}: {str(e)}")
        
        self.setStyleSheet(combined_style)

    def create_navigation(self):
        # 创建导航面板
        nav_frame = QFrame()
        nav_frame.setObjectName("navigationInterface")
        nav_layout = QVBoxLayout(nav_frame)
        
        # 模拟器选择
        emulator_group = QFrame()
        emulator_group.setObjectName("nav-group")
        emulator_layout = QVBoxLayout(emulator_group)
        
        group_title = QLabel("模拟器控制")
        group_title.setObjectName("nav-group-title")
        emulator_layout.addWidget(group_title)
        
        self.emulator_combo = QComboBox()
        self.emulator_combo.setObjectName("nav-item")
        emulator_layout.addWidget(self.emulator_combo)
        
        refresh_btn = QPushButton("刷新模拟器列表")
        refresh_btn.setObjectName("secondary-button")
        refresh_btn.clicked.connect(self.refresh_emulators)
        emulator_layout.addWidget(refresh_btn)
        
        start_emu_btn = QPushButton("启动模拟器")
        start_emu_btn.setObjectName("primary-button")
        start_emu_btn.clicked.connect(self.start_emulator)
        emulator_layout.addWidget(start_emu_btn)
        
        stop_emu_btn = QPushButton("关闭模拟器")
        stop_emu_btn.setObjectName("danger-button")
        stop_emu_btn.clicked.connect(self.stop_emulator)
        emulator_layout.addWidget(stop_emu_btn)
        
        nav_layout.addWidget(emulator_group)
        
        # 自动化控制
        automation_group = QFrame()
        automation_group.setObjectName("nav-group")
        automation_layout = QVBoxLayout(automation_group)
        
        group_title = QLabel("自动化控制")
        group_title.setObjectName("nav-group-title")
        automation_layout.addWidget(group_title)
        
        self.start_button = QPushButton("开始自动化")
        self.start_button.setObjectName("primary-button")
        self.start_button.clicked.connect(self.start_automation)
        automation_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止自动化")
        self.stop_button.setObjectName("danger-button")
        self.stop_button.clicked.connect(self.stop_automation)
        self.stop_button.setEnabled(False)
        automation_layout.addWidget(self.stop_button)
        
        nav_layout.addWidget(automation_group)
        nav_layout.addStretch()
        
        self.main_layout.addWidget(nav_frame)

    def create_main_content(self):
        # 创建主内容区
        main_frame = QFrame()
        main_frame.setObjectName("mainContent")
        main_layout = QVBoxLayout(main_frame)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        tab_widget.setObjectName("game-card")
        
        # 状态监控选项卡
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        
        # 添加状态卡片
        status_card = QFrame()
        status_card.setObjectName("game-card")
        status_layout.addWidget(status_card)
        
        tab_widget.addTab(status_tab, "状态监控")
        
        # 日志选项卡
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("game-card")
        log_layout.addWidget(self.log_area)
        
        tab_widget.addTab(log_tab, "运行日志")
        
        main_layout.addWidget(tab_widget)
        self.main_layout.addWidget(main_frame)

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        import_action = QAction("导入配置", self)
        import_action.triggered.connect(self.import_config)
        file_menu.addAction(import_action)
        
        export_action = QAction("导出日志", self)
        export_action.triggered.connect(self.export_log)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        self.statusBar().showMessage("状态: 未初始化")

    def init_game_engine(self):
        try:
            self.game_engine = GameEngine('config/strategies.json')
            self.refresh_emulators()
            self.log("游戏引擎初始化成功")
        except Exception as e:
            self.log(f"初始化游戏引擎失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"初始化游戏引擎失败: {str(e)}")

    def refresh_emulators(self):
        if self.game_engine:
            self.game_engine.detect_emulators()
            emulators = [e['name'] for e in self.game_engine.emulator_settings['emulators']]
            self.emulator_combo.clear()
            self.emulator_combo.addItems(emulators)
            if emulators:
                self.emulator_combo.setCurrentIndex(0)
                self.on_emulator_selected(0)
            self.log("模拟器列表已刷新")
        else:
            self.log("游戏引擎未初始化，无法刷新模拟器列表")

    def on_emulator_selected(self, index):
        selected_name = self.emulator_combo.currentText()
        selected_emulator = next(
            (e for e in self.game_engine.emulator_settings['emulators'] if e['name'] == selected_name),
            None
        )
        if selected_emulator:
            self.game_engine.select_emulator(selected_emulator['serial'])
            self.game_engine.update_screen_size()
            self.log(f"已选择模拟器: {selected_name}")
            self.statusBar().showMessage(f"状态: 已连接到 {selected_name}")
        else:
            self.log("未找到选择的模拟器")

    def start_emulator(self):
        if self.game_engine and self.game_engine.adb_device:
            self.game_engine.start_emulator()
            self.log("正在启动模拟器...")
        else:
            QMessageBox.warning(self, "警告", "请先选择一个模拟器")

    def stop_emulator(self):
        if self.game_engine and self.game_engine.adb_device:
            self.log("正在关闭模拟器...")
        else:
            QMessageBox.warning(self, "警告", "请先选择一个模拟器")

    def start_automation(self):
        if not self.game_engine or not self.game_engine.adb_device:
            QMessageBox.warning(self, "警告", "请先选择一个模拟器")
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 创建新线程运行自动化
        self.automation_thread = AutomationThread(self.game_engine)
        self.automation_thread.log_signal.connect(self.log)
        self.automation_thread.finished.connect(self.on_automation_finished)
        self.automation_thread.start()

    def stop_automation(self):
        if self.game_engine:
            self.game_engine.stop_automation = True
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_automation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def log(self, message):
        self.log_area.append(message)
        self.logger.info(message)

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入配置",
            "",
            "JSON files (*.json)"
        )
        if file_path:
            try:
                self.log(f"已导入配置: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入配置失败: {str(e)}")

    def export_log(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出日志",
            "",
            "Text files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.log_area.toPlainText())
                self.log(f"日志已导出到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出日志失败: {str(e)}")

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "游戏自动化控制系统\n版本 1.0\n作者: OpenAI"
        )

class AutomationThread(QThread):
    log_signal = pyqtSignal(str)
    
    def __init__(self, game_engine):
        super().__init__()
        self.game_engine = game_engine
    
    def run(self):
        try:
            self.game_engine.run_game_loop()
        except Exception as e:
            self.log_signal.emit(f"自动化过程出错: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameAutomationWindow()
    window.show()
    sys.exit(app.exec())
