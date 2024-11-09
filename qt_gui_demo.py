import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QTabWidget,
    QMenuBar, QMenu, QStatusBar, QMessageBox,
    QFrame, QScrollArea
)
from PyQt6.QtCore import Qt

class GameAutomationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("GameAutomationWindow")
        self.setWindowTitle("游戏自动化控制面板")
        self.resize(1000, 700)
        
        # 加载样式表
        self.load_stylesheets()
        
        # 设置主窗口布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建界面元素
        self.create_navigation()
        self.create_main_content()
        self.create_menu_bar()
        self.create_status_bar()

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
                with open(os.path.join(style_dir, file), 'r', encoding='utf-8') as f:
                    combined_style += f.read() + "\n"
            except Exception as e:
                print(f"加载样式表失败 {file}: {str(e)}")
        
        self.setStyleSheet(combined_style)

    def create_navigation(self):
        # 创建导航面板
        nav_frame = QFrame()
        nav_frame.setObjectName("navigationInterface")
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        
        # 模拟器选择
        emulator_group = QFrame()
        emulator_group.setObjectName("nav-group")
        emulator_layout = QVBoxLayout(emulator_group)
        emulator_layout.setSpacing(8)
        emulator_layout.setContentsMargins(16, 16, 16, 16)
        
        group_title = QLabel("模拟器控制")
        group_title.setObjectName("nav-group-title")
        emulator_layout.addWidget(group_title)
        
        self.emulator_combo = QComboBox()
        self.emulator_combo.setObjectName("nav-item")
        self.emulator_combo.addItems(["模拟器1", "模拟器2", "模拟器3"])
        emulator_layout.addWidget(self.emulator_combo)
        
        refresh_btn = QPushButton("刷新模拟器列表")
        refresh_btn.setObjectName("secondary-button")
        emulator_layout.addWidget(refresh_btn)
        
        start_emu_btn = QPushButton("启动模拟器")
        start_emu_btn.setObjectName("primary-button")
        emulator_layout.addWidget(start_emu_btn)
        
        stop_emu_btn = QPushButton("关闭模拟器")
        stop_emu_btn.setObjectName("danger-button")
        emulator_layout.addWidget(stop_emu_btn)
        
        nav_layout.addWidget(emulator_group)
        
        # 自动化控制
        automation_group = QFrame()
        automation_group.setObjectName("nav-group")
        automation_layout = QVBoxLayout(automation_group)
        automation_layout.setSpacing(8)
        automation_layout.setContentsMargins(16, 16, 16, 16)
        
        group_title = QLabel("自动化控制")
        group_title.setObjectName("nav-group-title")
        automation_layout.addWidget(group_title)
        
        start_button = QPushButton("开始自动化")
        start_button.setObjectName("primary-button")
        automation_layout.addWidget(start_button)
        
        stop_button = QPushButton("停止自动化")
        stop_button.setObjectName("danger-button")
        automation_layout.addWidget(stop_button)
        
        nav_layout.addWidget(automation_group)
        nav_layout.addStretch()
        
        self.main_layout.addWidget(nav_frame)

    def create_main_content(self):
        # 创建主内容区
        main_frame = QFrame()
        main_frame.setObjectName("mainContent")
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        tab_widget.setObjectName("game-card")
        
        # 状态监控选项卡
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(16)
        
        # 添加状态卡片
        status_card = QFrame()
        status_card.setObjectName("game-card")
        card_layout = QVBoxLayout(status_card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(8)
        
        # 添加一些状态信息
        status_title = QLabel("运行状态")
        status_title.setObjectName("game-card-header")
        card_layout.addWidget(status_title)
        
        status_content = QLabel("当前状态: 正常运行中")
        status_content.setObjectName("game-card-content")
        card_layout.addWidget(status_content)
        
        # 添加进度条示例
        progress_bar = QFrame()
        progress_bar.setObjectName("status-bar")
        progress_bar.setFixedHeight(8)
        
        progress_value = QFrame(progress_bar)
        progress_value.setObjectName("status-bar-progress")
        progress_value.setFixedHeight(8)
        progress_value.setFixedWidth(int(progress_bar.width() * 0.7))  # 70% 进度
        
        card_layout.addWidget(progress_bar)
        
        status_layout.addWidget(status_card)
        status_layout.addStretch()
        
        tab_widget.addTab(status_tab, "状态监控")
        
        # 日志选项卡
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(16, 16, 16, 16)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("game-card")
        self.log_area.setText("系统启动完成...\n准备就绪...")
        log_layout.addWidget(self.log_area)
        
        tab_widget.addTab(log_tab, "运行日志")
        
        main_layout.addWidget(tab_widget)
        self.main_layout.addWidget(main_frame)

    def create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setObjectName("nav-group")
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("导入配置")
        file_menu.addAction("导出日志")
        file_menu.addSeparator()
        file_menu.addAction("退出")
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于")

    def create_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setObjectName("nav-group")
        status_bar.showMessage("状态: 就绪")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameAutomationWindow()
    window.show()
    print("程序启动成功")  # 添加调试信息
    sys.exit(app.exec())
