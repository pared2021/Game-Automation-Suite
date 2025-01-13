"""
GUI styles and themes
"""

DARK_THEME = """
QMainWindow {
    background-color: #2b2b2b;
    color: #ffffff;
}

QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
}

QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    padding: 5px 10px;
}

QPushButton:hover {
    background-color: #4c4c4c;
}

QPushButton:pressed {
    background-color: #2c2c2c;
}

QPushButton:disabled {
    background-color: #2c2c2c;
    color: #808080;
}

QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    padding: 5px;
}

QComboBox:hover {
    background-color: #4c4c4c;
}

QComboBox::drop-down {
    border: none;
}

QComboBox::down-arrow {
    image: url(down_arrow.png);
    width: 12px;
    height: 12px;
}

QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    padding: 5px;
}

QLabel {
    color: #ffffff;
}

QGroupBox {
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 15px;
    color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}

QScrollArea {
    border: none;
}

QMenuBar {
    background-color: #2b2b2b;
    color: #ffffff;
}

QMenuBar::item {
    background-color: transparent;
    padding: 5px 10px;
}

QMenuBar::item:selected {
    background-color: #3c3c3c;
}

QMenu {
    background-color: #2b2b2b;
    border: 1px solid #555555;
    color: #ffffff;
}

QMenu::item {
    padding: 5px 20px;
}

QMenu::item:selected {
    background-color: #3c3c3c;
}

QStatusBar {
    background-color: #2b2b2b;
    color: #ffffff;
}

QTabWidget::pane {
    border: 1px solid #555555;
}

QTabBar::tab {
    background-color: #2b2b2b;
    border: 1px solid #555555;
    border-bottom: none;
    color: #ffffff;
    padding: 5px 10px;
}

QTabBar::tab:selected {
    background-color: #3c3c3c;
}

QTabBar::tab:hover {
    background-color: #4c4c4c;
}
"""

LIGHT_THEME = """
QMainWindow {
    background-color: #f0f0f0;
    color: #000000;
}

QWidget {
    background-color: #f0f0f0;
    color: #000000;
}

QPushButton {
    background-color: #e0e0e0;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    color: #000000;
    padding: 5px 10px;
}

QPushButton:hover {
    background-color: #d0d0d0;
}

QPushButton:pressed {
    background-color: #c0c0c0;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #808080;
}

QComboBox {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    color: #000000;
    padding: 5px;
}

QComboBox:hover {
    background-color: #f5f5f5;
}

QTextEdit {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    color: #000000;
    padding: 5px;
}

QLabel {
    color: #000000;
}

QGroupBox {
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 15px;
    color: #000000;
}

QMenuBar {
    background-color: #f0f0f0;
    color: #000000;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    color: #000000;
}

QStatusBar {
    background-color: #f0f0f0;
    color: #000000;
}
"""

def apply_theme(app, theme="dark"):
    """Apply theme to the application
    
    Args:
        app: QApplication instance
        theme: Theme name ("dark" or "light")
    """
    if theme.lower() == "dark":
        app.setStyleSheet(DARK_THEME)
    else:
        app.setStyleSheet(LIGHT_THEME)
