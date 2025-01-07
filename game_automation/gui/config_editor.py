from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from utils.config_manager import config_manager

class ConfigEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # 创建配置树
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Configuration")
        layout.addWidget(self.tree)

        # 创建编辑区域
        edit_layout = QVBoxLayout()
        self.edit_key = QLineEdit()
        self.edit_value = QLineEdit()
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_config)
        edit_layout.addWidget(self.edit_key)
        edit_layout.addWidget(self.edit_value)
        edit_layout.addWidget(self.btn_save)
        layout.addLayout(edit_layout)

        self.setLayout(layout)
        self.setWindowTitle("Configuration Editor")
        self.load_config()

    def load_config(self):
        self.tree.clear()
        self.add_config_items(config_manager.config, self.tree.invisibleRootItem())

    def add_config_items(self, config, parent):
        for key, value in config.items():
            item = QTreeWidgetItem(parent)
            item.setText(0, key)
            if isinstance(value, dict):
                self.add_config_items(value, item)
            else:
                item.setText(1, str(value))

    def save_config(self):
        key = self.edit_key.text()
        value = self.edit_value.text()
        try:
            config_manager.set(key, eval(value))
            config_manager.save_config()
            self.load_config()
            QMessageBox.information(self, "Success", "Configuration saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save configuration: {str(e)}")

# 不在模块级别创建实例
