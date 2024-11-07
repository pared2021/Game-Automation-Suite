import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from game_automation.game_engine import GameEngine
from utils.logger import detailed_logger
from utils.data_handler import DataHandler

class GameAutomationGUI:
    def __init__(self, master):
        self.master = master
        master.title("游戏自动化控制面板")
        master.geometry("800x600")

        self.logger = detailed_logger
        self.game_engine = None
        self.data_handler = DataHandler()
        
        self.create_menu()
        self.create_widgets()
        self.init_game_engine()

    def create_menu(self):
        # 创建菜单栏
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导入配置", command=self.import_config)
        file_menu.add_command(label="导出日志", command=self.export_log)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def create_widgets(self):
        # 创建主界面控件
        notebook = ttk.Notebook(self.master)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # 模拟器控制页面
        emulator_frame = ttk.Frame(notebook)
        notebook.add(emulator_frame, text="模拟器控制")

        ttk.Label(emulator_frame, text="选择模拟器:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.emulator_combobox = ttk.Combobox(emulator_frame, state="readonly")
        self.emulator_combobox.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        self.emulator_combobox.bind("<<ComboboxSelected>>", self.on_emulator_selected)

        ttk.Button(emulator_frame, text="刷新模拟器列表", command=self.refresh_emulators).grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(emulator_frame, text="启动模拟器", command=self.start_emulator).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(emulator_frame, text="关闭模拟器", command=self.stop_emulator).grid(row=1, column=1, padx=5, pady=5)

        # 自动化控制页面
        automation_frame = ttk.Frame(notebook)
        notebook.add(automation_frame, text="自动化控制")

        self.start_button = ttk.Button(automation_frame, text="开始自动化", command=self.start_automation)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(automation_frame, text="停止自动化", command=self.stop_automation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # 日志显示区域
        log_frame = ttk.Frame(self.master)
        log_frame.pack(expand=True, fill="both", padx=10, pady=10)

        ttk.Label(log_frame, text="日志:").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=10)
        self.log_area.pack(expand=True, fill="both")

        # 状态栏
        self.status_label = ttk.Label(self.master, text="状态: 未初始化", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def init_game_engine(self):
        # 初始化游戏引擎
        try:
            self.game_engine = GameEngine('config/strategies.json')
            self.refresh_emulators()
            self.log("游戏引擎初始化成功")
        except Exception as e:
            self.log(f"初始化游戏引擎失败: {str(e)}")
            messagebox.showerror("错误", f"初始化游戏引擎失败: {str(e)}")

    def refresh_emulators(self):
        # 刷新模拟器列表
        if self.game_engine:
            self.game_engine.detect_emulators()
            emulators = [e['name'] for e in self.game_engine.emulator_settings['emulators']]
            self.emulator_combobox['values'] = emulators
            if emulators:
                self.emulator_combobox.set(emulators[0])
                self.on_emulator_selected(None)
            self.log("模拟器列表已刷新")
        else:
            self.log("游戏引擎未初始化，无法刷新模拟器列表")

    def on_emulator_selected(self, event):
        # 处理模拟器选择事件
        selected_name = self.emulator_combobox.get()
        selected_emulator = next((e for e in self.game_engine.emulator_settings['emulators'] if e['name'] == selected_name), None)
        if selected_emulator:
            self.game_engine.select_emulator(selected_emulator['serial'])
            self.game_engine.update_screen_size()
            self.log(f"已选择模拟器: {selected_name}")
            self.status_label.config(text=f"状态: 已连接到 {selected_name}")
        else:
            self.log("未找到选择的模拟器")

    def start_emulator(self):
        # 启动模拟器
        if self.game_engine and self.game_engine.adb_device:
            self.game_engine.start_emulator()
            self.log("正在启动模拟器...")
        else:
            messagebox.showerror("错误", "请先选择一个模拟器")

    def stop_emulator(self):
        # 停止模拟器
        if self.game_engine and self.game_engine.adb_device:
            # 这里需要实现停止模拟器的逻辑
            self.log("正在关闭模拟器...")
        else:
            messagebox.showerror("错误", "请先选择一个模拟器")

    def start_automation(self):
        # 开始自动化
        if not self.game_engine or not self.game_engine.adb_device:
            messagebox.showerror("错误", "请先选择一个模拟器")
            return
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.run_automation, daemon=True).start()

    def stop_automation(self):
        # 停止自动化
        if self.game_engine:
            self.game_engine.stop_automation = True
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def run_automation(self):
        # 运行自动化
        try:
            self.game_engine.run_game_loop()
        except Exception as e:
            self.log(f"自动化过程出错: {str(e)}")
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def log(self, message):
        # 记录日志
        self.log_area.insert(tk.END, message + '\n')
        self.log_area.see(tk.END)
        self.logger.info(message)

    def import_config(self):
        # 导入配置
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                # 这里需要实现导入配置的逻辑
                self.log(f"已导入配置: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导入配置失败: {str(e)}")

    def export_log(self):
        # 导出日志
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.log_area.get("1.0", tk.END))
                self.log(f"日志已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出日志失败: {str(e)}")

    def show_about(self):
        # 显示关于信息
        messagebox.showinfo("关于", "游戏自动化控制系统\n版本 1.0\n作者: OpenAI")

if __name__ == "__main__":
    root = tk.Tk()
    gui = GameAutomationGUI(root)
    root.mainloop()
