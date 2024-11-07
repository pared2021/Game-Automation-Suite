import cv2
import numpy as np
import subprocess
import json
import os
import asyncio
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError

class EmulatorManager:
    def __init__(self):
        self.logger = detailed_logger
        self.adb_device = None
        self.screen_width = 0
        self.screen_height = 0
        self.emulator_settings = {}
        
    @log_exception
    async def connect(self):
        """初始化连接到模拟器"""
        try:
            await self.load_emulator_settings()
            await self.detect_emulators()
            await self.select_emulator()
            await self.update_screen_size()
            self.logger.info(f"Successfully connected to emulator: {self.adb_device}")
        except Exception as e:
            raise GameAutomationError(f"Failed to connect to emulator: {str(e)}")

    @log_exception
    async def load_emulator_settings(self):
        """加载模拟器配置"""
        try:
            config_path = 'config/emulator_settings.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.emulator_settings = json.load(f)
            else:
                self.emulator_settings = {"emulators": []}
        except Exception as e:
            raise GameAutomationError(f"Error loading emulator settings: {str(e)}")

    @log_exception
    async def detect_emulators(self):
        """检测连接的模拟器"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            self.emulator_settings['emulators'] = []
            for line in lines:
                if '\t' in line:
                    serial, status = line.split('\t')
                    self.emulator_settings['emulators'].append({
                        "name": f"Emulator_{serial}",
                        "serial": serial,
                        "status": status
                    })
        except Exception as e:
            raise GameAutomationError(f"Error detecting emulators: {str(e)}")

    @log_exception
    async def select_emulator(self, adb_device=None):
        """选择要使用的模拟器"""
        try:
            if adb_device:
                emulator = next((e for e in self.emulator_settings['emulators'] 
                               if e['serial'] == adb_device), None)
                if emulator:
                    self.adb_device = adb_device
                else:
                    raise GameAutomationError(f"Specified ADB device {adb_device} not found")
            else:
                if self.emulator_settings['emulators']:
                    self.adb_device = self.emulator_settings['emulators'][0]['serial']
                else:
                    raise GameAutomationError("No ADB devices detected")
        except Exception as e:
            raise GameAutomationError(f"Error selecting emulator: {str(e)}")

    @log_exception
    async def update_screen_size(self):
        """更新屏幕尺寸信息"""
        try:
            output = subprocess.check_output(['adb', '-s', self.adb_device, 'shell', 'wm', 'size'])
            size = output.decode().strip().split(':')[-1].strip().split('x')
            self.screen_width, self.screen_height = map(int, size)
        except Exception as e:
            raise GameAutomationError(f"Error updating screen size: {str(e)}")

    @log_exception
    async def capture_screen(self):
        """捕获屏幕截图"""
        try:
            subprocess.run(['adb', '-s', self.adb_device, 'shell', 'screencap', '-p', '/sdcard/screen.png'])
            subprocess.run(['adb', '-s', self.adb_device, 'pull', '/sdcard/screen.png', 'screen.png'])
            return cv2.imread('screen.png')
        except Exception as e:
            raise GameAutomationError(f"Error capturing screen: {str(e)}")

    @log_exception
    async def capture_audio(self):
        """捕获音频数据"""
        # TODO: 实现音频捕获功能
        return None

    @log_exception
    async def tap(self, x, y):
        """点击指定坐标"""
        try:
            subprocess.run(['adb', '-s', self.adb_device, 'shell', 'input', 'tap', str(x), str(y)])
        except Exception as e:
            raise GameAutomationError(f"Error performing tap action: {str(e)}")

    @log_exception
    async def swipe(self, start_x, start_y, end_x, end_y, duration=100):
        """滑动操作"""
        try:
            subprocess.run(['adb', '-s', self.adb_device, 'shell', 'input', 'swipe', 
                          str(start_x), str(start_y), str(end_x), str(end_y), str(duration)])
        except Exception as e:
            raise GameAutomationError(f"Error performing swipe action: {str(e)}")

    @log_exception
    async def input_text(self, text):
        """输入文本"""
        try:
            subprocess.run(['adb', '-s', self.adb_device, 'shell', 'input', 'text', text])
        except Exception as e:
            raise GameAutomationError(f"Error inputting text: {str(e)}")

    @log_exception
    async def get_emulator_info(self):
        """获取当前模拟器信息"""
        return next((e for e in self.emulator_settings['emulators'] 
                    if e['serial'] == self.adb_device), None)

    @log_exception
    async def start_emulator(self):
        """启动模拟器"""
        self.logger.info("Starting emulator...")
        # TODO: 实现启动模拟器的具体逻辑

emulator_manager = EmulatorManager()
