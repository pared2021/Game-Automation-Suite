import cv2
import numpy as np
import subprocess
import json
import os
import time
import re
import asyncio
import traceback
from .controllers.touch_controller import TouchController
from .controllers.auto_battle_strategy import AutoBattleStrategy
from .ocr_prediction.ocr_utils import OCRUtils
from .image_recognition import ImageRecognition
from utils.logger import setup_logger, log_exception
from utils.data_handler import DataHandler

class GameEngine:
    def __init__(self, strategy_file, adb_device=None):
        self.logger = setup_logger()
        try:
            self.adb_device = adb_device
            self.touch_controller = None
            self.battle_strategy = AutoBattleStrategy(strategy_file)
            self.ocr_utils = OCRUtils()
            self.image_recognition = ImageRecognition()
            self.data_handler = DataHandler()
            self.stop_automation = False
            self.screen_width = 0
            self.screen_height = 0
            self.load_emulator_settings()
            self.detect_emulators()
            self.select_emulator(adb_device)
            self.update_screen_size()
        except Exception as e:
            log_exception(self.logger, e)
            raise

    def load_emulator_settings(self):
        try:
            with open('config/emulator_settings.json', 'r') as f:
                self.emulator_settings = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading emulator settings: {str(e)}")
            raise

    def detect_emulators(self):
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
            self.logger.error(f"Error detecting emulators: {str(e)}")
            raise

    def select_emulator(self, adb_device):
        try:
            if adb_device:
                emulator = next((e for e in self.emulator_settings['emulators'] if e['serial'] == adb_device), None)
                if emulator:
                    self.adb_device = adb_device
                else:
                    raise ValueError(f"指定的 ADB 设备 {adb_device} 未找到")
            else:
                if self.emulator_settings['emulators']:
                    self.adb_device = self.emulator_settings['emulators'][0]['serial']
                else:
                    raise ValueError("未检测到任何 ADB 设备")
            
            self.touch_controller = TouchController(self.adb_device)
        except Exception as e:
            self.logger.error(f"Error selecting emulator: {str(e)}")
            raise

    def update_screen_size(self):
        try:
            output = subprocess.check_output(['adb', '-s', self.adb_device, 'shell', 'wm', 'size'])
            size = output.decode().strip().split(':')[-1].strip().split('x')
            self.screen_width, self.screen_height = map(int, size)
        except Exception as e:
            self.logger.error(f"Error updating screen size: {str(e)}")
            raise

    async def run_game_loop(self):
        session_start = time.time()
        session_id = self.data_handler.save_game_session(session_start, None, None)
        
        try:
            while not self.stop_automation:
                try:
                    screen = self.capture_screen()
                    text = self.ocr_utils.recognize_text(screen)
                    
                    if "战斗" in text:
                        await self.battle_strategy.execute_strategy("battle", self)
                        self.data_handler.save_game_event(session_id, "battle", {"type": "normal"})
                    elif "Boss" in text:
                        await self.battle_strategy.execute_strategy("boss_battle", self)
                        self.data_handler.save_game_event(session_id, "battle", {"type": "boss"})
                    else:
                        await self.battle_strategy.execute_strategy("default", self)
                    
                    await asyncio.sleep(1)
                except Exception as e:
                    log_exception(self.logger, e)
                    self.logger.warning("Continuing game loop despite error")
        finally:
            session_end = time.time()
            session_duration = int(session_end - session_start)
            self.data_handler.save_game_session(session_start, session_end, session_duration)

    def capture_screen(self):
        try:
            subprocess.run(['adb', '-s', self.adb_device, 'shell', 'screencap', '-p', '/sdcard/screen.png'])
            subprocess.run(['adb', '-s', self.adb_device, 'pull', '/sdcard/screen.png', 'screen.png'])
            return cv2.imread('screen.png')
        except Exception as e:
            self.logger.error(f"Error capturing screen: {str(e)}")
            raise

    async def find_game_object(self, template_path):
        try:
            screen = self.capture_screen()
            return self.image_recognition.template_matching(screen, template_path)
        except Exception as e:
            self.logger.error(f"Error finding game object: {str(e)}")
            raise

    async def wait_for_game_object(self, template_path, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.find_game_object(template_path):
                return True
            await asyncio.sleep(1)
        return False

    async def perform_action_sequence(self, actions):
        for action in actions:
            if action['type'] == 'tap':
                self.touch_controller.tap(action['x'], action['y'])
            elif action['type'] == 'wait':
                await asyncio.sleep(action['duration'])
            elif action['type'] == 'find_and_tap':
                objects = await self.find_game_object(action['template'])
                if objects:
                    x, y, w, h = objects[0]
                    self.touch_controller.tap(x + w//2, y + h//2)

    async def detect_game_state(self):
        # 实现游戏状态检测逻辑
        pass

    async def get_resource_value(self, resource):
        # 实现资源值获取逻辑
        pass

    def get_emulator_info(self):
        return next((e for e in self.emulator_settings['emulators'] if e['serial'] == self.adb_device), None)

    def start_emulator(self):
        # 实现启动模拟器的逻辑
        self.logger.info("Starting emulator...")
        # 这里需要添加实际启动模拟器的代码

if __name__ == "__main__":
    try:
        engine = GameEngine('config/strategies.json')
        print(f"检测到的模拟器: {engine.emulator_settings['emulators']}")
        print(f"选择的模拟器: {engine.adb_device}")
        print(f"屏幕尺寸: {engine.screen_width}x{engine.screen_height}")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())