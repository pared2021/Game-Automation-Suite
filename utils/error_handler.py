import logging
import os
from .config_manager import config_manager  # 更新导入路径

class GameAutomationError(Exception):
    """游戏自动化过程中的基础异常类"""
    def __init__(self, message="游戏自动化过程中发生错误"):
        self.message = message
        super().__init__(self.message)

class DeviceConnectionError(GameAutomationError):
    """设备连接相关的错误"""
    pass

class ImageRecognitionError(GameAutomationError):
    """图像识别相关的错误"""
    pass

class TaskExecutionError(GameAutomationError):
    """任务执行相关的错误"""
    pass

def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception occurred in {func.__name__}: {e}")
            raise
    return wrapper
