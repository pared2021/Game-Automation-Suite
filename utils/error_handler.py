import logging
import os
import functools
import asyncio
from typing import Callable, Any
from .config_manager import config_manager

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

def log_exception(func: Callable) -> Callable:
    """支持同步和异步函数的错误处理装饰器"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
