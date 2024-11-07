import logging
import os
from config_manager import config_manager  # 更新导入路径

def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception occurred in {func.__name__}: {e}")
            raise
    return wrapper
