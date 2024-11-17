import time
from typing import Tuple, Optional, Dict
import uiautomator2 as u2
from utils.error_handler import log_exception, DeviceConnectionError
from utils.logger import detailed_logger

class UIAutomator:
    """UI自动化控制器，提供设备交互功能"""

    def __init__(self, device_id: str):
        """初始化UI自动化控制器
        
        Args:
            device_id: 设备ID
            
        Raises:
            DeviceConnectionError: 设备连接失败时抛出
        """
        try:
            self.device = u2.connect(device_id)
            self.device.implicitly_wait(10.0)  # 设置默认等待时间
            self._screen_size = self.device.window_size()
            detailed_logger.info(f"UI自动化控制器初始化成功，屏幕分辨率: {self._screen_size}")
        except Exception as e:
            raise DeviceConnectionError(f"UI自动化控制器初始化失败: {str(e)}")

    @log_exception
    def click(self, x: int, y: int, timeout: float = 10.0, retry: int = 3) -> bool:
        """在指定坐标点击
        
        Args:
            x: X坐标
            y: Y坐标
            timeout: 操作超时时间（秒）
            retry: 重试次数
            
        Returns:
            bool: 点击是否成功
        """
        if not self._validate_coordinates(x, y):
            detailed_logger.warning(f"点击坐标 ({x}, {y}) 超出屏幕范围")
            return False

        start_time = time.time()
        for attempt in range(retry):
            try:
                if time.time() - start_time > timeout:
                    detailed_logger.warning(f"点击操作超时 ({x}, {y})")
                    return False
                
                self.device.click(x, y)
                detailed_logger.info(f"成功点击坐标 ({x}, {y})")
                return True
                
            except Exception as e:
                if attempt < retry - 1:
                    detailed_logger.warning(f"点击失败，正在重试 ({attempt + 1}/{retry}): {str(e)}")
                    time.sleep(1)
                else:
                    detailed_logger.error(f"点击操作失败 ({x}, {y}): {str(e)}")
                    return False
        
        return False

    @log_exception
    def swipe(self, 
              from_x: int, 
              from_y: int, 
              to_x: int, 
              to_y: int, 
              duration: float = 0.1,
              timeout: float = 10.0) -> bool:
        """执行滑动操作
        
        Args:
            from_x: 起始X坐标
            from_y: 起始Y坐标
            to_x: 结束X坐标
            to_y: 结束Y坐标
            duration: 滑动持续时间（秒）
            timeout: 操作超时时间（秒）
            
        Returns:
            bool: 滑动是否成功
        """
        if not all(self._validate_coordinates(x, y) for x, y in [(from_x, from_y), (to_x, to_y)]):
            detailed_logger.warning(f"滑动坐标超出屏幕范围: ({from_x}, {from_y}) -> ({to_x}, {to_y})")
            return False

        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.device.swipe(from_x, from_y, to_x, to_y, duration)
                    detailed_logger.info(f"成功执行滑动: ({from_x}, {from_y}) -> ({to_x}, {to_y})")
                    return True
                except Exception as e:
                    if time.time() - start_time + 1 < timeout:  # 留出1秒重试时间
                        detailed_logger.warning(f"滑动失败，正在重试: {str(e)}")
                        time.sleep(1)
                    else:
                        raise
            
            detailed_logger.warning("滑动操作超时")
            return False
            
        except Exception as e:
            detailed_logger.error(f"滑动操作失败: {str(e)}")
            return False

    @log_exception
    def wait_and_click(self, x: int, y: int, timeout: float = 10.0) -> bool:
        """等待并点击指定坐标
        
        Args:
            x: X坐标
            y: Y坐标
            timeout: 等待超时时间（秒）
            
        Returns:
            bool: 操作是否成功
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.click(x, y, timeout=1):
                return True
            time.sleep(0.5)
        
        detailed_logger.warning(f"等待点击超时 ({x}, {y})")
        return False

    @log_exception
    def find_element_by_text(self, text: str, timeout: float = 10.0) -> Optional[Dict]:
        """通过文本查找元素
        
        Args:
            text: 要查找的文本
            timeout: 查找超时时间（秒）
            
        Returns:
            Optional[Dict]: 元素信息，包含bounds等属性；未找到返回None
        """
        try:
            element = self.device(text=text).wait(timeout=timeout)
            if element.exists:
                info = element.info
                detailed_logger.info(f"找到文本元素: {text}")
                return info
            
            detailed_logger.warning(f"未找到文本元素: {text}")
            return None
            
        except Exception as e:
            detailed_logger.error(f"查找文本元素失败: {str(e)}")
            return None

    @log_exception
    def click_text(self, text: str, timeout: float = 10.0) -> bool:
        """点击包含指定文本的元素
        
        Args:
            text: 要点击的文本
            timeout: 操作超时时间（秒）
            
        Returns:
            bool: 是否成功点击
        """
        try:
            element = self.device(text=text).wait(timeout=timeout)
            if element.exists:
                element.click()
                detailed_logger.info(f"成功点击文本元素: {text}")
                return True
            
            detailed_logger.warning(f"未找到要点击的文本元素: {text}")
            return False
            
        except Exception as e:
            detailed_logger.error(f"点击文本元素失败: {str(e)}")
            return False

    @log_exception
    def screenshot(self, filename: str) -> bool:
        """截取屏幕截图
        
        Args:
            filename: 保存的文件名
            
        Returns:
            bool: 截图是否成功
        """
        try:
            self.device.screenshot(filename)
            detailed_logger.info(f"成功保存截图: {filename}")
            return True
        except Exception as e:
            detailed_logger.error(f"截图失败: {str(e)}")
            return False

    @log_exception
    def get_device_info(self) -> Dict:
        """获取设备信息
        
        Returns:
            Dict: 设备信息字典
        """
        return self.device.info

    def _validate_coordinates(self, x: int, y: int) -> bool:
        """验证坐标是否在屏幕范围内
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            bool: 坐标是否有效
        """
        width, height = self._screen_size
        return 0 <= x <= width and 0 <= y <= height

    @property
    def screen_size(self) -> Tuple[int, int]:
        """获取屏幕分辨率
        
        Returns:
            Tuple[int, int]: (宽度, 高度)
        """
        return self._screen_size
