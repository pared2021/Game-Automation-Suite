from typing import Optional, Tuple
from pathlib import Path
import asyncio
import subprocess
import tempfile
from datetime import datetime
import json
import os

from PIL import Image
import numpy as np

class DeviceError(Exception):
    """设备错误"""
    pass

class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self._initialized = False
        self._connected = False
        self._device_id = None
        self._resolution = (1920, 1080)
        self._config = None
        self._screenshot_path = None
        
    async def initialize(self):
        """初始化设备管理器"""
        if not self._initialized:
            try:
                # 加载配置
                config_path = Path("config/config.json")
                if not config_path.exists():
                    raise DeviceError("配置文件不存在")
                    
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                    
                # 设置分辨率
                self._resolution = (
                    self._config["device"]["resolution"]["width"],
                    self._config["device"]["resolution"]["height"]
                )
                
                # 创建临时目录
                self._screenshot_path = Path(tempfile.gettempdir()) / "game_automation"
                self._screenshot_path.mkdir(parents=True, exist_ok=True)
                
                self._initialized = True
                
            except Exception as e:
                raise DeviceError(f"初始化失败: {str(e)}")
                
    async def cleanup(self):
        """清理资源"""
        if self._initialized:
            try:
                # 断开设备
                if self._connected:
                    await self.disconnect()
                    
                # 删除临时文件
                if self._screenshot_path and self._screenshot_path.exists():
                    for file in self._screenshot_path.glob("*.png"):
                        file.unlink()
                        
                self._initialized = False
                
            except Exception as e:
                raise DeviceError(f"清理失败: {str(e)}")
                
    async def connect(self):
        """连接设备"""
        if not self._initialized:
            raise DeviceError("未初始化")
            
        if self._connected:
            return
            
        try:
            # 获取设备列表
            process = await asyncio.create_subprocess_exec(
                "adb", "devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise DeviceError(f"获取设备列表失败: {stderr.decode()}")
                
            # 解析设备ID
            lines = stdout.decode().strip().split("\n")[1:]
            devices = [
                line.split("\t")[0]
                for line in lines
                if line.strip() and "\tdevice" in line
            ]
            
            if not devices:
                raise DeviceError("未找到设备")
                
            # 使用第一个设备
            self._device_id = devices[0]
            
            # 获取分辨率
            process = await asyncio.create_subprocess_exec(
                "adb", "-s", self._device_id, "shell", "wm", "size",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                size = stdout.decode().strip()
                if "Physical size:" in size:
                    width, height = map(
                        int,
                        size.split("Physical size:")[1].strip().split("x")
                    )
                    self._resolution = (width, height)
                    
            self._connected = True
            
        except Exception as e:
            raise DeviceError(f"连接设备失败: {str(e)}")
            
    async def disconnect(self):
        """断开设备"""
        if not self._initialized:
            raise DeviceError("未初始化")
            
        if not self._connected:
            return
            
        try:
            # 断开设备
            process = await asyncio.create_subprocess_exec(
                "adb", "disconnect", self._device_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            self._connected = False
            self._device_id = None
            
        except Exception as e:
            raise DeviceError(f"断开设备失败: {str(e)}")
            
    async def get_screenshot(self) -> Image.Image:
        """获取屏幕截图
        
        Returns:
            Image.Image: 截图对象
        """
        if not self._initialized:
            raise DeviceError("未初始化")
            
        if not self._connected:
            raise DeviceError("未连接设备")
            
        try:
            # 生成临时文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_path = self._screenshot_path / f"screenshot_{timestamp}.png"
            
            # 截图
            process = await asyncio.create_subprocess_exec(
                "adb", "-s", self._device_id, "shell", "screencap", "-p",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise DeviceError(f"截图失败: {stderr.decode()}")
                
            # 保存截图
            with open(temp_path, "wb") as f:
                f.write(stdout.replace(b"\r\n", b"\n"))
                
            # 读取图片
            image = Image.open(temp_path)
            
            # 调整大小
            if image.size != self._resolution:
                image = image.resize(self._resolution)
                
            # 删除临时文件
            temp_path.unlink()
            
            return image
            
        except Exception as e:
            raise DeviceError(f"获取截图失败: {str(e)}")
            
    async def tap(self, x: int, y: int):
        """点击屏幕
        
        Args:
            x: X坐标
            y: Y坐标
        """
        if not self._initialized:
            raise DeviceError("未初始化")
            
        if not self._connected:
            raise DeviceError("未连接设备")
            
        try:
            # 执行点击
            process = await asyncio.create_subprocess_exec(
                "adb", "-s", self._device_id, "shell", "input", "tap",
                str(x), str(y),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise DeviceError(f"点击失败: {stderr.decode()}")
                
            # 等待点击延迟
            await asyncio.sleep(self._config["device"]["tap_delay"])
            
        except Exception as e:
            raise DeviceError(f"点击失败: {str(e)}")
            
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = None
    ):
        """滑动屏幕
        
        Args:
            start_x: 起始X坐标
            start_y: 起始Y坐标
            end_x: 结束X坐标
            end_y: 结束Y坐标
            duration: 持续时间(秒)
        """
        if not self._initialized:
            raise DeviceError("未初始化")
            
        if not self._connected:
            raise DeviceError("未连接设备")
            
        try:
            # 设置持续时间
            if duration is None:
                duration = self._config["device"]["swipe_duration"]
                
            # 转换为毫秒
            duration_ms = int(duration * 1000)
            
            # 执行滑动
            process = await asyncio.create_subprocess_exec(
                "adb", "-s", self._device_id, "shell", "input", "swipe",
                str(start_x), str(start_y),
                str(end_x), str(end_y),
                str(duration_ms),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise DeviceError(f"滑动失败: {stderr.decode()}")
                
            # 等待滑动完成
            await asyncio.sleep(duration)
            
        except Exception as e:
            raise DeviceError(f"滑动失败: {str(e)}")
            
    async def get_resolution(self) -> Tuple[int, int]:
        """获取屏幕分辨率
        
        Returns:
            Tuple[int, int]: (宽度, 高度)
        """
        return self._resolution
        
    async def is_connected(self) -> bool:
        """检查设备是否已连接
        
        Returns:
            bool: 是否已连接
        """
        return self._connected
