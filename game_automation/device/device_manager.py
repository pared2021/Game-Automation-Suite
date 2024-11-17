import os
import asyncio
import subprocess
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime

from utils.error_handler import log_exception, DeviceConnectionError
from utils.logger import detailed_logger
from .ui_automator import UIAutomator

class DeviceManager:
    """设备管理器，负责设备连接、状态监控和重连"""
    
    def __init__(self):
        self.device_id: Optional[str] = None
        self.ui_automator: Optional[UIAutomator] = None
        self.connected: bool = False
        self.monitor_task = None
        self._adb_path = self._get_adb_path()
        
        # 并发控制
        self._operation_lock = asyncio.Lock()
        self._resource_lock = asyncio.Lock()
        self._operation_queue = deque()
        self._active_operations: Dict[str, datetime] = {}
        self._max_concurrent_operations = 3
        self._operation_timeout = 30  # 操作超时时间(秒)
        
        # 资源管理
        self._resources: Dict[str, any] = {}
        self._cleanup_scheduled = False

    @log_exception
    def _get_adb_path(self) -> str:
        """获取ADB路径
        
        Returns:
            str: ADB可执行文件的路径
        
        Raises:
            DeviceConnectionError: 如果找不到ADB
        """
        # 首先检查环境变量中的ADB
        adb_path = os.environ.get('ANDROID_HOME')
        if adb_path:
            adb_path = os.path.join(adb_path, 'platform-tools', 'adb')
            if os.path.exists(adb_path):
                return adb_path

        # 检查常见的ADB安装位置
        common_paths = [
            r'C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe',
            r'C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe'
        ]
        
        for path in common_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path

        raise DeviceConnectionError("找不到ADB可执行文件")

    async def _acquire_operation_slot(self) -> bool:
        """获取操作槽位
        
        Returns:
            bool: 是否成功获取操作槽位
        """
        async with self._operation_lock:
            # 清理超时操作
            current_time = datetime.now()
            timed_out_ops = [
                op_id for op_id, start_time in self._active_operations.items()
                if (current_time - start_time).total_seconds() > self._operation_timeout
            ]
            for op_id in timed_out_ops:
                detailed_logger.warning(f"操作超时并被清理: {op_id}")
                del self._active_operations[op_id]

            # 检查是否有可用槽位
            if len(self._active_operations) >= self._max_concurrent_operations:
                return False

            # 分配新操作槽位
            op_id = f"op_{len(self._active_operations) + 1}"
            self._active_operations[op_id] = current_time
            return True

    async def _release_operation_slot(self, op_id: str) -> None:
        """释放操作槽位
        
        Args:
            op_id: 操作ID
        """
        async with self._operation_lock:
            if op_id in self._active_operations:
                del self._active_operations[op_id]

    async def _cleanup_resources(self) -> None:
        """清理设备资源"""
        async with self._resource_lock:
            for resource in self._resources.values():
                try:
                    if hasattr(resource, 'close'):
                        await resource.close()
                    elif hasattr(resource, '__del__'):
                        resource.__del__()
                except Exception as e:
                    detailed_logger.error(f"资源清理失败: {str(e)}")
            self._resources.clear()

    @log_exception
    async def connect(self, device_id: Optional[str] = None) -> None:
        """连接到设备
        
        Args:
            device_id: 可选的设备ID。如果未提供，将连接到第一个可用设备
        
        Raises:
            DeviceConnectionError: 连接失败时抛出
        """
        try:
            if device_id is None:
                # 获取已连接设备列表
                result = subprocess.run(
                    [self._adb_path, 'devices'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # 解析设备列表
                lines = result.stdout.strip().split('\n')[1:]  # 跳过第一行 "List of devices attached"
                devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
                
                if not devices:
                    raise DeviceConnectionError("未找到已连接的设备")
                
                device_id = devices[0]
            
            # 清理现有资源
            if self.connected:
                await self._cleanup_resources()
            
            # 尝试连接设备
            self.ui_automator = UIAutomator(device_id)
            self.device_id = device_id
            self.connected = True
            
            detailed_logger.info(f"成功连接到设备: {device_id}")
            
            # 启动设备监控
            if self.monitor_task is None:
                self.monitor_task = asyncio.create_task(self._monitor_device())
        
        except Exception as e:
            self.connected = False
            raise DeviceConnectionError(f"连接设备失败: {str(e)}")

    @log_exception
    async def disconnect(self) -> None:
        """断开设备连接"""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
        
        await self._cleanup_resources()
        
        self.connected = False
        self.device_id = None
        self.ui_automator = None
        detailed_logger.info("设备已断开连接")

    @log_exception
    async def check_connection(self) -> bool:
        """检查设备连接状态
        
        Returns:
            bool: 设备是否正常连接
        """
        if not self.device_id or not self.ui_automator:
            return False
            
        try:
            # 尝试获取设备信息来验证连接
            device_info = self.ui_automator.get_device_info()
            return bool(device_info)
        except Exception as e:
            detailed_logger.warning(f"设备连接检查失败: {str(e)}")
            return False

    async def _monitor_device(self) -> None:
        """监控设备状态并处理断开重连"""
        while True:
            try:
                if not await self.check_connection():
                    detailed_logger.warning("检测到设备断开连接")
                    self.connected = False
                    
                    # 清理资源
                    await self._cleanup_resources()
                    
                    # 尝试重新连接
                    retry_count = 0
                    while retry_count < 3 and not self.connected:
                        try:
                            detailed_logger.info(f"尝试重新连接设备 (尝试 {retry_count + 1}/3)")
                            await self.connect(self.device_id)
                            break
                        except DeviceConnectionError as e:
                            detailed_logger.error(f"重连失败: {str(e)}")
                            retry_count += 1
                            await asyncio.sleep(2)  # 等待2秒后重试
                    
                    if not self.connected:
                        detailed_logger.error("设备重连失败，停止监控")
                        break
                
                # 检查并清理超时操作
                await self._check_operation_timeouts()
                
                await asyncio.sleep(5)  # 每5秒检查一次设备状态
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                detailed_logger.error(f"设备监控异常: {str(e)}")
                await asyncio.sleep(5)  # 发生异常时等待5秒后继续监控

    async def _check_operation_timeouts(self) -> None:
        """检查并清理超时操作"""
        current_time = datetime.now()
        async with self._operation_lock:
            timed_out_ops = [
                op_id for op_id, start_time in self._active_operations.items()
                if (current_time - start_time).total_seconds() > self._operation_timeout
            ]
            for op_id in timed_out_ops:
                detailed_logger.warning(f"操作超时并被清理: {op_id}")
                del self._active_operations[op_id]

    async def queue_operation(self, operation_func, *args, **kwargs) -> any:
        """将操作加入队列并等待执行
        
        Args:
            operation_func: 要执行的操作函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            any: 操作结果
        """
        # 创建操作任务
        async def execute_operation():
            while True:
                if await self._acquire_operation_slot():
                    op_id = f"op_{len(self._active_operations)}"
                    try:
                        result = await operation_func(*args, **kwargs)
                        return result
                    finally:
                        await self._release_operation_slot(op_id)
                await asyncio.sleep(0.1)  # 短暂等待后重试
        
        # 将操作加入队列
        task = asyncio.create_task(execute_operation())
        self._operation_queue.append(task)
        
        try:
            return await task
        finally:
            self._operation_queue.remove(task)

    async def register_resource(self, resource_id: str, resource: any) -> None:
        """注册需要管理的资源
        
        Args:
            resource_id: 资源标识符
            resource: 资源对象
        """
        async with self._resource_lock:
            self._resources[resource_id] = resource

    async def unregister_resource(self, resource_id: str) -> None:
        """注销资源
        
        Args:
            resource_id: 资源标识符
        """
        async with self._resource_lock:
            if resource_id in self._resources:
                del self._resources[resource_id]

    @property
    def is_connected(self) -> bool:
        """获取设备连接状态
        
        Returns:
            bool: 设备是否已连接
        """
        return self.connected

    def get_ui_automator(self) -> Optional[UIAutomator]:
        """获取UI自动化控制器
        
        Returns:
            Optional[UIAutomator]: UI自动化控制器实例，如果未连接则返回None
        """
        return self.ui_automator if self.connected else None

    @property
    def active_operations_count(self) -> int:
        """获取当前活动操作数量
        
        Returns:
            int: 活动操作数量
        """
        return len(self._active_operations)

    @property
    def queued_operations_count(self) -> int:
        """获取队列中的操作数量
        
        Returns:
            int: 队列中的操作数量
        """
        return len(self._operation_queue)
