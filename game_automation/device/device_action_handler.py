from typing import Dict, Optional, Any
import asyncio
from datetime import datetime

from game_automation.core.decision_maker import Action, Condition
from .device_manager import DeviceManager
from utils.logger import detailed_logger
from utils.error_handler import log_exception

class DeviceActionHandler:
    """处理设备相关的Action和Condition"""

    def __init__(self, device_manager: DeviceManager):
        """初始化设备动作处理器
        
        Args:
            device_manager: 设备管理器实例
        """
        self.device_manager = device_manager
        self._operation_results: Dict[str, Any] = {}
        self._cleanup_tasks = []

    async def _execute_with_cleanup(self, operation_func, *args, **kwargs) -> Any:
        """执行操作并确保清理
        
        Args:
            operation_func: 要执行的操作函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 操作结果
        """
        operation_id = f"op_{datetime.now().timestamp()}"
        try:
            # 将操作加入队列并等待执行
            result = await self.device_manager.queue_operation(operation_func, *args, **kwargs)
            self._operation_results[operation_id] = result
            return result
        except Exception as e:
            detailed_logger.error(f"操作执行失败: {str(e)}")
            self._operation_results[operation_id] = None
            raise
        finally:
            # 添加清理任务
            cleanup_task = asyncio.create_task(self._cleanup_operation(operation_id))
            self._cleanup_tasks.append(cleanup_task)

    async def _cleanup_operation(self, operation_id: str) -> None:
        """清理操作相关资源
        
        Args:
            operation_id: 操作ID
        """
        try:
            # 清理操作结果
            if operation_id in self._operation_results:
                del self._operation_results[operation_id]
            
            # 执行其他必要的清理工作
            await self.device_manager.unregister_resource(operation_id)
        except Exception as e:
            detailed_logger.error(f"操作清理失败: {str(e)}")
        finally:
            # 从清理任务列表中移除
            for task in self._cleanup_tasks:
                if task.done():
                    self._cleanup_tasks.remove(task)

    @log_exception
    async def handle_click(self, action: Action) -> bool:
        """处理点击动作
        
        Args:
            action: 包含点击参数的Action对象
                   必需参数: x, y
                   可选参数: timeout, retry, wait_after
        
        Returns:
            bool: 点击是否成功
        """
        params = action.parameters
        if 'x' not in params or 'y' not in params:
            detailed_logger.error("点击动作缺少必需参数: x, y")
            return False

        ui_automator = self.device_manager.get_ui_automator()
        if not ui_automator:
            detailed_logger.error("设备未连接")
            return False

        async def click_operation():
            success = ui_automator.click(
                x=params['x'],
                y=params['y'],
                timeout=params.get('timeout', 10.0),
                retry=params.get('retry', 3)
            )
            if success and params.get('wait_after', 0) > 0:
                await asyncio.sleep(params['wait_after'])
            return success

        return await self._execute_with_cleanup(click_operation)

    @log_exception
    async def handle_swipe(self, action: Action) -> bool:
        """处理滑动动作
        
        Args:
            action: 包含滑动参数的Action对象
                   必需参数: from_x, from_y, to_x, to_y
                   可选参数: duration, timeout, wait_after
        
        Returns:
            bool: 滑动是否成功
        """
        params = action.parameters
        required_params = ['from_x', 'from_y', 'to_x', 'to_y']
        if not all(param in params for param in required_params):
            detailed_logger.error(f"滑动动作缺少必需参数: {required_params}")
            return False

        ui_automator = self.device_manager.get_ui_automator()
        if not ui_automator:
            detailed_logger.error("设备未连接")
            return False

        async def swipe_operation():
            success = ui_automator.swipe(
                from_x=params['from_x'],
                from_y=params['from_y'],
                to_x=params['to_x'],
                to_y=params['to_y'],
                duration=params.get('duration', 0.1),
                timeout=params.get('timeout', 10.0)
            )
            if success and params.get('wait_after', 0) > 0:
                await asyncio.sleep(params['wait_after'])
            return success

        return await self._execute_with_cleanup(swipe_operation)

    @log_exception
    async def handle_click_text(self, action: Action) -> bool:
        """处理文本点击动作
        
        Args:
            action: 包含文本参数的Action对象
                   必需参数: text
                   可选参数: timeout, wait_after
        
        Returns:
            bool: 点击是否成功
        """
        params = action.parameters
        if 'text' not in params:
            detailed_logger.error("文本点击动作缺少必需参数: text")
            return False

        ui_automator = self.device_manager.get_ui_automator()
        if not ui_automator:
            detailed_logger.error("设备未连接")
            return False

        async def click_text_operation():
            success = ui_automator.click_text(
                text=params['text'],
                timeout=params.get('timeout', 10.0)
            )
            if success and params.get('wait_after', 0) > 0:
                await asyncio.sleep(params['wait_after'])
            return success

        return await self._execute_with_cleanup(click_text_operation)

    @log_exception
    async def handle_screenshot(self, action: Action) -> bool:
        """处理截图动作
        
        Args:
            action: 包含截图参数的Action对象
                   必需参数: filename
                   可选参数: wait_after
        
        Returns:
            bool: 截图是否成功
        """
        params = action.parameters
        if 'filename' not in params:
            detailed_logger.error("截图动作缺少必需参数: filename")
            return False

        ui_automator = self.device_manager.get_ui_automator()
        if not ui_automator:
            detailed_logger.error("设备未连接")
            return False

        async def screenshot_operation():
            success = ui_automator.screenshot(params['filename'])
            if success and params.get('wait_after', 0) > 0:
                await asyncio.sleep(params['wait_after'])
            return success

        return await self._execute_with_cleanup(screenshot_operation)

    @log_exception
    def evaluate_device_connection(self, condition: Condition, context: Dict) -> bool:
        """评估设备连接状态条件
        
        Args:
            condition: 条件对象
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        return self.device_manager.is_connected

    @log_exception
    async def evaluate_text_exists(self, condition: Condition, context: Dict) -> bool:
        """评估文本存在条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - text: 要查找的文本
                      - timeout: 可选的超时时间
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'text' not in params:
            detailed_logger.error("文本存在条件缺少必需参数: text")
            return False

        ui_automator = self.device_manager.get_ui_automator()
        if not ui_automator:
            return False

        async def find_text_operation():
            element = ui_automator.find_element_by_text(
                text=params['text'],
                timeout=params.get('timeout', 10.0)
            )
            return element is not None

        try:
            return await self._execute_with_cleanup(find_text_operation)
        except Exception:
            return False

    def register_handlers(self, decision_maker) -> None:
        """注册设备相关的动作和条件处理器
        
        Args:
            decision_maker: DecisionMaker实例
        """
        # 注册动作处理器
        decision_maker.register_action_handler("device_click", self.handle_click)
        decision_maker.register_action_handler("device_swipe", self.handle_swipe)
        decision_maker.register_action_handler("device_click_text", self.handle_click_text)
        decision_maker.register_action_handler("device_screenshot", self.handle_screenshot)

        # 注册条件处理器
        decision_maker.register_condition_handler("device_connected", self.evaluate_device_connection)
        decision_maker.register_condition_handler("text_exists", self.evaluate_text_exists)

    async def cleanup(self) -> None:
        """清理所有操作和资源"""
        # 等待所有清理任务完成
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        self._cleanup_tasks.clear()
        self._operation_results.clear()
