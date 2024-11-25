from typing import Dict, Optional, Type
from datetime import datetime
import asyncio
import time

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .task_manager import Task, TaskStatus, TaskError
from ..device.device_manager import DeviceManager
from ..scene_understanding.advanced_scene_analyzer import AdvancedSceneAnalyzer

class TaskExecutionError(GameAutomationError):
    """任务执行错误"""
    pass

class GameTask(Task):
    """游戏任务基类"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 device_manager: DeviceManager,
                 scene_analyzer: AdvancedSceneAnalyzer,
                 **kwargs):
        """初始化游戏任务
        
        Args:
            task_id: 任务ID
            name: 任务名称
            device_manager: 设备管理器实例
            scene_analyzer: 场景分析器实例
            **kwargs: 其他Task参数
        """
        super().__init__(task_id, name, **kwargs)
        self.device_manager = device_manager
        self.scene_analyzer = scene_analyzer
        self.execution_data: Dict = {}  # 存储执行过程中的数据
        self.timeout = kwargs.get('timeout', 300)  # 默认5分钟超时

    async def verify_prerequisites(self) -> bool:
        """验证任务执行前提条件
        
        Returns:
            bool: 前提条件是否满足
        """
        # 验证设备连接
        if not self.device_manager.is_connected:
            detailed_logger.error("设备未连接")
            return False
            
        # 验证UI自动化控制器
        if not self.device_manager.get_ui_automator():
            detailed_logger.error("UI自动化控制器未初始化")
            return False
            
        return True

    async def cleanup(self) -> None:
        """任务清理工作"""
        self.execution_data.clear()

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, 
                 device_manager: DeviceManager,
                 scene_analyzer: AdvancedSceneAnalyzer):
        """初始化任务执行器
        
        Args:
            device_manager: 设备管理器实例
            scene_analyzer: 场景分析器实例
        """
        self.device_manager = device_manager
        self.scene_analyzer = scene_analyzer
        self.current_task: Optional[GameTask] = None
        self._task_types: Dict[str, Type[GameTask]] = {}
        
        # 资源管理
        self.max_concurrent_tasks = 5
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.active_tasks = 0
        self.last_resource_check = 0
        self.resource_check_interval = 60  # 资源检查间隔(秒)

    @log_exception
    def register_task_type(self, task_type: str, task_class: Type[GameTask]) -> None:
        """注册任务类型
        
        Args:
            task_type: 任务类型标识
            task_class: 任务类
        """
        if not issubclass(task_class, GameTask):
            raise TaskExecutionError(f"任务类必须继承自GameTask: {task_class}")
            
        self._task_types[task_type] = task_class
        detailed_logger.info(f"注册任务类型: {task_type}")

    @log_exception
    async def create_task(self, 
                         task_type: str,
                         task_id: str,
                         name: str,
                         **kwargs) -> GameTask:
        """创建任务实例
        
        Args:
            task_type: 任务类型
            task_id: 任务ID
            name: 任务名称
            **kwargs: 其他任务参数
            
        Returns:
            GameTask: 任务实例
        """
        if task_type not in self._task_types:
            raise TaskExecutionError(f"未注册的任务类型: {task_type}")
            
        task_class = self._task_types[task_type]
        task = task_class(
            task_id=task_id,
            name=name,
            device_manager=self.device_manager,
            scene_analyzer=self.scene_analyzer,
            **kwargs
        )
        
        return task

    async def _check_resources(self) -> bool:
        """检查系统资源使用情况
        
        Returns:
            bool: 资源是否充足
        """
        current_time = time.time()
        if current_time - self.last_resource_check < self.resource_check_interval:
            return True
            
        self.last_resource_check = current_time
        
        # TODO: 添加具体的资源检查逻辑
        if self.active_tasks >= self.max_concurrent_tasks:
            detailed_logger.warning("达到最大并发任务数限制")
            return False
            
        return True

    @log_exception
    async def execute_task(self, task: GameTask) -> bool:
        """执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            bool: 任务是否执行成功
        """
        if self.current_task:
            raise TaskExecutionError("当前有正在执行的任务")
            
        # 检查资源
        if not await self._check_resources():
            raise TaskExecutionError("系统资源不足")
            
        async with self.task_semaphore:
            self.current_task = task
            self.active_tasks += 1
            detailed_logger.info(f"开始执行任务: {task.name} ({task.task_id})")
            
            try:
                # 验证前提条件
                if not await task.verify_prerequisites():
                    raise TaskExecutionError("任务前提条件不满足")
                
                # 执行任务
                success = await self._execute_task_with_monitoring(task)
                
                # 任务清理
                await task.cleanup()
                
                if success:
                    detailed_logger.info(f"任务执行成功: {task.name}")
                else:
                    detailed_logger.error(f"任务执行失败: {task.name}")
                
                return success
                
            except Exception as e:
                detailed_logger.error(f"任务执行异常: {str(e)}")
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                return False
                
            finally:
                self.current_task = None
                self.active_tasks -= 1

    async def _execute_task_with_monitoring(self, task: GameTask) -> bool:
        """执行任务并进行监控
        
        Args:
            task: 要执行的任务
            
        Returns:
            bool: 任务是否执行成功
        """
        try:
            # 添加超时控制
            async with asyncio.timeout(task.timeout):
                # 创建监控任务
                monitor_task = asyncio.create_task(self._monitor_task_execution(task))
                
                try:
                    # 执行任务
                    success = await task.execute()
                    
                    # 取消监控
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                    
                    return success
                    
                except Exception as e:
                    # 确保监控任务被取消
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                    raise
                    
        except asyncio.TimeoutError:
            detailed_logger.error(f"任务执行超时: {task.name}")
            task.status = TaskStatus.FAILED
            task.error_message = "Task execution timeout"
            return False

    async def _monitor_task_execution(self, task: GameTask) -> None:
        """监控任务执行
        
        Args:
            task: 正在执行的任务
        """
        try:
            while True:
                # 检查任务状态
                if task.status == TaskStatus.FAILED:
                    detailed_logger.warning(f"任务执行失败: {task.name}")
                    break
                    
                if task.status == TaskStatus.COMPLETED:
                    detailed_logger.info(f"任务执行完成: {task.name}")
                    break
                    
                # 检查超时
                if task.timeout:
                    elapsed = (datetime.now() - task.start_time).total_seconds()
                    if elapsed > task.timeout:
                        raise TaskExecutionError(f"任务执行超时: {task.name}")
                
                # 更新场景分析
                if self.device_manager.is_connected:
                    ui = self.device_manager.get_ui_automator()
                    if ui:
                        screenshot_path = f"temp/task_{task.task_id}_monitor.png"
                        if ui.screenshot(screenshot_path):
                            # 分析场景
                            try:
                                import cv2
                                screenshot = cv2.imread(screenshot_path)
                                if screenshot is not None:
                                    analysis = self.scene_analyzer.analyze_screenshot(screenshot)
                                    task.execution_data['last_scene_analysis'] = analysis
                            except Exception as e:
                                detailed_logger.error(f"场景分析失败: {str(e)}")
                
                await asyncio.sleep(1)  # 监控间隔
                
        except asyncio.CancelledError:
            detailed_logger.info(f"任务监控已取消: {task.name}")
        except Exception as e:
            detailed_logger.error(f"任务监控异常: {str(e)}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

    @property
    def is_busy(self) -> bool:
        """是否有任务正在执行
        
        Returns:
            bool: 是否忙碌
        """
        return self.current_task is not None

    def get_registered_task_types(self) -> Dict[str, Type[GameTask]]:
        """获取已注册的任务类型
        
        Returns:
            Dict[str, Type[GameTask]]: 任务类型字典
        """
        return self._task_types.copy()
