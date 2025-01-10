from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from ..context_manager import ContextManager, GameState
from ..task_executor import TaskExecutor, TaskType, TaskPriority, TaskStatus
from ...device.device_manager import DeviceManager
from ...scene_understanding.advanced_scene_analyzer import AdvancedSceneAnalyzer
from ..error.error_manager import (
    ErrorManager,
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)

class GameEngine:
    """游戏引擎"""
    
    def __init__(
        self,
        device_manager: DeviceManager,
        scene_analyzer: AdvancedSceneAnalyzer,
        context_manager: ContextManager,
        task_executor: TaskExecutor,
        error_manager: Optional[ErrorManager] = None
    ):
        self._device_manager = device_manager
        self._scene_analyzer = scene_analyzer
        self._context_manager = context_manager
        self._task_executor = task_executor
        self._error_manager = error_manager or ErrorManager()
        
        self._initialized = False
        self._running = False
        self._scene_check_interval = 1.0
        self._scene_check_task = None
        
    async def initialize(self):
        """初始化引擎"""
        if not self._initialized:
            try:
                # 初始化组件
                await self._error_manager.initialize()
                await self._device_manager.initialize()
                await self._scene_analyzer.initialize()
                await self._context_manager.initialize()
                await self._task_executor.initialize()
                
                # 注册事件处理器
                self._register_event_handlers()
                
                self._initialized = True
                
            except Exception as e:
                raise GameAutomationError(
                    message=f"引擎初始化失败: {str(e)}",
                    error_code="ENG001",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_error=e
                )
                
    async def cleanup(self):
        """清理资源"""
        if self._initialized:
            try:
                # 停止引擎
                await self.stop()
                
                # 清理组件
                await self._task_executor.cleanup()
                await self._context_manager.cleanup()
                await self._scene_analyzer.cleanup()
                await self._device_manager.cleanup()
                await self._error_manager.cleanup()
                
                self._initialized = False
                
            except Exception as e:
                raise GameAutomationError(
                    message=f"引擎清理失败: {str(e)}",
                    error_code="ENG002",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_error=e
                )
                
    async def start(self):
        """启动引擎"""
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG003",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        if self._running:
            return
            
        try:
            # 连接设备
            await self._device_manager.connect()
            
            # 启动任务执行器
            await self._task_executor.start()
            
            # 启动场景检查
            self._running = True
            self._scene_check_task = asyncio.create_task(
                self._check_scene_loop()
            )
            
            # 设置初始状态
            await self._context_manager.set_state(GameState.UNKNOWN)
            
            logging.info("游戏引擎已启动")
            
        except Exception as e:
            self._running = False
            raise GameAutomationError(
                message=f"引擎启动失败: {str(e)}",
                error_code="ENG004",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
            
    async def stop(self):
        """停止引擎"""
        if not self._running:
            return
            
        try:
            # 停止场景检查
            self._running = False
            if self._scene_check_task:
                self._scene_check_task.cancel()
                try:
                    await self._scene_check_task
                except asyncio.CancelledError:
                    pass
                self._scene_check_task = None
                
            # 停止任务执行器
            await self._task_executor.stop()
            
            # 断开设备
            await self._device_manager.disconnect()
            
            logging.info("游戏引擎已停止")
            
        except Exception as e:
            raise GameAutomationError(
                message=f"引擎停止失败: {str(e)}",
                error_code="ENG005",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
            
    async def set_scene_check_interval(self, interval: float):
        """设置场景检查间隔
        
        Args:
            interval: 间隔时间(秒)
        """
        if interval <= 0:
            raise GameAutomationError(
                message="间隔时间必须大于0",
                error_code="ENG029",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR
            )
            
        self._scene_check_interval = interval
        logging.info(f"场景检查间隔已设置为: {interval}秒")
            
    async def get_performance_metrics(self) -> Dict:
        """获取性能指标
        
        Returns:
            Dict: 性能指标数据
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG030",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            metrics = {
                'scene_analysis': {
                    'total_checks': await self._scene_analyzer.get_total_checks(),
                    'avg_analysis_time': await self._scene_analyzer.get_avg_analysis_time(),
                    'success_rate': await self._scene_analyzer.get_success_rate()
                },
                'task_execution': {
                    'total_tasks': await self._task_executor.get_total_tasks(),
                    'completed_tasks': await self._task_executor.get_completed_tasks(),
                    'failed_tasks': await self._task_executor.get_failed_tasks(),
                    'avg_execution_time': await self._task_executor.get_avg_execution_time()
                },
                'device': {
                    'connection_status': await self._device_manager.get_connection_status(),
                    'response_time': await self._device_manager.get_response_time(),
                    'error_count': await self._device_manager.get_error_count()
                },
                'error_stats': {
                    'total_errors': len(await self._error_manager.get_errors()),
                    'critical_errors': len(await self._error_manager.get_errors(severity=ErrorSeverity.CRITICAL)),
                    'error_categories': await self._get_error_category_stats()
                }
            }
            
            return metrics
            
        except Exception as e:
            raise GameAutomationError(
                message=f"获取性能指标失败: {str(e)}",
                error_code="ENG031",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def get_engine_status(self) -> Dict:
        """获取引擎状态
        
        Returns:
            Dict: 引擎状态信息
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG032",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            return {
                'initialized': self._initialized,
                'running': self._running,
                'scene_check_interval': self._scene_check_interval,
                'current_state': await self.get_state(),
                'device_connected': await self._device_manager.is_connected(),
                'task_executor_running': await self._task_executor.is_running(),
                'active_tasks': len(await self.get_tasks(status=TaskStatus.RUNNING)),
                'pending_tasks': len(await self.get_tasks(status=TaskStatus.PENDING)),
                'recent_errors': len(await self._error_manager.get_errors(
                    start_time=datetime.now().replace(
                        minute=0, second=0, microsecond=0
                    )
                ))
            }
            
        except Exception as e:
            raise GameAutomationError(
                message=f"获取引擎状态失败: {str(e)}",
                error_code="ENG033",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def get_state(self) -> GameState:
        """获取游戏状态
        
        Returns:
            GameState: 游戏状态
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG019",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            return await self._context_manager.get_state()
            
        except Exception as e:
            raise GameAutomationError(
                message=f"获取状态失败: {str(e)}",
                error_code="ENG020",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def add_task(
        self,
        name: str,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        params: Dict = None
    ) -> str:
        """添加任务
        
        Args:
            name: 任务名称
            task_type: 任务类型
            priority: 优先级
            params: 参数
            
        Returns:
            str: 任务ID
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG011",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            return await self._task_executor.add_task(
                name,
                task_type,
                priority,
                params
            )
            
        except Exception as e:
            raise GameAutomationError(
                message=f"添加任务失败: {str(e)}",
                error_code="ENG012",
                category=ErrorCategory.TASK,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def cancel_task(self, task_id: str):
        """取消任务
        
        Args:
            task_id: 任务ID
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG013",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            await self._task_executor.cancel_task(task_id)
            
        except Exception as e:
            raise GameAutomationError(
                message=f"取消任务失败: {str(e)}",
                error_code="ENG014",
                category=ErrorCategory.TASK,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 任务信息
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG015",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            return await self._task_executor.get_task(task_id)
            
        except Exception as e:
            raise GameAutomationError(
                message=f"获取任务失败: {str(e)}",
                error_code="ENG016",
                category=ErrorCategory.TASK,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    async def get_tasks(
        self,
        task_type: TaskType = None,
        status: TaskStatus = None
    ) -> List[Dict]:
        """获取任务列表
        
        Args:
            task_type: 任务类型
            status: 任务状态
            
        Returns:
            List[Dict]: 任务列表
        """
        if not self._initialized:
            raise GameAutomationError(
                message="引擎未初始化",
                error_code="ENG017",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL
            )
            
        try:
            return await self._task_executor.get_tasks(task_type, status)
            
        except Exception as e:
            raise GameAutomationError(
                message=f"获取任务列表失败: {str(e)}",
                error_code="ENG018",
                category=ErrorCategory.TASK,
                severity=ErrorSeverity.ERROR,
                original_error=e
            )
            
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 状态变化事件
        self._context_manager.add_listener(
            'state_changed',
            self._handle_state_changed
        )
        
        # 任务状态事件
        self._context_manager.add_listener(
            'task_status_changed',
            self._handle_task_status_changed
        )
        
        # 错误事件
        self._context_manager.add_listener(
            'error_occurred',
            self._handle_error_occurred
        )
            
    async def _check_scene_loop(self):
        """场景检查循环"""
        while self._running:
            try:
                # 获取截图
                screenshot = await self._device_manager.get_screenshot()
                
                # 分析场景
                scene_info = await self._scene_analyzer.analyze_scene(screenshot)
                
                # 更新状态
                if scene_info['type'] != 'unknown':
                    try:
                        new_state = GameState[scene_info['type'].upper()]
                        current_state = await self._context_manager.get_state()
                        
                        if new_state != current_state:
                            await self._context_manager.set_state(new_state)
                            
                    except KeyError:
                        pass
                        
                # 等待下次检查
                await asyncio.sleep(self._scene_check_interval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                await self._error_manager.handle_error(
                    GameAutomationError(
                        message=f"场景检查失败: {str(e)}",
                        error_code="ENG006",
                        category=ErrorCategory.SCENE,
                        severity=ErrorSeverity.ERROR,
                        original_error=e
                    )
                )
                await asyncio.sleep(self._scene_check_interval)
                
    async def _handle_state_changed(self, data: Dict):
        """处理状态变化事件
        
        Args:
            data: 事件数据
        """
        old_state = data['old_state']
        new_state = data['new_state']
        
        logging.info(f"游戏状态变化: {old_state.name} -> {new_state.name}")
        
        try:
            # 根据状态执行相应操作
            if new_state == GameState.ERROR:
                # 停止所有任务
                await self._task_executor.stop()
                
            elif new_state == GameState.MAIN:
                # 恢复任务执行
                await self._task_executor.start()
                
        except Exception as e:
            await self._error_manager.handle_error(
                GameAutomationError(
                    message=f"状态处理失败: {str(e)}",
                    error_code="ENG007",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.ERROR,
                    original_error=e
                )
            )
            
    async def _handle_task_status_changed(self, data: Dict):
        """处理任务状态事件
        
        Args:
            data: 事件数据
        """
        task_id = data['task_id']
        old_status = data['old_status']
        new_status = data['new_status']
        
        logging.info(
            f"任务状态变化: {task_id} "
            f"{old_status.name} -> {new_status.name}"
        )
        
        try:
            # 根据状态执行相应操作
            if new_status == TaskStatus.FAILED:
                # 处理任务失败
                task = await self._task_executor.get_task(task_id)
                if task:
                    await self._error_manager.handle_error(
                        GameAutomationError(
                            message=f"任务失败: {task['name']}",
                            error_code="ENG008",
                            category=ErrorCategory.TASK,
                            severity=ErrorSeverity.ERROR,
                            context=task
                        )
                    )
                    
        except Exception as e:
            await self._error_manager.handle_error(
                GameAutomationError(
                    message=f"任务状态处理失败: {str(e)}",
                    error_code="ENG009",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.ERROR,
                    original_error=e
                )
            )
            
    async def _handle_error_occurred(self, data: Dict):
        """处理错误事件
        
        Args:
            data: 事件数据
        """
        error = data['error']
        
        logging.error(
            f"发生错误: {error.message}\n"
            f"错误代码: {error.error_code}\n"
            f"错误类别: {error.category.name}\n"
            f"严重程度: {error.severity.name}"
        )
        
        try:
            # 根据错误处理
            if error.severity >= ErrorSeverity.CRITICAL:
                # 停止引擎
                await self.stop()
                
        except Exception as e:
            await self._error_manager.handle_error(
                GameAutomationError(
                    message=f"错误处理失败: {str(e)}",
                    error_code="ENG010",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_error=e
                )
            )
            
    async def _get_error_category_stats(self) -> Dict[str, int]:
        """获取错误类别统计
        
        Returns:
            Dict[str, int]: 错误类别统计
        """
        stats = {}
        errors = await self._error_manager.get_errors()
        
        for error in errors:
            category = error.category.name
            stats[category] = stats.get(category, 0) + 1
            
        return stats
