"""Task scheduling and resource management."""
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from enum import Enum, auto
import logging

from game_automation.core.error.error_manager import (
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)
from .task_types import Task, TaskStatus
from .task_lifecycle import TaskState
from .task_dependency import TaskDependencyManager
from .task_monitor import TaskMonitor
from .task_priority import PriorityManager, PriorityLevel

class SchedulingStrategy(Enum):
    """调度策略类型"""
    PRIORITY = auto()       # 基于优先级
    FAIR = auto()          # 公平调度
    RESOURCE = auto()      # 基于资源
    TIME = auto()          # 基于时间
    HYBRID = auto()        # 混合策略

class ResourceType(Enum):
    """资源类型"""
    CPU = auto()           # CPU 资源
    MEMORY = auto()        # 内存资源
    GPU = auto()          # GPU 资源
    IO = auto()           # IO 资源
    NETWORK = auto()      # 网络资源

class SchedulerError(GameAutomationError):
    """调度器相关错误"""
    pass

class TaskScheduler:
    """任务调度器"""
    
    def __init__(
        self,
        dependency_manager: TaskDependencyManager,
        task_monitor: Optional[TaskMonitor] = None,
        strategy: SchedulingStrategy = SchedulingStrategy.HYBRID,
        max_parallel_tasks: int = 3
    ):
        """初始化调度器
        
        Args:
            dependency_manager: 依赖管理器
            task_monitor: 任务监控器
            strategy: 调度策略
            max_parallel_tasks: 最大并行任务数
        """
        self.dependency = dependency_manager
        self.monitor = task_monitor
        self.strategy = strategy
        self.max_parallel_tasks = max_parallel_tasks
        
        # 任务队列 - 按优先级排序
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # 优先级管理器
        self._priority = PriorityManager()
        
        # 资源限制
        self._resource_limits: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.8,      # 80% CPU
            ResourceType.MEMORY: 0.7,    # 70% Memory
            ResourceType.GPU: 0.9,      # 90% GPU
            ResourceType.IO: 0.6,       # 60% IO
            ResourceType.NETWORK: 0.5    # 50% Network
        }
        
        # 任务资源使用
        self._task_resources: Dict[str, Dict[ResourceType, float]] = {}
        
        # 任务时间片配额
        self._time_quotas: Dict[PriorityLevel, float] = {
            PriorityLevel.LOWEST: 1.0,       # 1 second
            PriorityLevel.VERY_LOW: 1.5,     # 1.5 seconds
            PriorityLevel.LOW: 2.0,          # 2 seconds
            PriorityLevel.BELOW_NORMAL: 3.0, # 3 seconds
            PriorityLevel.NORMAL: 4.0,       # 4 seconds
            PriorityLevel.ABOVE_NORMAL: 5.0, # 5 seconds
            PriorityLevel.HIGH: 6.0,         # 6 seconds
            PriorityLevel.VERY_HIGH: 8.0,    # 8 seconds
            PriorityLevel.HIGHEST: 10.0,     # 10 seconds
            PriorityLevel.CRITICAL: 15.0     # 15 seconds
        }
        
        # 任务运行时统计
        self._task_stats: Dict[str, Dict[str, Any]] = {}
        
        # 调度器状态
        self._running = False
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # 日志
        self._logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """初始化调度器"""
        self._running = True
        asyncio.create_task(self._scheduler_loop())
        
    async def cleanup(self):
        """清理资源"""
        self._running = False
        
        # 取消所有运行中的任务
        for task_id, task in self._running_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        self._running_tasks.clear()
        self._tasks.clear()
        self._task_stats.clear()
        self._task_resources.clear()
        
    async def add_task(self, task: Task) -> None:
        """添加任务
        
        Args:
            task: 任务实例
        """
        # 检查依赖
        if not self.dependency.are_dependencies_satisfied(task.task_id):
            raise SchedulerError(
                f"Task {task.task_id} dependencies not satisfied"
            )
            
        # 添加到任务字典
        self._tasks[task.task_id] = task
        
        # 注册到优先级管理器
        self._priority.register_task(task, task.priority)
        
        # 计算调度优先级
        priority = self._calculate_priority(task)
        
        # 添加到队列
        await self._task_queue.put((priority, task.task_id))
        
        # 初始化统计信息
        self._task_stats[task.task_id] = {
            'queued_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'wait_time': 0.0,
            'run_time': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'priority_changes': 0
        }
        
        self._logger.info(
            f"Added task {task.name} ({task.task_id}) with priority {priority}"
        )
        
    def remove_task(self, task_id: str) -> None:
        """移除任务
        
        Args:
            task_id: 任务ID
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            
        # 从优先级管理器中移除
        self._priority.unregister_task(task_id)
            
        if task_id in self._task_stats:
            del self._task_stats[task_id]
            
        if task_id in self._task_resources:
            del self._task_resources[task_id]
            
        self._logger.info(f"Removed task {task_id}")
        
    def set_resource_limit(
        self,
        resource_type: ResourceType,
        limit: float
    ) -> None:
        """设置资源限制
        
        Args:
            resource_type: 资源类型
            limit: 限制值 (0.0-1.0)
        """
        if not 0.0 <= limit <= 1.0:
            raise ValueError("Resource limit must be between 0.0 and 1.0")
            
        self._resource_limits[resource_type] = limit
        
    def set_time_quota(
        self,
        priority: PriorityLevel,
        quota: float
    ) -> None:
        """设置时间配额
        
        Args:
            priority: 任务优先级
            quota: 时间配额(秒)
        """
        if quota <= 0:
            raise ValueError("Time quota must be positive")
            
        self._time_quotas[priority] = quota
        
    def get_task_stats(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务统计信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 统计信息
        """
        return self._task_stats.get(task_id)
        
    def _calculate_priority(self, task: Task) -> float:
        """计算调度优先级
        
        Args:
            task: 任务实例
            
        Returns:
            float: 优先级值
        """
        # 获取有效优先级
        priority = self._priority.get_priority(task.task_id)
        base_priority = priority.value
        
        # 考虑等待时间
        stats = self._task_stats.get(task.task_id)
        if stats and stats['queued_at']:
            wait_time = (datetime.now() - stats['queued_at']).total_seconds()
            # 每等待60秒提升0.1优先级
            priority_boost = wait_time / 600
            base_priority += priority_boost
            
        # 考虑资源使用
        if task.task_id in self._task_resources:
            resources = self._task_resources[task.task_id]
            # 资源使用率越低优先级越高
            resource_factor = 1.0
            for res_type, usage in resources.items():
                limit = self._resource_limits[res_type]
                if usage > limit:
                    resource_factor *= 0.8
            base_priority *= resource_factor
            
        # 考虑依赖关系
        if self.dependency:
            # 关键路径上的任务优先级提升
            critical_path = self.dependency.get_critical_path(task.task_id)
            if critical_path:
                base_priority += 0.5
                
        return base_priority
        
    async def _scheduler_loop(self):
        """调度器主循环"""
        try:
            while self._running:
                try:
                    # 更新优先级管理器
                    self._priority.update()
                    
                    # 检查是否可以调度新任务
                    if len(self._running_tasks) >= self.max_parallel_tasks:
                        await asyncio.sleep(0.1)
                        continue
                        
                    # 获取下一个任务
                    priority, task_id = await self._task_queue.get()
                    task = self._tasks.get(task_id)
                    if not task:
                        continue
                        
                    # 检查依赖是否满足
                    if not self.dependency.are_dependencies_satisfied(task_id):
                        # 重新入队，降低优先级
                        new_priority = priority * 0.9
                        await self._task_queue.put((new_priority, task_id))
                        continue
                        
                    # 检查资源是否满足
                    if not self._check_resources(task):
                        # 重新入队，降低优先级
                        new_priority = priority * 0.8
                        await self._task_queue.put((new_priority, task_id))
                        continue
                        
                    # 启动任务
                    await self._start_task(task)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._logger.error(f"Scheduler error: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            pass
            
    def _check_resources(self, task: Task) -> bool:
        """检查资源是否满足
        
        Args:
            task: 任务实例
            
        Returns:
            bool: 是否满足
        """
        if not self.monitor:
            return True
            
        current_usage = self.monitor.get_resource_usage()
        task_usage = self._task_resources.get(task.task_id, {})
        
        for res_type, limit in self._resource_limits.items():
            current = current_usage.get(res_type, 0.0)
            needed = task_usage.get(res_type, 0.2)  # 默认20%
            if current + needed > limit:
                return False
                
        return True
        
    async def _start_task(self, task: Task):
        """启动任务
        
        Args:
            task: 任务实例
        """
        # 更新统计信息
        stats = self._task_stats[task.task_id]
        stats['started_at'] = datetime.now()
        stats['wait_time'] = (
            stats['started_at'] - stats['queued_at']
        ).total_seconds()
        
        # 创建任务
        self._running_tasks[task.task_id] = asyncio.create_task(
            self._run_task(task)
        )
        
        self._logger.info(f"Started task {task.name} ({task.task_id})")
        
    async def _run_task(self, task: Task):
        """运行任务
        
        Args:
            task: 任务实例
        """
        try:
            # 获取时间配额
            priority = self._priority.get_priority(task.task_id)
            time_quota = self._time_quotas[priority]
            deadline = datetime.now() + timedelta(seconds=time_quota)
            
            # 运行任务
            while datetime.now() < deadline:
                # 检查资源使用
                if self.monitor:
                    usage = self.monitor.get_task_resource_usage(task.task_id)
                    self._task_resources[task.task_id] = usage
                    
                    # 更新统计信息
                    stats = self._task_stats[task.task_id]
                    stats['cpu_usage'] = usage.get(ResourceType.CPU, 0.0)
                    stats['memory_usage'] = usage.get(ResourceType.MEMORY, 0.0)
                    
                # 模拟任务执行
                await asyncio.sleep(0.1)
                
            # 完成任务
            self._complete_task(task)
            
        except asyncio.CancelledError:
            self._cancel_task(task)
            raise
            
        except Exception as e:
            self._fail_task(task, str(e))
            
        finally:
            # 清理资源
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
                
    def _complete_task(self, task: Task):
        """完成任务
        
        Args:
            task: 任务实例
        """
        task.status = TaskStatus.COMPLETED
        stats = self._task_stats[task.task_id]
        stats['completed_at'] = datetime.now()
        stats['run_time'] = (
            stats['completed_at'] - stats['started_at']
        ).total_seconds()
        
        self._logger.info(f"Completed task {task.name} ({task.task_id})")
        
    def _cancel_task(self, task: Task):
        """取消任务
        
        Args:
            task: 任务实例
        """
        task.status = TaskStatus.CANCELLED
        stats = self._task_stats[task.task_id]
        stats['completed_at'] = datetime.now()
        stats['run_time'] = (
            stats['completed_at'] - stats['started_at']
        ).total_seconds()
        
        self._logger.info(f"Cancelled task {task.name} ({task.task_id})")
        
    def _fail_task(self, task: Task, error: str):
        """任务失败
        
        Args:
            task: 任务实例
            error: 错误信息
        """
        task.status = TaskStatus.FAILED
        task.error = error
        stats = self._task_stats[task.task_id]
        stats['completed_at'] = datetime.now()
        stats['run_time'] = (
            stats['completed_at'] - stats['started_at']
        ).total_seconds()
        
        self._logger.error(
            f"Task {task.name} ({task.task_id}) failed: {error}"
        )
