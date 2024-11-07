from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from ..core.scene_analyzer import SceneContext
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TaskRequirement:
    """任务需求数据类"""
    type: str
    value: Any
    description: str

@dataclass
class TaskReward:
    """任务奖励数据类"""
    type: str
    value: Any
    description: str

@dataclass
class Task:
    """任务数据类"""
    id: str
    name: str
    description: str
    type: str
    priority: TaskPriority
    status: TaskStatus
    progress: float
    requirements: List[TaskRequirement]
    rewards: List[TaskReward]
    parent_task: Optional[str]
    subtasks: List[str]
    completion_criteria: Dict[str, Any]
    failure_conditions: Dict[str, Any]
    timeout: Optional[float]
    start_time: Optional[float]
    completion_time: Optional[float]

class TaskManager:
    """
    任务管理器核心类
    负责游戏任务的创建、分配、监控和执行
    """
    def __init__(self):
        self.logger = detailed_logger
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.task_dependencies: Dict[str, List[str]] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()

    @log_exception
    async def initialize(self) -> None:
        """初始化任务管理器"""
        try:
            # 加载任务配置
            await self._load_task_configurations()
            
            # 启动任务处理循环
            asyncio.create_task(self._task_processing_loop())
            
            self.logger.info("Task Manager initialized successfully")
        except Exception as e:
            raise GameAutomationError(f"Failed to initialize task manager: {str(e)}")

    @log_exception
    async def create_task(self, task_config: Dict[str, Any]) -> Task:
        """
        创建新任务
        :param task_config: 任务配置
        :return: 创建的任务
        """
        try:
            task = Task(
                id=task_config['id'],
                name=task_config['name'],
                description=task_config['description'],
                type=task_config['type'],
                priority=TaskPriority[task_config['priority']],
                status=TaskStatus.PENDING,
                progress=0.0,
                requirements=[
                    TaskRequirement(**req) for req in task_config.get('requirements', [])
                ],
                rewards=[
                    TaskReward(**reward) for reward in task_config.get('rewards', [])
                ],
                parent_task=task_config.get('parent_task'),
                subtasks=task_config.get('subtasks', []),
                completion_criteria=task_config.get('completion_criteria', {}),
                failure_conditions=task_config.get('failure_conditions', {}),
                timeout=task_config.get('timeout'),
                start_time=None,
                completion_time=None
            )
            
            self.tasks[task.id] = task
            await self._update_task_dependencies(task)
            
            return task
            
        except Exception as e:
            raise GameAutomationError(f"Failed to create task: {str(e)}")

    @log_exception
    async def start_task(self, task_id: str) -> None:
        """
        开始执行任务
        :param task_id: 任务ID
        """
        try:
            if task_id not in self.tasks:
                raise GameAutomationError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]
            
            # 检查任务是否可以开始
            if not await self._can_start_task(task):
                task.status = TaskStatus.BLOCKED
                return
                
            # 更新任务状态
            task.status = TaskStatus.IN_PROGRESS
            task.start_time = asyncio.get_event_loop().time()
            
            # 添加到活动任务列表
            if task_id not in self.active_tasks:
                self.active_tasks.append(task_id)
                
            # 将任务添加到处理队列
            await self._task_queue.put(task_id)
            
            self.logger.info(f"Started task: {task.name}")
            
        except Exception as e:
            raise GameAutomationError(f"Failed to start task: {str(e)}")

    @log_exception
    async def update_task_progress(self,
                                 task_id: str,
                                 progress: float,
                                 scene_context: SceneContext) -> None:
        """
        更新任务进度
        :param task_id: 任务ID
        :param progress: 新的进度值
        :param scene_context: 当前场景上下文
        """
        try:
            if task_id not in self.tasks:
                raise GameAutomationError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]
            task.progress = progress
            
            # 检查任务是否完成
            if progress >= 100.0:
                await self.complete_task(task_id)
            # 检查失败条件
            elif await self._check_failure_conditions(task, scene_context):
                await self.fail_task(task_id)
                
        except Exception as e:
            raise GameAutomationError(f"Failed to update task progress: {str(e)}")

    @log_exception
    async def complete_task(self, task_id: str) -> None:
        """
        完成任务
        :param task_id: 任务ID
        """
        try:
            if task_id not in self.tasks:
                raise GameAutomationError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completion_time = asyncio.get_event_loop().time()
            
            # 更新任务列表
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            self.completed_tasks.append(task_id)
            
            # 处理子任务
            for subtask_id in task.subtasks:
                if subtask_id in self.tasks:
                    await self.start_task(subtask_id)
                    
            self.logger.info(f"Completed task: {task.name}")
            
        except Exception as e:
            raise GameAutomationError(f"Failed to complete task: {str(e)}")

    @log_exception
    async def fail_task(self, task_id: str) -> None:
        """
        标记任务失败
        :param task_id: 任务ID
        """
        try:
            if task_id not in self.tasks:
                raise GameAutomationError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED
            
            # 更新任务列表
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            self.failed_tasks.append(task_id)
            
            self.logger.warning(f"Task failed: {task.name}")
            
        except Exception as e:
            raise GameAutomationError(f"Failed to mark task as failed: {str(e)}")

    async def get_active_tasks(self) -> List[Task]:
        """获取当前活动的任务列表"""
        return [self.tasks[task_id] for task_id in self.active_tasks]

    async def get_task_chain(self, task_id: str) -> List[Task]:
        """获取任务链（包括父任务和子任务）"""
        task_chain = []
        try:
            task = self.tasks[task_id]
            
            # 添加父任务链
            current_parent = task.parent_task
            while current_parent and current_parent in self.tasks:
                parent_task = self.tasks[current_parent]
                task_chain.insert(0, parent_task)
                current_parent = parent_task.parent_task
            
            # 添加当前任务
            task_chain.append(task)
            
            # 添加子任务链
            for subtask_id in task.subtasks:
                if subtask_id in self.tasks:
                    task_chain.append(self.tasks[subtask_id])
                    
            return task_chain
            
        except Exception as e:
            self.logger.error(f"Error getting task chain: {str(e)}")
            return []

    async def _task_processing_loop(self) -> None:
        """任务处理循环"""
        while True:
            try:
                # 从队列获取任务
                task_id = await self._task_queue.get()
                task = self.tasks[task_id]
                
                # 检查任务超时
                if task.timeout and task.start_time:
                    elapsed_time = asyncio.get_event_loop().time() - task.start_time
                    if elapsed_time > task.timeout:
                        await self.fail_task(task_id)
                        continue
                
                # 处理任务
                if task.status == TaskStatus.IN_PROGRESS:
                    # 实际的任务处理逻辑将由具体的任务执行器实现
                    pass
                
                self._task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {str(e)}")
                await asyncio.sleep(1)

    async def _can_start_task(self, task: Task) -> bool:
        """检查任务是否可以开始"""
        try:
            # 检查依赖任务
            if task.id in self.task_dependencies:
                for dep_task_id in self.task_dependencies[task.id]:
                    if dep_task_id not in self.completed_tasks:
                        return False
            
            # 检查需求
            for req in task.requirements:
                if not await self._check_requirement(req):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if task can start: {str(e)}")
            return False

    async def _check_requirement(self, requirement: TaskRequirement) -> bool:
        """检查任务需求是否满足"""
        # TODO: 实现具体的需求检查逻辑
        return True

    async def _check_failure_conditions(self, task: Task, scene_context: SceneContext) -> bool:
        """检查任务失败条件"""
        try:
            for condition, value in task.failure_conditions.items():
                # TODO: 实现具体的失败条件检查逻辑
                pass
            return False
        except Exception as e:
            self.logger.error(f"Error checking failure conditions: {str(e)}")
            return False

    async def _update_task_dependencies(self, task: Task) -> None:
        """更新任务依赖关系"""
        if task.parent_task:
            if task.id not in self.task_dependencies:
                self.task_dependencies[task.id] = []
            self.task_dependencies[task.id].append(task.parent_task)

    async def _load_task_configurations(self) -> None:
        """加载任务配置"""
        # TODO: 实现任务配置加载逻辑
        pass

# 创建全局实例
task_manager = TaskManager()
