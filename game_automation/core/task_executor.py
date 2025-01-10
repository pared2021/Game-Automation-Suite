from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
from enum import Enum, auto
import uuid

from .error.error_manager import (
    GameAutomationError,
    ErrorCategory,
    ErrorSeverity
)

class TaskType(Enum):
    """任务类型"""
    DAILY = auto()      # 日常任务
    WEEKLY = auto()     # 周常任务
    EVENT = auto()      # 活动任务
    STORY = auto()      # 剧情任务
    BATTLE = auto()     # 战斗任务
    RESOURCE = auto()   # 资源任务
    CUSTOM = auto()     # 自定义任务

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()    # 等待中
    RUNNING = auto()    # 运行中
    COMPLETED = auto()  # 已完成
    FAILED = auto()     # 失败
    CANCELLED = auto()  # 已取消

class Task:
    """任务"""
    def __init__(
        self,
        task_id: str,
        name: str,
        task_type: TaskType,
        priority: TaskPriority,
        params: Dict = None
    ):
        self.task_id = task_id
        self.name = name
        self.task_type = task_type
        self.priority = priority
        self.params = params or {}
        
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        
    def to_dict(self) -> Dict:
        """转换为字典
        
        Returns:
            Dict: 任务信息字典
        """
        return {
            'id': self.task_id,
            'name': self.name,
            'type': self.task_type.name,
            'priority': self.priority.name,
            'status': self.status.name,
            'params': self.params,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        }

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_handlers: Dict[TaskType, Callable] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        
    async def initialize(self):
        """初始化执行器"""
        # 注册默认任务处理器
        self._register_default_handlers()
        
    async def cleanup(self):
        """清理资源"""
        # 取消所有运行中的任务
        for task in self._running_tasks.values():
            task.cancel()
            
        # 等待任务完成
        if self._running_tasks:
            await asyncio.gather(
                *self._running_tasks.values(),
                return_exceptions=True
            )
            
        # 清理数据
        self._tasks.clear()
        self._running_tasks.clear()
        self._task_handlers.clear()
        self._task_queue = asyncio.PriorityQueue()
        
    def _register_default_handlers(self):
        """注册默认任务处理器"""
        # TODO: 实现默认任务处理器
        pass
        
    def register_task_handler(
        self,
        task_type: TaskType,
        handler: Callable
    ):
        """注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self._task_handlers[task_type] = handler
        
    async def add_task(
        self,
        name: str,
        task_type: TaskType,
        priority: TaskPriority,
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
        # 创建任务
        task_id = str(uuid.uuid4())
        task = Task(task_id, name, task_type, priority, params)
        
        # 保存任务
        self._tasks[task_id] = task
        
        # 加入队列
        await self._task_queue.put(
            (
                -priority.value,  # 优先级取反，值越小优先级越高
                task.created_at.timestamp(),
                task_id
            )
        )
        
        return task_id
        
    async def cancel_task(self, task_id: str):
        """取消任务
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self._tasks:
            return
            
        task = self._tasks[task_id]
        
        # 取消运行中的任务
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            await asyncio.sleep(0)  # 让出控制权
            
        # 更新状态
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 任务信息
        """
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None
        
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
        tasks = []
        
        for task in self._tasks.values():
            if task_type and task.task_type != task_type:
                continue
                
            if status and task.status != status:
                continue
                
            tasks.append(task.to_dict())
            
        return tasks
        
    async def start(self):
        """启动执行器"""
        if self._running:
            return
            
        self._running = True
        asyncio.create_task(self._process_tasks())
        
    async def stop(self):
        """停止执行器"""
        self._running = False
        
    async def _process_tasks(self):
        """处理任务队列"""
        while self._running:
            try:
                # 获取任务
                _, _, task_id = await self._task_queue.get()
                
                # 检查任务是否存在
                if task_id not in self._tasks:
                    continue
                    
                task = self._tasks[task_id]
                
                # 检查任务状态
                if task.status != TaskStatus.PENDING:
                    continue
                    
                # 获取处理器
                handler = self._task_handlers.get(task.task_type)
                if not handler:
                    task.status = TaskStatus.FAILED
                    task.error = "No handler registered for task type"
                    continue
                    
                # 执行任务
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                task_coroutine = handler(task)
                self._running_tasks[task_id] = asyncio.create_task(
                    task_coroutine
                )
                
                try:
                    await self._running_tasks[task_id]
                    task.status = TaskStatus.COMPLETED
                    
                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    
                finally:
                    task.completed_at = datetime.now()
                    del self._running_tasks[task_id]
                    
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                # 记录错误但继续处理
                print(f"Task processing error: {e}")
                await asyncio.sleep(1)
