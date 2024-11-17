from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import os
from enum import Enum, auto

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .task_rule import TaskRuleManager, TaskRule
from .task_history import TaskHistory, TaskHistoryEntry

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = auto()    # 等待执行
    RUNNING = auto()    # 正在执行
    COMPLETED = auto()  # 执行完成
    FAILED = auto()     # 执行失败
    PAUSED = auto()     # 已暂停

class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class TaskError(GameAutomationError):
    """任务相关错误"""
    pass

class Task:
    """任务基类"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 dependencies: List[str] = None,
                 timeout: float = None):
        """初始化任务
        
        Args:
            task_id: 任务ID
            name: 任务名称
            priority: 任务优先级
            dependencies: 依赖的任务ID列表
            timeout: 任务超时时间（秒）
        """
        self.task_id = task_id
        self.name = name
        self.priority = priority
        self.dependencies = dependencies or []
        self.timeout = timeout
        
        self.status = TaskStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.progress: float = 0.0
        self.performance_metrics: Dict[str, Any] = {}
        
        self._on_complete_callbacks: List[Callable] = []
        self._on_fail_callbacks: List[Callable] = []
        self._on_progress_callbacks: List[Callable] = []

    def execute(self) -> bool:
        """执行任务
        
        Returns:
            bool: 任务是否执行成功
        """
        try:
            self.start_time = datetime.now()
            self.status = TaskStatus.RUNNING
            detailed_logger.info(f"开始执行任务: {self.name} ({self.task_id})")
            
            success = self._execute()
            
            self.end_time = datetime.now()
            if success:
                self.status = TaskStatus.COMPLETED
                self.progress = 1.0
                detailed_logger.info(f"任务执行成功: {self.name}")
                self._trigger_complete_callbacks()
            else:
                self.status = TaskStatus.FAILED
                detailed_logger.error(f"任务执行失败: {self.name}")
                self._trigger_fail_callbacks()
            
            return success
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error_message = str(e)
            self.end_time = datetime.now()
            detailed_logger.error(f"任务执行异常: {self.name} - {str(e)}")
            self._trigger_fail_callbacks()
            return False

    def _execute(self) -> bool:
        """实际的任务执行逻辑，子类需要实现此方法
        
        Returns:
            bool: 任务是否执行成功
        """
        raise NotImplementedError("Task._execute() must be implemented by subclass")

    def update_progress(self, progress: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """更新任务进度
        
        Args:
            progress: 进度值(0-1)
            metrics: 性能指标
        """
        self.progress = max(0.0, min(1.0, progress))
        if metrics:
            self.performance_metrics.update(metrics)
        self._trigger_progress_callbacks()

    def pause(self) -> None:
        """暂停任务"""
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.PAUSED
            detailed_logger.info(f"任务已暂停: {self.name}")

    def resume(self) -> None:
        """恢复任务"""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            detailed_logger.info(f"任务已恢复: {self.name}")

    def on_complete(self, callback: Callable) -> None:
        """添加任务完成回调
        
        Args:
            callback: 回调函数
        """
        self._on_complete_callbacks.append(callback)

    def on_fail(self, callback: Callable) -> None:
        """添加任务失败回调
        
        Args:
            callback: 回调函数
        """
        self._on_fail_callbacks.append(callback)

    def on_progress(self, callback: Callable) -> None:
        """添加进度更新回调
        
        Args:
            callback: 回调函数
        """
        self._on_progress_callbacks.append(callback)

    def _trigger_complete_callbacks(self) -> None:
        """触发完成回调"""
        for callback in self._on_complete_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"任务完成回调执行失败: {str(e)}")

    def _trigger_fail_callbacks(self) -> None:
        """触发失败回调"""
        for callback in self._on_fail_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"任务失败回调执行失败: {str(e)}")

    def _trigger_progress_callbacks(self) -> None:
        """触发进度更新回调"""
        for callback in self._on_progress_callbacks:
            try:
                callback(self)
            except Exception as e:
                detailed_logger.error(f"进度更新回调执行失败: {str(e)}")

    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 任务信息字典
        """
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority.name,
            'status': self.status.name,
            'dependencies': self.dependencies,
            'progress': self.progress,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }

class TaskManager:
    """任务管理器"""
    
    def __init__(self, state_dir: str = "data/task_state"):
        """初始化任务管理器
        
        Args:
            state_dir: 状态保存目录
        """
        self.state_dir = state_dir
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.running_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # 规则管理器
        self.rule_manager = TaskRuleManager()
        
        # 历史记录管理器
        self.history = TaskHistory()
        
        # 监控配置
        self.auto_save_interval = 300  # 5分钟自动保存
        self.last_save_time = datetime.now()

    @log_exception
    def add_task(self, task: Task) -> None:
        """添加任务
        
        Args:
            task: 任务实例
        """
        if task.task_id in self.tasks:
            raise TaskError(f"任务ID已存在: {task.task_id}")
            
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        self._sort_queue()
        
        # 创建任务规则
        self.rule_manager.create_rule(task)
        
        # 添加任务监控回调
        task.on_complete(self._on_task_complete)
        task.on_fail(self._on_task_fail)
        task.on_progress(self._on_task_progress)
        
        detailed_logger.info(f"添加任务: {task.name} ({task.task_id})")

    def _on_task_complete(self, task: Task) -> None:
        """任务完成回调
        
        Args:
            task: 完成的任务
        """
        # 添加历史记录
        self._add_history_entry(task)
        
        # 检查是否需要自动保存
        self._check_auto_save()

    def _on_task_fail(self, task: Task) -> None:
        """任务失败回调
        
        Args:
            task: 失败的任务
        """
        # 添加历史记录
        self._add_history_entry(task)
        
        # 检查是否需要自动保存
        self._check_auto_save()

    def _on_task_progress(self, task: Task) -> None:
        """任务进度更新回调
        
        Args:
            task: 更新进度的任务
        """
        # 检查超时
        if task.timeout and task.start_time:
            elapsed = datetime.now() - task.start_time
            if elapsed.total_seconds() > task.timeout:
                task.status = TaskStatus.FAILED
                task.error_message = "任务执行超时"
                detailed_logger.warning(f"任务超时: {task.name}")
                self._on_task_fail(task)

    def _add_history_entry(self, task: Task) -> None:
        """添加任务历史记录
        
        Args:
            task: 任务实例
        """
        if task.start_time:
            duration = None
            if task.end_time:
                duration = task.end_time - task.start_time
                
            entry = TaskHistoryEntry(
                task_id=task.task_id,
                task_name=task.name,
                status=task.status.name,
                start_time=task.start_time,
                end_time=task.end_time,
                duration=duration,
                error_message=task.error_message,
                performance_metrics=task.performance_metrics
            )
            
            self.history.add_entry(entry)

    def _check_auto_save(self) -> None:
        """检查是否需要自动保存状态"""
        current_time = datetime.now()
        if (current_time - self.last_save_time).total_seconds() >= self.auto_save_interval:
            self._auto_save_state()
            self.last_save_time = current_time

    def _auto_save_state(self) -> None:
        """自动保存状态"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.state_dir, f"task_state_{timestamp}.json")
            self.save_state(filepath)
            
            # 清理旧的自动保存文件
            self._cleanup_old_states()
            
        except Exception as e:
            detailed_logger.error(f"自动保存状态失败: {str(e)}")

    def _cleanup_old_states(self, keep_days: int = 7) -> None:
        """清理旧的状态文件
        
        Args:
            keep_days: 保留天数
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            
            for filename in os.listdir(self.state_dir):
                if filename.startswith("task_state_"):
                    filepath = os.path.join(self.state_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        detailed_logger.info(f"清理旧状态文件: {filename}")
                        
        except Exception as e:
            detailed_logger.error(f"清理旧状态文件失败: {str(e)}")

    @log_exception
    def remove_task(self, task_id: str) -> None:
        """移除任务
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.tasks:
            raise TaskError(f"任务不存在: {task_id}")
            
        task = self.tasks[task_id]
        
        # 从各个列表中移除
        if task in self.task_queue:
            self.task_queue.remove(task)
        if task in self.running_tasks:
            self.running_tasks.remove(task)
        if task in self.completed_tasks:
            self.completed_tasks.remove(task)
        if task in self.failed_tasks:
            self.failed_tasks.remove(task)
            
        # 移除任务规则
        self.rule_manager.remove_rule(task_id)
        
        del self.tasks[task_id]
        detailed_logger.info(f"移除任务: {task.name} ({task_id})")

    @log_exception
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Task]: 任务实例，不存在返回None
        """
        return self.tasks.get(task_id)

    @log_exception
    def execute_next_task(self) -> bool:
        """执行下一个任务
        
        Returns:
            bool: 是否成功开始执行任务
        """
        if not self.task_queue:
            return False
            
        # 获取下一个可执行的任务
        next_task = self._get_next_executable_task()
        if not next_task:
            return False
            
        # 从队列移到运行列表
        self.task_queue.remove(next_task)
        self.running_tasks.append(next_task)
        
        # 执行任务
        success = next_task.execute()
        
        # 更新任务状态
        self.running_tasks.remove(next_task)
        if success:
            self.completed_tasks.append(next_task)
        else:
            self.failed_tasks.append(next_task)
        
        return True

    @log_exception
    def execute_all_tasks(self) -> None:
        """执行所有任务"""
        while self.task_queue:
            self.execute_next_task()

    def _sort_queue(self) -> None:
        """对任务队列进行排序"""
        self.task_queue.sort(key=lambda x: (
            -x.priority.value,  # 优先级高的先执行
            len(x.dependencies),  # 依赖少的先执行
            x.task_id  # 相同情况下按ID排序
        ))

    def _get_next_executable_task(self) -> Optional[Task]:
        """获取下一个可执行的任务
        
        Returns:
            Optional[Task]: 可执行的任务，如果没有返回None
        """
        for task in self.task_queue:
            # 检查依赖是否都已完成
            dependencies_met = all(
                self.get_task(dep_id) in self.completed_tasks
                for dep_id in task.dependencies
            )
            
            # 检查任务规则是否满足
            rule = self.rule_manager.get_rule(task.task_id)
            if rule and not rule.evaluate({'task_manager': self}):
                continue
                
            if dependencies_met:
                return task
        return None

    @log_exception
    def save_state(self, filepath: str) -> None:
        """保存任务状态到文件
        
        Args:
            filepath: 保存路径
        """
        state = {
            'tasks': [task.to_dict() for task in self.tasks.values()],
            'queue': [task.task_id for task in self.task_queue],
            'running': [task.task_id for task in self.running_tasks],
            'completed': [task.task_id for task in self.completed_tasks],
            'failed': [task.task_id for task in self.failed_tasks],
            'last_save_time': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            detailed_logger.info(f"任务状态已保存: {filepath}")
        except Exception as e:
            raise TaskError(f"保存任务状态失败: {str(e)}")

    @log_exception
    def load_state(self, filepath: str) -> None:
        """从文件加载任务状态
        
        Args:
            filepath: 状态文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 清空当前状态
            self.tasks.clear()
            self.task_queue.clear()
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            
            # 重建任务
            for task_data in state['tasks']:
                task = self._create_task_from_dict(task_data)
                self.tasks[task.task_id] = task
                
                # 重新创建任务规则
                self.rule_manager.create_rule(task)
            
            # 恢复队列
            self.task_queue = [self.tasks[task_id] for task_id in state['queue']]
            self.running_tasks = [self.tasks[task_id] for task_id in state['running']]
            self.completed_tasks = [self.tasks[task_id] for task_id in state['completed']]
            self.failed_tasks = [self.tasks[task_id] for task_id in state['failed']]
            
            # 更新最后保存时间
            if 'last_save_time' in state:
                self.last_save_time = datetime.fromisoformat(state['last_save_time'])
            
            detailed_logger.info(f"任务状态已加载: {filepath}")
            
        except Exception as e:
            raise TaskError(f"加载任务状态失败: {str(e)}")

    def _create_task_from_dict(self, data: Dict) -> Task:
        """从字典创建任务实例
        
        Args:
            data: 任务数据字典
            
        Returns:
            Task: 任务实例
        """
        task = Task(
            task_id=data['task_id'],
            name=data['name'],
            priority=TaskPriority[data['priority']],
            dependencies=data['dependencies']
        )
        
        task.status = TaskStatus[data['status']]
        task.progress = data['progress']
        
        if data['start_time']:
            task.start_time = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            task.end_time = datetime.fromisoformat(data['end_time'])
            
        task.error_message = data['error_message']
        task.performance_metrics = data.get('performance_metrics', {})
        
        return task

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskStatus]: 任务状态，不存在返回None
        """
        task = self.get_task(task_id)
        return task.status if task else None

    def get_task_progress(self, task_id: str) -> Optional[float]:
        """获取任务进度
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[float]: 任务进度，不存在返回None
        """
        task = self.get_task(task_id)
        return task.progress if task else None

    def get_statistics(self) -> Dict:
        """获取任务统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total': len(self.tasks),
            'pending': len(self.task_queue),
            'running': len(self.running_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }
        
        # 添加规则统计
        stats['rules'] = self.rule_manager.get_statistics()
        
        # 添加历史统计
        stats['history'] = self.history.get_statistics()
        
        return stats
