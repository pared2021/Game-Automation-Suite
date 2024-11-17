from typing import Dict, Optional
from datetime import datetime

from game_automation.core.decision_maker import Action, Condition
from game_automation.core.task_manager import TaskManager, Task, TaskStatus, TaskPriority
from utils.logger import detailed_logger

class TaskActionHandler:
    """处理任务相关的Action和Condition"""

    def __init__(self, task_manager: TaskManager):
        """初始化任务动作处理器
        
        Args:
            task_manager: 任务管理器实例
        """
        self.task_manager = task_manager

    def evaluate_task_status(self, condition: Condition, context: Dict) -> bool:
        """评估任务状态条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - task_id: 任务ID
                      - status: 期望的任务状态
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'task_id' not in params or 'status' not in params:
            detailed_logger.error("任务状态条件缺少必需参数: task_id, status")
            return False

        task_id = params['task_id']
        expected_status = TaskStatus[params['status']]
        
        task_status = self.task_manager.get_task_status(task_id)
        return task_status == expected_status if task_status else False

    def evaluate_task_progress(self, condition: Condition, context: Dict) -> bool:
        """评估任务进度条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - task_id: 任务ID
                      - threshold: 进度阈值
                      - operator: 比较运算符 (greater_than/less_than)
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'task_id' not in params or 'threshold' not in params:
            detailed_logger.error("任务进度条件缺少必需参数: task_id, threshold")
            return False

        task_id = params['task_id']
        threshold = params['threshold']
        operator = params.get('operator', 'greater_than')

        progress = self.task_manager.get_task_progress(task_id)
        if progress is None:
            return False

        if operator == 'greater_than':
            return progress > threshold
        elif operator == 'less_than':
            return progress < threshold
        else:
            detailed_logger.error(f"不支持的运算符: {operator}")
            return False

    def evaluate_task_dependencies(self, condition: Condition, context: Dict) -> bool:
        """评估任务依赖条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - task_id: 任务ID
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足（依赖是否都已完成）
        """
        params = condition.parameters
        if 'task_id' not in params:
            detailed_logger.error("任务依赖条件缺少必需参数: task_id")
            return False

        task = self.task_manager.get_task(params['task_id'])
        if not task:
            return False

        # 检查所有依赖任务是否完成
        for dep_id in task.dependencies:
            dep_task = self.task_manager.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    async def handle_add_task(self, action: Action) -> bool:
        """处理添加任务动作
        
        Args:
            action: 动作对象，参数需包含:
                   - task_id: 任务ID
                   - name: 任务名称
                   - priority: 任务优先级
                   - dependencies: 可选的依赖任务ID列表
        
        Returns:
            bool: 是否成功
        """
        params = action.parameters
        required_params = ['task_id', 'name']
        if not all(param in params for param in required_params):
            detailed_logger.error(f"添加任务动作缺少必需参数: {required_params}")
            return False

        try:
            priority = TaskPriority[params.get('priority', 'NORMAL')]
            task = Task(
                task_id=params['task_id'],
                name=params['name'],
                priority=priority,
                dependencies=params.get('dependencies', [])
            )
            self.task_manager.add_task(task)
            return True
        except Exception as e:
            detailed_logger.error(f"添加任务失败: {str(e)}")
            return False

    async def handle_remove_task(self, action: Action) -> bool:
        """处理移除任务动作
        
        Args:
            action: 动作对象，参数需包含:
                   - task_id: 任务ID
        
        Returns:
            bool: 是否成功
        """
        params = action.parameters
        if 'task_id' not in params:
            detailed_logger.error("移除任务动作缺少必需参数: task_id")
            return False

        try:
            self.task_manager.remove_task(params['task_id'])
            return True
        except Exception as e:
            detailed_logger.error(f"移除任务失败: {str(e)}")
            return False

    async def handle_execute_task(self, action: Action) -> bool:
        """处理执行任务动作
        
        Args:
            action: 动作对象，参数需包含:
                   - task_id: 任务ID
        
        Returns:
            bool: 是否成功
        """
        params = action.parameters
        if 'task_id' not in params:
            detailed_logger.error("执行任务动作缺少必需参数: task_id")
            return False

        task = self.task_manager.get_task(params['task_id'])
        if not task:
            return False

        return task.execute()

    def register_handlers(self, decision_maker) -> None:
        """注册任务相关的动作和条件处理器
        
        Args:
            decision_maker: DecisionMaker实例
        """
        # 注册条件处理器
        decision_maker.register_condition_handler("task_status", self.evaluate_task_status)
        decision_maker.register_condition_handler("task_progress", self.evaluate_task_progress)
        decision_maker.register_condition_handler("task_dependencies", self.evaluate_task_dependencies)

        # 注册动作处理器
        decision_maker.register_action_handler("add_task", self.handle_add_task)
        decision_maker.register_action_handler("remove_task", self.handle_remove_task)
        decision_maker.register_action_handler("execute_task", self.handle_execute_task)
