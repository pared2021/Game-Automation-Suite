from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from .decision_maker import Rule, Condition, Action
from .task_manager import Task, TaskStatus

@dataclass
class TaskCondition:
    """任务条件数据类"""
    condition_type: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class TaskAction:
    """任务动作数据类"""
    action_type: str
    parameters: Dict[str, Any]
    description: str

class TaskRule:
    """任务规则类，用于将任务转换为决策系统的规则"""

    def __init__(self, task: Task):
        """初始化任务规则
        
        Args:
            task: 任务实例
        """
        self.task = task
        self.conditions: List[TaskCondition] = []
        self.actions: List[TaskAction] = []
        self.priority = task.priority.value
        self.last_evaluation_time: Optional[datetime] = None
        self.evaluation_count = 0
        self.success_count = 0

    def add_condition(self, condition: TaskCondition) -> None:
        """添加任务条件
        
        Args:
            condition: 任务条件
        """
        self.conditions.append(condition)
        detailed_logger.info(f"添加任务条件: {condition.description}")

    def add_action(self, action: TaskAction) -> None:
        """添加任务动作
        
        Args:
            action: 任务动作
        """
        self.actions.append(action)
        detailed_logger.info(f"添加任务动作: {action.description}")

    def to_rule(self) -> Rule:
        """转换为决策系统规则
        
        Returns:
            Rule: 决策系统规则实例
        """
        # 创建规则条件
        rule_conditions = [
            Condition(
                condition_type=cond.condition_type,
                parameters=cond.parameters
            )
            for cond in self.conditions
        ]
        
        # 创建规则动作
        rule_actions = [
            Action(
                action_type=act.action_type,
                parameters=act.parameters
            )
            for act in self.actions
        ]
        
        # 添加任务状态检查条件
        rule_conditions.append(
            Condition(
                condition_type="task_status",
                parameters={
                    "task_id": self.task.task_id,
                    "expected_status": TaskStatus.PENDING.name
                }
            )
        )
        
        # 添加依赖检查条件
        if self.task.dependencies:
            rule_conditions.append(
                Condition(
                    condition_type="task_dependencies",
                    parameters={
                        "task_id": self.task.task_id,
                        "dependencies": self.task.dependencies
                    }
                )
            )
        
        return Rule(
            rule_id=f"task_rule_{self.task.task_id}",
            name=f"Task Rule - {self.task.name}",
            conditions=rule_conditions,
            actions=rule_actions,
            priority=self.priority
        )

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """评估任务规则
        
        Args:
            context: 评估上下文
            
        Returns:
            bool: 是否满足执行条件
        """
        try:
            self.last_evaluation_time = datetime.now()
            self.evaluation_count += 1
            
            # 检查任务状态
            if self.task.status != TaskStatus.PENDING:
                return False
            
            # 检查任务依赖
            if not self._check_dependencies(context):
                return False
            
            # 评估所有条件
            for condition in self.conditions:
                if not self._evaluate_condition(condition, context):
                    return False
            
            self.success_count += 1
            return True
            
        except Exception as e:
            detailed_logger.error(f"任务规则评估失败: {str(e)}")
            return False

    def _check_dependencies(self, context: Dict[str, Any]) -> bool:
        """检查任务依赖
        
        Args:
            context: 评估上下文
            
        Returns:
            bool: 依赖是否满足
        """
        if not self.task.dependencies:
            return True
            
        task_manager = context.get('task_manager')
        if not task_manager:
            return False
            
        return all(
            task_manager.get_task_status(dep_id) == TaskStatus.COMPLETED
            for dep_id in self.task.dependencies
        )

    def _evaluate_condition(self, condition: TaskCondition, context: Dict[str, Any]) -> bool:
        """评估单个条件
        
        Args:
            condition: 任务条件
            context: 评估上下文
            
        Returns:
            bool: 条件是否满足
        """
        try:
            # 获取条件处理器
            condition_handler = context.get('condition_handlers', {}).get(condition.condition_type)
            if not condition_handler:
                detailed_logger.warning(f"未找到条件处理器: {condition.condition_type}")
                return False
            
            # 创建条件对象
            rule_condition = Condition(
                condition_type=condition.condition_type,
                parameters=condition.parameters
            )
            
            # 评估条件
            return condition_handler(rule_condition, context)
            
        except Exception as e:
            detailed_logger.error(f"条件评估失败: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'task_id': self.task.task_id,
            'evaluation_count': self.evaluation_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / self.evaluation_count if self.evaluation_count > 0 else 0,
            'last_evaluation': self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            'conditions_count': len(self.conditions),
            'actions_count': len(self.actions)
        }

class TaskRuleManager:
    """任务规则管理器"""

    def __init__(self):
        """初始化任务规则管理器"""
        self.task_rules: Dict[str, TaskRule] = {}

    def create_rule(self, task: Task) -> TaskRule:
        """创建任务规则
        
        Args:
            task: 任务实例
            
        Returns:
            TaskRule: 任务规则实例
        """
        if task.task_id in self.task_rules:
            raise GameAutomationError(f"任务规则已存在: {task.task_id}")
            
        rule = TaskRule(task)
        self.task_rules[task.task_id] = rule
        detailed_logger.info(f"创建任务规则: {task.name} ({task.task_id})")
        return rule

    def get_rule(self, task_id: str) -> Optional[TaskRule]:
        """获取任务规则
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskRule]: 任务规则实例，不存在返回None
        """
        return self.task_rules.get(task_id)

    def remove_rule(self, task_id: str) -> None:
        """移除任务规则
        
        Args:
            task_id: 任务ID
        """
        if task_id in self.task_rules:
            del self.task_rules[task_id]
            detailed_logger.info(f"移除任务规则: {task_id}")

    def get_all_rules(self) -> List[Rule]:
        """获取所有决策系统规则
        
        Returns:
            List[Rule]: 规则列表
        """
        return [rule.to_rule() for rule in self.task_rules.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_rules': len(self.task_rules),
            'rules': [rule.get_statistics() for rule in self.task_rules.values()]
        }
        
        # 计算总体统计
        total_evaluations = sum(rule.evaluation_count for rule in self.task_rules.values())
        total_successes = sum(rule.success_count for rule in self.task_rules.values())
        
        stats['total_evaluations'] = total_evaluations
        stats['total_successes'] = total_successes
        stats['overall_success_rate'] = (
            total_successes / total_evaluations if total_evaluations > 0 else 0
        )
        
        return stats
