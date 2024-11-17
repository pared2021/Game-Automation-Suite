from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from game_automation.core.decision_maker import Action, Condition

class BehaviorType(Enum):
    """行为类型枚举"""
    COMBAT = auto()      # 战斗行为
    RESOURCE = auto()    # 资源收集
    MOVEMENT = auto()    # 移动行为
    INTERACTION = auto() # 交互行为
    STRATEGY = auto()    # 策略行为

class BehaviorPriority(Enum):
    """行为优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

@dataclass
class BehaviorContext:
    """行为上下文数据"""
    behavior_type: BehaviorType
    game_type: str
    start_time: datetime
    parameters: Dict[str, Any]
    state: Dict[str, Any]

class GameBehavior:
    """游戏行为基类"""
    
    def __init__(self, 
                 behavior_id: str,
                 name: str,
                 behavior_type: BehaviorType,
                 priority: BehaviorPriority = BehaviorPriority.NORMAL):
        """初始化游戏行为
        
        Args:
            behavior_id: 行为ID
            name: 行为名称
            behavior_type: 行为类型
            priority: 行为优先级
        """
        self.behavior_id = behavior_id
        self.name = name
        self.behavior_type = behavior_type
        self.priority = priority
        
        self.conditions: List[Condition] = []
        self.actions: List[Action] = []
        self.conflicts: Set[str] = set()  # 冲突的行为ID
        self.dependencies: Set[str] = set()  # 依赖的行为ID
        
        self.context: Optional[BehaviorContext] = None
        self.last_execution: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0

    def add_condition(self, condition: Condition) -> None:
        """添加行为条件
        
        Args:
            condition: 条件对象
        """
        self.conditions.append(condition)

    def add_action(self, action: Action) -> None:
        """添加行为动作
        
        Args:
            action: 动作对象
        """
        self.actions.append(action)

    def add_conflict(self, behavior_id: str) -> None:
        """添加冲突行为
        
        Args:
            behavior_id: 冲突的行为ID
        """
        self.conflicts.add(behavior_id)

    def add_dependency(self, behavior_id: str) -> None:
        """添加依赖行为
        
        Args:
            behavior_id: 依赖的行为ID
        """
        self.dependencies.add(behavior_id)

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """检查行为是否可以执行
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: 是否可以执行
        """
        try:
            # 检查所有条件
            for condition in self.conditions:
                handler = context.get('condition_handlers', {}).get(condition.condition_type)
                if not handler:
                    detailed_logger.warning(f"未找到条件处理器: {condition.condition_type}")
                    return False
                    
                if not handler(condition, context):
                    return False
            
            return True
        except Exception as e:
            detailed_logger.error(f"检查行为执行条件失败: {str(e)}")
            return False

    async def execute(self, context: Dict[str, Any]) -> bool:
        """执行行为
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: 是否执行成功
        """
        try:
            self.last_execution = datetime.now()
            self.execution_count += 1
            
            # 创建行为上下文
            self.context = BehaviorContext(
                behavior_type=self.behavior_type,
                game_type=context.get('game_type', 'unknown'),
                start_time=datetime.now(),
                parameters=context.get('parameters', {}),
                state={}
            )
            
            # 执行所有动作
            for action in self.actions:
                handler = context.get('action_handlers', {}).get(action.action_type)
                if not handler:
                    detailed_logger.warning(f"未找到动作处理器: {action.action_type}")
                    continue
                    
                if not await handler(action):
                    return False
            
            self.success_count += 1
            return True
            
        except Exception as e:
            detailed_logger.error(f"执行行为失败: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取行为统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'behavior_id': self.behavior_id,
            'name': self.name,
            'type': self.behavior_type.name,
            'priority': self.priority.name,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / self.execution_count if self.execution_count > 0 else 0,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'conditions_count': len(self.conditions),
            'actions_count': len(self.actions),
            'conflicts_count': len(self.conflicts),
            'dependencies_count': len(self.dependencies)
        }

class BehaviorTemplate:
    """行为模板类"""
    
    def __init__(self, template_id: str, name: str):
        """初始化行为模板
        
        Args:
            template_id: 模板ID
            name: 模板名称
        """
        self.template_id = template_id
        self.name = name
        self.conditions: List[Condition] = []
        self.actions: List[Action] = []
        self.parameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def create_behavior(self, behavior_id: str, name: str,
                       behavior_type: BehaviorType,
                       priority: BehaviorPriority = BehaviorPriority.NORMAL,
                       parameters: Optional[Dict[str, Any]] = None) -> GameBehavior:
        """从模板创建行为实例
        
        Args:
            behavior_id: 行为ID
            name: 行为名称
            behavior_type: 行为类型
            priority: 行为优先级
            parameters: 行为参数
            
        Returns:
            GameBehavior: 行为实例
        """
        behavior = GameBehavior(behavior_id, name, behavior_type, priority)
        
        # 复制条件和动作
        for condition in self.conditions:
            behavior.add_condition(condition)
        for action in self.actions:
            behavior.add_action(action)
            
        # 更新参数
        if parameters:
            self.parameters.update(parameters)
            
        return behavior

class BehaviorManager:
    """行为管理器"""
    
    def __init__(self):
        """初始化行为管理器"""
        self.behaviors: Dict[str, GameBehavior] = {}
        self.templates: Dict[str, BehaviorTemplate] = {}
        self.active_behaviors: Set[str] = set()

    def add_behavior(self, behavior: GameBehavior) -> None:
        """添加行为
        
        Args:
            behavior: 行为实例
        """
        if behavior.behavior_id in self.behaviors:
            raise GameAutomationError(f"行为ID已存在: {behavior.behavior_id}")
            
        self.behaviors[behavior.behavior_id] = behavior
        detailed_logger.info(f"添加行为: {behavior.name} ({behavior.behavior_id})")

    def remove_behavior(self, behavior_id: str) -> None:
        """移除行为
        
        Args:
            behavior_id: 行为ID
        """
        if behavior_id in self.behaviors:
            del self.behaviors[behavior_id]
            self.active_behaviors.discard(behavior_id)
            detailed_logger.info(f"移除行为: {behavior_id}")

    def add_template(self, template: BehaviorTemplate) -> None:
        """添加行为模板
        
        Args:
            template: 行为模板
        """
        if template.template_id in self.templates:
            raise GameAutomationError(f"模板ID已存在: {template.template_id}")
            
        self.templates[template.template_id] = template
        detailed_logger.info(f"添加行为模板: {template.name} ({template.template_id})")

    def create_behavior_from_template(self, template_id: str,
                                    behavior_id: str,
                                    name: str,
                                    behavior_type: BehaviorType,
                                    priority: BehaviorPriority = BehaviorPriority.NORMAL,
                                    parameters: Optional[Dict[str, Any]] = None) -> GameBehavior:
        """从模板创建行为
        
        Args:
            template_id: 模板ID
            behavior_id: 行为ID
            name: 行为名称
            behavior_type: 行为类型
            priority: 行为优先级
            parameters: 行为参数
            
        Returns:
            GameBehavior: 行为实例
        """
        if template_id not in self.templates:
            raise GameAutomationError(f"模板不存在: {template_id}")
            
        template = self.templates[template_id]
        behavior = template.create_behavior(
            behavior_id=behavior_id,
            name=name,
            behavior_type=behavior_type,
            priority=priority,
            parameters=parameters
        )
        
        self.add_behavior(behavior)
        return behavior

    def get_behavior(self, behavior_id: str) -> Optional[GameBehavior]:
        """获取行为
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            Optional[GameBehavior]: 行为实例
        """
        return self.behaviors.get(behavior_id)

    def check_conflicts(self, behavior_id: str) -> List[str]:
        """检查行为冲突
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            List[str]: 冲突的行为ID列表
        """
        behavior = self.behaviors.get(behavior_id)
        if not behavior:
            return []
            
        conflicts = []
        for active_id in self.active_behaviors:
            if active_id in behavior.conflicts:
                conflicts.append(active_id)
                
        return conflicts

    def activate_behavior(self, behavior_id: str) -> bool:
        """激活行为
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            bool: 是否成功激活
        """
        behavior = self.behaviors.get(behavior_id)
        if not behavior:
            return False
            
        # 检查冲突
        conflicts = self.check_conflicts(behavior_id)
        if conflicts:
            detailed_logger.warning(f"行为存在冲突: {behavior_id} -> {conflicts}")
            return False
            
        self.active_behaviors.add(behavior_id)
        detailed_logger.info(f"激活行为: {behavior.name} ({behavior_id})")
        return True

    def deactivate_behavior(self, behavior_id: str) -> None:
        """停用行为
        
        Args:
            behavior_id: 行为ID
        """
        if behavior_id in self.active_behaviors:
            self.active_behaviors.remove(behavior_id)
            behavior = self.behaviors.get(behavior_id)
            if behavior:
                detailed_logger.info(f"停用行为: {behavior.name} ({behavior_id})")

    def get_active_behaviors(self) -> List[GameBehavior]:
        """获取所有激活的行为
        
        Returns:
            List[GameBehavior]: 激活的行为列表
        """
        return [
            self.behaviors[behavior_id]
            for behavior_id in self.active_behaviors
            if behavior_id in self.behaviors
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取行为统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_behaviors': len(self.behaviors),
            'active_behaviors': len(self.active_behaviors),
            'total_templates': len(self.templates),
            'behaviors': [behavior.get_statistics() for behavior in self.behaviors.values()]
        }
        
        # 计算总体统计
        total_executions = sum(behavior.execution_count for behavior in self.behaviors.values())
        total_successes = sum(behavior.success_count for behavior in self.behaviors.values())
        
        stats['total_executions'] = total_executions
        stats['total_successes'] = total_successes
        stats['overall_success_rate'] = (
            total_successes / total_executions if total_executions > 0 else 0
        )
        
        return stats
