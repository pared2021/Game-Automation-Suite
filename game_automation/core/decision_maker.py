from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import json
from datetime import datetime

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class RuleError(GameAutomationError):
    """规则相关错误"""
    pass

class Condition:
    """条件类，用于规则判断"""
    
    def __init__(self, 
                 condition_type: str,
                 parameters: Dict[str, Any],
                 operator: str = "equals"):
        """初始化条件
        
        Args:
            condition_type: 条件类型
            parameters: 条件参数
            operator: 比较运算符
        """
        self.type = condition_type
        self.parameters = parameters
        self.operator = operator

    def evaluate(self, context: Dict[str, Any], handler: Optional[Callable] = None) -> bool:
        """评估条件
        
        Args:
            context: 评估上下文
            handler: 可选的条件处理器
            
        Returns:
            bool: 条件是否满足
        """
        try:
            # 如果提供了处理器，使用处理器评估
            if handler:
                return handler(self, context)
                
            # 默认评估逻辑
            if self.type not in context:
                return False
                
            actual_value = context[self.type]
            expected_value = self.parameters.get("value")
            
            if self.operator == "equals":
                return actual_value == expected_value
            elif self.operator == "not_equals":
                return actual_value != expected_value
            elif self.operator == "greater_than":
                return actual_value > expected_value
            elif self.operator == "less_than":
                return actual_value < expected_value
            elif self.operator == "contains":
                return expected_value in actual_value
            elif self.operator == "not_contains":
                return expected_value not in actual_value
            else:
                detailed_logger.warning(f"未知的操作符: {self.operator}")
                return False
                
        except Exception as e:
            detailed_logger.error(f"条件评估失败: {str(e)}")
            return False

    @classmethod
    def from_dict(cls, data: Dict) -> 'Condition':
        """从字典创建条件实例
        
        Args:
            data: 条件数据字典
            
        Returns:
            Condition: 条件实例
        """
        return cls(
            condition_type=data["type"],
            parameters=data["parameters"],
            operator=data.get("operator", "equals")
        )

class Action:
    """动作类，表示规则触发的动作"""
    
    def __init__(self, 
                 action_type: str,
                 parameters: Dict[str, Any]):
        """初始化动作
        
        Args:
            action_type: 动作类型
            parameters: 动作参数
        """
        self.type = action_type
        self.parameters = parameters

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        """从字典创建动作实例
        
        Args:
            data: 动作数据字典
            
        Returns:
            Action: 动作实例
        """
        return cls(
            action_type=data["type"],
            parameters=data["parameters"]
        )

class Rule:
    """规则类，包含条件和动作"""
    
    def __init__(self, 
                 rule_id: str,
                 name: str,
                 conditions: List[Condition],
                 actions: List[Action],
                 priority: int = 0):
        """初始化规则
        
        Args:
            rule_id: 规则ID
            name: 规则名称
            conditions: 条件列表
            actions: 动作列表
            priority: 规则优先级
        """
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.last_triggered: Optional[datetime] = None

    def evaluate(self, context: Dict[str, Any], condition_handlers: Dict[str, Callable]) -> bool:
        """评估规则
        
        Args:
            context: 评估上下文
            condition_handlers: 条件处理器字典
            
        Returns:
            bool: 规则是否满足
        """
        return all(
            condition.evaluate(
                context,
                condition_handlers.get(condition.type)
            )
            for condition in self.conditions
        )

    @classmethod
    def from_dict(cls, data: Dict) -> 'Rule':
        """从字典创建规则实例
        
        Args:
            data: 规则数据字典
            
        Returns:
            Rule: 规则实例
        """
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            conditions=[Condition.from_dict(c) for c in data["conditions"]],
            actions=[Action.from_dict(a) for a in data["actions"]],
            priority=data.get("priority", 0)
        )

class BehaviorState(Enum):
    """行为状态枚举"""
    IDLE = auto()      # 空闲
    ACTIVE = auto()    # 活动
    BLOCKED = auto()   # 阻塞
    COMPLETED = auto() # 完成

class Behavior:
    """行为类，表示一组相关的规则和状态"""
    
    def __init__(self, 
                 behavior_id: str,
                 name: str,
                 rules: List[Rule],
                 priority: int = 0):
        """初始化行为
        
        Args:
            behavior_id: 行为ID
            name: 行为名称
            rules: 规则列表
            priority: 行为优先级
        """
        self.behavior_id = behavior_id
        self.name = name
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.priority = priority
        self.state = BehaviorState.IDLE
        self.context: Dict[str, Any] = {}

    def update(self, context: Dict[str, Any], condition_handlers: Dict[str, Callable]) -> Optional[List[Action]]:
        """更新行为状态并返回要执行的动作
        
        Args:
            context: 更新上下文
            condition_handlers: 条件处理器字典
            
        Returns:
            Optional[List[Action]]: 要执行的动作列表
        """
        self.context.update(context)
        
        if self.state == BehaviorState.BLOCKED:
            return None
            
        for rule in self.rules:
            if rule.evaluate(self.context, condition_handlers):
                rule.last_triggered = datetime.now()
                self.state = BehaviorState.ACTIVE
                return rule.actions
                
        self.state = BehaviorState.IDLE
        return None

class DecisionMaker:
    """决策器，管理行为和规则"""
    
    def __init__(self):
        """初始化决策器"""
        self.behaviors: Dict[str, Behavior] = {}
        self.global_context: Dict[str, Any] = {}
        self.action_handlers: Dict[str, Callable] = {}
        self.condition_handlers: Dict[str, Callable] = {}

    @log_exception
    def register_behavior(self, behavior: Behavior) -> None:
        """注册行为
        
        Args:
            behavior: 行为实例
        """
        if behavior.behavior_id in self.behaviors:
            raise RuleError(f"行为ID已存在: {behavior.behavior_id}")
            
        self.behaviors[behavior.behavior_id] = behavior
        detailed_logger.info(f"注册行为: {behavior.name}")

    @log_exception
    def register_action_handler(self, 
                              action_type: str,
                              handler: Callable[[Action], None]) -> None:
        """注册动作处理器
        
        Args:
            action_type: 动作类型
            handler: 处理函数
        """
        self.action_handlers[action_type] = handler
        detailed_logger.info(f"注册动作处理器: {action_type}")

    @log_exception
    def register_condition_handler(self,
                                 condition_type: str,
                                 handler: Callable[[Condition, Dict[str, Any]], bool]) -> None:
        """注册条件处理器
        
        Args:
            condition_type: 条件类型
            handler: 处理函数
        """
        self.condition_handlers[condition_type] = handler
        detailed_logger.info(f"注册条件处理器: {condition_type}")

    @log_exception
    def update(self, context_update: Dict[str, Any]) -> None:
        """更新决策器状态
        
        Args:
            context_update: 上下文更新
        """
        # 更新全局上下文
        self.global_context.update(context_update)
        
        # 按优先级排序行为
        sorted_behaviors = sorted(
            self.behaviors.values(),
            key=lambda b: b.priority,
            reverse=True
        )
        
        # 更新行为并执行动作
        for behavior in sorted_behaviors:
            actions = behavior.update(self.global_context, self.condition_handlers)
            if actions:
                self._execute_actions(actions)
                break  # 只执行最高优先级的行为

    def _execute_actions(self, actions: List[Action]) -> None:
        """执行动作列表
        
        Args:
            actions: 要执行的动作列表
        """
        for action in actions:
            handler = self.action_handlers.get(action.type)
            if handler:
                try:
                    handler(action)
                except Exception as e:
                    detailed_logger.error(f"执行动作失败: {str(e)}")
            else:
                detailed_logger.warning(f"未找到动作处理器: {action.type}")

    @log_exception
    def load_rules(self, filepath: str) -> None:
        """从文件加载规则
        
        Args:
            filepath: 规则文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for behavior_data in data.get("behaviors", []):
                rules = [
                    Rule.from_dict(rule_data)
                    for rule_data in behavior_data.get("rules", [])
                ]
                
                behavior = Behavior(
                    behavior_id=behavior_data["behavior_id"],
                    name=behavior_data["name"],
                    rules=rules,
                    priority=behavior_data.get("priority", 0)
                )
                
                self.register_behavior(behavior)
                
            detailed_logger.info(f"从文件加载规则: {filepath}")
            
        except Exception as e:
            raise RuleError(f"加载规则失败: {str(e)}")

    @log_exception
    def save_rules(self, filepath: str) -> None:
        """保存规则到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            behaviors_data = []
            for behavior in self.behaviors.values():
                rules_data = []
                for rule in behavior.rules:
                    rules_data.append({
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "conditions": [
                            {
                                "type": c.type,
                                "parameters": c.parameters,
                                "operator": c.operator
                            }
                            for c in rule.conditions
                        ],
                        "actions": [
                            {
                                "type": a.type,
                                "parameters": a.parameters
                            }
                            for a in rule.actions
                        ],
                        "priority": rule.priority
                    })
                
                behaviors_data.append({
                    "behavior_id": behavior.behavior_id,
                    "name": behavior.name,
                    "rules": rules_data,
                    "priority": behavior.priority
                })
            
            data = {"behaviors": behaviors_data}
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            detailed_logger.info(f"规则已保存: {filepath}")
            
        except Exception as e:
            raise RuleError(f"保存规则失败: {str(e)}")

    def get_behavior_state(self, behavior_id: str) -> Optional[BehaviorState]:
        """获取行为状态
        
        Args:
            behavior_id: 行为ID
            
        Returns:
            Optional[BehaviorState]: 行为状态
        """
        behavior = self.behaviors.get(behavior_id)
        return behavior.state if behavior else None

    def get_active_behaviors(self) -> List[Behavior]:
        """获取当前活动的行为列表
        
        Returns:
            List[Behavior]: 活动行为列表
        """
        return [
            behavior
            for behavior in self.behaviors.values()
            if behavior.state == BehaviorState.ACTIVE
        ]
