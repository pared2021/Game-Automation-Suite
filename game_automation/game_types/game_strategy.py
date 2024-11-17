from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from game_automation.core.decision_maker import Rule, Action, Condition
from .game_behavior import GameBehavior, BehaviorManager, BehaviorType

class StrategyType(Enum):
    """策略类型枚举"""
    COMBAT = auto()      # 战斗策略
    RESOURCE = auto()    # 资源策略
    EXPLORATION = auto() # 探索策略
    DEVELOPMENT = auto() # 发展策略
    SOCIAL = auto()      # 社交策略

class StrategyState(Enum):
    """策略状态枚举"""
    INACTIVE = auto()    # 未激活
    ACTIVE = auto()      # 已激活
    SUSPENDED = auto()   # 已暂停
    COMPLETED = auto()   # 已完成
    FAILED = auto()      # 已失败

@dataclass
class StrategyMetrics:
    """策略性能指标"""
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    resource_usage: Dict[str, float] = None
    efficiency_score: float = 0.0

class GameStrategy:
    """游戏策略类"""
    
    def __init__(self, 
                 strategy_id: str,
                 name: str,
                 strategy_type: StrategyType,
                 behavior_manager: BehaviorManager):
        """初始化游戏策略
        
        Args:
            strategy_id: 策略ID
            name: 策略名称
            strategy_type: 策略类型
            behavior_manager: 行为管理器实例
        """
        self.strategy_id = strategy_id
        self.name = name
        self.strategy_type = strategy_type
        self.behavior_manager = behavior_manager
        
        self.state = StrategyState.INACTIVE
        self.behaviors: List[str] = []  # 行为ID列表
        self.rules: List[Rule] = []
        self.conditions: List[Condition] = []
        self.metrics = StrategyMetrics()
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.last_execution: Optional[datetime] = None
        
        self.parameters: Dict[str, Any] = {}
        self.optimization_settings: Dict[str, Any] = {}

    def add_behavior(self, behavior_id: str) -> None:
        """添加策略行为
        
        Args:
            behavior_id: 行为ID
        """
        if behavior_id not in self.behavior_manager.behaviors:
            raise GameAutomationError(f"行为不存在: {behavior_id}")
            
        if behavior_id not in self.behaviors:
            self.behaviors.append(behavior_id)
            detailed_logger.info(f"添加策略行为: {behavior_id} -> {self.name}")

    def add_rule(self, rule: Rule) -> None:
        """添加策略规则
        
        Args:
            rule: 规则对象
        """
        self.rules.append(rule)
        detailed_logger.info(f"添加策略规则: {rule.name} -> {self.name}")

    def add_condition(self, condition: Condition) -> None:
        """添加策略条件
        
        Args:
            condition: 条件对象
        """
        self.conditions.append(condition)
        detailed_logger.info(f"添加策略条件: {condition.condition_type} -> {self.name}")

    async def activate(self, context: Dict[str, Any]) -> bool:
        """激活策略
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: 是否成功激活
        """
        try:
            # 检查条件
            if not self._check_conditions(context):
                return False
                
            # 激活所有行为
            for behavior_id in self.behaviors:
                if not self.behavior_manager.activate_behavior(behavior_id):
                    return False
            
            self.state = StrategyState.ACTIVE
            self.start_time = datetime.now()
            detailed_logger.info(f"激活策略: {self.name}")
            return True
            
        except Exception as e:
            detailed_logger.error(f"激活策略失败: {str(e)}")
            return False

    def suspend(self) -> None:
        """暂停策略"""
        if self.state == StrategyState.ACTIVE:
            # 停用所有行为
            for behavior_id in self.behaviors:
                self.behavior_manager.deactivate_behavior(behavior_id)
                
            self.state = StrategyState.SUSPENDED
            detailed_logger.info(f"暂停策略: {self.name}")

    def resume(self) -> None:
        """恢复策略"""
        if self.state == StrategyState.SUSPENDED:
            self.state = StrategyState.ACTIVE
            detailed_logger.info(f"恢复策略: {self.name}")

    def complete(self) -> None:
        """完成策略"""
        self.state = StrategyState.COMPLETED
        self.end_time = datetime.now()
        
        # 停用所有行为
        for behavior_id in self.behaviors:
            self.behavior_manager.deactivate_behavior(behavior_id)
            
        detailed_logger.info(f"完成策略: {self.name}")

    def fail(self, error_message: str) -> None:
        """标记策略失败
        
        Args:
            error_message: 错误信息
        """
        self.state = StrategyState.FAILED
        self.end_time = datetime.now()
        
        # 停用所有行为
        for behavior_id in self.behaviors:
            self.behavior_manager.deactivate_behavior(behavior_id)
            
        detailed_logger.error(f"策略失败: {self.name} - {error_message}")

    def _check_conditions(self, context: Dict[str, Any]) -> bool:
        """检查策略条件
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: 条件是否满足
        """
        try:
            for condition in self.conditions:
                handler = context.get('condition_handlers', {}).get(condition.condition_type)
                if not handler:
                    detailed_logger.warning(f"未找到条件处理器: {condition.condition_type}")
                    return False
                    
                if not handler(condition, context):
                    return False
            
            return True
        except Exception as e:
            detailed_logger.error(f"检查策略条件失败: {str(e)}")
            return False

    def update_metrics(self, execution_time: float, success: bool,
                      resource_usage: Optional[Dict[str, float]] = None) -> None:
        """更新策略指标
        
        Args:
            execution_time: 执行时间
            success: 是否成功
            resource_usage: 资源使用情况
        """
        self.metrics.execution_count += 1
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1
            
        self.metrics.total_duration += execution_time
        self.metrics.average_duration = (
            self.metrics.total_duration / self.metrics.execution_count
        )
        
        if resource_usage:
            if not self.metrics.resource_usage:
                self.metrics.resource_usage = {}
            for resource, usage in resource_usage.items():
                current = self.metrics.resource_usage.get(resource, 0)
                self.metrics.resource_usage[resource] = current + usage
                
        # 计算效率分数
        success_rate = self.metrics.success_count / self.metrics.execution_count
        time_efficiency = 1.0 / (self.metrics.average_duration + 1)  # 避免除零
        self.metrics.efficiency_score = (success_rate + time_efficiency) / 2

    def optimize(self) -> None:
        """优化策略配置"""
        if not self.optimization_settings:
            return
            
        try:
            # 基于性能指标调整参数
            if self.metrics.efficiency_score < 0.5:
                # 效率较低，调整参数
                if 'resource_threshold' in self.parameters:
                    self.parameters['resource_threshold'] *= 1.2
                if 'retry_count' in self.parameters:
                    self.parameters['retry_count'] += 1
                    
            elif self.metrics.efficiency_score > 0.8:
                # 效率较高，优化资源使用
                if 'resource_threshold' in self.parameters:
                    self.parameters['resource_threshold'] *= 0.9
                    
            detailed_logger.info(f"优化策略参数: {self.name}")
            
        except Exception as e:
            detailed_logger.error(f"优化策略失败: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'type': self.strategy_type.name,
            'state': self.state.name,
            'behaviors_count': len(self.behaviors),
            'rules_count': len(self.rules),
            'conditions_count': len(self.conditions),
            'metrics': {
                'execution_count': self.metrics.execution_count,
                'success_count': self.metrics.success_count,
                'failure_count': self.metrics.failure_count,
                'success_rate': (
                    self.metrics.success_count / self.metrics.execution_count
                    if self.metrics.execution_count > 0 else 0
                ),
                'average_duration': self.metrics.average_duration,
                'efficiency_score': self.metrics.efficiency_score,
                'resource_usage': self.metrics.resource_usage
            },
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None
        }

class StrategyManager:
    """策略管理器"""
    
    def __init__(self, behavior_manager: BehaviorManager):
        """初始化策略管理器
        
        Args:
            behavior_manager: 行为管理器实例
        """
        self.behavior_manager = behavior_manager
        self.strategies: Dict[str, GameStrategy] = {}
        self.active_strategies: Set[str] = set()
        
        # 优化设置
        self.optimization_interval = 300  # 5分钟优化一次
        self.last_optimization = datetime.now()

    def add_strategy(self, strategy: GameStrategy) -> None:
        """添加策略
        
        Args:
            strategy: 策略实例
        """
        if strategy.strategy_id in self.strategies:
            raise GameAutomationError(f"策略ID已存在: {strategy.strategy_id}")
            
        self.strategies[strategy.strategy_id] = strategy
        detailed_logger.info(f"添加策略: {strategy.name} ({strategy.strategy_id})")

    def remove_strategy(self, strategy_id: str) -> None:
        """移除策略
        
        Args:
            strategy_id: 策略ID
        """
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            
            # 如果策略处于活动状态，先停用
            if strategy_id in self.active_strategies:
                strategy.suspend()
                self.active_strategies.remove(strategy_id)
            
            del self.strategies[strategy_id]
            detailed_logger.info(f"移除策略: {strategy.name} ({strategy_id})")

    def get_strategy(self, strategy_id: str) -> Optional[GameStrategy]:
        """获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[GameStrategy]: 策略实例
        """
        return self.strategies.get(strategy_id)

    async def activate_strategy(self, strategy_id: str,
                              context: Dict[str, Any]) -> bool:
        """激活策略
        
        Args:
            strategy_id: 策略ID
            context: 执行上下文
            
        Returns:
            bool: 是否成功激活
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return False
            
        if await strategy.activate(context):
            self.active_strategies.add(strategy_id)
            return True
            
        return False

    def suspend_strategy(self, strategy_id: str) -> None:
        """暂停策略
        
        Args:
            strategy_id: 策略ID
        """
        strategy = self.strategies.get(strategy_id)
        if strategy:
            strategy.suspend()
            self.active_strategies.discard(strategy_id)

    def resume_strategy(self, strategy_id: str) -> None:
        """恢复策略
        
        Args:
            strategy_id: 策略ID
        """
        strategy = self.strategies.get(strategy_id)
        if strategy:
            strategy.resume()
            self.active_strategies.add(strategy_id)

    def check_conflicts(self, strategy_id: str) -> List[str]:
        """检查策略冲突
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            List[str]: 冲突的策略ID列表
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return []
            
        conflicts = []
        for active_id in self.active_strategies:
            active_strategy = self.strategies.get(active_id)
            if not active_strategy:
                continue
                
            # 检查行为冲突
            for behavior_id in strategy.behaviors:
                if self.behavior_manager.check_conflicts(behavior_id):
                    conflicts.append(active_id)
                    break
                    
        return conflicts

    def optimize_strategies(self) -> None:
        """优化所有策略"""
        current_time = datetime.now()
        if (current_time - self.last_optimization).total_seconds() < self.optimization_interval:
            return
            
        try:
            for strategy in self.strategies.values():
                strategy.optimize()
                
            self.last_optimization = current_time
            detailed_logger.info("完成策略优化")
            
        except Exception as e:
            detailed_logger.error(f"策略优化失败: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'strategies': [
                strategy.get_statistics()
                for strategy in self.strategies.values()
            ]
        }
        
        # 计算总体统计
        total_executions = sum(
            strategy.metrics.execution_count
            for strategy in self.strategies.values()
        )
        total_successes = sum(
            strategy.metrics.success_count
            for strategy in self.strategies.values()
        )
        
        stats['total_executions'] = total_executions
        stats['total_successes'] = total_successes
        stats['overall_success_rate'] = (
            total_successes / total_executions if total_executions > 0 else 0
        )
        
        return stats
