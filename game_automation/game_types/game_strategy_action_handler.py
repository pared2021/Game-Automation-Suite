from typing import Dict, Optional
from datetime import datetime

from game_automation.core.decision_maker import Action, Condition
from game_automation.game_types.game_type_manager import game_type_manager
from utils.logger import detailed_logger

class GameStrategyActionHandler:
    """处理游戏策略相关的Action和Condition"""

    def __init__(self, game_engine):
        """初始化游戏策略动作处理器
        
        Args:
            game_engine: 游戏引擎实例
        """
        self.game_engine = game_engine
        self.last_action_time = None

    def evaluate_game_type(self, condition: Condition, context: Dict) -> bool:
        """评估游戏类型条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - game_type: 期望的游戏类型 (rpg/strategy/action)
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'game_type' not in params:
            detailed_logger.error("游戏类型条件缺少必需参数: game_type")
            return False

        current_type = game_type_manager.get_current_game_type()
        if not current_type:
            return False

        return current_type.__class__.__name__.lower().replace('game', '') == params['game_type']

    def evaluate_action_available(self, condition: Condition, context: Dict) -> bool:
        """评估游戏动作是否可用
        
        Args:
            condition: 条件对象，参数需包含:
                      - action_name: 动作名称
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'action_name' not in params:
            detailed_logger.error("动作可用性条件缺少必需参数: action_name")
            return False

        current_type = game_type_manager.get_current_game_type()
        if not current_type:
            return False

        return params['action_name'] in current_type.get_game_specific_actions()

    def evaluate_cooldown(self, condition: Condition, context: Dict) -> bool:
        """评估动作冷却条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - cooldown: 冷却时间（秒）
            context: 上下文数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'cooldown' not in params:
            detailed_logger.error("冷却条件缺少必需参数: cooldown")
            return False

        if not self.last_action_time:
            return True

        elapsed = (datetime.now() - self.last_action_time).total_seconds()
        return elapsed >= params['cooldown']

    async def handle_rpg_action(self, action: Action) -> bool:
        """处理RPG游戏动作
        
        Args:
            action: 动作对象，参数因具体动作类型而异
        
        Returns:
            bool: 是否成功
        """
        current_type = game_type_manager.get_current_game_type()
        if not isinstance(current_type, game_type_manager.game_types['rpg'].__class__):
            detailed_logger.error("当前不是RPG游戏类型")
            return False

        try:
            action_type = action.parameters.get('type')
            if action_type == 'use_skill':
                await current_type.use_skill(self.game_engine, action.parameters.get('skill_name'))
            elif action_type == 'open_inventory':
                await current_type.open_inventory(self.game_engine)
            elif action_type == 'talk_to_npc':
                await current_type.talk_to_npc(self.game_engine, action.parameters.get('npc_name'))
            else:
                detailed_logger.error(f"未知的RPG动作类型: {action_type}")
                return False

            self.last_action_time = datetime.now()
            return True
        except Exception as e:
            detailed_logger.error(f"执行RPG动作失败: {str(e)}")
            return False

    async def handle_strategy_action(self, action: Action) -> bool:
        """处理策略游戏动作
        
        Args:
            action: 动作对象，参数因具体动作类型而异
        
        Returns:
            bool: 是否成功
        """
        current_type = game_type_manager.get_current_game_type()
        if not isinstance(current_type, game_type_manager.game_types['strategy'].__class__):
            detailed_logger.error("当前不是策略游戏类型")
            return False

        try:
            action_type = action.parameters.get('type')
            if action_type == 'build_structure':
                await current_type.build_structure(self.game_engine, action.parameters.get('structure_name'))
            elif action_type == 'train_unit':
                await current_type.train_unit(self.game_engine, action.parameters.get('unit_name'))
            elif action_type == 'research_technology':
                await current_type.research_technology(self.game_engine, action.parameters.get('tech_name'))
            else:
                detailed_logger.error(f"未知的策略游戏动作类型: {action_type}")
                return False

            self.last_action_time = datetime.now()
            return True
        except Exception as e:
            detailed_logger.error(f"执行策略游戏动作失败: {str(e)}")
            return False

    async def handle_action_game_action(self, action: Action) -> bool:
        """处理动作游戏动作
        
        Args:
            action: 动作对象，参数因具体动作类型而异
        
        Returns:
            bool: 是否成功
        """
        current_type = game_type_manager.get_current_game_type()
        if not isinstance(current_type, game_type_manager.game_types['action'].__class__):
            detailed_logger.error("当前不是动作游戏类型")
            return False

        try:
            action_type = action.parameters.get('type')
            if action_type == 'jump':
                await current_type.jump(self.game_engine)
            elif action_type == 'dodge':
                await current_type.dodge(self.game_engine)
            elif action_type == 'use_special_move':
                await current_type.use_special_move(self.game_engine, action.parameters.get('move_name'))
            else:
                detailed_logger.error(f"未知的动作游戏动作类型: {action_type}")
                return False

            self.last_action_time = datetime.now()
            return True
        except Exception as e:
            detailed_logger.error(f"执行动作游戏动作失败: {str(e)}")
            return False

    def register_handlers(self, decision_maker) -> None:
        """注册游戏策略相关的动作和条件处理器
        
        Args:
            decision_maker: DecisionMaker实例
        """
        # 注册条件处理器
        decision_maker.register_condition_handler("game_type", self.evaluate_game_type)
        decision_maker.register_condition_handler("action_available", self.evaluate_action_available)
        decision_maker.register_condition_handler("cooldown", self.evaluate_cooldown)

        # 注册动作处理器
        decision_maker.register_action_handler("rpg_action", self.handle_rpg_action)
        decision_maker.register_action_handler("strategy_action", self.handle_strategy_action)
        decision_maker.register_action_handler("action_game_action", self.handle_action_game_action)
