import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from game_automation.core.decision_maker import DecisionMaker, Action, Condition
from game_automation.game_types.game_type_manager import game_type_manager
from game_automation.game_types.game_strategy_action_handler import GameStrategyActionHandler

@pytest.fixture
def game_engine():
    """创建游戏引擎Mock"""
    return Mock()

@pytest.fixture
def game_strategy_handler(game_engine):
    """创建游戏策略处理器实例"""
    return GameStrategyActionHandler(game_engine)

@pytest.fixture
def decision_maker(game_strategy_handler):
    """创建决策器实例并注册策略处理器"""
    dm = DecisionMaker()
    game_strategy_handler.register_handlers(dm)
    return dm

def test_game_type_condition(game_strategy_handler):
    """测试游戏类型条件"""
    # 设置当前游戏类型为RPG
    game_type_manager.set_game_type('rpg')
    
    condition = Condition(
        condition_type="game_type",
        parameters={
            "game_type": "rpg"
        }
    )
    
    assert game_strategy_handler.evaluate_game_type(condition, {})

def test_action_available_condition(game_strategy_handler):
    """测试动作可用性条件"""
    # 设置当前游戏类型为RPG
    game_type_manager.set_game_type('rpg')
    
    condition = Condition(
        condition_type="action_available",
        parameters={
            "action_name": "use_skill"
        }
    )
    
    assert game_strategy_handler.evaluate_action_available(condition, {})

def test_cooldown_condition(game_strategy_handler):
    """测试冷却条件"""
    condition = Condition(
        condition_type="cooldown",
        parameters={
            "cooldown": 5.0
        }
    )
    
    # 初始状态应该通过
    assert game_strategy_handler.evaluate_cooldown(condition, {})
    
    # 设置上次动作时间
    game_strategy_handler.last_action_time = datetime.now() - timedelta(seconds=3)
    assert not game_strategy_handler.evaluate_cooldown(condition, {})
    
    game_strategy_handler.last_action_time = datetime.now() - timedelta(seconds=6)
    assert game_strategy_handler.evaluate_cooldown(condition, {})

@pytest.mark.asyncio
async def test_rpg_action(game_strategy_handler):
    """测试RPG动作"""
    game_type_manager.set_game_type('rpg')
    
    action = Action(
        action_type="rpg_action",
        parameters={
            "type": "use_skill",
            "skill_name": "fireball"
        }
    )
    
    assert await game_strategy_handler.handle_rpg_action(action)

@pytest.mark.asyncio
async def test_strategy_action(game_strategy_handler):
    """测试策略游戏动作"""
    game_type_manager.set_game_type('strategy')
    
    action = Action(
        action_type="strategy_action",
        parameters={
            "type": "build_structure",
            "structure_name": "barracks"
        }
    )
    
    assert await game_strategy_handler.handle_strategy_action(action)

@pytest.mark.asyncio
async def test_action_game_action(game_strategy_handler):
    """测试动作游戏动作"""
    game_type_manager.set_game_type('action')
    
    action = Action(
        action_type="action_game_action",
        parameters={
            "type": "dodge"
        }
    )
    
    assert await game_strategy_handler.handle_action_game_action(action)

def test_condition_handler_registration(decision_maker):
    """测试条件处理器注册"""
    assert "game_type" in decision_maker.condition_handlers
    assert "action_available" in decision_maker.condition_handlers
    assert "cooldown" in decision_maker.condition_handlers

def test_action_handler_registration(decision_maker):
    """测试动作处理器注册"""
    assert "rpg_action" in decision_maker.action_handlers
    assert "strategy_action" in decision_maker.action_handlers
    assert "action_game_action" in decision_maker.action_handlers

def test_rule_integration(decision_maker):
    """测试规则集成"""
    # 加载游戏策略规则文件
    decision_maker.load_rules("config/game_strategy_rules.json")
    
    # 验证行为已加载
    assert "rpg_combat" in decision_maker.behaviors
    assert "strategy_base_building" in decision_maker.behaviors
    assert "action_combat" in decision_maker.behaviors

def test_wrong_game_type_actions(game_strategy_handler):
    """测试错误游戏类型的动作处理"""
    # 设置为RPG类型但尝试执行策略游戏动作
    game_type_manager.set_game_type('rpg')
    
    action = Action(
        action_type="strategy_action",
        parameters={
            "type": "build_structure",
            "structure_name": "barracks"
        }
    )
    
    assert not game_strategy_handler.handle_strategy_action(action)

def test_invalid_action_type(game_strategy_handler):
    """测试无效的动作类型"""
    game_type_manager.set_game_type('rpg')
    
    action = Action(
        action_type="rpg_action",
        parameters={
            "type": "invalid_action"
        }
    )
    
    assert not game_strategy_handler.handle_rpg_action(action)

def test_missing_parameters(game_strategy_handler):
    """测试缺少参数的情况"""
    # 测试缺少游戏类型参数的条件
    condition = Condition(
        condition_type="game_type",
        parameters={}
    )
    assert not game_strategy_handler.evaluate_game_type(condition, {})
    
    # 测试缺少动作名称参数的条件
    condition = Condition(
        condition_type="action_available",
        parameters={}
    )
    assert not game_strategy_handler.evaluate_action_available(condition, {})
    
    # 测试缺少冷却时间参数的条件
    condition = Condition(
        condition_type="cooldown",
        parameters={}
    )
    assert not game_strategy_handler.evaluate_cooldown(condition, {})
