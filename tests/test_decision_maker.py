import pytest
import json
import os
from typing import List, Dict, Any
from datetime import datetime

from game_automation.core.decision_maker import (
    DecisionMaker, Behavior, Rule, Condition, Action,
    BehaviorState, RuleError
)

@pytest.fixture
def decision_maker():
    """创建决策器实例"""
    return DecisionMaker()

@pytest.fixture
def sample_condition():
    """创建示例条件"""
    return Condition(
        condition_type="scene_type",
        parameters={"value": "battle"},
        operator="equals"
    )

@pytest.fixture
def sample_action():
    """创建示例动作"""
    return Action(
        action_type="click",
        parameters={"x": 100, "y": 200}
    )

@pytest.fixture
def sample_rule(sample_condition, sample_action):
    """创建示例规则"""
    return Rule(
        rule_id="test_rule",
        name="Test Rule",
        conditions=[sample_condition],
        actions=[sample_action],
        priority=1
    )

@pytest.fixture
def sample_behavior(sample_rule):
    """创建示例行为"""
    return Behavior(
        behavior_id="test_behavior",
        name="Test Behavior",
        rules=[sample_rule],
        priority=1
    )

def test_condition_evaluation():
    """测试条件评估"""
    # 测试相等条件
    condition = Condition("value", {"value": 10}, "equals")
    assert condition.evaluate({"value": 10})
    assert not condition.evaluate({"value": 20})
    
    # 测试大于条件
    condition = Condition("value", {"value": 10}, "greater_than")
    assert condition.evaluate({"value": 20})
    assert not condition.evaluate({"value": 5})
    
    # 测试包含条件
    condition = Condition("list", {"value": "item"}, "contains")
    assert condition.evaluate({"list": ["item", "other"]})
    assert not condition.evaluate({"list": ["other"]})

def test_action_creation():
    """测试动作创建"""
    action_data = {
        "type": "click",
        "parameters": {"x": 100, "y": 200}
    }
    
    action = Action.from_dict(action_data)
    assert action.type == "click"
    assert action.parameters == {"x": 100, "y": 200}

def test_rule_evaluation(sample_rule):
    """测试规则评估"""
    # 测试条件满足
    context = {"scene_type": "battle"}
    assert sample_rule.evaluate(context)
    
    # 测试条件不满足
    context = {"scene_type": "menu"}
    assert not sample_rule.evaluate(context)

def test_behavior_update(sample_behavior):
    """测试行为更新"""
    # 测试条件满足时的行为
    context = {"scene_type": "battle"}
    actions = sample_behavior.update(context)
    
    assert actions is not None
    assert len(actions) == 1
    assert sample_behavior.state == BehaviorState.ACTIVE
    
    # 测试条件不满足时的行为
    context = {"scene_type": "menu"}
    actions = sample_behavior.update(context)
    
    assert actions is None
    assert sample_behavior.state == BehaviorState.IDLE

def test_decision_maker_registration(decision_maker, sample_behavior):
    """测试决策器注册功能"""
    # 测试行为注册
    decision_maker.register_behavior(sample_behavior)
    assert sample_behavior.behavior_id in decision_maker.behaviors
    
    # 测试重复注册
    with pytest.raises(RuleError):
        decision_maker.register_behavior(sample_behavior)
    
    # 测试动作处理器注册
    def handler(action: Action):
        pass
    
    decision_maker.register_action_handler("click", handler)
    assert "click" in decision_maker.action_handlers

def test_decision_maker_update(decision_maker, sample_behavior):
    """测试决策器更新"""
    # 注册行为和动作处理器
    decision_maker.register_behavior(sample_behavior)
    
    action_executed = False
    def handler(action: Action):
        nonlocal action_executed
        action_executed = True
    
    decision_maker.register_action_handler("click", handler)
    
    # 测试更新触发动作
    context = {"scene_type": "battle"}
    decision_maker.update(context)
    assert action_executed
    
    # 测试更新不触发动作
    action_executed = False
    context = {"scene_type": "menu"}
    decision_maker.update(context)
    assert not action_executed

def test_rule_persistence(decision_maker, sample_behavior, tmp_path):
    """测试规则持久化"""
    # 注册行为
    decision_maker.register_behavior(sample_behavior)
    
    # 保存规则
    rules_file = tmp_path / "rules.json"
    decision_maker.save_rules(str(rules_file))
    
    # 创建新的决策器并加载规则
    new_decision_maker = DecisionMaker()
    new_decision_maker.load_rules(str(rules_file))
    
    # 验证加载的规则
    assert len(new_decision_maker.behaviors) == 1
    loaded_behavior = new_decision_maker.behaviors[sample_behavior.behavior_id]
    assert loaded_behavior.name == sample_behavior.name
    assert len(loaded_behavior.rules) == len(sample_behavior.rules)

def test_behavior_priority(decision_maker):
    """测试行为优先级"""
    # 创建两个行为
    high_priority = Behavior(
        behavior_id="high",
        name="High Priority",
        rules=[
            Rule(
                rule_id="high_rule",
                name="High Rule",
                conditions=[
                    Condition("value", {"value": 10}, "equals")
                ],
                actions=[
                    Action("high_action", {})
                ]
            )
        ],
        priority=2
    )
    
    low_priority = Behavior(
        behavior_id="low",
        name="Low Priority",
        rules=[
            Rule(
                rule_id="low_rule",
                name="Low Rule",
                conditions=[
                    Condition("value", {"value": 10}, "equals")
                ],
                actions=[
                    Action("low_action", {})
                ]
            )
        ],
        priority=1
    )
    
    # 注册行为
    decision_maker.register_behavior(high_priority)
    decision_maker.register_behavior(low_priority)
    
    # 记录执行的动作
    executed_actions = []
    def handler(action: Action):
        executed_actions.append(action.type)
    
    decision_maker.register_action_handler("high_action", handler)
    decision_maker.register_action_handler("low_action", handler)
    
    # 更新决策器
    context = {"value": 10}
    decision_maker.update(context)
    
    # 验证只执行了高优先级的动作
    assert len(executed_actions) == 1
    assert executed_actions[0] == "high_action"

def test_behavior_state_tracking(decision_maker, sample_behavior):
    """测试行为状态跟踪"""
    decision_maker.register_behavior(sample_behavior)
    
    # 初始状态
    assert decision_maker.get_behavior_state(sample_behavior.behavior_id) == BehaviorState.IDLE
    
    # 触发行为
    context = {"scene_type": "battle"}
    decision_maker.update(context)
    assert decision_maker.get_behavior_state(sample_behavior.behavior_id) == BehaviorState.ACTIVE
    
    # 行为不满足条件
    context = {"scene_type": "menu"}
    decision_maker.update(context)
    assert decision_maker.get_behavior_state(sample_behavior.behavior_id) == BehaviorState.IDLE

def test_active_behaviors_tracking(decision_maker):
    """测试活动行为跟踪"""
    # 创建两个行为
    behavior1 = Behavior(
        behavior_id="b1",
        name="Behavior 1",
        rules=[
            Rule(
                rule_id="r1",
                name="Rule 1",
                conditions=[
                    Condition("value", {"value": 10}, "equals")
                ],
                actions=[Action("action1", {})]
            )
        ]
    )
    
    behavior2 = Behavior(
        behavior_id="b2",
        name="Behavior 2",
        rules=[
            Rule(
                rule_id="r2",
                name="Rule 2",
                conditions=[
                    Condition("value", {"value": 20}, "equals")
                ],
                actions=[Action("action2", {})]
            )
        ]
    )
    
    # 注册行为
    decision_maker.register_behavior(behavior1)
    decision_maker.register_behavior(behavior2)
    
    # 初始状态
    assert len(decision_maker.get_active_behaviors()) == 0
    
    # 触发第一个行为
    context = {"value": 10}
    decision_maker.update(context)
    active = decision_maker.get_active_behaviors()
    assert len(active) == 1
    assert active[0].behavior_id == "b1"

def test_cleanup():
    """清理测试资源"""
    # 如果测试过程中创建了任何临时文件，在这里清理
    pass
