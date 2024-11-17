import pytest
import numpy as np
from datetime import datetime

from game_automation.core.decision_maker import DecisionMaker, Action, Condition
from game_automation.scene_understanding.scene_analyzer import SceneAnalyzer
from game_automation.scene_understanding.scene_action_handler import SceneActionHandler

@pytest.fixture
def scene_analyzer():
    """创建场景分析器实例"""
    return SceneAnalyzer()

@pytest.fixture
def scene_action_handler(scene_analyzer):
    """创建场景动作处理器实例"""
    return SceneActionHandler(scene_analyzer)

@pytest.fixture
def decision_maker(scene_action_handler):
    """创建决策器实例并注册场景处理器"""
    dm = DecisionMaker()
    scene_action_handler.register_handlers(dm)
    return dm

@pytest.fixture
def sample_screenshot():
    """创建示例截图"""
    return np.zeros((100, 100, 3), dtype=np.uint8)  # 创建黑色图像作为测试用例

def test_scene_type_condition(scene_action_handler, sample_screenshot):
    """测试场景类型条件"""
    condition = Condition(
        condition_type="scene_type",
        parameters={
            "scene_type": "battle"
        }
    )
    
    context = {"screenshot": sample_screenshot}
    result = scene_action_handler.evaluate_scene_type(condition, context)
    assert isinstance(result, bool)

def test_scene_changed_condition(scene_action_handler, sample_screenshot):
    """测试场景切换条件"""
    condition = Condition(
        condition_type="scene_changed",
        parameters={}
    )
    
    context = {"screenshot": sample_screenshot}
    result = scene_action_handler.evaluate_scene_changed(condition, context)
    assert isinstance(result, bool)

def test_scene_brightness_condition(scene_action_handler, sample_screenshot):
    """测试场景亮度条件"""
    condition = Condition(
        condition_type="scene_brightness",
        parameters={
            "threshold": 128,
            "operator": "less_than"
        }
    )
    
    context = {"screenshot": sample_screenshot}
    result = scene_action_handler.evaluate_scene_brightness(condition, context)
    assert isinstance(result, bool)

def test_scene_complexity_condition(scene_action_handler, sample_screenshot):
    """测试场景复杂度条件"""
    condition = Condition(
        condition_type="scene_complexity",
        parameters={
            "threshold": 50,
            "operator": "greater_than"
        }
    )
    
    context = {"screenshot": sample_screenshot}
    result = scene_action_handler.evaluate_scene_complexity(condition, context)
    assert isinstance(result, bool)

def test_save_template_action(scene_action_handler, sample_screenshot):
    """测试保存模板动作"""
    action = Action(
        action_type="save_template",
        parameters={
            "template_name": "test_template",
            "screenshot": sample_screenshot
        }
    )
    
    result = scene_action_handler.handle_save_template(action)
    assert isinstance(result, bool)

def test_condition_handler_registration(decision_maker):
    """测试条件处理器注册"""
    assert "scene_type" in decision_maker.condition_handlers
    assert "scene_changed" in decision_maker.condition_handlers
    assert "scene_brightness" in decision_maker.condition_handlers
    assert "scene_complexity" in decision_maker.condition_handlers

def test_action_handler_registration(decision_maker):
    """测试动作处理器注册"""
    assert "save_template" in decision_maker.action_handlers

def test_rule_integration(decision_maker):
    """测试规则集成"""
    # 加载场景规则文件
    decision_maker.load_rules("config/scene_rules.json")
    
    # 验证行为已加载
    assert "scene_monitoring" in decision_maker.behaviors
    assert "scene_type_reaction" in decision_maker.behaviors
    assert "scene_state_monitoring" in decision_maker.behaviors

def test_scene_analysis_caching(scene_action_handler, sample_screenshot):
    """测试场景分析结果缓存"""
    condition = Condition(
        condition_type="scene_type",
        parameters={
            "scene_type": "battle"
        }
    )
    
    context = {"screenshot": sample_screenshot}
    
    # 第一次分析
    scene_action_handler.evaluate_scene_type(condition, context)
    first_analysis_time = scene_action_handler.last_analysis_time
    
    # 第二次分析
    scene_action_handler.evaluate_scene_type(condition, context)
    second_analysis_time = scene_action_handler.last_analysis_time
    
    # 验证分析结果被更新
    assert second_analysis_time > first_analysis_time

def test_invalid_parameters(scene_action_handler):
    """测试无效参数处理"""
    # 测试缺少必需参数的场景类型条件
    condition = Condition(
        condition_type="scene_type",
        parameters={}
    )
    context = {}
    assert not scene_action_handler.evaluate_scene_type(condition, context)
    
    # 测试缺少必需参数的保存模板动作
    action = Action(
        action_type="save_template",
        parameters={}
    )
    assert not scene_action_handler.handle_save_template(action)
