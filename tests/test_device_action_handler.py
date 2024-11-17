import pytest
from game_automation.core.decision_maker import DecisionMaker, Action, Condition
from game_automation.device.device_manager import DeviceManager
from game_automation.device.device_action_handler import DeviceActionHandler

@pytest.fixture
def device_manager():
    """创建设备管理器实例"""
    return DeviceManager()

@pytest.fixture
def device_action_handler(device_manager):
    """创建设备动作处理器实例"""
    return DeviceActionHandler(device_manager)

@pytest.fixture
def decision_maker(device_action_handler):
    """创建决策器实例并注册设备处理器"""
    dm = DecisionMaker()
    device_action_handler.register_handlers(dm)
    return dm

def test_device_click_action(device_action_handler):
    """测试设备点击动作"""
    action = Action(
        action_type="device_click",
        parameters={
            "x": 100,
            "y": 200,
            "timeout": 5.0,
            "retry": 3
        }
    )
    
    # 设备未连接时应返回False
    assert not device_action_handler.handle_click(action)

def test_device_swipe_action(device_action_handler):
    """测试设备滑动动作"""
    action = Action(
        action_type="device_swipe",
        parameters={
            "from_x": 100,
            "from_y": 500,
            "to_x": 100,
            "to_y": 100,
            "duration": 0.5,
            "timeout": 5.0
        }
    )
    
    # 设备未连接时应返回False
    assert not device_action_handler.handle_swipe(action)

def test_device_click_text_action(device_action_handler):
    """测试设备文本点击动作"""
    action = Action(
        action_type="device_click_text",
        parameters={
            "text": "开始游戏",
            "timeout": 5.0
        }
    )
    
    # 设备未连接时应返回False
    assert not device_action_handler.handle_click_text(action)

def test_device_screenshot_action(device_action_handler):
    """测试设备截图动作"""
    action = Action(
        action_type="device_screenshot",
        parameters={
            "filename": "test.png"
        }
    )
    
    # 设备未连接时应返回False
    assert not device_action_handler.handle_screenshot(action)

def test_device_connection_condition(device_action_handler):
    """测试设备连接状态条件"""
    condition = Condition(
        condition_type="device_connected",
        parameters={},
        operator="equals"
    )
    
    # 初始状态应为未连接
    assert not device_action_handler.evaluate_device_connection(condition, {})

def test_text_exists_condition(device_action_handler):
    """测试文本存在条件"""
    condition = Condition(
        condition_type="text_exists",
        parameters={
            "text": "开始游戏",
            "timeout": 5.0
        }
    )
    
    # 设备未连接时应返回False
    assert not device_action_handler.evaluate_text_exists(condition, {})

def test_action_handler_registration(decision_maker, device_action_handler):
    """测试动作处理器注册"""
    # 验证所有设备动作类型都已注册
    assert "device_click" in decision_maker.action_handlers
    assert "device_swipe" in decision_maker.action_handlers
    assert "device_click_text" in decision_maker.action_handlers
    assert "device_screenshot" in decision_maker.action_handlers

def test_condition_handler_registration(decision_maker, device_action_handler):
    """测试条件处理器注册"""
    # 验证所有设备条件类型都已注册
    assert "device_connected" in decision_maker.condition_handlers
    assert "text_exists" in decision_maker.condition_handlers

def test_rule_integration(decision_maker):
    """测试规则集成"""
    # 加载设备规则文件
    decision_maker.load_rules("config/device_rules.json")
    
    # 验证行为已加载
    assert "device_connection_check" in decision_maker.behaviors
    assert "text_interaction" in decision_maker.behaviors
    assert "screen_interaction" in decision_maker.behaviors
    
    # 验证规则评估
    context = {}
    decision_maker.update(context)
    
    # 由于设备未连接，不应有活动行为
    assert len(decision_maker.get_active_behaviors()) == 0
