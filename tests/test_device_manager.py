import pytest
import asyncio
from datetime import datetime
from game_automation.device.device_manager import DeviceManager
from game_automation.device.device_action_handler import DeviceActionHandler
from game_automation.device.emulator_manager import EmulatorManager
from game_automation.core.decision_maker import Action, Condition
from utils.error_handler import DeviceConnectionError

@pytest.mark.asyncio
async def test_device_connection():
    """测试设备连接功能"""
    device_manager = DeviceManager()
    
    # 测试设备连接
    try:
        await device_manager.connect()
        assert device_manager.is_connected
        assert device_manager.get_ui_automator() is not None
    except DeviceConnectionError:
        # 如果没有真实设备，尝试连接模拟器
        emulator_manager = EmulatorManager()
        detected = await emulator_manager.detect_emulators()
        
        if detected:
            await emulator_manager.connect()
            assert emulator_manager.is_connected
            
            # 使用检测到的模拟器连接设备管理器
            await device_manager.connect(f"127.0.0.1:{await emulator_manager._get_emulator_adb_port()}")
            assert device_manager.is_connected
            assert device_manager.get_ui_automator() is not None

@pytest.mark.asyncio
async def test_concurrent_operations():
    """测试并发操作控制"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        
        # 创建多个并发操作
        async def dummy_operation(delay: float):
            await asyncio.sleep(delay)
            return True
        
        # 同时启动多个操作
        tasks = []
        for i in range(5):  # 超过最大并发数(3)
            task = device_manager.queue_operation(dummy_operation, 0.1)
            tasks.append(task)
        
        # 等待所有操作完成
        results = await asyncio.gather(*tasks)
        
        # 验证所有操作都成功完成
        assert all(results)
        
        # 验证活动操作数量不超过最大限制
        assert device_manager.active_operations_count <= 3
        
    except DeviceConnectionError:
        pytest.skip("No device available for concurrent operation testing")

@pytest.mark.asyncio
async def test_resource_management():
    """测试资源管理功能"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        
        # 注册测试资源
        class TestResource:
            def __init__(self):
                self.closed = False
            
            async def close(self):
                self.closed = True
        
        resource = TestResource()
        await device_manager.register_resource("test_resource", resource)
        
        # 断开连接应该触发资源清理
        await device_manager.disconnect()
        
        # 验证资源被正确清理
        assert resource.closed
        
    except DeviceConnectionError:
        pytest.skip("No device available for resource management testing")

@pytest.mark.asyncio
async def test_operation_timeout():
    """测试操作超时处理"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        
        # 创建一个会超时的操作
        async def long_operation():
            await asyncio.sleep(35)  # 超过默认30秒超时
            return True
        
        # 启动操作并等待
        task = device_manager.queue_operation(long_operation)
        
        # 等待操作超时
        await asyncio.sleep(32)
        
        # 验证超时操作被清理
        assert device_manager.active_operations_count == 0
        
    except DeviceConnectionError:
        pytest.skip("No device available for timeout testing")

@pytest.mark.asyncio
async def test_action_handler():
    """测试设备动作处理器"""
    device_manager = DeviceManager()
    action_handler = DeviceActionHandler(device_manager)
    
    try:
        await device_manager.connect()
        
        # 测试点击动作
        click_action = Action(
            action_type="device_click",
            parameters={
                "x": 100,
                "y": 100,
                "wait_after": 1
            }
        )
        result = await action_handler.handle_click(click_action)
        assert result is True
        
        # 测试滑动动作
        swipe_action = Action(
            action_type="device_swipe",
            parameters={
                "from_x": 100,
                "from_y": 200,
                "to_x": 100,
                "to_y": 100,
                "duration": 0.5,
                "wait_after": 1
            }
        )
        result = await action_handler.handle_swipe(swipe_action)
        assert result is True
        
        # 测试截图动作
        screenshot_action = Action(
            action_type="device_screenshot",
            parameters={
                "filename": "test_screenshot.png"
            }
        )
        result = await action_handler.handle_screenshot(screenshot_action)
        assert result is True
        
        # 测试文本检测条件
        text_condition = Condition(
            condition_type="text_exists",
            parameters={
                "text": "测试文本",
                "timeout": 5
            }
        )
        result = await action_handler.evaluate_text_exists(text_condition, {})
        assert isinstance(result, bool)
        
        # 测试清理
        await action_handler.cleanup()
        assert len(action_handler._cleanup_tasks) == 0
        assert len(action_handler._operation_results) == 0
        
    except DeviceConnectionError:
        pytest.skip("No device available for action handler testing")

@pytest.mark.asyncio
async def test_error_recovery():
    """测试错误恢复机制"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        
        # 模拟设备断开
        device_manager.connected = False
        
        # 等待监控任务检测到断开并尝试重连
        await asyncio.sleep(6)  # 等待一个监控周期
        
        # 验证重连尝试
        assert device_manager.is_connected or device_manager.monitor_task is None
        
    except DeviceConnectionError:
        pytest.skip("No device available for error recovery testing")

@pytest.mark.asyncio
async def test_device_monitoring():
    """测试设备监控功能"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        assert device_manager.is_connected
        
        # 测试连接检查
        assert await device_manager.check_connection()
        
        # 测试断开连接
        await device_manager.disconnect()
        assert not device_manager.is_connected
        assert not await device_manager.check_connection()
        
        # 验证监控任务被正确清理
        assert device_manager.monitor_task is None
        
    except DeviceConnectionError:
        pytest.skip("No device available for monitoring testing")

@pytest.mark.asyncio
async def test_operation_queue():
    """测试操作队列功能"""
    device_manager = DeviceManager()
    
    try:
        await device_manager.connect()
        
        # 创建多个操作并检查队列状态
        async def test_operation(delay: float):
            await asyncio.sleep(delay)
            return True
        
        # 添加多个操作到队列
        tasks = []
        for i in range(5):
            task = device_manager.queue_operation(test_operation, 0.2)
            tasks.append(task)
        
        # 验证队列中的操作数量
        assert device_manager.queued_operations_count > 0
        
        # 等待所有操作完成
        results = await asyncio.gather(*tasks)
        
        # 验证所有操作都成功完成
        assert all(results)
        
        # 验证队列被清空
        assert device_manager.queued_operations_count == 0
        
    except DeviceConnectionError:
        pytest.skip("No device available for queue testing")
