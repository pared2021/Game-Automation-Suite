import pytest
import asyncio
import cv2
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from game_automation.core.task_executor import (
    TaskExecutor, GameTask, TaskExecutionError
)
from game_automation.device.device_manager import DeviceManager
from game_automation.scene_understanding.advanced_scene_analyzer import AdvancedSceneAnalyzer
from game_automation.core.task_manager import TaskStatus, TaskPriority

# Mock classes for testing
class MockUIAutomator:
    def __init__(self):
        self.screenshots_taken = 0
        
    def screenshot(self, filename: str) -> bool:
        self.screenshots_taken += 1
        # 创建一个简单的测试图像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(filename, image)
        return True

class MockDeviceManager:
    def __init__(self, connected: bool = True):
        self.connected = connected
        self.ui_automator = MockUIAutomator()
    
    @property
    def is_connected(self) -> bool:
        return self.connected
        
    def get_ui_automator(self):
        return self.ui_automator if self.connected else None

class MockSceneAnalyzer:
    def __init__(self):
        self.analysis_count = 0
        
    def analyze_screenshot(self, screenshot: np.ndarray) -> Dict:
        self.analysis_count += 1
        return {
            'scene_type': 'test_scene',
            'scene_state': {'test': True},
            'scene_changed': False,
            'timestamp': datetime.now().isoformat()
        }

class TestGameTask(GameTask):
    def __init__(self, 
                 task_id: str,
                 name: str,
                 device_manager,
                 scene_analyzer,
                 should_succeed: bool = True,
                 execution_time: float = 0.1,
                 **kwargs):
        super().__init__(task_id, name, device_manager, scene_analyzer, **kwargs)
        self.should_succeed = should_succeed
        self.execution_time = execution_time
        self.prerequisites_met = True
        
    async def verify_prerequisites(self) -> bool:
        return await super().verify_prerequisites() and self.prerequisites_met
        
    def _execute(self) -> bool:
        # 模拟任务执行
        asyncio.sleep(self.execution_time)
        return self.should_succeed

@pytest.fixture
def device_manager():
    return MockDeviceManager()

@pytest.fixture
def scene_analyzer():
    return MockSceneAnalyzer()

@pytest.fixture
def task_executor(device_manager, scene_analyzer):
    executor = TaskExecutor(device_manager, scene_analyzer)
    executor.register_task_type('test_task', TestGameTask)
    return executor

@pytest.mark.asyncio
async def test_task_type_registration(task_executor):
    """测试任务类型注册"""
    # 验证已注册的任务类型
    task_types = task_executor.get_registered_task_types()
    assert 'test_task' in task_types
    assert task_types['test_task'] == TestGameTask
    
    # 测试注册无效任务类型
    class InvalidTask:
        pass
    
    with pytest.raises(TaskExecutionError):
        task_executor.register_task_type('invalid_task', InvalidTask)

@pytest.mark.asyncio
async def test_task_creation(task_executor):
    """测试任务创建"""
    task = await task_executor.create_task(
        'test_task',
        'task1',
        'Test Task',
        priority=TaskPriority.NORMAL
    )
    
    assert isinstance(task, TestGameTask)
    assert task.task_id == 'task1'
    assert task.name == 'Test Task'
    assert task.status == TaskStatus.PENDING
    
    # 测试创建未注册的任务类型
    with pytest.raises(TaskExecutionError):
        await task_executor.create_task('unknown_type', 'task2', 'Unknown Task')

@pytest.mark.asyncio
async def test_task_execution_success(task_executor):
    """测试任务成功执行"""
    task = await task_executor.create_task(
        'test_task',
        'success_task',
        'Success Task',
        should_succeed=True
    )
    
    success = await task_executor.execute_task(task)
    assert success
    assert task.status == TaskStatus.COMPLETED
    assert task.error_message is None

@pytest.mark.asyncio
async def test_task_execution_failure(task_executor):
    """测试任务执行失败"""
    task = await task_executor.create_task(
        'test_task',
        'fail_task',
        'Fail Task',
        should_succeed=False
    )
    
    success = await task_executor.execute_task(task)
    assert not success
    assert task.status == TaskStatus.FAILED

@pytest.mark.asyncio
async def test_task_prerequisites(task_executor):
    """测试任务前提条件检查"""
    # 测试设备未连接的情况
    disconnected_device = MockDeviceManager(connected=False)
    executor = TaskExecutor(disconnected_device, MockSceneAnalyzer())
    executor.register_task_type('test_task', TestGameTask)
    
    task = await executor.create_task(
        'test_task',
        'prereq_task',
        'Prerequisite Test Task'
    )
    
    success = await executor.execute_task(task)
    assert not success
    assert task.status == TaskStatus.FAILED
    assert "前提条件不满足" in task.error_message

@pytest.mark.asyncio
async def test_task_timeout(task_executor):
    """测试任务超时"""
    task = await task_executor.create_task(
        'test_task',
        'timeout_task',
        'Timeout Test Task',
        execution_time=2.0,
        timeout=1.0
    )
    
    success = await task_executor.execute_task(task)
    assert not success
    assert task.status == TaskStatus.FAILED
    assert "超时" in task.error_message

@pytest.mark.asyncio
async def test_concurrent_task_execution(task_executor):
    """测试并发任务执行限制"""
    task1 = await task_executor.create_task(
        'test_task',
        'task1',
        'Task 1',
        execution_time=1.0
    )
    
    task2 = await task_executor.create_task(
        'test_task',
        'task2',
        'Task 2'
    )
    
    # 开始执行第一个任务
    execution1 = asyncio.create_task(task_executor.execute_task(task1))
    
    # 尝试同时执行第二个任务
    with pytest.raises(TaskExecutionError, match="当前有正在执行的任务"):
        await task_executor.execute_task(task2)
    
    # 等待第一个任务完成
    await execution1

@pytest.mark.asyncio
async def test_task_monitoring(task_executor):
    """测试任务监控"""
    task = await task_executor.create_task(
        'test_task',
        'monitor_task',
        'Monitor Test Task',
        execution_time=2.0
    )
    
    await task_executor.execute_task(task)
    
    # 验证监控过程中的场景分析
    assert task_executor.scene_analyzer.analysis_count > 0
    assert 'last_scene_analysis' in task.execution_data

@pytest.mark.asyncio
async def test_task_cleanup(task_executor):
    """测试任务清理"""
    task = await task_executor.create_task(
        'test_task',
        'cleanup_task',
        'Cleanup Test Task'
    )
    
    # 添加一些执行数据
    task.execution_data['test_data'] = 'test'
    
    await task_executor.execute_task(task)
    
    # 验证数据被清理
    assert len(task.execution_data) == 0

def test_executor_busy_state(task_executor):
    """测试执行器忙碌状态"""
    assert not task_executor.is_busy
    
    async def check_busy_state():
        task = await task_executor.create_task(
            'test_task',
            'busy_test',
            'Busy Test Task',
            execution_time=1.0
        )
        
        # 开始执行任务
        execution = asyncio.create_task(task_executor.execute_task(task))
        
        # 验证忙碌状态
        assert task_executor.is_busy
        assert task_executor.current_task is task
        
        # 等待任务完成
        await execution
        
        # 验证空闲状态
        assert not task_executor.is_busy
        assert task_executor.current_task is None
    
    asyncio.run(check_busy_state())
