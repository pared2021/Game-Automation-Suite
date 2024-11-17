import pytest
import os
from datetime import datetime
from typing import List

from game_automation.core.task_manager import (
    Task, TaskManager, TaskStatus, TaskPriority, TaskError
)

class TestTask(Task):
    """用于测试的任务实现"""
    def __init__(self, task_id: str, name: str, should_succeed: bool = True, **kwargs):
        super().__init__(task_id, name, **kwargs)
        self.should_succeed = should_succeed
        self.executed = False

    def _execute(self) -> bool:
        self.executed = True
        return self.should_succeed

@pytest.fixture
def task_manager():
    """创建任务管理器实例"""
    return TaskManager()

@pytest.fixture
def sample_tasks() -> List[TestTask]:
    """创建测试任务列表"""
    return [
        TestTask("task1", "Task 1", priority=TaskPriority.NORMAL),
        TestTask("task2", "Task 2", priority=TaskPriority.HIGH),
        TestTask("task3", "Task 3", priority=TaskPriority.LOW),
        TestTask("task4", "Task 4", should_succeed=False)
    ]

def test_task_creation():
    """测试任务创建"""
    task = TestTask("test1", "Test Task")
    
    assert task.task_id == "test1"
    assert task.name == "Test Task"
    assert task.status == TaskStatus.PENDING
    assert task.progress == 0.0
    assert task.start_time is None
    assert task.end_time is None

def test_task_execution():
    """测试任务执行"""
    # 测试成功的任务
    success_task = TestTask("success", "Success Task", should_succeed=True)
    assert success_task.execute()
    assert success_task.status == TaskStatus.COMPLETED
    assert success_task.progress == 1.0
    assert success_task.start_time is not None
    assert success_task.end_time is not None
    
    # 测试失败的任务
    fail_task = TestTask("fail", "Fail Task", should_succeed=False)
    assert not fail_task.execute()
    assert fail_task.status == TaskStatus.FAILED
    assert fail_task.start_time is not None
    assert fail_task.end_time is not None

def test_task_callbacks():
    """测试任务回调"""
    task = TestTask("callback", "Callback Task")
    
    complete_called = False
    fail_called = False
    
    def on_complete(t):
        nonlocal complete_called
        complete_called = True
    
    def on_fail(t):
        nonlocal fail_called
        fail_called = True
    
    task.on_complete(on_complete)
    task.on_fail(on_fail)
    
    task.execute()
    assert complete_called
    assert not fail_called

def test_task_manager_basic(task_manager, sample_tasks):
    """测试任务管理器基本功能"""
    # 添加任务
    for task in sample_tasks:
        task_manager.add_task(task)
    
    assert len(task_manager.tasks) == len(sample_tasks)
    
    # 验证任务获取
    task = task_manager.get_task("task1")
    assert task is not None
    assert task.task_id == "task1"
    
    # 验证任务移除
    task_manager.remove_task("task1")
    assert "task1" not in task_manager.tasks
    assert len(task_manager.tasks) == len(sample_tasks) - 1

def test_task_execution_order(task_manager, sample_tasks):
    """测试任务执行顺序"""
    # 添加任务
    for task in sample_tasks:
        task_manager.add_task(task)
    
    # 执行所有任务
    task_manager.execute_all_tasks()
    
    # 验证执行结果
    assert len(task_manager.completed_tasks) == 3  # 3个成功任务
    assert len(task_manager.failed_tasks) == 1     # 1个失败任务
    
    # 验证优先级顺序
    completed_order = [task.task_id for task in task_manager.completed_tasks]
    assert completed_order.index("task2") < completed_order.index("task1")  # HIGH在NORMAL之前
    assert completed_order.index("task1") < completed_order.index("task3")  # NORMAL在LOW之前

def test_task_dependencies(task_manager):
    """测试任务依赖关系"""
    # 创建带依赖的任务
    task1 = TestTask("dep1", "Dep Task 1")
    task2 = TestTask("dep2", "Dep Task 2", dependencies=["dep1"])
    task3 = TestTask("dep3", "Dep Task 3", dependencies=["dep2"])
    
    # 添加任务
    task_manager.add_task(task3)  # 先添加依赖于其他任务的任务
    task_manager.add_task(task2)
    task_manager.add_task(task1)
    
    # 执行任务
    task_manager.execute_all_tasks()
    
    # 验证执行顺序
    completed = task_manager.completed_tasks
    assert completed[0].task_id == "dep1"
    assert completed[1].task_id == "dep2"
    assert completed[2].task_id == "dep3"

def test_task_state_persistence(task_manager, sample_tasks, tmp_path):
    """测试任务状态持久化"""
    # 添加任务
    for task in sample_tasks:
        task_manager.add_task(task)
    
    # 执行部分任务
    task_manager.execute_next_task()
    task_manager.execute_next_task()
    
    # 保存状态
    state_file = tmp_path / "task_state.json"
    task_manager.save_state(str(state_file))
    
    # 创建新的管理器并加载状态
    new_manager = TaskManager()
    new_manager.load_state(str(state_file))
    
    # 验证状态恢复
    assert len(new_manager.tasks) == len(task_manager.tasks)
    assert len(new_manager.completed_tasks) == len(task_manager.completed_tasks)
    assert len(new_manager.failed_tasks) == len(task_manager.failed_tasks)
    
    # 验证任务状态
    for task_id, task in task_manager.tasks.items():
        loaded_task = new_manager.get_task(task_id)
        assert loaded_task is not None
        assert loaded_task.status == task.status
        assert loaded_task.progress == task.progress

def test_task_statistics(task_manager, sample_tasks):
    """测试任务统计信息"""
    # 添加任务
    for task in sample_tasks:
        task_manager.add_task(task)
    
    # 执行部分任务
    task_manager.execute_next_task()
    task_manager.execute_next_task()
    
    # 获取统计信息
    stats = task_manager.get_statistics()
    
    assert stats['total'] == len(sample_tasks)
    assert stats['completed'] + stats['failed'] + stats['pending'] == len(sample_tasks)
    assert stats['running'] == 0

def test_error_handling(task_manager):
    """测试错误处理"""
    # 测试重复任务ID
    task1 = TestTask("same_id", "Task 1")
    task2 = TestTask("same_id", "Task 2")
    
    task_manager.add_task(task1)
    with pytest.raises(TaskError):
        task_manager.add_task(task2)
    
    # 测试移除不存在的任务
    with pytest.raises(TaskError):
        task_manager.remove_task("non_existent")
    
    # 测试无效的状态文件
    with pytest.raises(TaskError):
        task_manager.load_state("non_existent_file.json")

def test_task_pause_resume():
    """测试任务暂停和恢复"""
    task = TestTask("pause_test", "Pause Test Task")
    
    # 测试暂停
    task.status = TaskStatus.RUNNING
    task.pause()
    assert task.status == TaskStatus.PAUSED
    
    # 测试恢复
    task.resume()
    assert task.status == TaskStatus.RUNNING

def test_cleanup():
    """清理测试资源"""
    # 如果测试过程中创建了任何临时文件，在这里清理
    pass
