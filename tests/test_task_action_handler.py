import pytest
from datetime import datetime

from game_automation.core.decision_maker import DecisionMaker, Action, Condition
from game_automation.core.task_manager import TaskManager, Task, TaskStatus, TaskPriority
from game_automation.core.task_action_handler import TaskActionHandler

@pytest.fixture
def task_manager():
    """创建任务管理器实例"""
    return TaskManager()

@pytest.fixture
def task_action_handler(task_manager):
    """创建任务动作处理器实例"""
    return TaskActionHandler(task_manager)

@pytest.fixture
def decision_maker(task_action_handler):
    """创建决策器实例并注册任务处理器"""
    dm = DecisionMaker()
    task_action_handler.register_handlers(dm)
    return dm

@pytest.fixture
def sample_task():
    """创建示例任务"""
    return Task(
        task_id="test_task",
        name="Test Task",
        priority=TaskPriority.NORMAL
    )

def test_task_status_condition(task_action_handler, task_manager, sample_task):
    """测试任务状态条件"""
    task_manager.add_task(sample_task)
    
    condition = Condition(
        condition_type="task_status",
        parameters={
            "task_id": "test_task",
            "status": "PENDING"
        }
    )
    
    assert task_action_handler.evaluate_task_status(condition, {})

def test_task_progress_condition(task_action_handler, task_manager, sample_task):
    """测试任务进度条件"""
    task_manager.add_task(sample_task)
    sample_task.progress = 0.6
    
    condition = Condition(
        condition_type="task_progress",
        parameters={
            "task_id": "test_task",
            "threshold": 0.5,
            "operator": "greater_than"
        }
    )
    
    assert task_action_handler.evaluate_task_progress(condition, {})

def test_task_dependencies_condition(task_action_handler, task_manager):
    """测试任务依赖条件"""
    # 创建依赖任务
    dep_task = Task(
        task_id="dep_task",
        name="Dependency Task",
        priority=TaskPriority.NORMAL
    )
    dep_task.status = TaskStatus.COMPLETED
    task_manager.add_task(dep_task)
    
    # 创建主任务
    main_task = Task(
        task_id="main_task",
        name="Main Task",
        priority=TaskPriority.NORMAL,
        dependencies=["dep_task"]
    )
    task_manager.add_task(main_task)
    
    condition = Condition(
        condition_type="task_dependencies",
        parameters={
            "task_id": "main_task"
        }
    )
    
    assert task_action_handler.evaluate_task_dependencies(condition, {})

def test_add_task_action(task_action_handler):
    """测试添加任务动作"""
    action = Action(
        action_type="add_task",
        parameters={
            "task_id": "new_task",
            "name": "New Task",
            "priority": "HIGH",
            "dependencies": []
        }
    )
    
    assert task_action_handler.handle_add_task(action)
    assert task_action_handler.task_manager.get_task("new_task") is not None

def test_remove_task_action(task_action_handler, task_manager, sample_task):
    """测试移除任务动作"""
    task_manager.add_task(sample_task)
    
    action = Action(
        action_type="remove_task",
        parameters={
            "task_id": "test_task"
        }
    )
    
    assert task_action_handler.handle_remove_task(action)
    assert task_action_handler.task_manager.get_task("test_task") is None

def test_execute_task_action(task_action_handler, task_manager, sample_task):
    """测试执行任务动作"""
    task_manager.add_task(sample_task)
    
    action = Action(
        action_type="execute_task",
        parameters={
            "task_id": "test_task"
        }
    )
    
    # 由于示例任务没有实现_execute方法，应该返回False
    assert not task_action_handler.handle_execute_task(action)

def test_condition_handler_registration(decision_maker):
    """测试条件处理器注册"""
    assert "task_status" in decision_maker.condition_handlers
    assert "task_progress" in decision_maker.condition_handlers
    assert "task_dependencies" in decision_maker.condition_handlers

def test_action_handler_registration(decision_maker):
    """测试动作处理器注册"""
    assert "add_task" in decision_maker.action_handlers
    assert "remove_task" in decision_maker.action_handlers
    assert "execute_task" in decision_maker.action_handlers

def test_rule_integration(decision_maker):
    """测试规则集成"""
    # 加载任务规则文件
    decision_maker.load_rules("config/task_rules.json")
    
    # 验证行为已加载
    assert "task_execution_control" in decision_maker.behaviors
    assert "task_progress_monitoring" in decision_maker.behaviors
    assert "task_status_monitoring" in decision_maker.behaviors

def test_invalid_parameters(task_action_handler):
    """测试无效参数处理"""
    # 测试缺少必需参数的任务状态条件
    condition = Condition(
        condition_type="task_status",
        parameters={}
    )
    assert not task_action_handler.evaluate_task_status(condition, {})
    
    # 测试缺少必需参数的添加任务动作
    action = Action(
        action_type="add_task",
        parameters={}
    )
    assert not task_action_handler.handle_add_task(action)

def test_task_lifecycle(task_action_handler, task_manager):
    """测试任务生命周期"""
    # 添加任务
    add_action = Action(
        action_type="add_task",
        parameters={
            "task_id": "lifecycle_task",
            "name": "Lifecycle Test Task",
            "priority": "NORMAL"
        }
    )
    assert task_action_handler.handle_add_task(add_action)
    
    # 检查任务状态
    status_condition = Condition(
        condition_type="task_status",
        parameters={
            "task_id": "lifecycle_task",
            "status": "PENDING"
        }
    )
    assert task_action_handler.evaluate_task_status(status_condition, {})
    
    # 执行任务
    execute_action = Action(
        action_type="execute_task",
        parameters={
            "task_id": "lifecycle_task"
        }
    )
    task_action_handler.handle_execute_task(execute_action)
    
    # 移除任务
    remove_action = Action(
        action_type="remove_task",
        parameters={
            "task_id": "lifecycle_task"
        }
    )
    assert task_action_handler.handle_remove_task(remove_action)
