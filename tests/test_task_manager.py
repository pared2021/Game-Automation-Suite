import unittest
from game_automation.controllers.task_manager import TaskManager, Task

class TestTaskManager(unittest.TestCase):
    def setUp(self):
        self.task_manager = TaskManager(None)
        self.task_manager.tasks = {
            "main_quests": [],
            "side_quests": [],
            "daily_tasks": [],
            "weekly_tasks": [],
            "monthly_tasks": [],
            "events": []
        }

    def test_load_tasks(self):
        # 测试加载任务
        task_data = {
            "id": 1,
            "name": "Test Task",
            "description": "This is a test task.",
            "reward": 100,
            "task_type": "main_quests",
            "prerequisites": [],
            "subtasks": []
        }
        self.task_manager.tasks["main_quests"].append(Task(**task_data))
        self.assertEqual(len(self.task_manager.tasks["main_quests"]), 1)

    def test_complete_task(self):
        # 测试完成任务
        task = Task(1, "Test Task", "This is a test task.", 100, "main_quests")
        self.task_manager.tasks["main_quests"].append(task)
        self.task_manager.complete_task(task)
        self.assertEqual(task.status, "已完成")

    def test_update_task_progress(self):
        # 测试更新任务进度
        task = Task(1, "Test Task", "This is a test task.", 100, "main_quests")
        self.task_manager.tasks["main_quests"].append(task)
        self.task_manager.update_task_progress(1, 50)
        self.assertEqual(task.progress, 50)

if __name__ == '__main__':
    unittest.main()
