import asyncio
from datetime import datetime, timedelta
from utils.logger import detailed_logger
from utils.error_handler import log_exception
from utils.config_manager import config_manager
from game_automation.controllers.task_utils import TaskUtils

class Task:
    def __init__(self, task_id, name, description, reward, task_type, start_time=None, end_time=None, prerequisites=None, subtasks=None):
        self.id = task_id
        self.name = name
        self.description = description
        self.reward = reward
        self.task_type = task_type
        self.status = "未开始"
        self.progress = 0
        self.start_time = start_time
        self.end_time = end_time
        self.prerequisites = prerequisites or []
        self.subtasks = subtasks or []

class TaskManager:
    def __init__(self, game_engine):
        self.logger = detailed_logger
        self.game_engine = game_engine
        self.config = config_manager.get('task_manager', {})
        self.tasks = {
            "main_quests": [],
            "side_quests": [],
            "daily_tasks": [],
            "weekly_tasks": [],
            "monthly_tasks": [],
            "events": []
        }

    @log_exception
    async def load_tasks(self):
        try:
            tasks_data = self.config.get('predefined_tasks', {})
            for task_type, tasks in tasks_data.items():
                for task_data in tasks:
                    task = Task(
                        task_id=task_data['id'],
                        name=task_data['name'],
                        description=task_data['description'],
                        reward=task_data['reward'],
                        task_type=task_type,
                        start_time=datetime.fromisoformat(task_data.get('start_time', datetime.now().isoformat())),
                        end_time=datetime.fromisoformat(task_data.get('end_time', (datetime.now() + timedelta(days=30)).isoformat())),
                        prerequisites=task_data.get('prerequisites', []),
                        subtasks=task_data.get('subtasks', [])
                    )
                    self.tasks[task_type].append(task)
            self.logger.info(f"Loaded {sum(len(tasks) for tasks in self.tasks.values())} tasks")
        except Exception as e:
            self.logger.error(f"Failed to load tasks: {e}")

    async def get_task_status(self):
        return {task_type: [TaskUtils.task_to_dict(task) for task in tasks] for task_type, tasks in self.tasks.items()}

    @log_exception
    async def complete_task(self, task):
        try:
            if await TaskUtils.check_prerequisites(task, self):
                if all(subtask.status == "已完成" for subtask in task.subtasks):
                    task.status = "已完成"
                    task.progress = 100
                    self.logger.info(f"Task completed: {task.name}")
                    await self.game_engine.reward_player(task.reward)
                else:
                    self.logger.warning(f"Cannot complete task {task.name}: not all subtasks are completed")
            else:
                self.logger.warning(f"Cannot complete task {task.name}: prerequisites not met")
        except Exception as e:
            self.logger.error(f"Failed to complete task {task.name}: {e}")

    @log_exception
    async def update_task_progress(self, task_id, progress):
        try:
            for task_list in self.tasks.values():
                for task in task_list:
                    if task.id == task_id:
                        task.progress = min(progress, 100)
                        if task.progress == 100:
                            await self.complete_task(task)
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to update task progress for ID {task_id}: {e}")

    # 其他方法保持不变...

task_manager = TaskManager(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用
