import asyncio
from datetime import datetime, timedelta
from utils.logger import detailed_logger
from utils.error_handler import log_exception
from utils.config_manager import config_manager

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
        # 从配置文件或数据库加载任务
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

    async def get_task_status(self):
        return {task_type: [self.task_to_dict(task) for task in tasks] for task_type, tasks in self.tasks.items()}

    def task_to_dict(self, task):
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "reward": task.reward,
            "status": task.status,
            "progress": task.progress,
            "priority": self.get_task_priority(task),
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "prerequisites": task.prerequisites,
            "subtasks": [self.task_to_dict(subtask) for subtask in task.subtasks]
        }

    def get_task_priority(self, task):
        if task.task_type == "main_quests":
            return "high"
        elif task.task_type in ["daily_tasks", "events"] and task.status != "已完成":
            return "medium"
        else:
            return "low"

    @log_exception
    async def complete_task_by_id(self, task_id):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id:
                    await self.complete_task(task)
                    return True
        return False

    @log_exception
    async def complete_task(self, task):
        if all(await self.is_prerequisite_completed(prereq) for prereq in task.prerequisites):
            if all(subtask.status == "已完成" for subtask in task.subtasks):
                task.status = "已完成"
                task.progress = 100
                self.logger.info(f"Task completed: {task.name}")
                await self.game_engine.reward_player(task.reward)
            else:
                self.logger.warning(f"Cannot complete task {task.name}: not all subtasks are completed")
        else:
            self.logger.warning(f"Cannot complete task {task.name}: prerequisites not met")

    async def is_prerequisite_completed(self, task_id):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id:
                    return task.status == "已完成"
        return False

    @log_exception
    async def update_task_progress(self, task_id, progress):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id:
                    task.progress = min(progress, 100)
                    if task.progress == 100:
                        await self.complete_task(task)
                    return True
        return False

    @log_exception
    async def check_and_refresh_tasks(self):
        now = datetime.now()
        for task_type in ["daily_tasks", "weekly_tasks", "monthly_tasks", "events"]:
            for task in self.tasks[task_type]:
                if task.end_time and now > task.end_time:
                    await self.refresh_task(task)

    async def refresh_task(self, task):
        if task.task_type == "daily_tasks":
            task.start_time = datetime.now()
            task.end_time = task.start_time + timedelta(days=1)
        elif task.task_type == "weekly_tasks":
            task.start_time = datetime.now()
            task.end_time = task.start_time + timedelta(weeks=1)
        elif task.task_type == "monthly_tasks":
            task.start_time = datetime.now()
            task.end_time = task.start_time + timedelta(days=30)
        task.status = "未开始"
        task.progress = 0
        for subtask in task.subtasks:
            subtask.status = "未开始"
            subtask.progress = 0
        self.logger.info(f"Task refreshed: {task.name}")

    @log_exception
    async def add_new_task(self, task):
        self.tasks[f"{task.task_type}"].append(task)
        self.logger.info(f"New task added: {task.name}")

    @log_exception
    async def remove_task(self, task_id):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id:
                    task_list.remove(task)
                    self.logger.info(f"Task removed: {task.name}")
                    return True
        return False

    async def get_available_tasks(self):
        available_tasks = []
        for task_list in self.tasks.values():
            for task in task_list:
                if task.status == "未开始" and all(await self.is_prerequisite_completed(prereq) for prereq in task.prerequisites):
                    available_tasks.append(task)
        return available_tasks

    async def start_task(self, task_id):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id and task.status == "未开始":
                    task.status = "进行中"
                    task.start_time = datetime.now()
                    self.logger.info(f"Task started: {task.name}")
                    return True
        return False

    async def abandon_task(self, task_id):
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id and task.status == "进行中":
                    task.status = "已放弃"
                    self.logger.info(f"Task abandoned: {task.name}")
                    return True
        return False

    async def get_task_chain(self, task_id):
        task_chain = []
        for task_list in self.tasks.values():
            for task in task_list:
                if task.id == task_id:
                    task_chain.append(task)
                    for prereq_id in task.prerequisites:
                        task_chain.extend(await self.get_task_chain(prereq_id))
                    break
        return task_chain

    async def optimize_task_order(self):
        available_tasks = await self.get_available_tasks()
        optimized_order = sorted(available_tasks, key=lambda t: (self.get_task_priority(t), -len(t.prerequisites), t.end_time or datetime.max))
        return optimized_order

    def all_tasks_completed(self):
        return all(task.status == "已完成" for task_list in self.tasks.values() for task in task_list)

    async def reset_tasks(self):
        for task_list in self.tasks.values():
            for task in task_list:
                task.status = "未开始"
                task.progress = 0
        self.logger.info("All tasks have been reset")

task_manager = TaskManager(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用