class TaskUtils:
    @staticmethod
    async def check_prerequisites(task, task_manager):
        for prereq in task['prerequisites']:
            if not await task_manager.is_prerequisite_completed(prereq):
                return False
        return True

    @staticmethod
    def task_to_dict(task):
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "reward": task.reward,
            "status": task.status,
            "progress": task.progress,
            "priority": TaskUtils.get_task_priority(task),
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "prerequisites": task.prerequisites,
            "subtasks": [TaskUtils.task_to_dict(subtask) for subtask in task.subtasks]
        }

    @staticmethod
    def get_task_priority(task):
        if task.task_type == "main_quests":
            return "high"
        elif task.task_type in ["daily_tasks", "events"] and task.status != "已完成":
            return "medium"
        else:
            return "low"
