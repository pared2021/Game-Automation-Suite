from .base_action import GameAction
from utils.error_handler import log_exception

class CompleteTaskAction(GameAction):
    """
    Action for completing a task in the game.
    """

    @log_exception
    async def execute(self, game_engine, task_id):
        """
        Execute the complete task action.

        Args:
            game_engine: The GameEngine instance.
            task_id (str): The ID of the task to complete.

        Returns:
            int: The reward obtained from completing the task.
        """
        game_engine.logger.info(f"Completing task: {task_id}")
        
        # Implement the logic for completing the task
        # This is a placeholder implementation
        task = await self._get_task(game_engine, task_id)
        if task:
            reward = await self._complete_task(game_engine, task)
        else:
            game_engine.logger.warning(f"Task not found: {task_id}")
            reward = 0

        return reward

    async def _get_task(self, game_engine, task_id):
        # Implement logic to retrieve task details
        # This is a placeholder and should be replaced with actual task retrieval logic
        tasks = {
            "task_1": {"name": "Defeat 5 enemies", "reward": 50},
            "task_2": {"name": "Collect 10 resources", "reward": 30},
            "task_3": {"name": "Explore new area", "reward": 40}
        }
        return tasks.get(task_id)

    async def _complete_task(self, game_engine, task):
        # Implement logic to complete the task and update game state
        # This is a placeholder and should be replaced with actual task completion logic
        game_engine.logger.info(f"Completed task: {task['name']}")
        # Update game state (e.g., increase experience, add rewards)
        return task['reward']

    def get_description(self):
        return "Complete a task or quest in the game"