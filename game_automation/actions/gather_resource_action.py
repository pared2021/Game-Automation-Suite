from .base_action import GameAction
from utils.error_handler import log_exception
import asyncio

class GatherResourceAction(GameAction):
    @log_exception
    async def execute(self, game_engine, resource):
        game_engine.logger.info(f"Gathering resource: {resource}")
        
        resource_info = await self._get_resource_info(game_engine, resource)
        if resource_info:
            reward = await self._gather_resource(game_engine, resource_info)
        else:
            game_engine.logger.warning(f"Unknown resource: {resource}")
            reward = 0

        return reward

    async def _get_resource_info(self, game_engine, resource):
        resources = {
            "wood": {"name": "Wood", "gather_time": 2, "reward": 5},
            "stone": {"name": "Stone", "gather_time": 3, "reward": 8},
            "herb": {"name": "Herb", "gather_time": 1, "reward": 3}
        }
        return resources.get(resource)

    async def _gather_resource(self, game_engine, resource_info):
        game_engine.logger.info(f"Gathering {resource_info['name']} for {resource_info['gather_time']} seconds")
        await asyncio.sleep(resource_info['gather_time'])
        game_engine.logger.info(f"Gathered {resource_info['name']}")
        await self._update_inventory(game_engine, resource_info['name'], 1)
        return resource_info['reward']

    async def _update_inventory(self, game_engine, resource_name, amount):
        # Implement inventory update logic
        pass

    def get_description(self):
        return "Gather a resource from the game world"