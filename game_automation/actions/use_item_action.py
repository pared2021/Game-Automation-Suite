import asyncio
from .base_action import GameAction
from utils.error_handler import log_exception

class UseItemAction(GameAction):
    """
    Action for using an item in the game.
    """

    @log_exception
    async def execute(self, game_engine, item):
        """
        Execute the use item action.

        Args:
            game_engine: The GameEngine instance.
            item (str): The name of the item to use.

        Returns:
            int: The reward obtained from using the item.
        """
        game_engine.logger.info(f"Using item: {item}")
        
        # Implement the logic for using the item
        # This is a placeholder implementation
        if item == "health_potion":
            reward = await self._use_health_potion(game_engine)
        elif item == "mana_potion":
            reward = await self._use_mana_potion(game_engine)
        else:
            game_engine.logger.warning(f"Unknown item: {item}")
            reward = 0

        return reward

    async def _use_health_potion(self, game_engine):
        # Implement health potion usage logic
        game_state = await game_engine.get_game_state()
        if game_state['health'] < 100:
            new_health = min(game_state['health'] + 50, 100)
            # Update game state
            # This is a placeholder and should be replaced with actual game state update logic
            game_engine.logger.info(f"Health increased from {game_state['health']} to {new_health}")
            await asyncio.sleep(1)  # Simulate potion use time
            return 10
        else:
            game_engine.logger.info("Health is already full")
            return 0

    async def _use_mana_potion(self, game_engine):
        # Implement mana potion usage logic
        game_state = await game_engine.get_game_state()
        if game_state.get('mana', 0) < 100:
            new_mana = min(game_state.get('mana', 0) + 50, 100)
            # Update game state
            # This is a placeholder and should be replaced with actual game state update logic
            game_engine.logger.info(f"Mana increased from {game_state.get('mana', 0)} to {new_mana}")
            await asyncio.sleep(1)  # Simulate potion use time
            return 8
        else:
            game_engine.logger.info("Mana is already full")
            return 0

    def get_description(self):
        return "Use an item from the inventory"