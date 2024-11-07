from abc import ABC, abstractmethod

class GameAction(ABC):
    """
    Abstract base class for all game actions.
    
    All action classes should inherit from this class and implement the execute method.
    """

    @abstractmethod
    async def execute(self, game_engine, *args):
        """
        Execute the action in the game.

        Args:
            game_engine: The GameEngine instance.
            *args: Additional arguments specific to the action.

        Returns:
            int: The reward obtained from executing the action.
        """
        pass

    @abstractmethod
    def get_description(self):
        """
        Get a description of the action.

        Returns:
            str: A human-readable description of the action.
        """
        pass