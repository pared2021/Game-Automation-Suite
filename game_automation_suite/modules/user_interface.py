from ..utils import log
from ..config import Config

class UserInterface:
    def __init__(self, config):
        self.config = config

    def show(self):
        log("Showing user interface...")
        # Add user interface logic here
        print("Game Automation Suite User Interface")
        print(f"Game Window Title: {self.config.GAME_WINDOW_TITLE}")
        print(f"Debug Mode: {self.config.DEBUG}")