from ..utils import wait, log
from ..config import Config

class GameControl:
    def __init__(self, config):
        self.config = config

    def perform_action(self):
        log("Performing game action...")
        # Add game control logic here
        wait(self.config.GAME_CONTROL_DELAY)