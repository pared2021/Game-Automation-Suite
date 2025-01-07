from .config import Config
from .modules.game_control import GameControl
from .modules.image_recognition import ImageRecognition
from .modules.user_interface import UserInterface

class GameAutomationSuite:
    def __init__(self):
        self.config = Config()
        self.game_control = GameControl(self.config)
        self.image_recognition = ImageRecognition(self.config)
        self.user_interface = UserInterface(self.config)

    def run(self):
        self.user_interface.show()
        while True:
            if self.config.DEBUG:
                print("Running game automation suite...")
            self.game_control.perform_action()
            self.image_recognition.detect_objects()