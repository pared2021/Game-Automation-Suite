import pytest
from game_automation_suite.modules.game_control import GameControl
from game_automation_suite.modules.image_recognition import ImageRecognition
from game_automation_suite.modules.user_interface import UserInterface

def test_game_control():
    config = GameControl.Config()
    game_control = GameControl(config)
    game_control.perform_action()

def test_image_recognition():
    config = ImageRecognition.Config()
    image_recognition = ImageRecognition(config)
    image_recognition.detect_objects()

def test_user_interface():
    config = UserInterface.Config()
    user_interface = UserInterface(config)
    user_interface.show()