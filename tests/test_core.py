import pytest
from game_automation_suite.core import GameAutomationSuite

def test_game_automation_suite():
    suite = GameAutomationSuite()
    assert isinstance(suite.game_control, GameAutomationSuite.game_control.__class__)
    assert isinstance(suite.image_recognition, GameAutomationSuite.image_recognition.__class__)
    assert isinstance(suite.user_interface, GameAutomationSuite.user_interface.__class__)