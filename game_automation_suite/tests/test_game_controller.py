import unittest
from unittest.mock import patch, MagicMock
from game_automation_suite.core.game_controller import GameController

class TestGameController(unittest.TestCase):
    @patch('pyautogui.getWindowsWithTitle')
    def test_find_game_window(self, mock_get_windows):
        mock_window = MagicMock()
        mock_window.title = "Test Game"
        mock_get_windows.return_value = [mock_window]
        controller = GameController("Test Game")
        controller.find_game_window()
        mock_get_windows.assert_called_with("Test Game")
        mock_window.activate.assert_called_once()

    @patch('pyautogui.click')
    def test_click(self, mock_click):
        controller = GameController("Test Game")
        controller.click(100, 200)
        mock_click.assert_called_with(100, 200, clicks=1, interval=0.0, button='left')

    @patch('pyautogui.moveTo')
    def test_move_to(self, mock_move_to):
        controller = GameController("Test Game")
        controller.move_to(100, 200)
        mock_move_to.assert_called_with(100, 200, duration=0.0)

    @patch('pyautogui.dragTo')
    def test_drag_to(self, mock_drag_to):
        controller = GameController("Test Game")
        controller.drag_to(100, 200)
        mock_drag_to.assert_called_with(100, 200, duration=0.0, button='left')

    @patch('pyautogui.typewrite')
    def test_type_text(self, mock_typewrite):
        controller = GameController("Test Game")
        controller.type_text("Hello, World!")
        mock_typewrite.assert_called_with("Hello, World!", interval=0.0)

    @patch('time.sleep')
    def test_wait(self, mock_sleep):
        controller = GameController("Test Game")
        controller.wait(5)
        mock_sleep.assert_called_with(5)

if __name__ == '__main__':
    unittest.main()