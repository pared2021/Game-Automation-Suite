import pyautogui
import time

class GameController:
    def __init__(self, game_window_title):
        self.game_window_title = game_window_title
        self.game_window = None

    def find_game_window(self):
        """
        查找游戏窗口并设置为当前活动窗口。
        """
        try:
            self.game_window = pyautogui.getWindowsWithTitle(self.game_window_title)[0]
            self.game_window.activate()
            print(f"Game window '{self.game_window_title}' found and activated.")
        except IndexError:
            print(f"Game window '{self.game_window_title}' not found.")
        except Exception as e:
            print(f"An error occurred while finding the game window: {e}")

    def click(self, x, y, clicks=1, interval=0.0, button='left'):
        """
        在游戏窗口中点击指定位置。
        """
        try:
            pyautogui.click(x, y, clicks, interval, button)
            print(f"Clicked at ({x}, {y})")
        except Exception as e:
            print(f"An error occurred while clicking: {e}")

    def move_to(self, x, y, duration=0.0):
        """
        将鼠标移动到指定位置。
        """
        try:
            pyautogui.moveTo(x, y, duration)
            print(f"Moved mouse to ({x}, {y})")
        except Exception as e:
            print(f"An error occurred while moving the mouse: {e}")

    def drag_to(self, x, y, duration=0.0, button='left'):
        """
        在游戏窗口中拖动鼠标到指定位置。
        """
        try:
            pyautogui.dragTo(x, y, duration, button)
            print(f"Dragged mouse to ({x}, {y})")
        except Exception as e:
            print(f"An error occurred while dragging the mouse: {e}")

    def type_text(self, text, interval=0.0):
        """
        在游戏窗口中输入文本。
        """
        try:
            pyautogui.typewrite(text, interval)
            print(f"Typed text: {text}")
        except Exception as e:
            print(f"An error occurred while typing text: {e}")

    def wait(self, seconds):
        """
        等待指定秒数。
        """
        try:
            time.sleep(seconds)
            print(f"Waited for {seconds} seconds")
        except Exception as e:
            print(f"An error occurred while waiting: {e}")