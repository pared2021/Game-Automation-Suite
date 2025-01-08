from game_automation_suite.core.game_controller import GameController
from game_automation_suite.utils.config import Config

def main():
    config = Config("config.json")
    game_window_title = config.get("game_window_title", "Default Game")
    controller = GameController(game_window_title)

    try:
        controller.find_game_window()
        controller.click(100, 200)
        controller.move_to(150, 250)
        controller.drag_to(200, 300)
        controller.type_text("Hello, World!")
        controller.wait(5)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == '__main__':
    main()