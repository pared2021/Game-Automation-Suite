# Game Automation Control API Documentation

## GameEngine

The `GameEngine` class is the core of the automation system. It provides the following main methods:

### `async def initialize()`

Set up the game engine and its components.

### `async def run_automation_loop()`

Start the main automation loop. This method continuously gets the game state, makes decisions, and executes actions.

### `async def get_game_state() -> Dict[str, Any]`

Retrieve the current game state. Returns a dictionary containing various game state information such as health, gold, level, enemy count, etc.

### `async def execute_action(action: Dict[str, Any])`

Execute a game action. The action is a dictionary containing the action type and any necessary parameters.

## AIDecisionMaker

The `AIDecisionMaker` class is responsible for making decisions based on the current game state.

### `async def make_decision(game_state: Dict[str, Any]) -> Dict[str, Any]`

Make a decision based on the current game state. Returns an action to be executed.

## DeviceManager

The `DeviceManager` class handles interactions with the device running the game.

### `async def initialize()`

Initialize the device manager.

### `async def capture_screen()`

Capture the current screen of the device.

### `async def tap(x: int, y: int)`

Perform a tap action at the specified coordinates.

### `async def swipe(start_x: int, start_y: int, end_x: int, end_y: int)`

Perform a swipe action from the start coordinates to the end coordinates.

## OCRManager

The `OCRManager` class handles Optical Character Recognition (OCR) operations.

### `async def initialize()`

Initialize the OCR manager.

### `async def recognize_text(image: Any) -> str`

Recognize text in the given image.

## ImageRecognitionManager

The `ImageRecognitionManager` class handles image recognition operations.

### `async def initialize()`

Initialize the image recognition manager.

### `async def detect_objects(image: Any) -> List[Dict[str, Any]]`

Detect objects in the given image. Returns a list of detected objects with their properties.

## BattleManager

The `BattleManager` class manages battle-related operations.

### `async def initialize()`

Initialize the battle manager.

### `async def use_item(item: str)`

Use the specified item in battle.

### `async def use_skill(skill: str)`

Use the specified skill in battle.

For detailed usage examples and more information, please refer to the individual class and method documentation in the source code.