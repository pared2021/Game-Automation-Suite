import os

class Config:
    DEBUG = os.getenv('DEBUG', False)
    GAME_WINDOW_TITLE = os.getenv('GAME_WINDOW_TITLE', 'Game Window')
    IMAGE_RECOGNITION_THRESHOLD = float(os.getenv('IMAGE_RECOGNITION_THRESHOLD', 0.8))
    GAME_CONTROL_DELAY = float(os.getenv('GAME_CONTROL_DELAY', 0.1))