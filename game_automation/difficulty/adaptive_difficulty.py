from utils.logger import detailed_logger
from utils.config_manager import config_manager

class AdaptiveDifficulty:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('difficulty', {})
        self.current_difficulty = 1.0
        self.player_performance = []

    def update_difficulty(self, player_performance):
        self.player_performance.append(player_performance)
        if len(self.player_performance) >= self.config.get('performance_window', 10):
            avg_performance = sum(self.player_performance) / len(self.player_performance)
            if avg_performance > self.config.get('high_performance_threshold', 0.8):
                self.increase_difficulty()
            elif avg_performance < self.config.get('low_performance_threshold', 0.3):
                self.decrease_difficulty()
            self.player_performance = []

    def increase_difficulty(self):
        self.current_difficulty *= 1.1
        self.logger.info(f"Difficulty increased to {self.current_difficulty}")

    def decrease_difficulty(self):
        self.current_difficulty *= 0.9
        self.logger.info(f"Difficulty decreased to {self.current_difficulty}")

    def get_current_difficulty(self):
        return self.current_difficulty

adaptive_difficulty = AdaptiveDifficulty()
