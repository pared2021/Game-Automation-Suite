import random
from ..blessing.advanced_blessing_manager import AdvancedBlessingManager

class RogueManager:
    def __init__(self):
        self.current_level = 0
        self.max_level = 10
        self.player_health = 100
        self.player_gold = 0
        self.blessing_manager = AdvancedBlessingManager()
        self.is_running = False

    def start_new_run(self):
        self.current_level = 0
        self.player_health = 100
        self.player_gold = 0
        self.blessing_manager.active_blessings.clear()
        self.is_running = True

    def advance_level(self):
        if not self.is_running:
            return False
        self.current_level += 1
        self.player_gold += random.randint(10, 50)
        if self.current_level % 3 == 0:
            self.offer_blessing()
        return self.current_level <= self.max_level

    def offer_blessing(self):
        tiers = ['common', 'rare', 'epic', 'legendary']
        weights = [0.6, 0.3, 0.08, 0.02]
        chosen_tier = random.choices(tiers, weights=weights)[0]
        blessing = self.blessing_manager.get_random_blessing(chosen_tier)
        if blessing:
            self.blessing_manager.activate_blessing(blessing)

    def take_damage(self, amount):
        if not self.is_running:
            return
        self.player_health -= amount
        if self.player_health <= 0:
            self.end_run()

    def end_run(self):
        self.is_running = False

    def get_current_state(self):
        return {
            "level": self.current_level,
            "health": self.player_health,
            "gold": self.player_gold,
            "blessings": [str(blessing) for blessing in self.blessing_manager.active_blessings],
            "is_running": self.is_running
        }

rogue_manager = RogueManager()