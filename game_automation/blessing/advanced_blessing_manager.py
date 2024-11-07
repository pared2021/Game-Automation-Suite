import random
from .blessing_manager import BlessingManager

class Blessing:
    def __init__(self, name, effect, tier):
        self.name = name
        self.effect = effect
        self.tier = tier

    def apply(self, game_state):
        return self.effect(game_state)

    def __str__(self):
        return f"{self.name} ({self.tier})"

class AdvancedBlessingManager(BlessingManager):
    def __init__(self):
        super().__init__()
        self.blessing_tiers = {
            'common': [],
            'rare': [],
            'epic': [],
            'legendary': []
        }
        self.initialize_blessings()

    def initialize_blessings(self):
        self.add_blessing(Blessing("Health Boost", lambda state: {**state, "health": state["health"] * 1.1}, "common"))
        self.add_blessing(Blessing("Gold Rush", lambda state: {**state, "gold": state["gold"] * 1.2}, "rare"))
        self.add_blessing(Blessing("Double Strike", lambda state: {**state, "damage": state.get("damage", 1) * 2}, "epic"))
        self.add_blessing(Blessing("Immortality", lambda state: {**state, "health": float('inf')}, "legendary"))

    def add_blessing(self, blessing):
        if blessing.tier in self.blessing_tiers:
            self.blessing_tiers[blessing.tier].append(blessing)
        else:
            raise ValueError(f"Invalid blessing tier: {blessing.tier}")

    def get_random_blessing(self, tier):
        if tier in self.blessing_tiers and self.blessing_tiers[tier]:
            return random.choice(self.blessing_tiers[tier])
        return None

    def apply_blessings(self, game_state):
        for blessing in self.active_blessings:
            game_state = blessing.apply(game_state)
        return game_state

    def upgrade_blessing(self, blessing):
        tiers = list(self.blessing_tiers.keys())
        current_index = tiers.index(blessing.tier)
        if current_index < len(tiers) - 1:
            next_tier = tiers[current_index + 1]
            blessing.tier = next_tier
            return True
        return False