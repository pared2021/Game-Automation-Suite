class BlessingManager:
    def __init__(self):
        self.active_blessings = []

    def activate_blessing(self, blessing):
        self.active_blessings.append(blessing)

    def deactivate_blessing(self, blessing):
        if blessing in self.active_blessings:
            self.active_blessings.remove(blessing)

    def get_active_blessings(self):
        return self.active_blessings

    def apply_blessings(self, game_state):
        for blessing in self.active_blessings:
            game_state = blessing.apply(game_state)
        return game_state