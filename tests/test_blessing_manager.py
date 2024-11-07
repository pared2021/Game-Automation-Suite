import unittest
from game_automation.blessing.blessing_manager import BlessingManager

class TestBlessingManager(unittest.TestCase):
    def setUp(self):
        self.blessing_manager = BlessingManager()

    def test_activate_blessing(self):
        self.blessing_manager.activate_blessing("Test Blessing")
        self.assertIn("Test Blessing", self.blessing_manager.active_blessings)

    def test_deactivate_blessing(self):
        self.blessing_manager.activate_blessing("Test Blessing")
        self.blessing_manager.deactivate_blessing("Test Blessing")
        self.assertNotIn("Test Blessing", self.blessing_manager.active_blessings)

    def test_get_active_blessings(self):
        self.blessing_manager.activate_blessing("Blessing 1")
        self.blessing_manager.activate_blessing("Blessing 2")
        active_blessings = self.blessing_manager.get_active_blessings()
        self.assertEqual(set(active_blessings), set(["Blessing 1", "Blessing 2"]))

    def test_apply_blessings(self):
        class TestBlessing:
            def apply(self, game_state):
                game_state["health"] += 10
                return game_state

        self.blessing_manager.activate_blessing(TestBlessing())
        game_state = {"health": 100}
        new_state = self.blessing_manager.apply_blessings(game_state)
        self.assertEqual(new_state["health"], 110)

if __name__ == '__main__':
    unittest.main()