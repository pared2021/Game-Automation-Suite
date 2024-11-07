import unittest
from game_automation.rogue.rogue_manager import RogueManager
from game_automation.blessing.advanced_blessing_manager import AdvancedBlessingManager

class TestRogueManager(unittest.TestCase):
    def setUp(self):
        self.rogue_manager = RogueManager()

    def test_start_new_run(self):
        self.rogue_manager.start_new_run()
        self.assertEqual(self.rogue_manager.current_level, 0)
        self.assertEqual(self.rogue_manager.player_health, 100)
        self.assertEqual(self.rogue_manager.player_gold, 0)

    def test_advance_level(self):
        self.rogue_manager.start_new_run()
        initial_gold = self.rogue_manager.player_gold
        self.rogue_manager.advance_level()
        self.assertEqual(self.rogue_manager.current_level, 1)
        self.assertGreater(self.rogue_manager.player_gold, initial_gold)

    def test_take_damage(self):
        self.rogue_manager.start_new_run()
        initial_health = self.rogue_manager.player_health
        damage = 20
        self.rogue_manager.take_damage(damage)
        self.assertEqual(self.rogue_manager.player_health, initial_health - damage)

    def test_end_run(self):
        self.rogue_manager.start_new_run()
        self.rogue_manager.current_level = 5
        self.rogue_manager.end_run()
        self.assertEqual(self.rogue_manager.current_level, 0)

    def test_offer_blessing(self):
        self.rogue_manager.start_new_run()
        self.rogue_manager.offer_blessing()
        self.assertGreater(len(self.rogue_manager.blessing_manager.active_blessings), 0)

    def test_get_current_state(self):
        self.rogue_manager.start_new_run()
        state = self.rogue_manager.get_current_state()
        self.assertIn('level', state)
        self.assertIn('health', state)
        self.assertIn('gold', state)
        self.assertIn('blessings', state)

if __name__ == '__main__':
    unittest.main()