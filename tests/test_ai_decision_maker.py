import unittest
from unittest.mock import Mock, patch
import asyncio
from game_automation.ai.ai_decision_maker import AIDecisionMaker

class TestAIDecisionMaker(unittest.TestCase):
    def setUp(self):
        self.mock_game_engine = Mock()
        self.ai_decision_maker = AIDecisionMaker(self.mock_game_engine)

    @patch('random.random')
    async def test_make_decision(self, mock_random):
        mock_random.return_value = 0.5
        game_state = {'health': 20, 'gold': 50, 'enemy_count': 2}
        decision = await self.ai_decision_maker.make_decision(game_state)
        self.assertEqual(decision, 'use_health_potion')

        game_state = {'health': 100, 'gold': 150, 'enemy_count': 2}
        mock_random.return_value = 0.2
        decision = await self.ai_decision_maker.make_decision(game_state)
        self.assertEqual(decision, 'upgrade_equipment')

        game_state = {'health': 100, 'gold': 50, 'enemy_count': 4}
        decision = await self.ai_decision_maker.make_decision(game_state)
        self.assertEqual(decision, 'use_area_attack')

        game_state = {'health': 100, 'gold': 50, 'enemy_count': 2}
        decision = await self.ai_decision_maker.make_decision(game_state)
        self.assertEqual(decision, 'attack_strongest_enemy')

    async def test_execute_decision(self):
        await self.ai_decision_maker.execute_decision('use_health_potion')
        self.mock_game_engine.use_item.assert_called_once_with('health_potion')

        await self.ai_decision_maker.execute_decision('upgrade_equipment')
        self.mock_game_engine.upgrade_equipment.assert_called_once()

        await self.ai_decision_maker.execute_decision('use_area_attack')
        self.mock_game_engine.use_skill.assert_called_once_with('area_attack')

        self.mock_game_engine.find_strongest_enemy.return_value = {'id': 1}
        await self.ai_decision_maker.execute_decision('attack_strongest_enemy')
        self.mock_game_engine.attack.assert_called_once_with({'id': 1})

        with self.assertRaises(ValueError):
            await self.ai_decision_maker.execute_decision('unknown_decision')

if __name__ == '__main__':
    unittest.main()