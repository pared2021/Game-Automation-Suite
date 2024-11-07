import unittest
from unittest.mock import Mock, patch
import asyncio
from game_automation.game_engine import GameEngine

class TestGameScenarios(unittest.TestCase):
    def setUp(self):
        self.game_engine = GameEngine('test_strategy.json')

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_action')
    async def test_normal_battle_scenario(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [
            {'state': 'battle', 'health': 100, 'enemies': 2},
            {'state': 'battle', 'health': 80, 'enemies': 1},
            {'state': 'victory', 'health': 80, 'enemies': 0}
        ]
        self.game_engine.ai_decision_maker.make_decision = Mock(return_value={'type': 'use_skill', 'skill': 'attack'})

        await self.game_engine.run_automation_loop()

        self.assertEqual(mock_execute.call_count, 3)
        mock_execute.assert_any_call({'type': 'use_skill', 'skill': 'attack'})

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_action')
    async def test_boss_battle_scenario(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [
            {'state': 'boss_battle', 'health': 100, 'boss_health': 1000},
            {'state': 'boss_battle', 'health': 70, 'boss_health': 800},
            {'state': 'boss_battle', 'health': 50, 'boss_health': 500},
            {'state': 'victory', 'health': 30, 'boss_health': 0}
        ]
        self.game_engine.ai_decision_maker.make_decision = Mock(side_effect=[
            {'type': 'use_skill', 'skill': 'ultimate'},
            {'type': 'use_item', 'item': 'health_potion'},
            {'type': 'use_skill', 'skill': 'ultimate'}
        ])

        await self.game_engine.run_automation_loop()

        self.assertEqual(mock_execute.call_count, 3)
        mock_execute.assert_any_call({'type': 'use_skill', 'skill': 'ultimate'})
        mock_execute.assert_any_call({'type': 'use_item', 'item': 'health_potion'})

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_action')
    async def test_low_health_scenario(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [
            {'state': 'battle', 'health': 20, 'enemies': 2},
            {'state': 'battle', 'health': 70, 'enemies': 2},
            {'state': 'victory', 'health': 70, 'enemies': 0}
        ]
        self.game_engine.ai_decision_maker.make_decision = Mock(side_effect=[
            {'type': 'use_item', 'item': 'health_potion'},
            {'type': 'use_skill', 'skill': 'attack'}
        ])

        await self.game_engine.run_automation_loop()

        self.assertEqual(mock_execute.call_count, 2)
        mock_execute.assert_any_call({'type': 'use_item', 'item': 'health_potion'})

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_action')
    async def test_game_over_scenario(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [
            {'state': 'battle', 'health': 10, 'enemies': 2},
            {'state': 'game_over', 'health': 0, 'enemies': 2}
        ]
        self.game_engine.ai_decision_maker.make_decision = Mock(return_value={'type': 'use_item', 'item': 'health_potion'})

        await self.game_engine.run_automation_loop()

        self.assertEqual(mock_execute.call_count, 1)
        self.assertFalse(self.game_engine.running)

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_action')
    async def test_invincible_state_scenario(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [
            {'state': 'battle', 'health': 100, 'enemies': 2, 'invincible': True},
            {'state': 'battle', 'health': 100, 'enemies': 1, 'invincible': True},
            {'state': 'victory', 'health': 100, 'enemies': 0, 'invincible': False}
        ]
        self.game_engine.ai_decision_maker.make_decision = Mock(return_value={'type': 'use_skill', 'skill': 'attack'})

        await self.game_engine.run_automation_loop()

        self.assertEqual(mock_execute.call_count, 3)
        for call in mock_execute.call_args_list:
            self.assertEqual(call[0][0], {'type': 'use_skill', 'skill': 'attack'})

if __name__ == '__main__':
    unittest.main()