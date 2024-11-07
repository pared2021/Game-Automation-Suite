import unittest
from unittest.mock import Mock, patch
import asyncio
from game_automation.controllers.advanced_battle_strategy import AdvancedBattleStrategy

class TestAdvancedBattleStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AdvancedBattleStrategy('config/advanced_strategies.yaml')

    @patch('game_automation.controllers.advanced_battle_strategy.time')
    async def test_execute_loop(self, mock_time):
        mock_time.time.side_effect = [0, 1, 2, 61]
        mock_game_engine = Mock()
        mock_game_engine.execute_action = Mock()

        loop_phase = {
            'type': 'loop',
            'duration': 60,
            'actions': [{'type': 'tap', 'x': 100, 'y': 100}]
        }

        await self.strategy.execute_loop(loop_phase, mock_game_engine)
        self.assertEqual(mock_game_engine.execute_action.call_count, 3)

    async def test_execute_combo(self):
        mock_game_engine = Mock()
        mock_game_engine.execute_action = Mock()

        combo_actions = [
            {'type': 'tap', 'x': 100, 'y': 100, 'min_combo': 2},
            {'type': 'tap', 'x': 200, 'y': 200, 'min_combo': 3}
        ]

        self.strategy.combo_counter = 2
        await self.strategy.execute_combo(combo_actions, mock_game_engine)
        mock_game_engine.execute_action.assert_called_once_with({'type': 'tap', 'x': 100, 'y': 100, 'min_combo': 2})

    @patch('game_automation.controllers.advanced_battle_strategy.AdvancedBattleStrategy.evaluate_condition')
    async def test_execute_strategy(self, mock_evaluate_condition):
        mock_game_engine = Mock()
        mock_game_engine.execute_action = Mock()
        mock_evaluate_condition.return_value = True

        self.strategy.strategies['test'] = [
            {'type': 'action_sequence', 'actions': [{'type': 'tap', 'x': 100, 'y': 100}]},
            {'type': 'conditional', 'condition': {'type': 'health_percentage', 'value': 50}, 'actions': [{'type': 'tap', 'x': 200, 'y': 200}]},
            {'type': 'loop', 'duration': 10, 'actions': [{'type': 'tap', 'x': 300, 'y': 300}]},
            {'type': 'combo', 'actions': [{'type': 'tap', 'x': 400, 'y': 400, 'min_combo': 1}]}
        ]

        await self.strategy.execute_strategy('test', mock_game_engine)
        self.assertEqual(mock_game_engine.execute_action.call_count, 4)

    async def test_adaptive_action(self):
        mock_game_engine = Mock()
        mock_game_engine.get_game_state = Mock(return_value={'health': 50})
        mock_game_engine.execute_action = Mock()

        adaptive_phase = {
            'type': 'adaptive',
            'actions': [
                {'type': 'attack', 'damage': 10},
                {'type': 'heal', 'healing': 20}
            ]
        }

        await self.strategy.execute_adaptive_action(adaptive_phase, mock_game_engine)
        mock_game_engine.execute_action.assert_called_once_with({'type': 'heal', 'healing': 20})

if __name__ == '__main__':
    unittest.main()