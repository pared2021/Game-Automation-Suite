import unittest
from unittest.mock import Mock, patch
import asyncio
from game_automation.controllers.auto_battle_strategy import AutoBattleStrategy

class TestAutoBattleStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AutoBattleStrategy('config/strategies.json')

    def test_get_strategy(self):
        default_strategy = self.strategy.get_strategy('default')
        self.assertIsNotNone(default_strategy)
        
        boss_strategy = self.strategy.get_strategy('boss_battle')
        self.assertIsNotNone(boss_strategy)

    @patch('game_automation.controllers.auto_battle_strategy.GameEngine')
    async def test_execute_strategy(self, mock_game_engine):
        mock_game_engine.perform_action_sequence = Mock()
        await self.strategy.execute_strategy('default', mock_game_engine)
        mock_game_engine.perform_action_sequence.assert_called()

    @patch('game_automation.controllers.auto_battle_strategy.GameEngine')
    async def test_conditional_strategy(self, mock_game_engine):
        mock_game_engine.find_game_object.return_value = [True]
        mock_game_engine.perform_action_sequence = Mock()
        
        conditional_strategy = [{
            'type': 'conditional',
            'condition': {'type': 'object_present', 'template': 'test.png'},
            'actions': [{'type': 'tap', 'x': 100, 'y': 100}]
        }]
        
        self.strategy.strategies['test_conditional'] = conditional_strategy
        await self.strategy.execute_strategy('test_conditional', mock_game_engine)
        mock_game_engine.perform_action_sequence.assert_called_once()

    @patch('game_automation.controllers.auto_battle_strategy.GameEngine')
    @patch('game_automation.controllers.auto_battle_strategy.time')
    async def test_loop_strategy(self, mock_time, mock_game_engine):
        mock_time.time.side_effect = [0, 1, 2, 61]  # Simulate time passing
        mock_game_engine.perform_action_sequence = Mock()
        
        loop_strategy = [{
            'type': 'loop',
            'duration': 60,
            'actions': [{'type': 'tap', 'x': 100, 'y': 100}]
        }]
        
        self.strategy.strategies['test_loop'] = loop_strategy
        await self.strategy.execute_strategy('test_loop', mock_game_engine)
        self.assertEqual(mock_game_engine.perform_action_sequence.call_count, 3)

if __name__ == '__main__':
    unittest.main()