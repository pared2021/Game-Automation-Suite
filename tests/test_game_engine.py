import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from game_automation.game_engine import GameEngine
from game_automation.actions.use_item_action import UseItemAction
from game_automation.actions.complete_task_action import CompleteTaskAction
from game_automation.actions.gather_resource_action import GatherResourceAction
from game_automation.actions.explore_area_action import ExploreAreaAction
from game_automation.actions.attack_enemy_action import AttackEnemyAction
from utils.error_handler import GameAutomationError, InputError, DeviceError, OCRError, NetworkError

class TestGameEngine(unittest.TestCase):
    def setUp(self):
        self.game_engine = GameEngine('test_strategy.json')

    @patch('game_automation.game_engine.AIDecisionMaker')
    @patch('game_automation.game_engine.DeviceManager')
    @patch('game_automation.game_engine.ImageRecognition')
    @patch('game_automation.game_engine.OCRUtils')
    async def test_run_game_loop(self, mock_ocr, mock_image, mock_device, mock_ai):
        mock_ai.return_value.make_decision.return_value = "use_health_potion"
        self.game_engine.get_game_state = AsyncMock(return_value={'health': 50})
        self.game_engine.execute_decision = AsyncMock(return_value=10)
        
        # Run the game loop for a short time
        asyncio.create_task(self.game_engine.run_game_loop())
        await asyncio.sleep(0.1)
        self.game_engine.stop_automation = True
        await asyncio.sleep(0.1)

        self.game_engine.get_game_state.assert_called()
        mock_ai.return_value.make_decision.assert_called()
        self.game_engine.execute_decision.assert_called_with("use_health_potion")

    async def test_execute_decision(self):
        self.game_engine.actions['use'] = AsyncMock(return_value=5)
        self.game_engine.actions['complete_task'] = AsyncMock(return_value=10)
        self.game_engine.actions['gather'] = AsyncMock(return_value=8)
        self.game_engine.actions['explore'] = AsyncMock(return_value=15)
        self.game_engine.actions['attack'] = AsyncMock(return_value=20)

        self.assertEqual(await self.game_engine.execute_decision("use_health_potion"), 5)
        self.assertEqual(await self.game_engine.execute_decision("complete_task_1"), 10)
        self.assertEqual(await self.game_engine.execute_decision("gather_wood"), 8)
        self.assertEqual(await self.game_engine.execute_decision("explore"), 15)
        self.assertEqual(await self.game_engine.execute_decision("attack"), 20)

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    @patch('game_automation.game_engine.GameEngine.execute_decision')
    async def test_error_handling(self, mock_execute, mock_get_state):
        mock_get_state.side_effect = [DeviceError("Test device error"), {'health': 100}]
        mock_execute.return_value = 10

        # Run the game loop for a short time
        asyncio.create_task(self.game_engine.run_game_loop())
        await asyncio.sleep(0.2)
        self.game_engine.stop_automation = True
        await asyncio.sleep(0.1)

        # Check that the loop continued after the error
        self.assertEqual(mock_get_state.call_count, 2)
        mock_execute.assert_called()

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    async def test_get_game_state(self, mock_get_state):
        mock_get_state.return_value = {
            'health': 80,
            'enemy_count': 3,
            'gold': 100,
            'level': 5,
            'text': 'Sample text',
            'objects': [{'type': 'enemy', 'position': (100, 100)}]
        }

        game_state = await self.game_engine.get_game_state()
        self.assertEqual(game_state['health'], 80)
        self.assertEqual(game_state['enemy_count'], 3)
        self.assertEqual(game_state['gold'], 100)
        self.assertEqual(game_state['level'], 5)
        self.assertEqual(game_state['text'], 'Sample text')
        self.assertEqual(len(game_state['objects']), 1)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.dump')
    async def test_save_game_state(self, mock_json_dump, mock_open):
        mock_game_state = {'health': 100, 'gold': 500}
        self.game_engine.get_game_state = AsyncMock(return_value=mock_game_state)

        await self.game_engine.save_game_state('test_save.json')

        mock_open.assert_called_with('test_save.json', 'w')
        mock_json_dump.assert_called_with(mock_game_state, mock_open())

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"health": 90, "gold": 450}')
    @patch('json.load')
    async def test_load_game_state(self, mock_json_load, mock_open):
        mock_game_state = {'health': 90, 'gold': 450}
        mock_json_load.return_value = mock_game_state

        loaded_state = await self.game_engine.load_game_state('test_load.json')

        mock_open.assert_called_with('test_load.json', 'r')
        self.assertEqual(loaded_state, mock_game_state)

    @patch('game_automation.game_engine.GameEngine.get_game_state')
    async def test_game_state_components(self, mock_get_state):
        mock_get_state.return_value = {
            'health': 75,
            'enemy_count': 2,
            'gold': 200,
            'level': 3,
            'text': 'Battle in progress',
            'objects': [{'type': 'enemy', 'position': (100, 100)}, {'type': 'item', 'position': (200, 200)}]
        }

        game_state = await self.game_engine.get_game_state()
        
        self.assertIn('health', game_state)
        self.assertIn('enemy_count', game_state)
        self.assertIn('gold', game_state)
        self.assertIn('level', game_state)
        self.assertIn('text', game_state)
        self.assertIn('objects', game_state)
        
        self.assertEqual(len(game_state['objects']), 2)
        self.assertEqual(game_state['objects'][0]['type'], 'enemy')
        self.assertEqual(game_state['objects'][1]['type'], 'item')

if __name__ == '__main__':
    unittest.main()