import unittest
import asyncio
from game_automation.game_engine import GameEngine
from utils.config_manager import config_manager
from utils.performance_monitor import performance_monitor

class AutomatedTestSuite(unittest.TestCase):
    def setUp(self):
        self.game_engine = GameEngine()

    async def test_initialization(self):
        await self.game_engine.initialize()
        self.assertTrue(self.game_engine.emulator_manager.connected_emulator)
        self.assertIsNotNone(self.game_engine.task_manager.tasks)

    async def test_game_loop(self):
        await self.game_engine.initialize()
        asyncio.create_task(self.game_engine.run_game_loop())
        await asyncio.sleep(60)  # Run for 1 minute
        self.game_engine.stop_automation = True
        await asyncio.sleep(1)  # Allow time for the loop to stop
        self.assertGreater(len(self.game_engine.ai_decision_maker.memory), 0)

    async def test_scene_analysis(self):
        await self.game_engine.initialize()
        game_state = await self.game_engine.get_game_state()
        self.assertIn('scene_type', game_state)
        self.assertIn('key_elements', game_state)
        self.assertIn('potential_actions', game_state)

    async def test_task_management(self):
        await self.game_engine.initialize()
        tasks = self.game_engine.get_tasks()
        self.assertGreater(len(tasks), 0)
        current_task = self.game_engine.get_current_task()
        self.assertIsNotNone(current_task)

    async def test_performance_monitoring(self):
        await self.game_engine.initialize()
        asyncio.create_task(self.game_engine.run_game_loop())
        await asyncio.sleep(30)  # Run for 30 seconds
        self.game_engine.stop_automation = True
        performance_stats = performance_monitor.get_stats()
        self.assertIn('cpu_percent', performance_stats)
        self.assertIn('memory_percent', performance_stats)

    async def test_error_handling(self):
        await self.game_engine.initialize()
        self.game_engine.error_count = self.game_engine.max_errors
        await self.game_engine.check_and_handle_errors()
        self.assertEqual(self.game_engine.recovery_attempts, 1)

    async def test_run_mode_adjustment(self):
        await self.game_engine.initialize()
        self.game_engine.set_run_mode('power_saving')
        self.assertEqual(self.game_engine.run_mode, 'power_saving')
        self.game_engine.set_run_mode('high_performance')
        self.assertEqual(self.game_engine.run_mode, 'high_performance')

    def run_async_test(self, test_method):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_method())

    def test_all(self):
        self.run_async_test(self.test_initialization)
        self.run_async_test(self.test_game_loop)
        self.run_async_test(self.test_scene_analysis)
        self.run_async_test(self.test_task_management)
        self.run_async_test(self.test_performance_monitoring)
        self.run_async_test(self.test_error_handling)
        self.run_async_test(self.test_run_mode_adjustment)

if __name__ == '__main__':
    unittest.main()