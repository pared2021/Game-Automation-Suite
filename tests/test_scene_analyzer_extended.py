import unittest
import asyncio
from unittest.mock import Mock, patch
import numpy as np
from game_automation.scene_understanding.scene_analyzer import SceneAnalyzer

class TestSceneAnalyzer(unittest.TestCase):
    def setUp(self):
        self.scene_analyzer = SceneAnalyzer()
        self.mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.mock_game_state = {
            'scene_type': 'battle',
            'elements': ['player_character', 'enemy', 'health_bar']
        }

    async def test_scene_classification(self):
        result = await self.scene_analyzer.analyze_scene_type(self.mock_image)
        self.assertEqual(result, 'unknown')  # 根据当前逻辑，默认返回unknown

    async def test_element_detection(self):
        elements = await self.scene_analyzer.detect_elements(self.mock_image)
        self.assertIsInstance(elements, list)

    async def test_action_validation(self):
        valid_actions = await self.scene_analyzer.get_valid_actions(self.mock_game_state)
        self.assertIsInstance(valid_actions, list)

    async def test_state_analysis(self):
        state = await self.scene_analyzer.analyze_game_state(self.mock_game_state)
        self.assertIn('current_scene', state)

    async def test_scene_change_detection(self):
        previous_state = {'elements': ['player_character']}
        new_state = {'elements': ['player_character', 'enemy']}
        changes = await self.scene_analyzer.detect_scene_changes(previous_state, new_state)
        self.assertTrue(changes['has_changes'])

    async def test_risk_assessment(self):
        risk_level = await self.scene_analyzer.assess_risk(self.mock_game_state)
        self.assertIsInstance(risk_level, float)

    async def test_priority_calculation(self):
        priorities = await self.scene_analyzer.calculate_priorities(self.mock_game_state)
        self.assertIsInstance(priorities, dict)

    async def test_resource_analysis(self):
        resources = await self.scene_analyzer.analyze_resources(self.mock_game_state)
        self.assertIsInstance(resources, dict)

    async def test_pattern_recognition(self):
        patterns = await self.scene_analyzer.recognize_patterns([self.mock_game_state] * 5)
        self.assertIsInstance(patterns, list)

    async def test_concurrent_analysis(self):
        async def analyze_concurrent():
            tasks = []
            for _ in range(10):
                tasks.append(asyncio.create_task(
                    self.scene_analyzer.analyze_game_scene(self.mock_image, "Test text")
                ))
            results = await asyncio.gather(*tasks)
            return results

        results = await analyze_concurrent()
        self.assertEqual(len(results), 10)

    async def test_error_handling(self):
        with self.assertRaises(ValueError):
            await self.scene_analyzer.analyze_scene_type(None)

    def run_async_test(self, test_method):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_method())

    def test_all_async(self):
        async def run_all_tests():
            await self.test_scene_classification()
            await self.test_element_detection()
            await self.test_action_validation()
            await self.test_state_analysis()
            await self.test_scene_change_detection()
            await self.test_risk_assessment()
            await self.test_priority_calculation()
            await self.test_resource_analysis()
            await self.test_pattern_recognition()
            await self.test_concurrent_analysis()
            await self.test_error_handling()

        asyncio.run(run_all_tests())

if __name__ == '__main__':
    unittest.main()
