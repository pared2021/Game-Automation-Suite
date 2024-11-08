import unittest
import asyncio
from unittest.mock import Mock, patch
import numpy as np
import cv2
from game_automation.image_recognition import EnhancedImageRecognition

class TestImageRecognition(unittest.TestCase):
    def setUp(self):
        self.image_recognition = EnhancedImageRecognition()
        self.mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.mock_image, (30, 30), (70, 70), (255, 255, 255), -1)

    async def test_template_matching(self):
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.mock_image
            result = await self.image_recognition.analyze_scene(self.mock_image)
            self.assertIsNotNone(result)
            self.assertTrue(isinstance(result, list))

    async def test_color_detection(self):
        dominant_color = await self.image_recognition.analyze_color_scheme(self.mock_image)
        self.assertIsInstance(dominant_color, list)
        self.assertEqual(len(dominant_color), 3)  # RGB值应该有3个分量

    async def test_edge_detection(self):
        edges = await self.image_recognition.detect_edges(self.mock_image)
        self.assertEqual(edges.shape, (100, 100))  # 应该保持原始图像的尺寸

    async def test_image_segmentation(self):
        segments = await self.image_recognition.segment_image(self.mock_image)
        self.assertEqual(segments.shape, (100, 100))  # 分割结果应该保持原始图像的尺寸

    async def test_error_handling(self):
        with self.assertRaises(ValueError):
            await self.image_recognition.analyze_scene(None)

    async def test_performance(self):
        start_time = asyncio.get_event_loop().time()
        for _ in range(10):  # 减少迭代次数以加快测试
            await self.image_recognition.detect_edges(self.mock_image)
        processing_time = asyncio.get_event_loop().time() - start_time
        self.assertLess(processing_time, 2.0)  # 确保10次处理在2秒内完成

    def test_all(self):
        async def run_all_tests():
            await self.test_template_matching()
            await self.test_color_detection()
            await self.test_edge_detection()
            await self.test_image_segmentation()
            await self.test_error_handling()
            await self.test_performance()

        asyncio.run(run_all_tests())

if __name__ == '__main__':
    unittest.main()
