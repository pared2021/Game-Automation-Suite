import asyncio
import unittest
from game_automation.ocr_prediction.ocr_utils import recognize_text, recognize_text_multilingual

class TestOCRUtils(unittest.TestCase):

    def setUp(self):
        # 设置测试环境
        self.test_image = "path/to/test/image.png"  # 替换为实际测试图像路径

    def test_recognize_text(self):
        result = asyncio.run(recognize_text(self.test_image))
        self.assertIsNotNone(result)  # 检查返回结果不为None

    def test_recognize_text_multilingual(self):
        result = asyncio.run(recognize_text_multilingual(self.test_image, languages=["en", "zh"]))
        self.assertIsNotNone(result)  # 检查返回结果不为None

if __name__ == '__main__':
    unittest.main()
