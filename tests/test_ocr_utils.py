import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from game_automation.ocr_prediction.ocr_utils import OCRUtils

class TestOCRUtils(unittest.TestCase):
    def setUp(self):
        self.ocr_utils = OCRUtils()

    @patch('game_automation.ocr_prediction.ocr_utils.PaddleOCR')
    def test_recognize_text(self, mock_paddle_ocr):
        mock_paddle_ocr.return_value.ocr.return_value = [
            [[[0, 0], [100, 0], [100, 40], [0, 40]], ('Hello', 0.99)],
            [[[0, 50], [100, 50], [100, 90], [0, 90]], ('World', 0.98)]
        ]
        result = self.ocr_utils.recognize_text('test_image.png')
        self.assertEqual(result, 'Hello World')

    @patch('game_automation.ocr_prediction.ocr_utils.PaddleOCR')
    def test_extract_text_from_region(self, mock_paddle_ocr):
        mock_paddle_ocr.return_value.ocr.return_value = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], ('Test', 0.95)]
        ]
        result = self.ocr_utils.extract_text_from_region('test_image.png', (0, 0, 50, 20))
        self.assertEqual(result, 'Test')

    @patch('game_automation.ocr_prediction.ocr_utils.PaddleOCR')
    def test_find_text_location(self, mock_paddle_ocr):
        mock_paddle_ocr.return_value.ocr.return_value = [
            [[[10, 10], [60, 10], [60, 30], [10, 30]], ('Target', 0.97)]
        ]
        result = self.ocr_utils.find_text_location('test_image.png', 'Target')
        self.assertEqual(result, [[10, 10], [60, 10], [60, 30], [10, 30]])

    @patch('game_automation.ocr_prediction.ocr_utils.PaddleOCR')
    def test_get_text_confidence(self, mock_paddle_ocr):
        mock_paddle_ocr.return_value.ocr.return_value = [
            [[[0, 0], [100, 0], [100, 40], [0, 40]], ('Confidence', 0.95)]
        ]
        result = self.ocr_utils.get_text_confidence('test_image.png', 'Confidence')
        self.assertEqual(result, 0.95)

if __name__ == '__main__':
    unittest.main()