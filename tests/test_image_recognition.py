import unittest
import numpy as np
import cv2
from game_automation.image_recognition import ImageRecognition

class TestImageRecognition(unittest.TestCase):
    def setUp(self):
        self.image_recognition = ImageRecognition()

    def test_template_matching(self):
        screen = np.zeros((100, 100, 3), dtype=np.uint8)
        template = np.ones((10, 10, 3), dtype=np.uint8) * 255
        screen[45:55, 45:55] = template

        cv2.imwrite('test_screen.png', screen)
        cv2.imwrite('test_template.png', template)

        result = self.image_recognition.template_matching('test_screen.png', 'test_template.png')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (45, 45, 10, 10))

    def test_find_color(self):
        screen = np.zeros((100, 100, 3), dtype=np.uint8)
        screen[30:40, 30:40] = [0, 0, 255]  # Red square

        cv2.imwrite('test_color_screen.png', screen)

        result = self.image_recognition.find_color('test_color_screen.png', [0, 0, 255])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (30, 30, 10, 10))

    def test_detect_edges(self):
        screen = np.zeros((100, 100), dtype=np.uint8)
        screen[40:60, 40:60] = 255

        cv2.imwrite('test_edge_screen.png', screen)

        edges = self.image_recognition.detect_edges('test_edge_screen.png')
        self.assertTrue(np.any(edges))

    def test_find_contours(self):
        screen = np.zeros((100, 100), dtype=np.uint8)
        screen[30:70, 30:70] = 255

        cv2.imwrite('test_contour_screen.png', screen)

        contours = self.image_recognition.find_contours('test_contour_screen.png')
        self.assertEqual(len(contours), 1)

    def test_match_feature_points(self):
        screen1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        screen2 = np.roll(screen1, 10)

        cv2.imwrite('test_feature_screen1.png', screen1)
        cv2.imwrite('test_feature_screen2.png', screen2)

        matches = self.image_recognition.match_feature_points('test_feature_screen1.png', 'test_feature_screen2.png')
        self.assertTrue(len(matches) > 0)

if __name__ == '__main__':
    unittest.main()