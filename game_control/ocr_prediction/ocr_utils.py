import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
from functools import lru_cache

class OCRUtils:
    def __init__(self):
        with open('config/data_processing.json', 'r') as f:
            config = json.load(f)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.confidence_threshold = config['ocr_confidence_threshold']

    @lru_cache(maxsize=32)
    def recognize_text(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("输入必须是图像文件路径或 numpy 数组")

        result = self.ocr.ocr(image, cls=True)
        recognized_text = []
        for line in result:
            if line[1][1] >= self.confidence_threshold:
                recognized_text.append(line[1][0])
        return ' '.join(recognized_text)

    def extract_text_from_region(self, image, region):
        if isinstance(image, str):
            image = cv2.imread(image)
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        result = self.ocr.ocr(roi, cls=True)
        return ' '.join([line[1][0] for line in result if line[1][1] >= self.confidence_threshold])

    def find_text_location(self, image, target_text):
        if isinstance(image, str):
            image = cv2.imread(image)
        result = self.ocr.ocr(image, cls=True)
        for line in result:
            if target_text in line[1][0] and line[1][1] >= self.confidence_threshold:
                return line[0]  # 返回文本框的坐标
        return None

    def get_text_confidence(self, image, target_text):
        if isinstance(image, str):
            image = cv2.imread(image)
        result = self.ocr.ocr(image, cls=True)
        for line in result:
            if target_text == line[1][0]:
                return line[1][1]  # 返回置信度
        return 0.0

if __name__ == "__main__":
    # 测试 OCRUtils 类
    ocr_utils = OCRUtils()
    text = ocr_utils.recognize_text('test_image.png')
    print(f"Recognized text: {text}")
    region_text = ocr_utils.extract_text_from_region('test_image.png', (100, 100, 200, 50))
    print(f"Text in region: {region_text}")
    location = ocr_utils.find_text_location('test_image.png', '测试文本')
    print(f"Text location: {location}")
    confidence = ocr_utils.get_text_confidence('test_image.png', '测试文本')
    print(f"Text confidence: {confidence}")