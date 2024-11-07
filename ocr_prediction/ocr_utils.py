import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRUtils:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    def recognize_text(self, image):
        if isinstance(image, str):
            # 如果输入是文件路径，则读取图像
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            # 如果输入已经是 numpy 数组，则直接使用
            pass
        else:
            raise ValueError("输入必须是图像文件路径或 numpy 数组")

        result = self.ocr.ocr(image, cls=True)
        recognized_text = []
        for line in result:
            recognized_text.append(line[1][0])
        return ' '.join(recognized_text)

    def extract_text_from_region(self, image, region):
        if isinstance(image, str):
            image = cv2.imread(image)
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        result = self.ocr.ocr(roi, cls=True)
        return ' '.join([line[1][0] for line in result])