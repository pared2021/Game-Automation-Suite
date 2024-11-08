import cv2
import numpy as np
from enum import Enum
from typing import Union, Tuple
from paddleocr import PaddleOCR

class OCREngine(Enum):
    PADDLE = "paddle"
    ONNX = "onnx"

class OCRUtils:
    def __init__(self, engine: Union[str, OCREngine] = OCREngine.PADDLE):
        """
        初始化OCR工具类
        Args:
            engine: OCR引擎类型，可选 'paddle' 或 'onnx'
        """
        if isinstance(engine, str):
            engine = OCREngine(engine.lower())
        
        self.engine_type = engine
        
        if engine == OCREngine.PADDLE:
            self.engine = PaddleOCR(use_angle_cls=True, lang='ch')
        else:
            from onnx_ocr import ONNXOCREngine  # 使用绝对导入
            self.engine = ONNXOCREngine()

    def recognize_text(self, image: Union[str, np.ndarray]) -> str:
        """
        识别图像中的文本
        Args:
            image: 图像文件路径或numpy数组
        Returns:
            识别出的文本字符串
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("输入必须是图像文件路径或numpy数组")

        if self.engine_type == OCREngine.PADDLE:
            result = self.engine.ocr(image, cls=True)
            recognized_text = []
            for line in result:
                recognized_text.append(line[1][0])  # 提取文本内容
            print(f"识别结果: {recognized_text}")  # 输出识别结果
            return ' '.join([text for text in recognized_text if isinstance(text, str)])  # 确保返回字符串
        else:
            return self.engine.recognize_text(image)

    def extract_text_from_region(self, image: Union[str, np.ndarray], region: Tuple[int, int, int, int]) -> str:
        """
        从图像指定区域提取文本
        Args:
            image: 图像文件路径或numpy数组
            region: 区域坐标(x, y, w, h)
        Returns:
            识别出的文本字符串
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        if self.engine_type == OCREngine.PADDLE:
            result = self.engine.ocr(roi, cls=True)
            return ' '.join([line[1][0] for line in result])
        else:
            return self.engine.extract_text_from_region(image, region)

    @staticmethod
    def get_available_engines() -> list:
        """
        获取可用的OCR引擎列表
        Returns:
            可用引擎列表
        """
        engines = [OCREngine.PADDLE]
        try:
            from onnx_ocr import ONNXOCREngine  # 使用绝对导入
            engines.append(OCREngine.ONNX)
        except ImportError:
            pass
        return engines
