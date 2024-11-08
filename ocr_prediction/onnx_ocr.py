import cv2
import numpy as np
import os
from enum import Enum  # 添加对Enum的导入
from typing import List, Tuple, Union

class TextBoxType(Enum):
    QUAD = "quad"
    POLY = "poly"

class RecognitionAlgorithm(Enum):
    CRNN = "crnn"
    RARE = "rare"
    SAR = "sar"

class ONNXOCREngine:
    def __init__(self, models_dir: str = None, use_angle_cls: bool = False, detection_params: dict = None, recognition_algorithm: RecognitionAlgorithm = RecognitionAlgorithm.CRNN):
        """
        初始化ONNX OCR引擎
        Args:
            models_dir: ONNX模型文件目录
            use_angle_cls: 是否使用文字方向分类
            detection_params: 检测参数字典
            recognition_algorithm: 识别算法类型
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models', 'onnx_ocr')
        
        self.models_dir = models_dir
        self.det_model_dir = os.path.join(models_dir, 'det.onnx')
        self.rec_model_dir = os.path.join(models_dir, 'rec.onnx')
        self.cls_model_dir = os.path.join(models_dir, 'cls.onnx')
        self.char_dict_path = os.path.join(models_dir, 'ppocr_keys_v1.txt')
        self.use_angle_cls = use_angle_cls
        self.recognition_algorithm = recognition_algorithm
        
        from onnxocr.onnx_paddleocr import ONNXPaddleOcr
        self.ocr = ONNXPaddleOcr(
            use_angle_cls=self.use_angle_cls,
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            cls_model_dir=self.cls_model_dir,
            rec_char_dict_path=self.char_dict_path,
            detection_params=detection_params or {}
        )

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

        result = self.ocr.ocr(image)
        if not result or not result[0]:
            return ""
            
        recognized_text = []
        for line in result[0]:
            if isinstance(line, list) and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], tuple) else line[1]
                recognized_text.append(text)
        return ' '.join(recognized_text)

    def extract_text_from_region(self, image: Union[str, np.ndarray], region: Tuple[int, int, int, int], box_type: TextBoxType = TextBoxType.QUAD) -> str:
        """
        从图像指定区域提取文本
        Args:
            image: 图像文件路径或numpy数组
            region: 区域坐标(x, y, w, h)
            box_type: 文本框类型
        Returns:
            识别出的文本字符串
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        return self.recognize_text(roi)
