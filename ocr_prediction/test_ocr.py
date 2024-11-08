import cv2
import numpy as np
import sys
import os

# 将当前目录添加到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from onnx_ocr import ONNXOCREngine, TextBoxType  # 使用绝对导入

def test_ocr_engine():
    # 初始化OCR引擎
    ocr_engine = ONNXOCREngine(use_angle_cls=True)

    # 测试图像路径
    test_image_path = "test_image.jpg"  # 请确保该图像存在于当前目录

    # 识别整个图像中的文本
    recognized_text = ocr_engine.recognize_text(test_image_path)
    print("识别的文本:", recognized_text)

    # 从指定区域提取文本
    region = (50, 50, 200, 100)  # 示例区域坐标
    extracted_text = ocr_engine.extract_text_from_region(test_image_path, region, TextBoxType.QUAD)
    print("提取的文本:", extracted_text)

if __name__ == "__main__":
    test_ocr_engine()
