import sys
import os
import cv2
import numpy as np
import asyncio
from game_automation.image_recognition import enhanced_image_recognition

# 确保模块路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_analyze_scene():
    image = cv2.imread('game_automation/resources/test_image.jpg')  # 替换为测试图像的路径
    results = await enhanced_image_recognition.analyze_scene(image)
    print("Scene Analysis Results:", results)

async def test_recognize_text():
    image = cv2.imread('game_automation/resources/test_image.jpg')  # 替换为测试图像的路径
    text = await enhanced_image_recognition.recognize_text(image)
    print("Recognized Text:", text)

async def test_detect_objects():
    image = cv2.imread('game_automation/resources/test_image.jpg')  # 替换为测试图像的路径
    objects = await enhanced_image_recognition.detect_objects(image)
    print("Detected Objects:", objects)

async def test_analyze_color_scheme():
    image = cv2.imread('game_automation/resources/test_image.jpg')  # 替换为测试图像的路径
    color_scheme = await enhanced_image_recognition.analyze_color_scheme(image)
    print("Dominant Color Scheme:", color_scheme)

async def test_segment_image():
    image = cv2.imread('game_automation/resources/test_image.jpg')  # 替换为测试图像的路径
    markers = await enhanced_image_recognition.segment_image(image)
    print("Image Segmentation Markers:", markers)

async def run_tests():
    await test_analyze_scene()
    await test_recognize_text()
    await test_detect_objects()
    await test_analyze_color_scheme()
    await test_segment_image()

if __name__ == "__main__":
    asyncio.run(run_tests())
