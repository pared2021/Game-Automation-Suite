import sys
import os
import asyncio
import cv2

# 添加模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game_automation.image_recognition import EnhancedImageRecognition

async def test_analyze_scene():
    recognizer = EnhancedImageRecognition()
    # 使用有效的测试图像进行测试
    image = cv2.imread('C:/Users/www11/Documents/leidian64/Pictures/Screenshots/Screenshot_20241108-115456.png')
    result = await recognizer.analyze_scene(image)
    print("Analyze Scene Result:", result)

async def test_detect_objects():
    recognizer = EnhancedImageRecognition()
    # 使用有效的测试图像进行测试
    image = cv2.imread('C:/Users/www11/Documents/leidian64/Pictures/Screenshots/Screenshot_20241108-115456.png')
    result = await recognizer.detect_objects(image)
    print("Detect Objects Result:", result)

async def test_analyze_color_scheme():
    recognizer = EnhancedImageRecognition()
    # 使用有效的测试图像进行测试
    image = cv2.imread('C:/Users/www11/Documents/leidian64/Pictures/Screenshots/Screenshot_20241108-115456.png')
    result = await recognizer.analyze_color_scheme(image)
    print("Analyze Color Scheme Result:", result)

async def test_detect_edges():
    recognizer = EnhancedImageRecognition()
    # 使用有效的测试图像进行测试
    image = cv2.imread('C:/Users/www11/Documents/leidian64/Pictures/Screenshots/Screenshot_20241108-115456.png')
    result = await recognizer.detect_edges(image)
    print("Detect Edges Result:", result)

async def test_segment_image():
    recognizer = EnhancedImageRecognition()
    # 使用有效的测试图像进行测试
    image = cv2.imread('C:/Users/www11/Documents/leidian64/Pictures/Screenshots/Screenshot_20241108-115456.png')
    result = await recognizer.segment_image(image)
    print("Segment Image Result:", result)

if __name__ == "__main__":
    await test_analyze_scene()
    await test_detect_objects()
    await test_analyze_color_scheme()
    await test_detect_edges()
    await test_segment_image()
