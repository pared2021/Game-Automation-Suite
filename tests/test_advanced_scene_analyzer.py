import pytest
import cv2
import numpy as np
import os
from datetime import datetime

from game_automation.scene_understanding.advanced_scene_analyzer import AdvancedSceneAnalyzer

@pytest.fixture
def advanced_analyzer():
    """创建高级场景分析器实例"""
    test_template_dir = "tests/resources/templates"
    if not os.path.exists(test_template_dir):
        os.makedirs(test_template_dir)
    return AdvancedSceneAnalyzer(template_dir=test_template_dir)

@pytest.fixture
def test_image():
    """创建测试用图像"""
    # 创建一个包含文本和物体的测试图像
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 添加一个矩形物体
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    
    # 添加一个圆形物体
    cv2.circle(image, (300, 100), 50, (0, 0, 255), -1)
    
    # 添加文本区域（模拟文本的矩形）
    cv2.putText(image, "Test Text", (100, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

@pytest.fixture
def motion_sequence():
    """创建用于测试运动检测的图像序列"""
    sequence = []
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 创建一个移动的物体序列
    for i in range(3):
        frame = image.copy()
        cv2.circle(frame, (50 + i*50, 100), 20, (255, 255, 255), -1)
        sequence.append(frame)
    
    return sequence

def test_feature_extraction(advanced_analyzer, test_image):
    """测试特征提取功能"""
    result = advanced_analyzer.analyze_screenshot(test_image)
    
    # 验证特征提取结果
    assert 'features' in result
    features = result['features']
    
    assert 'keypoint_count' in features
    assert features['keypoint_count'] > 0
    
    assert 'color_distribution' in features
    assert len(features['color_distribution']) > 0
    
    assert 'edge_density' in features
    assert 0 <= features['edge_density'] <= 1

def test_motion_detection(advanced_analyzer, motion_sequence):
    """测试运动检测功能"""
    # 分析连续帧
    for frame in motion_sequence:
        result = advanced_analyzer.analyze_screenshot(frame)
    
    # 验证运动检测结果
    assert 'motion' in result
    motion = result['motion']
    
    assert 'motion_level' in motion
    assert 0 <= motion['motion_level'] <= 1
    
    assert 'direction' in motion
    assert motion['direction'] in [None, 'approaching', 'receding', 'static']

def test_object_detection(advanced_analyzer, test_image):
    """测试物体检测功能"""
    result = advanced_analyzer.analyze_screenshot(test_image)
    
    # 验证物体检测结果
    assert 'objects' in result
    objects = result['objects']
    
    assert isinstance(objects, list)
    assert len(objects) > 0  # 应该至少检测到一个物体
    
    for obj in objects:
        assert 'position' in obj
        assert 'size' in obj
        assert 'area' in obj
        assert 'aspect_ratio' in obj
        
        # 验证位置和尺寸的有效性
        x, y = obj['position']
        w, h = obj['size']
        assert 0 <= x < test_image.shape[1]
        assert 0 <= y < test_image.shape[0]
        assert w > 0 and h > 0

def test_text_region_detection(advanced_analyzer, test_image):
    """测试文本区域检测功能"""
    result = advanced_analyzer.analyze_screenshot(test_image)
    
    # 验证文本区域检测结果
    assert 'text_regions' in result
    text_regions = result['text_regions']
    
    assert isinstance(text_regions, list)
    assert len(text_regions) > 0  # 应该至少检测到一个文本区域
    
    for region in text_regions:
        assert 'position' in region
        assert 'size' in region
        assert 'confidence' in region
        
        # 验证置信度范围
        assert 0 <= region['confidence'] <= 1

def test_scene_history(advanced_analyzer, test_image):
    """测试场景历史记录功能"""
    # 分析多个场景
    for _ in range(5):
        advanced_analyzer.analyze_screenshot(test_image)
    
    # 获取历史记录
    history = advanced_analyzer.get_scene_history()
    
    assert isinstance(history, list)
    assert len(history) > 0
    
    for entry in history:
        assert 'timestamp' in entry
        assert isinstance(entry['timestamp'], datetime)
        assert 'scene_type' in entry
        assert 'features' in entry

def test_history_size_limit(advanced_analyzer, test_image):
    """测试场景历史记录大小限制"""
    history_size = advanced_analyzer.scene_history.maxlen
    
    # 添加超过限制的场景
    for _ in range(history_size + 5):
        advanced_analyzer.analyze_screenshot(test_image)
    
    history = advanced_analyzer.get_scene_history()
    assert len(history) == history_size

def test_clear_history(advanced_analyzer, test_image):
    """测试清除历史记录功能"""
    # 添加一些场景
    for _ in range(5):
        advanced_analyzer.analyze_screenshot(test_image)
    
    # 清除历史
    advanced_analyzer.clear_history()
    
    history = advanced_analyzer.get_scene_history()
    assert len(history) == 0

def test_color_histogram(advanced_analyzer, test_image):
    """测试颜色直方图计算"""
    result = advanced_analyzer.analyze_screenshot(test_image)
    
    # 验证颜色直方图
    features = result['features']
    assert 'color_distribution' in features
    histogram = features['color_distribution']
    
    assert isinstance(histogram, list)
    assert len(histogram) > 0
    assert all(0 <= value <= 1 for value in histogram)

def test_cleanup():
    """清理测试资源"""
    test_template_dir = "tests/resources/templates"
    if os.path.exists(test_template_dir):
        for file in os.listdir(test_template_dir):
            os.remove(os.path.join(test_template_dir, file))
        os.rmdir(test_template_dir)
    
    test_resources_dir = "tests/resources"
    if os.path.exists(test_resources_dir):
        os.rmdir(test_resources_dir)
