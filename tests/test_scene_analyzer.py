import pytest
import cv2
import numpy as np
import os
from pathlib import Path

from game_automation.scene_understanding.scene_analyzer import SceneAnalyzer, SceneAnalysisError

@pytest.fixture
def scene_analyzer():
    """创建场景分析器实例"""
    # 使用测试专用的模板目录
    test_template_dir = "tests/resources/templates"
    if not os.path.exists(test_template_dir):
        os.makedirs(test_template_dir)
    return SceneAnalyzer(template_dir=test_template_dir)

@pytest.fixture
def test_image():
    """创建测试用图像"""
    # 创建一个简单的测试图像
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 添加一些特征
    cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
    cv2.circle(image, (50, 50), 20, (0, 0, 255), -1)
    return image

def test_image_preprocessing(scene_analyzer, test_image):
    """测试图像预处理功能"""
    # 预处理图像
    processed = scene_analyzer._preprocess_image(test_image)
    
    # 验证预处理结果
    assert processed is not None
    assert isinstance(processed, np.ndarray)
    assert processed.shape[:2] == test_image.shape[:2]  # 确保尺寸相同
    assert len(processed.shape) == 2  # 确保是灰度图

def test_scene_matching(scene_analyzer, test_image):
    """测试场景匹配功能"""
    # 保存测试图像作为模板
    scene_analyzer.save_template(test_image, "test_scene")
    
    # 创建一个相似的图像进行匹配
    similar_image = test_image.copy()
    # 添加一些噪声
    noise = np.random.normal(0, 10, similar_image.shape).astype(np.uint8)
    similar_image = cv2.add(similar_image, noise)
    
    # 分析场景
    result = scene_analyzer.analyze_screenshot(similar_image)
    
    # 验证结果
    assert result is not None
    assert isinstance(result, dict)
    assert 'scene_type' in result
    assert result['scene_type'] in ['test_scene', 'unknown']

def test_scene_state_extraction(scene_analyzer, test_image):
    """测试场景状态提取功能"""
    # 分析场景状态
    result = scene_analyzer.analyze_screenshot(test_image)
    
    # 验证状态信息
    assert 'scene_state' in result
    assert isinstance(result['scene_state'], dict)
    assert 'brightness' in result['scene_state']
    assert 'complexity' in result['scene_state']
    
    # 验证数值合理性
    assert 0 <= result['scene_state']['brightness'] <= 255
    assert result['scene_state']['complexity'] >= 0

def test_scene_change_detection(scene_analyzer, test_image):
    """测试场景切换检测功能"""
    # 第一次分析
    first_result = scene_analyzer.analyze_screenshot(test_image)
    assert not first_result['scene_changed']  # 首次分析应该返回False
    
    # 创建一个不同的图像
    different_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(different_image, (50, 50), 40, (255, 255, 255), -1)
    
    # 分析新图像
    second_result = scene_analyzer.analyze_screenshot(different_image)
    assert second_result['scene_changed']  # 场景改变应该返回True

def test_template_management(scene_analyzer, test_image):
    """测试模板管理功能"""
    # 保存模板
    success = scene_analyzer.save_template(test_image, "test_template")
    assert success
    
    # 验证模板是否被正确保存
    template_path = os.path.join(scene_analyzer.template_dir, "test_template.png")
    assert os.path.exists(template_path)
    
    # 验证模板是否可以被加载
    loaded_template = cv2.imread(template_path)
    assert loaded_template is not None
    assert loaded_template.shape == test_image.shape

def test_scene_location(scene_analyzer, test_image):
    """测试场景位置检测功能"""
    # 保存模板
    scene_analyzer.save_template(test_image, "location_test")
    
    # 创建一个大图，将测试图像放在特定位置
    large_image = np.zeros((200, 200, 3), dtype=np.uint8)
    x, y = 50, 50
    large_image[y:y+100, x:x+100] = test_image
    
    # 获取场景位置
    location = scene_analyzer.get_scene_location("location_test", large_image)
    
    # 验证位置
    assert location is not None
    center_x, center_y = location
    assert 95 <= center_x <= 105  # 允许一定的误差
    assert 95 <= center_y <= 105

def test_error_handling(scene_analyzer):
    """测试错误处理"""
    # 测试无效图像
    with pytest.raises(SceneAnalysisError):
        scene_analyzer.analyze_screenshot(None)
    
    # 测试无效模板名称
    assert scene_analyzer.get_scene_location("non_existent", np.zeros((100, 100, 3))) is None

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
