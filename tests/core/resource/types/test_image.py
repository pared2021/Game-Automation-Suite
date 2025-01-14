"""图像资源测试"""

import os
import cv2
import pytest
import numpy as np
from pathlib import Path

from game_automation.core.resource.types.image import ImageResource
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_image_path(tmp_path):
    """创建测试图像"""
    image_path = tmp_path / "test.png"
    # 创建一个简单的测试图像
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def test_template_path(tmp_path):
    """创建测试模板图像"""
    template_path = tmp_path / "template.png"
    # 创建一个小的白色方块作为模板
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(template, (5, 5), (15, 15), (255, 255, 255), -1)
    cv2.imwrite(str(template_path), template)
    return template_path


@pytest.mark.asyncio
async def test_load_image(test_image_path):
    """测试加载图像"""
    resource = ImageResource('test', str(test_image_path))
    
    # 测试加载前的状态
    assert resource.image is None
    assert resource.gray is None
    assert resource.size is None
    
    # 加载图像
    await resource.load()
    
    # 测试加载后的状态
    assert resource.image is not None
    assert resource.gray is not None
    assert resource.size == (100, 100)
    
    # 测试图像内容
    assert resource.image.shape == (100, 100, 3)
    assert resource.gray.shape == (100, 100)


@pytest.mark.asyncio
async def test_load_nonexistent_image():
    """测试加载不存在的图像"""
    resource = ImageResource('test', 'nonexistent.png')
    
    with pytest.raises(ResourceLoadError):
        await resource.load()


@pytest.mark.asyncio
async def test_image_operations(test_image_path):
    """测试图像操作"""
    resource = ImageResource('test', str(test_image_path))
    await resource.load()
    
    # 测试调整大小
    resized = resource.resize(50, 50)
    assert resized.shape == (50, 50, 3)
    
    # 测试裁剪
    cropped = resource.crop(20, 20, 40, 40)
    assert cropped.shape == (40, 40, 3)
    
    # 测试无效裁剪
    with pytest.raises(ValueError):
        resource.crop(-1, -1, 10, 10)
    with pytest.raises(ValueError):
        resource.crop(90, 90, 20, 20)


@pytest.mark.asyncio
async def test_template_matching(test_image_path, test_template_path):
    """测试模板匹配"""
    # 加载主图像
    image = ImageResource('test', str(test_image_path))
    await image.load()
    
    # 加载模板图像
    template = ImageResource('template', str(test_template_path))
    await template.load()
    
    # 执行模板匹配
    result = image.match_template(template)
    assert result is not None
    
    # 解包结果
    x, y, score = result
    assert 0 <= x <= 100
    assert 0 <= y <= 100
    assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_unload_image(test_image_path):
    """测试释放图像"""
    resource = ImageResource('test', str(test_image_path))
    await resource.load()
    
    # 测试释放前的状态
    assert resource.image is not None
    assert resource.gray is not None
    
    # 释放图像
    await resource.unload()
    
    # 测试释放后的状态
    assert resource.image is None
    assert resource.gray is None
    assert resource.size is None
