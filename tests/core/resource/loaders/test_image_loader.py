"""图像加载器测试"""

import os
import cv2
import pytest
import numpy as np
from pathlib import Path

from game_automation.core.resource.types.image import ImageResource
from game_automation.core.resource.loaders.image_loader import ImageLoader
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
def image_loader(tmp_path):
    """创建图像加载器"""
    return ImageLoader(str(tmp_path))


@pytest.mark.asyncio
async def test_load_image(image_loader, test_image_path):
    """测试加载图像"""
    # 使用相对路径
    resource = await image_loader.load(
        'test',
        ImageResource,
        path=test_image_path.name
    )
    
    assert isinstance(resource, ImageResource)
    assert resource.key == 'test'
    assert resource.path == test_image_path
    
    # 加载图像
    await resource.load()
    assert resource.image is not None
    assert resource.size == (100, 100)


@pytest.mark.asyncio
async def test_load_with_absolute_path(image_loader, test_image_path):
    """测试使用绝对路径加载图像"""
    resource = await image_loader.load(
        'test',
        ImageResource,
        path=str(test_image_path)
    )
    
    assert isinstance(resource, ImageResource)
    assert resource.path == test_image_path


@pytest.mark.asyncio
async def test_load_with_metadata(image_loader, test_image_path):
    """测试加载带元数据的图像"""
    metadata = {
        'description': 'Test image',
        'tags': ['test', 'image']
    }
    
    resource = await image_loader.load(
        'test',
        ImageResource,
        path=str(test_image_path),
        metadata=metadata
    )
    
    assert resource.metadata == metadata


@pytest.mark.asyncio
async def test_load_nonexistent_image(image_loader):
    """测试加载不存在的图像"""
    with pytest.raises(ResourceLoadError):
        await image_loader.load(
            'test',
            ImageResource,
            path='nonexistent.png'
        )


@pytest.mark.asyncio
async def test_load_invalid_type(image_loader, test_image_path):
    """测试加载无效的资源类型"""
    class InvalidResource:
        pass
    
    with pytest.raises(ResourceLoadError):
        await image_loader.load(
            'test',
            InvalidResource,
            path=str(test_image_path)
        )


@pytest.mark.asyncio
async def test_unload_image(image_loader, test_image_path):
    """测试释放图像"""
    resource = await image_loader.load(
        'test',
        ImageResource,
        path=str(test_image_path)
    )
    
    # 加载图像
    await resource.load()
    assert resource.image is not None
    
    # 释放图像
    await image_loader.unload(resource)
    assert resource.image is None
