"""模型加载器测试"""

import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.types.model import ModelResource
from game_automation.core.resource.loaders.model_loader import ModelLoader
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_model_path(tmp_path) -> Path:
    """创建测试模型文件"""
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # 创建一个简单的 ONNX 模型
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 3])
        
        node = helper.make_node(
            'Identity',
            inputs=['X'],
            outputs=['Y']
        )
        
        graph = helper.make_graph(
            [node],
            'test-model',
            [X],
            [Y]
        )
        
        model = helper.make_model(graph)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))
        
        return model_path
        
    except ImportError:
        pytest.skip("ONNX not installed")


@pytest.fixture
def version_file(tmp_path) -> Path:
    """创建版本文件"""
    version_file = tmp_path / "versions.json"
    versions = {
        "test": "1.0.0"
    }
    with open(version_file, 'w', encoding='utf-8') as f:
        json.dump(versions, f)
    return version_file


@pytest.fixture
def model_loader(tmp_path, version_file):
    """创建模型加载器"""
    return ModelLoader(
        str(tmp_path),
        device='cpu',
        version_file=str(version_file)
    )


@pytest.mark.asyncio
async def test_load_model(model_loader, test_model_path):
    """测试加载模型"""
    # 使用相对路径
    resource = await model_loader.load(
        'test',
        ModelResource,
        path=test_model_path.name
    )
    
    assert isinstance(resource, ModelResource)
    assert resource.key == 'test'
    assert resource.path == test_model_path
    assert resource.version == '1.0.0'  # 从版本文件加载
    
    # 加载模型
    await resource.load()
    assert resource.model is not None


@pytest.mark.asyncio
async def test_load_with_absolute_path(model_loader, test_model_path):
    """测试使用绝对路径加载模型"""
    resource = await model_loader.load(
        'test',
        ModelResource,
        path=str(test_model_path)
    )
    
    assert isinstance(resource, ModelResource)
    assert resource.path == test_model_path


@pytest.mark.asyncio
async def test_load_with_version(model_loader, test_model_path):
    """测试使用指定版本加载模型"""
    resource = await model_loader.load(
        'test',
        ModelResource,
        path=test_model_path.name,
        version='2.0.0'
    )
    
    assert resource.version == '2.0.0'  # 使用指定版本


@pytest.mark.asyncio
async def test_load_with_device(model_loader, test_model_path):
    """测试使用指定设备加载模型"""
    resource = await model_loader.load(
        'test',
        ModelResource,
        path=test_model_path.name,
        device='cuda'
    )
    
    assert resource.device == 'cuda'


@pytest.mark.asyncio
async def test_load_invalid_type(model_loader, test_model_path):
    """测试加载无效的资源类型"""
    class InvalidResource:
        pass
    
    with pytest.raises(ResourceLoadError):
        await model_loader.load(
            'test',
            InvalidResource,
            path=str(test_model_path)
        )


@pytest.mark.asyncio
async def test_unload_model(model_loader, test_model_path):
    """测试释放模型"""
    resource = await model_loader.load(
        'test',
        ModelResource,
        path=test_model_path.name
    )
    
    # 加载模型
    await resource.load()
    assert resource.model is not None
    
    # 释放模型
    await model_loader.unload(resource)
    assert resource.model is None


def test_version_management(tmp_path):
    """测试版本管理"""
    version_file = tmp_path / "versions.json"
    loader = ModelLoader(version_file=str(version_file))
    
    # 更新版本
    loader.update_version('test', '1.0.0')
    assert os.path.exists(version_file)
    
    # 检查版本文件
    with open(version_file, 'r', encoding='utf-8') as f:
        versions = json.load(f)
        assert versions['test'] == '1.0.0'
        
    # 更新版本
    loader.update_version('test', '2.0.0')
    with open(version_file, 'r', encoding='utf-8') as f:
        versions = json.load(f)
        assert versions['test'] == '2.0.0'
