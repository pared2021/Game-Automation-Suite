"""模型资源测试"""

import os
import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from game_automation.core.resource.types.model import ModelResource
from game_automation.core.resource.errors import ResourceLoadError


@pytest.fixture
def test_onnx_model(tmp_path) -> Path:
    """创建测试 ONNX 模型文件"""
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
def test_pytorch_model(tmp_path) -> Path:
    """创建测试 PyTorch 模型文件"""
    try:
        import torch
        import torch.nn as nn
        
        # 创建一个简单的 PyTorch 模型
        class SimpleModel(nn.Module):
            def forward(self, x):
                return x
                
        model = SimpleModel()
        model_path = tmp_path / "model.pth"
        torch.save(model, model_path)
        
        return model_path
        
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def test_tensorflow_model(tmp_path) -> Path:
    """创建测试 TensorFlow 模型文件"""
    try:
        import tensorflow as tf
        
        # 创建一个简单的 TensorFlow 模型
        class SimpleModel(tf.keras.Model):
            def call(self, x):
                return x
                
        model = SimpleModel()
        model_path = tmp_path / "model"
        tf.saved_model.save(model, str(model_path))
        
        return model_path
        
    except ImportError:
        pytest.skip("TensorFlow not installed")


@pytest.mark.asyncio
async def test_load_onnx_model(test_onnx_model):
    """测试加载 ONNX 模型"""
    try:
        import onnxruntime as ort
        
        resource = ModelResource('test', str(test_onnx_model))
        
        # 测试加载前的状态
        assert resource.model is None
        assert resource.hash is None
        
        # 加载模型
        await resource.load()
        
        # 测试加载后的状态
        assert resource.model is not None
        assert isinstance(resource.model, ort.InferenceSession)
        assert resource.hash is not None
        
    except ImportError:
        pytest.skip("ONNX Runtime not installed")


@pytest.mark.asyncio
async def test_load_pytorch_model(test_pytorch_model):
    """测试加载 PyTorch 模型"""
    try:
        import torch
        
        resource = ModelResource('test', str(test_pytorch_model))
        await resource.load()
        
        assert resource.model is not None
        assert isinstance(resource.model, torch.nn.Module)
        assert resource.hash is not None
        
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.mark.asyncio
async def test_load_tensorflow_model(test_tensorflow_model):
    """测试加载 TensorFlow 模型"""
    try:
        import tensorflow as tf
        
        resource = ModelResource('test', str(test_tensorflow_model))
        await resource.load()
        
        assert resource.model is not None
        assert resource.hash is not None
        
    except ImportError:
        pytest.skip("TensorFlow not installed")


@pytest.mark.asyncio
async def test_model_verification(test_onnx_model):
    """测试模型验证"""
    try:
        resource = ModelResource('test', str(test_onnx_model))
        await resource.load()
        
        # 验证模型
        assert resource.verify()
        
        # 修改模型文件
        with open(test_onnx_model, 'ab') as f:
            f.write(b'invalid')
            
        # 验证应该失败
        assert not resource.verify()
        
    except ImportError:
        pytest.skip("ONNX Runtime not installed")


@pytest.mark.asyncio
async def test_model_device(test_pytorch_model):
    """测试模型设备配置"""
    try:
        import torch
        
        # 测试 CPU 设备
        resource = ModelResource('test', str(test_pytorch_model), device='cpu')
        await resource.load()
        assert str(next(resource.model.parameters()).device) == 'cpu'
        
        # 测试 CUDA 设备（如果可用）
        if torch.cuda.is_available():
            resource = ModelResource(
                'test',
                str(test_pytorch_model),
                device='cuda'
            )
            await resource.load()
            assert str(next(resource.model.parameters()).device).startswith('cuda')
            
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.mark.asyncio
async def test_load_nonexistent_model():
    """测试加载不存在的模型"""
    resource = ModelResource('test', 'nonexistent.onnx')
    with pytest.raises(ResourceLoadError):
        await resource.load()


@pytest.mark.asyncio
async def test_unload_model(test_onnx_model):
    """测试释放模型"""
    try:
        resource = ModelResource('test', str(test_onnx_model))
        await resource.load()
        
        # 测试释放前的状态
        assert resource.model is not None
        
        # 释放模型
        await resource.unload()
        
        # 测试释放后的状态
        assert resource.model is None
        
    except ImportError:
        pytest.skip("ONNX Runtime not installed")
