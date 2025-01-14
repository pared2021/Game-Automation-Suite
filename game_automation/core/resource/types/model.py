"""模型资源实现"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..base import ResourceBase, ResourceType
from ..errors import ResourceLoadError


class ModelFormat:
    """模型格式枚举"""
    ONNX = 'onnx'
    PYTORCH = 'pth'
    TENSORFLOW = 'pb'
    
    @staticmethod
    def from_extension(path: str) -> str:
        """从文件扩展名获取模型格式
        
        Args:
            path: 文件路径
            
        Returns:
            模型格式
            
        Raises:
            ValueError: 不支持的模型格式
        """
        ext = Path(path).suffix.lower()
        if ext in ['.onnx']:
            return ModelFormat.ONNX
        elif ext in ['.pth', '.pt']:
            return ModelFormat.PYTORCH
        elif ext in ['.pb']:
            return ModelFormat.TENSORFLOW
        else:
            raise ValueError(f"Unsupported model format: {ext}")


class ModelResource(ResourceBase):
    """模型资源
    
    支持的格式：
    - ONNX
    - PyTorch
    - TensorFlow
    
    特性：
    - 模型加载和释放
    - 模型版本管理
    - 模型校验
    - 模型元数据
    """
    
    def __init__(
        self,
        key: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        device: str = 'cpu'
    ):
        """初始化模型资源
        
        Args:
            key: 资源标识符
            path: 模型文件路径
            metadata: 资源元数据
            version: 模型版本
            device: 运行设备
        """
        super().__init__(key, ResourceType.MODEL, metadata)
        self._path = Path(path)
        self._format = ModelFormat.from_extension(str(path))
        self._version = version
        self._device = device
        self._model: Optional[Any] = None
        self._hash: Optional[str] = None
        
    @property
    def path(self) -> Path:
        """获取模型路径"""
        return self._path
        
    @property
    def format(self) -> str:
        """获取模型格式"""
        return self._format
        
    @property
    def version(self) -> Optional[str]:
        """获取模型版本"""
        return self._version
        
    @property
    def device(self) -> str:
        """获取运行设备"""
        return self._device
        
    @property
    def model(self) -> Optional[Any]:
        """获取模型对象"""
        return self._model
        
    @property
    def hash(self) -> Optional[str]:
        """获取模型哈希值"""
        return self._hash
        
    async def _do_load(self) -> None:
        """加载模型
        
        Raises:
            ResourceLoadError: 模型加载失败
        """
        try:
            # 检查文件是否存在
            if not self._path.exists():
                raise ResourceLoadError(
                    self.key,
                    f"Model file not found: {self._path}"
                )
                
            # 计算模型哈希值
            self._hash = self._compute_hash()
            
            # 加载模型
            if self._format == ModelFormat.ONNX:
                await self._load_onnx()
            elif self._format == ModelFormat.PYTORCH:
                await self._load_pytorch()
            elif self._format == ModelFormat.TENSORFLOW:
                await self._load_tensorflow()
                
        except Exception as e:
            if not isinstance(e, ResourceLoadError):
                raise ResourceLoadError(self.key, cause=e)
            raise
            
    async def _do_unload(self) -> None:
        """释放模型"""
        if self._model is not None:
            # 释放模型
            if hasattr(self._model, 'close'):
                self._model.close()
            elif hasattr(self._model, 'cpu'):
                self._model.cpu()
                
            self._model = None
            
    def _compute_hash(self) -> str:
        """计算模型哈希值
        
        Returns:
            哈希值
        """
        sha256 = hashlib.sha256()
        with open(self._path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
        
    async def _load_onnx(self) -> None:
        """加载 ONNX 模型"""
        try:
            import onnxruntime as ort
            
            # 创建推理会话
            providers = ['CPUExecutionProvider']
            if self._device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
                
            self._model = ort.InferenceSession(
                str(self._path),
                providers=providers
            )
            
        except ImportError:
            raise ResourceLoadError(
                self.key,
                "ONNX Runtime not installed"
            )
            
    async def _load_pytorch(self) -> None:
        """加载 PyTorch 模型"""
        try:
            import torch
            
            # 加载模型
            self._model = torch.load(
                self._path,
                map_location=self._device
            )
            
            # 设置为评估模式
            if hasattr(self._model, 'eval'):
                self._model.eval()
                
        except ImportError:
            raise ResourceLoadError(
                self.key,
                "PyTorch not installed"
            )
            
    async def _load_tensorflow(self) -> None:
        """加载 TensorFlow 模型"""
        try:
            import tensorflow as tf
            
            # 设置设备
            if self._device == 'cuda':
                physical_devices = tf.config.list_physical_devices('GPU')
                if not physical_devices:
                    raise ResourceLoadError(
                        self.key,
                        "No GPU devices available"
                    )
                    
            # 加载模型
            self._model = tf.saved_model.load(str(self._path))
            
        except ImportError:
            raise ResourceLoadError(
                self.key,
                "TensorFlow not installed"
            )
            
    def verify(self) -> bool:
        """验证模型
        
        Returns:
            验证是否通过
        """
        # 检查模型是否已加载
        if not self._model:
            return False
            
        # 验证模型哈希值
        current_hash = self._compute_hash()
        return current_hash == self._hash
