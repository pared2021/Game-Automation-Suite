"""模型加载器实现"""

import os
from typing import Type, Dict, Any, Optional, List
from pathlib import Path

from ..base import ResourceBase
from ..loader import ResourceLoader
from ..types.model import ModelResource
from ..errors import ResourceLoadError


class ModelLoader(ResourceLoader):
    """模型加载器
    
    功能：
    - 支持多种模型格式
    - 支持模型验证
    - 支持设备配置
    - 支持版本管理
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        device: str = 'cpu',
        version_file: Optional[str] = None
    ):
        """初始化模型加载器
        
        Args:
            base_path: 模型基础路径
            device: 默认运行设备
            version_file: 版本文件路径
        """
        self._base_path = Path(base_path) if base_path else None
        self._device = device
        self._version_file = Path(version_file) if version_file else None
        self._versions: Dict[str, str] = {}
        
        # 加载版本信息
        if self._version_file and self._version_file.exists():
            import json
            with open(self._version_file, 'r', encoding='utf-8') as f:
                self._versions = json.load(f)
                
    async def load(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """加载模型资源
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
                path: 模型路径（相对或绝对）
                metadata: 资源元数据
                version: 模型版本
                device: 运行设备
                
        Returns:
            模型资源对象
            
        Raises:
            ResourceLoadError: 资源加载失败
        """
        if not issubclass(resource_type, ModelResource):
            raise ResourceLoadError(
                key,
                f"Invalid resource type: {resource_type}"
            )
            
        # 获取模型路径
        path = kwargs.get('path')
        if not path:
            raise ResourceLoadError(key, "Model path not specified")
            
        # 处理相对路径
        if self._base_path and not os.path.isabs(path):
            path = self._base_path / path
            
        # 获取版本信息
        version = kwargs.get('version')
        if not version and key in self._versions:
            version = self._versions[key]
            
        # 获取运行设备
        device = kwargs.get('device', self._device)
        
        # 创建资源对象
        return ModelResource(
            key,
            str(path),
            metadata=kwargs.get('metadata'),
            version=version,
            device=device
        )
        
    async def unload(self, resource: ResourceBase) -> None:
        """释放模型资源
        
        Args:
            resource: 资源对象
        """
        if not isinstance(resource, ModelResource):
            return
            
        await resource.unload()
        
    def update_version(self, key: str, version: str) -> None:
        """更新模型版本
        
        Args:
            key: 资源标识符
            version: 模型版本
        """
        self._versions[key] = version
        
        # 保存版本信息
        if self._version_file:
            import json
            with open(self._version_file, 'w', encoding='utf-8') as f:
                json.dump(self._versions, f, indent=2)
