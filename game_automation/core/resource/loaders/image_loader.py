"""图像资源加载器实现"""

import os
from typing import Type, Dict, Any, Optional
from pathlib import Path

from ..base import ResourceBase
from ..loader import ResourceLoader
from ..types.image import ImageResource
from ..errors import ResourceLoadError


class ImageLoader(ResourceLoader):
    """图像资源加载器
    
    功能：
    - 支持多种图像格式
    - 支持图像预处理
    - 支持图像缓存
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """初始化图像加载器
        
        Args:
            base_path: 图像基础路径
        """
        self._base_path = Path(base_path) if base_path else None
        
    async def load(
        self,
        key: str,
        resource_type: Type[ResourceBase],
        **kwargs: Any
    ) -> ResourceBase:
        """加载图像资源
        
        Args:
            key: 资源标识符
            resource_type: 资源类型
            **kwargs: 加载参数
                path: 图像路径（相对或绝对）
                preprocess: 是否预处理
                metadata: 资源元数据
                
        Returns:
            图像资源对象
            
        Raises:
            ResourceLoadError: 资源加载失败
        """
        if not issubclass(resource_type, ImageResource):
            raise ResourceLoadError(
                key,
                f"Invalid resource type: {resource_type}"
            )
            
        # 获取图像路径
        path = kwargs.get('path')
        if not path:
            raise ResourceLoadError(key, "Image path not specified")
            
        # 处理相对路径
        if self._base_path and not os.path.isabs(path):
            path = self._base_path / path
            
        # 创建资源对象
        return ImageResource(
            key,
            str(path),
            metadata=kwargs.get('metadata'),
            preprocess=kwargs.get('preprocess', True)
        )
        
    async def unload(self, resource: ResourceBase) -> None:
        """释放图像资源
        
        Args:
            resource: 资源对象
        """
        if not isinstance(resource, ImageResource):
            return
            
        await resource.unload()
