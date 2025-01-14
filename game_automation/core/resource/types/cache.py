"""缓存资源实现"""

import os
import json
import pickle
import hashlib
import tempfile
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime, timedelta

from ..base import ResourceBase, ResourceType
from ..errors import ResourceLoadError


class CacheFormat:
    """缓存格式枚举"""
    JSON = 'json'
    PICKLE = 'pickle'
    CUSTOM = 'custom'
    
    @staticmethod
    def from_extension(path: str) -> str:
        """从文件扩展名获取缓存格式
        
        Args:
            path: 文件路径
            
        Returns:
            缓存格式
            
        Raises:
            ValueError: 不支持的缓存格式
        """
        ext = Path(path).suffix.lower()
        if ext in ['.json']:
            return CacheFormat.JSON
        elif ext in ['.pkl', '.pickle']:
            return CacheFormat.PICKLE
        else:
            return CacheFormat.CUSTOM


class CacheResource(ResourceBase):
    """缓存资源
    
    支持的格式：
    - JSON
    - Pickle
    - 自定义格式
    
    特性：
    - 缓存加载和保存
    - 缓存过期
    - 缓存验证
    - 缓存压缩
    """
    
    def __init__(
        self,
        key: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        compress: bool = False,
        serializer: Optional[Any] = None,
        deserializer: Optional[Any] = None
    ):
        """初始化缓存资源
        
        Args:
            key: 资源标识符
            path: 缓存文件路径
            metadata: 资源元数据
            ttl: 缓存生存时间（秒）
            compress: 是否压缩
            serializer: 自定义序列化函数
            deserializer: 自定义反序列化函数
        """
        super().__init__(key, ResourceType.CACHE, metadata)
        self._path = Path(path)
        self._format = CacheFormat.from_extension(str(path))
        self._ttl = ttl
        self._compress = compress
        self._serializer = serializer
        self._deserializer = deserializer
        self._data: Optional[Any] = None
        self._created_at: Optional[datetime] = None
        
    @property
    def path(self) -> Path:
        """获取缓存路径"""
        return self._path
        
    @property
    def format(self) -> str:
        """获取缓存格式"""
        return self._format
        
    @property
    def data(self) -> Optional[Any]:
        """获取缓存数据"""
        return self._data
        
    @property
    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if not self._ttl or not self._created_at:
            return False
        return datetime.now() > self._created_at + timedelta(seconds=self._ttl)
        
    async def _do_load(self) -> None:
        """加载缓存
        
        Raises:
            ResourceLoadError: 缓存加载失败
        """
        try:
            # 检查文件是否存在
            if not self._path.exists():
                raise ResourceLoadError(
                    self.key,
                    f"Cache file not found: {self._path}"
                )
                
            # 读取缓存文件
            with open(self._path, 'rb') as f:
                data = f.read()
                
            # 解压数据
            if self._compress:
                import gzip
                data = gzip.decompress(data)
                
            # 反序列化数据
            if self._format == CacheFormat.JSON:
                self._data = json.loads(data.decode('utf-8'))
            elif self._format == CacheFormat.PICKLE:
                self._data = pickle.loads(data)
            else:  # 自定义格式
                if not self._deserializer:
                    raise ResourceLoadError(
                        self.key,
                        "Deserializer not specified"
                    )
                self._data = self._deserializer(data)
                
            # 更新创建时间
            self._created_at = datetime.now()
            
        except Exception as e:
            if not isinstance(e, ResourceLoadError):
                raise ResourceLoadError(self.key, cause=e)
            raise
            
    async def _do_unload(self) -> None:
        """释放缓存"""
        self._data = None
        self._created_at = None
        
    async def save(self, data: Any) -> None:
        """保存缓存数据
        
        Args:
            data: 要保存的数据
            
        Raises:
            ResourceLoadError: 缓存保存失败
        """
        try:
            # 序列化数据
            if self._format == CacheFormat.JSON:
                raw_data = json.dumps(data).encode('utf-8')
            elif self._format == CacheFormat.PICKLE:
                raw_data = pickle.dumps(data)
            else:  # 自定义格式
                if not self._serializer:
                    raise ResourceLoadError(
                        self.key,
                        "Serializer not specified"
                    )
                raw_data = self._serializer(data)
                
            # 压缩数据
            if self._compress:
                import gzip
                raw_data = gzip.compress(raw_data)
                
            # 创建临时文件
            tmp_path = Path(tempfile.mktemp())
            try:
                # 写入临时文件
                with open(tmp_path, 'wb') as f:
                    f.write(raw_data)
                    
                # 原子性地移动文件
                tmp_path.replace(self._path)
                
                # 更新内存中的数据
                self._data = data
                self._created_at = datetime.now()
                
            finally:
                # 清理临时文件
                if tmp_path.exists():
                    tmp_path.unlink()
                    
        except Exception as e:
            if not isinstance(e, ResourceLoadError):
                raise ResourceLoadError(self.key, cause=e)
            raise
            
    def verify(self) -> bool:
        """验证缓存
        
        Returns:
            验证是否通过
        """
        # 检查缓存是否已加载
        if not self._data:
            return False
            
        # 检查缓存是否过期
        if self.is_expired:
            return False
            
        return True
        
    def clear(self) -> None:
        """清除缓存"""
        if self._path.exists():
            self._path.unlink()
        self._data = None
        self._created_at = None
