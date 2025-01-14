"""缓存压缩模块"""

import zlib
import json
import base64
from typing import Any, Optional


class CacheCompressor:
    """缓存压缩器
    
    特性：
    - 支持多种压缩级别
    - 支持压缩阈值
    - 支持自动压缩
    - 支持压缩统计
    """
    
    def __init__(
        self,
        compression_level: int = 6,
        compression_threshold: int = 1024,
        auto_compress: bool = True
    ):
        """初始化压缩器
        
        Args:
            compression_level: 压缩级别 (0-9)
            compression_threshold: 压缩阈值（字节）
            auto_compress: 是否自动压缩
        """
        if not 0 <= compression_level <= 9:
            raise ValueError('Compression level must be between 0 and 9')
            
        self._level = compression_level
        self._threshold = compression_threshold
        self._auto_compress = auto_compress
        self._stats = {
            'compressed_count': 0,
            'uncompressed_count': 0,
            'total_original_size': 0,
            'total_compressed_size': 0
        }
        
    def should_compress(self, data: str) -> bool:
        """判断是否需要压缩
        
        Args:
            data: 原始数据
            
        Returns:
            是否需要压缩
        """
        return (
            self._auto_compress and
            len(data.encode('utf-8')) >= self._threshold
        )
        
    def compress(self, data: Any) -> str:
        """压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            压缩后的数据
        """
        # 转换为 JSON 字符串
        json_str = json.dumps(data)
        
        # 判断是否需要压缩
        if not self.should_compress(json_str):
            self._stats['uncompressed_count'] += 1
            return json_str
            
        # 压缩数据
        original_size = len(json_str.encode('utf-8'))
        compressed = zlib.compress(
            json_str.encode('utf-8'),
            level=self._level
        )
        compressed_size = len(compressed)
        
        # 更新统计信息
        self._stats['compressed_count'] += 1
        self._stats['total_original_size'] += original_size
        self._stats['total_compressed_size'] += compressed_size
        
        # 编码为 Base64 字符串
        return base64.b64encode(compressed).decode('utf-8')
        
    def decompress(self, data: str) -> Any:
        """解压数据
        
        Args:
            data: 压缩数据
            
        Returns:
            原始数据
        """
        try:
            # 尝试解析为 JSON
            return json.loads(data)
        except json.JSONDecodeError:
            # 解压缩数据
            try:
                compressed = base64.b64decode(data.encode('utf-8'))
                decompressed = zlib.decompress(compressed)
                return json.loads(decompressed.decode('utf-8'))
            except Exception as e:
                raise ValueError(f'Failed to decompress data: {e}')
                
    def get_stats(self) -> dict:
        """获取压缩统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        if stats['compressed_count'] > 0:
            stats['compression_ratio'] = (
                stats['total_compressed_size'] /
                stats['total_original_size']
            )
        else:
            stats['compression_ratio'] = 1.0
        return stats
        
    def clear_stats(self) -> None:
        """清除统计信息"""
        self._stats = {
            'compressed_count': 0,
            'uncompressed_count': 0,
            'total_original_size': 0,
            'total_compressed_size': 0
        }
