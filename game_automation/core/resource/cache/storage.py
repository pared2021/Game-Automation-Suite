"""缓存存储模块"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class StorageStats:
    """存储统计信息"""
    
    def __init__(self):
        self.io_reads = 0
        self.io_writes = 0
        self.io_deletes = 0
        self.io_errors = 0
        self.total_read_time = 0.0
        self.total_write_time = 0.0
        self.total_delete_time = 0.0
        self.total_read_size = 0
        self.total_write_size = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'io_reads': self.io_reads,
            'io_writes': self.io_writes,
            'io_deletes': self.io_deletes,
            'io_errors': self.io_errors,
            'total_read_time': self.total_read_time,
            'total_write_time': self.total_write_time,
            'total_delete_time': self.total_delete_time,
            'total_read_size': self.total_read_size,
            'total_write_size': self.total_write_size,
        }
        
        # 计算平均值
        if self.io_reads > 0:
            stats['avg_read_time'] = self.total_read_time / self.io_reads
            stats['avg_read_size'] = self.total_read_size / self.io_reads
        else:
            stats['avg_read_time'] = 0.0
            stats['avg_read_size'] = 0
            
        if self.io_writes > 0:
            stats['avg_write_time'] = self.total_write_time / self.io_writes
            stats['avg_write_size'] = self.total_write_size / self.io_writes
        else:
            stats['avg_write_time'] = 0.0
            stats['avg_write_size'] = 0
            
        if self.io_deletes > 0:
            stats['avg_delete_time'] = self.total_delete_time / self.io_deletes
        else:
            stats['avg_delete_time'] = 0.0
            
        return stats
        
    def clear(self) -> None:
        """清除统计信息"""
        self.io_reads = 0
        self.io_writes = 0
        self.io_deletes = 0
        self.io_errors = 0
        self.total_read_time = 0.0
        self.total_write_time = 0.0
        self.total_delete_time = 0.0
        self.total_read_size = 0
        self.total_write_size = 0


class CacheStorage:
    """缓存存储
    
    特性：
    - 支持文件系统存储
    - 支持分片存储
    - 支持存储统计
    - 支持存储清理
    """
    
    def __init__(
        self,
        storage_dir: str,
        max_chunk_size: int = 1024 * 1024,  # 1MB
        max_storage_size: int = 1024 * 1024 * 1024  # 1GB
    ):
        """初始化存储
        
        Args:
            storage_dir: 存储目录
            max_chunk_size: 最大分片大小（字节）
            max_storage_size: 最大存储空间（字节）
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._max_chunk_size = max_chunk_size
        self._max_storage_size = max_storage_size
        self._stats = StorageStats()
        
    def _get_chunks_dir(self, key: str) -> Path:
        """获取分片目录
        
        Args:
            key: 缓存键
            
        Returns:
            分片目录路径
        """
        return self._storage_dir / key
        
    def _get_chunk_path(self, key: str, chunk_index: int) -> Path:
        """获取分片文件路径
        
        Args:
            key: 缓存键
            chunk_index: 分片索引
            
        Returns:
            分片文件路径
        """
        return self._get_chunks_dir(key) / f'chunk_{chunk_index:04d}.dat'
        
    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            元数据文件路径
        """
        return self._get_chunks_dir(key) / 'metadata.json'
        
    def _split_data(self, data: str) -> List[str]:
        """分割数据
        
        Args:
            data: 原始数据
            
        Returns:
            分片列表
        """
        chunks = []
        bytes_data = data.encode('utf-8')
        total_size = len(bytes_data)
        offset = 0
        
        while offset < total_size:
            chunk = bytes_data[offset:offset + self._max_chunk_size]
            chunks.append(chunk.decode('utf-8'))
            offset += self._max_chunk_size
            
        return chunks
        
    def _merge_chunks(self, chunks: List[str]) -> str:
        """合并分片
        
        Args:
            chunks: 分片列表
            
        Returns:
            合并后的数据
        """
        return ''.join(chunks)
        
    def read(self, key: str) -> Optional[str]:
        """读取数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据
        """
        start_time = time.time()
        try:
            # 读取元数据
            metadata_path = self._get_metadata_path(key)
            if not metadata_path.exists():
                return None
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # 读取所有分片
            chunks = []
            chunks_dir = self._get_chunks_dir(key)
            if not chunks_dir.exists():
                return None
                
            for i in range(metadata['chunks']):
                chunk_path = self._get_chunk_path(key, i)
                if not chunk_path.exists():
                    return None
                    
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunks.append(f.read())
                    
            # 合并分片
            data = self._merge_chunks(chunks)
            
            # 更新统计信息
            self._stats.io_reads += 1
            self._stats.total_read_time += time.time() - start_time
            self._stats.total_read_size += len(data.encode('utf-8'))
            
            return data
            
        except Exception as e:
            logger.error(f'Failed to read data for key {key}: {e}')
            self._stats.io_errors += 1
            return None
            
    def write(self, key: str, data: str) -> bool:
        """写入数据
        
        Args:
            key: 缓存键
            data: 缓存数据
            
        Returns:
            是否写入成功
        """
        start_time = time.time()
        try:
            # 检查存储空间
            if self.get_storage_size() + len(data.encode('utf-8')) > self._max_storage_size:
                return False
                
            # 分割数据
            chunks = self._split_data(data)
            
            # 创建分片目录
            chunks_dir = self._get_chunks_dir(key)
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分片
            for i, chunk in enumerate(chunks):
                chunk_path = self._get_chunk_path(key, i)
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                    
            # 写入元数据
            metadata = {
                'key': key,
                'chunks': len(chunks),
                'size': len(data.encode('utf-8')),
                'created_at': time.time()
            }
            with open(self._get_metadata_path(key), 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
                
            # 更新统计信息
            self._stats.io_writes += 1
            self._stats.total_write_time += time.time() - start_time
            self._stats.total_write_size += len(data.encode('utf-8'))
            
            return True
            
        except Exception as e:
            logger.error(f'Failed to write data for key {key}: {e}')
            self._stats.io_errors += 1
            return False
            
    def delete(self, key: str) -> bool:
        """删除数据
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        start_time = time.time()
        try:
            chunks_dir = self._get_chunks_dir(key)
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
                
            # 更新统计信息
            self._stats.io_deletes += 1
            self._stats.total_delete_time += time.time() - start_time
            
            return True
            
        except Exception as e:
            logger.error(f'Failed to delete data for key {key}: {e}')
            self._stats.io_errors += 1
            return False
            
    def clear(self) -> bool:
        """清除所有数据
        
        Returns:
            是否清除成功
        """
        try:
            if self._storage_dir.exists():
                shutil.rmtree(self._storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            return True
            
        except Exception as e:
            logger.error(f'Failed to clear storage: {e}')
            self._stats.io_errors += 1
            return False
            
    def get_storage_size(self) -> int:
        """获取存储大小
        
        Returns:
            存储大小（字节）
        """
        total_size = 0
        if self._storage_dir.exists():
            for path in self._storage_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        return total_size
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return self._stats.get_stats()
        
    def clear_stats(self) -> None:
        """清除统计信息"""
        self._stats.clear()
