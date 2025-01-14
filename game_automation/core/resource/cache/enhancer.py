"""缓存增强器"""

import zlib
import pickle
import hashlib
import logging
import threading
import time
from enum import Enum
from typing import Any, Optional, Callable, Dict, List, Tuple, Set
from datetime import datetime

from .monitor import CacheMonitor

logger = logging.getLogger(__name__)


class PreloadStrategy(Enum):
    """预热策略"""
    ALL = "all"  # 预热所有键
    PRIORITY = "priority"  # 按优先级预热
    RECENT = "recent"  # 预热最近使用的键
    CUSTOM = "custom"  # 自定义预热策略


class PreloadStats:
    """预热统计信息"""
    
    def __init__(self):
        self.total_keys = 0  # 总键数
        self.loaded_keys = 0  # 已加载键数
        self.failed_keys = 0  # 失败键数
        self.start_time = 0.0  # 开始时间
        self.end_time = 0.0  # 结束时间
        self.current_key = ""  # 当前正在加载的键
        self.status = "idle"  # 状态：idle, running, completed, failed
        
    @property
    def progress(self) -> float:
        """预热进度（百分比）"""
        if self.total_keys == 0:
            return 0.0
        return (self.loaded_keys / self.total_keys) * 100
        
    @property
    def duration(self) -> float:
        """预热持续时间（秒）"""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class CacheEnhancer:
    """缓存增强器
    
    特性：
    - 支持缓存压缩
    - 支持缓存验证
    - 支持缓存预热
    - 支持缓存监控
    """
    
    def __init__(
        self,
        compression_level: int = 6,
        compression_threshold: int = 1024,
        verify_data: bool = True,
        preload_keys: Optional[List[str]] = None,
        preload_callback: Optional[Callable[[str], Any]] = None,
        monitor_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        preload_strategy: PreloadStrategy = PreloadStrategy.ALL,
        preload_batch_size: int = 10,
        preload_interval: float = 0.1,
        monitor_save_dir: str = "data/monitor/cache"
    ):
        """初始化缓存增强器
        
        Args:
            compression_level: 压缩级别 (0-9)
            compression_threshold: 压缩阈值（字节）
            verify_data: 是否验证数据
            preload_keys: 预加载的键列表
            preload_callback: 预加载回调函数
            monitor_callback: 监控回调函数
            preload_strategy: 预热策略
            preload_batch_size: 预热批次大小
            preload_interval: 预热间隔（秒）
            monitor_save_dir: 监控数据保存目录
        """
        self._compression_level = compression_level
        self._compression_threshold = compression_threshold
        self._verify_data = verify_data
        self._preload_keys = set(preload_keys or [])
        self._preload_callback = preload_callback
        self._monitor_callback = monitor_callback
        self._preload_strategy = preload_strategy
        self._preload_batch_size = preload_batch_size
        self._preload_interval = preload_interval
        
        self._checksums: Dict[str, str] = {}
        self._access_times: Dict[str, float] = {}  # 键的最后访问时间
        self._key_priorities: Dict[str, int] = {}  # 键的优先级
        self._preload_stats = PreloadStats()
        self._preload_lock = threading.Lock()
        
        # 初始化监控器
        self._monitor = CacheMonitor(save_dir=monitor_save_dir)
        
    def compress(self, data: Any) -> Tuple[bytes, bool]:
        """压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            压缩后的数据和是否已压缩的标志
        """
        start_time = time.time()
        try:
            # 如果已经是字节，直接使用
            if isinstance(data, bytes):
                serialized = data
            else:
                serialized = pickle.dumps(data)
            
            # 如果数据大小小于阈值，不进行压缩
            if len(serialized) < self._compression_threshold:
                self._monitor.record_operation(
                    "compress",
                    "unknown",
                    True,
                    time.time() - start_time,
                    len(serialized),
                    "memory",
                    {
                        "compressed": False,
                        "original_size": len(serialized)
                    }
                )
                if self._monitor_callback:
                    self._monitor_callback("compress", {
                        "success": True,
                        "compressed": False,
                        "size": len(serialized)
                    })
                return serialized, False
                
            # 压缩数据
            compressed = zlib.compress(serialized, self._compression_level)
            self._monitor.record_operation(
                "compress",
                "unknown",
                True,
                time.time() - start_time,
                len(compressed),
                "memory",
                {
                    "compressed": True,
                    "original_size": len(serialized),
                    "compressed_size": len(compressed),
                    "compression_ratio": len(compressed) / len(serialized)
                }
            )
            if self._monitor_callback:
                self._monitor_callback("compress", {
                    "success": True,
                    "compressed": True,
                    "original_size": len(serialized),
                    "compressed_size": len(compressed)
                })
            return compressed, True
            
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            self._monitor.record_operation(
                "compress",
                "unknown",
                False,
                time.time() - start_time,
                0,
                "memory",
                {"error": str(e)}
            )
            if self._monitor_callback:
                self._monitor_callback("compress", {
                    "success": False,
                    "error": str(e)
                })
            raise
            
    def decompress(self, data: bytes, is_compressed: bool) -> Any:
        """解压数据
        
        Args:
            data: 压缩数据
            is_compressed: 是否已压缩
            
        Returns:
            解压后的数据
        """
        start_time = time.time()
        try:
            if is_compressed:
                data = zlib.decompress(data)
                
            # 如果是字节，尝试反序列化
            try:
                result = pickle.loads(data)
            except Exception:
                result = data
                
            self._monitor.record_operation(
                "decompress",
                "unknown",
                True,
                time.time() - start_time,
                len(data),
                "memory",
                {
                    "compressed": is_compressed,
                    "data_size": len(data)
                }
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            self._monitor.record_operation(
                "decompress",
                "unknown",
                False,
                time.time() - start_time,
                len(data),
                "memory",
                {"error": str(e)}
            )
            raise
            
    def verify(self, key: str, data: Any) -> bool:
        """验证数据
        
        Args:
            key: 缓存键
            data: 缓存数据
            
        Returns:
            验证是否通过
        """
        start_time = time.time()
        try:
            if not self._verify_data:
                return True
                
            # 计算数据校验和
            checksum = self._calculate_checksum(data)
            
            # 如果是新数据，保存校验和
            if key not in self._checksums:
                self._checksums[key] = checksum
                self._monitor.record_operation(
                    "verify",
                    key,
                    True,
                    time.time() - start_time,
                    len(pickle.dumps(data)),
                    "memory",
                    {"new_data": True}
                )
                return True
                
            # 验证校验和
            result = self._checksums[key] == checksum
            self._monitor.record_operation(
                "verify",
                key,
                result,
                time.time() - start_time,
                len(pickle.dumps(data)),
                "memory",
                {"new_data": False}
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to verify data for key {key}: {e}")
            self._monitor.record_operation(
                "verify",
                key,
                False,
                time.time() - start_time,
                0,
                "memory",
                {"error": str(e)}
            )
            return False
            
    def preload(self) -> None:
        """预热缓存"""
        if not self._preload_callback:
            return
            
        with self._preload_lock:
            # 如果正在预热，直接返回
            if self._preload_stats.status == "running":
                return
                
            # 初始化预热统计
            self._preload_stats = PreloadStats()
            self._preload_stats.status = "running"
            self._preload_stats.start_time = time.time()
            
            try:
                # 获取预热键
                keys = list(self._get_preload_keys())
                self._preload_stats.total_keys = len(keys)
                
                # 按顺序预热
                for key in keys:
                    try:
                        self._preload_key(key)
                        self._preload_stats.loaded_keys += 1
                    except Exception as e:
                        logger.error(f"Failed to preload cache key {key}: {e}")
                        self._preload_stats.failed_keys += 1
                        self._monitor.record_operation(
                            "preload",
                            key,
                            False,
                            time.time() - self._preload_stats.start_time,
                            0,
                            "memory",
                            {"error": str(e)}
                        )
                        if self._monitor_callback:
                            self._monitor_callback("preload", {
                                "success": False,
                                "key": key,
                                "error": str(e)
                            })
                            
                # 记录预热完成
                self._preload_stats.status = "completed"
                self._monitor.record_operation(
                    "preload",
                    "all",
                    True,
                    time.time() - self._preload_stats.start_time,
                    0,
                    "memory",
                    {
                        "total_keys": self._preload_stats.total_keys,
                        "loaded_keys": self._preload_stats.loaded_keys,
                        "failed_keys": self._preload_stats.failed_keys,
                        "strategy": self._preload_strategy.value
                    }
                )
                if self._monitor_callback:
                    self._monitor_callback("preload", {
                        "success": True,
                        "total_keys": self._preload_stats.total_keys,
                        "loaded_keys": self._preload_stats.loaded_keys,
                        "failed_keys": self._preload_stats.failed_keys,
                        "strategy": self._preload_strategy.value
                    })
                    
            except Exception as e:
                logger.error(f"Failed to preload cache: {e}")
                self._preload_stats.status = "failed"
                self._monitor.record_operation(
                    "preload",
                    "all",
                    False,
                    time.time() - self._preload_stats.start_time,
                    0,
                    "memory",
                    {"error": str(e)}
                )
                if self._monitor_callback:
                    self._monitor_callback("preload", {
                        "success": False,
                        "error": str(e)
                    })
                    
            finally:
                self._preload_stats.end_time = time.time()

    def _preload_key(self, key: str) -> None:
        """预热一个键
        
        Args:
            key: 键
        """
        start_time = time.time()
        try:
            data = self._preload_callback(key)
            if data is not None:
                # 计算并保存校验和
                if self._verify_data:
                    self._checksums[key] = self._calculate_checksum(data)
                # 更新访问时间
                self._access_times[key] = time.time()
                logger.debug(f"Preloaded cache key: {key}")
                
                self._monitor.record_operation(
                    "preload",
                    key,
                    True,
                    time.time() - start_time,
                    len(pickle.dumps(data)),
                    "memory",
                    None
                )
                
        except Exception as e:
            logger.error(f"Failed to preload cache key {key}: {e}")
            self._monitor.record_operation(
                "preload",
                key,
                False,
                time.time() - start_time,
                0,
                "memory",
                {"error": str(e)}
            )
            raise

    def _get_preload_keys(self) -> List[str]:
        """根据策略获取预热键列表
        
        Returns:
            预热键列表
        """
        keys = list(self._preload_keys)
        
        if self._preload_strategy == PreloadStrategy.ALL:
            return keys
            
        elif self._preload_strategy == PreloadStrategy.PRIORITY:
            # 按优先级排序（优先级高的在前，相同优先级按字母顺序）
            return sorted(
                keys,
                key=lambda k: (-self._key_priorities.get(k, 0), k)
            )
            
        elif self._preload_strategy == PreloadStrategy.RECENT:
            # 按最近访问时间排序（最近的在前，相同时间按字母顺序）
            return sorted(
                keys,
                key=lambda k: (-self._access_times.get(k, 0), k)
            )
            
        else:  # PreloadStrategy.CUSTOM
            return keys

    def set_key_priority(self, key: str, priority: int) -> None:
        """设置键的优先级
        
        Args:
            key: 缓存键
            priority: 优先级（越大越优先）
        """
        self._key_priorities[key] = priority
        
    def update_access_time(self, key: str) -> None:
        """更新键的访问时间
        
        Args:
            key: 缓存键
        """
        self._access_times[key] = time.time()
        
    def get_preload_stats(self) -> PreloadStats:
        """获取预热统计信息
        
        Returns:
            预热统计信息
        """
        return self._preload_stats
        
    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """获取监控指标
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            监控指标
        """
        return self._monitor.get_metrics(start_time, end_time)
        
    def monitor(self, key: str, stats: Dict[str, Any]) -> None:
        """监控缓存操作
        
        Args:
            key: 缓存键
            stats: 统计信息
        """
        if self._monitor_callback:
            try:
                self._monitor_callback(key, stats)
            except Exception as e:
                logger.error(f"Failed to monitor cache key {key}: {e}")
                
        # 记录操作
        self._monitor.record_operation(
            stats.get("operation", "unknown"),
            key,
            stats.get("success", True),
            stats.get("duration", 0.0),
            stats.get("size", 0),
            stats.get("source", "unknown"),
            stats.get("extra")
        )
        
    def _calculate_checksum(self, data: Any) -> str:
        """计算数据校验和
        
        Args:
            data: 数据
            
        Returns:
            校验和字符串
        """
        serialized = pickle.dumps(data)
        return hashlib.sha256(serialized).hexdigest()
