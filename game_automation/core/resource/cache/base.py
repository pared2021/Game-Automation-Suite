"""基础缓存模块"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0  # 命中次数
    misses: int = 0  # 未命中次数
    evictions: int = 0  # 淘汰次数
    size: int = 0  # 当前大小
    max_size: int = 0  # 最大大小
    total_time: float = 0.0  # 总访问时间
    max_time: float = 0.0  # 最大访问时间
    min_time: float = float('inf')  # 最小访问时间
    expired: int = 0  # 过期次数
    updates: int = 0  # 更新次数
    total_size: int = 0  # 总存储大小（字节）
    last_cleanup: Optional[datetime] = None  # 最后清理时间
    hourly_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # 每小时统计
    
    def record_access(self, start_time: float) -> None:
        """记录访问时间
        
        Args:
            start_time: 开始时间
        """
        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.max_time = max(self.max_time, elapsed)
        self.min_time = min(self.min_time, elapsed)
        
    def record_hourly(self, hit: bool = False, miss: bool = False, evict: bool = False) -> None:
        """记录每小时统计
        
        Args:
            hit: 是否命中
            miss: 是否未命中
            evict: 是否淘汰
        """
        hour = datetime.now().strftime('%Y-%m-%d %H:00')
        if hour not in self.hourly_stats:
            self.hourly_stats[hour] = {
                'hits': 0,
                'misses': 0,
                'evictions': 0
            }
            
        if hit:
            self.hourly_stats[hour]['hits'] += 1
        if miss:
            self.hourly_stats[hour]['misses'] += 1
        if evict:
            self.hourly_stats[hour]['evictions'] += 1
            
        # 清理过期统计
        cutoff = datetime.now() - timedelta(days=7)
        self.hourly_stats = {
            k: v for k, v in self.hourly_stats.items()
            if datetime.strptime(k, '%Y-%m-%d %H:00') > cutoff
        }
        
    def hit_rate(self) -> float:
        """获取命中率
        
        Returns:
            命中率
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def eviction_rate(self) -> float:
        """获取淘汰率
        
        Returns:
            淘汰率
        """
        total = self.hits + self.misses
        return self.evictions / total if total > 0 else 0.0
        
    def avg_access_time(self) -> float:
        """获取平均访问时间
        
        Returns:
            平均访问时间（秒）
        """
        total = self.hits + self.misses
        return self.total_time / total if total > 0 else 0.0
        
    def memory_usage(self) -> float:
        """获取内存使用率
        
        Returns:
            内存使用率
        """
        return self.size / self.max_size if self.max_size > 0 else 0.0
        
    def hourly_hit_rate(self, hour: Optional[str] = None) -> float:
        """获取指定小时的命中率
        
        Args:
            hour: 小时（格式：YYYY-MM-DD HH:00）
            
        Returns:
            命中率
        """
        if not hour:
            hour = datetime.now().strftime('%Y-%m-%d %H:00')
            
        if hour not in self.hourly_stats:
            return 0.0
            
        stats = self.hourly_stats[hour]
        total = stats['hits'] + stats['misses']
        return stats['hits'] / total if total > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            统计信息字典
        """
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'max_size': self.max_size,
            'total_time': self.total_time,
            'max_time': self.max_time,
            'min_time': self.min_time,
            'expired': self.expired,
            'updates': self.updates,
            'total_size': self.total_size,
            'hit_rate': self.hit_rate(),
            'eviction_rate': self.eviction_rate(),
            'avg_access_time': self.avg_access_time(),
            'memory_usage': self.memory_usage(),
            'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None,
            'hourly_stats': self.hourly_stats
        }
        
    def clear(self) -> None:
        """清除统计信息"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.min_time = float('inf')
        self.expired = 0
        self.updates = 0
        self.total_size = 0
        self.hourly_stats.clear()
