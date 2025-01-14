"""资源监控实现"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import ResourceBase, ResourceState


class ResourceUsage:
    """资源使用统计"""
    
    def __init__(self, resource: ResourceBase):
        """初始化资源使用统计
        
        Args:
            resource: 资源对象
        """
        self.key = resource.key
        self.type = resource.type
        self.load_count = 0
        self.unload_count = 0
        self.error_count = 0
        self.total_load_time = 0.0
        self.total_unload_time = 0.0
        self.last_load_time: Optional[datetime] = None
        self.last_unload_time: Optional[datetime] = None
        self.peak_ref_count = 0
        
    def update_load_stats(self, duration: float) -> None:
        """更新加载统计
        
        Args:
            duration: 加载耗时（秒）
        """
        self.load_count += 1
        self.total_load_time += duration
        self.last_load_time = datetime.now()
        
    def update_unload_stats(self, duration: float) -> None:
        """更新释放统计
        
        Args:
            duration: 释放耗时（秒）
        """
        self.unload_count += 1
        self.total_unload_time += duration
        self.last_unload_time = datetime.now()
        
    def update_error_stats(self) -> None:
        """更新错误统计"""
        self.error_count += 1
        
    def update_ref_count(self, ref_count: int) -> None:
        """更新引用计数
        
        Args:
            ref_count: 当前引用计数
        """
        self.peak_ref_count = max(self.peak_ref_count, ref_count)


class ResourceLeak:
    """资源泄漏信息"""
    
    def __init__(
        self,
        resource: ResourceBase,
        duration: float,
        ref_count: int
    ):
        """初始化资源泄漏信息
        
        Args:
            resource: 泄漏的资源
            duration: 泄漏持续时间（秒）
            ref_count: 当前引用计数
        """
        self.key = resource.key
        self.type = resource.type
        self.duration = duration
        self.ref_count = ref_count
        self.detected_at = datetime.now()


class MonitorManager:
    """监控管理器
    
    负责资源使用的监控和统计。
    """
    
    def __init__(self):
        """初始化监控管理器"""
        self._usage_stats: Dict[str, ResourceUsage] = {}
        
    async def record_usage(self, resource: ResourceBase) -> None:
        """记录资源使用
        
        Args:
            resource: 资源对象
        """
        if resource.key not in self._usage_stats:
            self._usage_stats[resource.key] = ResourceUsage(resource)
            
        stats = self._usage_stats[resource.key]
        stats.update_ref_count(resource.ref_count)
        
        if resource.state == ResourceState.ERROR:
            stats.update_error_stats()
            
    async def check_leaks(self) -> List[ResourceLeak]:
        """检查资源泄漏
        
        Returns:
            泄漏资源列表
        """
        leaks: List[ResourceLeak] = []
        now = datetime.now()
        
        for key, stats in self._usage_stats.items():
            if (
                stats.ref_count > 0 and
                stats.last_unload_time and
                (now - stats.last_unload_time).total_seconds() > 3600
            ):
                leaks.append(ResourceLeak(
                    stats.resource,
                    (now - stats.last_unload_time).total_seconds(),
                    stats.ref_count
                ))
                
        return leaks
        
    async def get_metrics(self) -> Dict[str, float]:
        """获取性能指标
        
        Returns:
            性能指标字典
        """
        total_resources = len(self._usage_stats)
        if total_resources == 0:
            return {
                'avg_load_time': 0.0,
                'avg_unload_time': 0.0,
                'error_rate': 0.0,
                'cache_hit_rate': 0.0
            }
            
        total_load_time = sum(
            stats.total_load_time
            for stats in self._usage_stats.values()
        )
        total_unload_time = sum(
            stats.total_unload_time
            for stats in self._usage_stats.values()
        )
        total_errors = sum(
            stats.error_count
            for stats in self._usage_stats.values()
        )
        total_loads = sum(
            stats.load_count
            for stats in self._usage_stats.values()
        )
        
        return {
            'avg_load_time': total_load_time / total_loads if total_loads > 0 else 0.0,
            'avg_unload_time': total_unload_time / total_resources,
            'error_rate': total_errors / total_loads if total_loads > 0 else 0.0,
            'cache_hit_rate': 0.0  # TODO: 实现缓存命中率统计
        }
