"""缓存监控"""

import os
import time
import shutil
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from .serializer import CacheSerializer

logger = logging.getLogger(__name__)


class CacheMetricType(Enum):
    """缓存指标类型"""
    COUNTER = "counter"  # 计数器，如命中次数
    GAUGE = "gauge"  # 仪表盘，如当前缓存大小
    HISTOGRAM = "histogram"  # 直方图，如响应时间分布
    SUMMARY = "summary"  # 摘要，如响应时间分位数


@dataclass
class CacheMetricValue:
    """缓存指标值"""
    type: CacheMetricType
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class CacheMetrics:
    """缓存指标"""
    
    def __init__(self):
        self.metrics: Dict[str, List[CacheMetricValue]] = {}
        
    def add_metric(
        self,
        name: str,
        type: CacheMetricType,
        value: Any,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """添加指标
        
        Args:
            name: 指标名称
            type: 指标类型
            value: 指标值
            labels: 指标标签
            timestamp: 指标时间戳
        """
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append(
            CacheMetricValue(
                type=type,
                value=value,
                labels=labels or {},
                timestamp=timestamp or datetime.now()
            )
        )
        
    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[CacheMetricValue]]:
        """获取指标
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            指标数据
        """
        if not start_time and not end_time:
            return self.metrics
            
        filtered = {}
        for name, values in self.metrics.items():
            filtered[name] = [
                v for v in values
                if (not start_time or v.timestamp >= start_time) and
                (not end_time or v.timestamp <= end_time)
            ]
        return filtered


class CacheMonitor:
    """缓存监控器
    
    特性：
    - 支持多种指标类型：计数器、仪表盘、直方图、摘要
    - 支持指标标签
    - 支持时间范围查询
    - 支持指标持久化
    - 支持自定义监控回调
    """
    
    def __init__(
        self,
        save_dir: str = "data/monitor/cache",
        save_interval: int = 300,  # 5分钟
        cleanup_interval: int = 86400,  # 1天
        cleanup_threshold: int = 7  # 7天
    ):
        """初始化缓存监控器
        
        Args:
            save_dir: 保存目录
            save_interval: 保存间隔（秒）
            cleanup_interval: 清理间隔（秒）
            cleanup_threshold: 清理阈值（天）
        """
        self._metrics = CacheMetrics()
        self._save_dir = save_dir
        self._save_interval = save_interval
        self._cleanup_interval = cleanup_interval
        self._cleanup_threshold = cleanup_threshold
        self._last_save = 0  # 从不保存开始
        self._last_cleanup = time.time()
        
        # 初始化序列化器
        self._serializer = CacheSerializer(save_dir)
        
        # 加载历史指标
        self._load_metrics()
        
    def record_operation(
        self,
        operation: str,
        key: str,
        success: bool,
        duration: float,
        size: int,
        source: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录缓存操作
        
        Args:
            operation: 操作类型（get、put、remove等）
            key: 缓存键
            success: 是否成功
            duration: 持续时间（秒）
            size: 数据大小（字节）
            source: 数据来源（memory、disk等）
            extra: 额外信息
        """
        labels = {
            "operation": operation,
            "key": key,
            "source": source
        }
        
        # 记录操作结果
        self._metrics.add_metric(
            "cache_operation_total",
            CacheMetricType.COUNTER,
            1,
            {**labels, "success": str(success)}
        )
        
        # 记录操作延迟
        self._metrics.add_metric(
            "cache_operation_duration_seconds",
            CacheMetricType.HISTOGRAM,
            duration,
            labels
        )
        
        # 记录数据大小
        if size > 0:
            self._metrics.add_metric(
                "cache_data_size_bytes",
                CacheMetricType.GAUGE,
                size,
                labels
            )
            
        # 记录额外信息
        if extra:
            for key, value in extra.items():
                if isinstance(value, (int, float)):
                    self._metrics.add_metric(
                        f"cache_{key}",
                        CacheMetricType.GAUGE,
                        value,
                        labels
                    )
                    
        # 检查是否需要保存
        self._check_save()
        
        # 检查是否需要清理
        self._check_cleanup()
        
    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[CacheMetricValue]]:
        """获取指标数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            指标数据
        """
        return self._metrics.get_metrics(start_time, end_time)
        
    def _check_save(self) -> None:
        """检查是否需要保存指标"""
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self._save_metrics()
            self._last_save = now
            
    def _check_cleanup(self) -> None:
        """检查是否需要清理指标"""
        now = time.time()
        if now - self._last_cleanup >= self._cleanup_interval:
            self._cleanup_metrics()
            self._last_cleanup = now
            
    def _save_metrics(self) -> None:
        """保存指标"""
        try:
            # 获取当前时间戳
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            
            # 将指标转换为字典
            metrics_dict = {}
            for name, values in self._metrics.metrics.items():
                metrics_dict[name] = [
                    {
                        "type": v.type.value,
                        "value": v.value,
                        "timestamp": v.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "labels": v.labels
                    }
                    for v in values
                ]
                
            # 保存指标
            if metrics_dict:
                filename = f"metrics_{timestamp}.json"
                success = self._serializer.save(metrics_dict, filename)
                if success:
                    logger.debug(f"Saved metrics to {filename}")
                    # 清理旧文件
                    self._cleanup_metrics()
                else:
                    logger.error(f"Failed to save metrics to {filename}")
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            
    def _cleanup_metrics(self) -> None:
        """清理指标"""
        try:
            # 获取清理阈值时间
            threshold = datetime.now() - timedelta(days=self._cleanup_threshold)
            
            # 获取所有指标文件
            files = self._serializer.list_files(pattern=".json")
            
            # 按时间排序
            file_times = []
            for filename in files:
                try:
                    # 从文件名中获取时间戳
                    parts = filename.split("_")
                    if len(parts) < 2:
                        continue
                        
                    date_str = parts[1].split(".")[0]
                    file_time = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                    file_times.append((filename, file_time))
                    
                except Exception as e:
                    logger.error(f"Failed to parse file time {filename}: {e}")
                    
            # 按时间排序（最新的在前）
            file_times.sort(key=lambda x: x[1], reverse=True)
            
            # 保留最新的文件，删除其他文件
            for i, (filename, file_time) in enumerate(file_times):
                if i > 0 or file_time < threshold:  # 保留最新的一个文件
                    if self._serializer.delete(filename):
                        logger.debug(f"Deleted metrics file: {filename}")
                        
            # 清理内存中的过期指标
            for name in list(self._metrics.metrics.keys()):
                self._metrics.metrics[name] = [
                    v for v in self._metrics.metrics[name]
                    if v.timestamp >= threshold
                ]
                
            # 重置时间戳
            self._last_cleanup = time.time()
                
        except Exception as e:
            logger.error(f"Failed to cleanup metrics: {e}")

    def _load_metrics(self) -> None:
        """加载历史指标"""
        try:
            # 获取所有指标文件
            files = self._serializer.list_files(pattern=".json")
            
            # 加载每个文件
            for filename in files:
                try:
                    # 加载指标数据
                    metrics_dict = self._serializer.load(filename)
                    if not metrics_dict:
                        continue
                        
                    # 添加指标
                    for name, values in metrics_dict.items():
                        for value in values:
                            try:
                                # 如果时间戳是字符串，解析它
                                timestamp = value["timestamp"]
                                if isinstance(timestamp, str):
                                    timestamp = datetime.strptime(
                                        timestamp,
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                elif isinstance(timestamp, datetime):
                                    pass
                                else:
                                    raise ValueError(
                                        f"Invalid timestamp type: {type(timestamp)}"
                                    )
                                    
                                self._metrics.add_metric(
                                    name,
                                    CacheMetricType(value["type"]),
                                    value["value"],
                                    value["labels"],
                                    timestamp
                                )
                            except (ValueError, KeyError) as e:
                                logger.error(
                                    f"Failed to parse metric value: {e}"
                                )
                                
                    logger.debug(f"Loaded metrics from {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to load metrics from {filename}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            
    def cleanup(self) -> None:
        """清理所有指标"""
        try:
            # 保存当前指标
            if self._metrics.metrics:
                self._save_metrics()
            
            # 清空内存中的指标
            self._metrics = CacheMetrics()
            
            # 清理所有文件
            if os.path.exists(self._save_dir):
                shutil.rmtree(self._save_dir)
            os.makedirs(self._save_dir, exist_ok=True)
            
            # 重置时间戳
            self._last_save = 0
            self._last_cleanup = time.time()
            
            logger.info("Cleaned up all metrics")
            
        except Exception as e:
            logger.error(f"Failed to cleanup all metrics: {e}")
            raise  # 重新抛出异常以便测试捕获
