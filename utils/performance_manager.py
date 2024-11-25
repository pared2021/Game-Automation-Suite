from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time
import psutil
import numpy as np
from collections import deque

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    thread_count: int
    io_read_bytes: int
    io_write_bytes: int
    network_sent_bytes: int
    network_recv_bytes: int
    custom_metrics: Dict[str, float]

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.max_history_size = 1000  # 限制历史记录大小
        self.metrics_history: deque = deque(maxlen=self.max_history_size)
        self.adaptive_thresholds = True  # 启用自适应阈值
        self.sampling_interval = 1.0  # 采样间隔(秒)
        
        # 资源限制
        self.resource_limits = {
            'cpu_threshold': 80.0,  # CPU使用率阈值(%)
            'memory_threshold': 80.0,  # 内存使用率阈值(%)
            'max_threads': 100  # 最大线程数
        }
        
        # 性能统计
        self.last_io_stats = None
        self.last_network_stats = None
        self.last_collection_time = None
        
        # 自适应阈值统计
        self.threshold_update_interval = 300  # 阈值更新间隔(秒)
        self.last_threshold_update = 0
        
        # 进程监控
        self.process = psutil.Process()

    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标
        
        Returns:
            PerformanceMetrics: 性能指标数据
        """
        current_time = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = self.process.cpu_percent()
            
            # 内存使用率
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # 线程数
            thread_count = self.process.num_threads()
            
            # IO统计
            io_counters = self.process.io_counters()
            io_read = io_counters.read_bytes
            io_write = io_counters.write_bytes
            
            # 网络统计
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
            
            # 计算IO和网络速率
            if self.last_collection_time:
                time_diff = (current_time - self.last_collection_time).total_seconds()
                if time_diff > 0:
                    io_read_rate = (io_read - self.last_io_stats[0]) / time_diff
                    io_write_rate = (io_write - self.last_io_stats[1]) / time_diff
                    net_sent_rate = (net_sent - self.last_network_stats[0]) / time_diff
                    net_recv_rate = (net_recv - self.last_network_stats[1]) / time_diff
                else:
                    io_read_rate = io_write_rate = net_sent_rate = net_recv_rate = 0
            else:
                io_read_rate = io_write_rate = net_sent_rate = net_recv_rate = 0
            
            # 更新上次统计
            self.last_io_stats = (io_read, io_write)
            self.last_network_stats = (net_sent, net_recv)
            self.last_collection_time = current_time
            
            # 创建指标对象
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                thread_count=thread_count,
                io_read_bytes=int(io_read_rate),
                io_write_bytes=int(io_write_rate),
                network_sent_bytes=int(net_sent_rate),
                network_recv_bytes=int(net_recv_rate),
                custom_metrics={}
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            detailed_logger.error(f"收集性能指标失败: {str(e)}")
            raise

    def _update_thresholds(self) -> None:
        """更新自适应阈值"""
        current_time = time.time()
        if not self.adaptive_thresholds or \
           current_time - self.last_threshold_update < self.threshold_update_interval:
            return
            
        try:
            if len(self.metrics_history) > 100:
                # 基于历史数据计算新阈值
                cpu_data = [m.cpu_usage for m in self.metrics_history]
                memory_data = [m.memory_usage for m in self.metrics_history]
                thread_data = [m.thread_count for m in self.metrics_history]
                
                # 使用95百分位值加上余量作为新阈值
                self.resource_limits['cpu_threshold'] = min(
                    95.0,  # 最大允许值
                    np.percentile(cpu_data, 95) + 10
                )
                self.resource_limits['memory_threshold'] = min(
                    95.0,  # 最大允许值
                    np.percentile(memory_data, 95) + 10
                )
                self.resource_limits['max_threads'] = min(
                    200,  # 最大允许值
                    int(np.percentile(thread_data, 95) + 5)
                )
                
                detailed_logger.info(
                    f"更新资源阈值: CPU={self.resource_limits['cpu_threshold']:.1f}%, "
                    f"Memory={self.resource_limits['memory_threshold']:.1f}%, "
                    f"Threads={self.resource_limits['max_threads']}"
                )
                
            self.last_threshold_update = current_time
            
        except Exception as e:
            detailed_logger.error(f"更新资源阈值失败: {str(e)}")

    def check_resource_limits(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """检查资源使用是否超过限制
        
        Args:
            metrics: 性能指标数据
            
        Returns:
            Dict[str, bool]: 各项资源是否超限
        """
        return {
            'cpu_exceeded': metrics.cpu_usage > self.resource_limits['cpu_threshold'],
            'memory_exceeded': metrics.memory_usage > self.resource_limits['memory_threshold'],
            'threads_exceeded': metrics.thread_count > self.resource_limits['max_threads']
        }

    @log_exception
    def monitor(self) -> Dict[str, Any]:
        """执行性能监控
        
        Returns:
            Dict[str, Any]: 监控结果
        """
        # 收集指标
        metrics = self._collect_metrics()
        
        # 更新自适应阈值
        self._update_thresholds()
        
        # 检查资源限制
        limit_checks = self.check_resource_limits(metrics)
        
        # 准备监控结果
        result = {
            'timestamp': metrics.timestamp.isoformat(),
            'metrics': {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'thread_count': metrics.thread_count,
                'io_read_rate': metrics.io_read_bytes,
                'io_write_rate': metrics.io_write_bytes,
                'network_sent_rate': metrics.network_sent_bytes,
                'network_recv_rate': metrics.network_recv_bytes
            },
            'thresholds': self.resource_limits.copy(),
            'limit_checks': limit_checks,
            'alerts': []
        }
        
        # 添加告警
        if limit_checks['cpu_exceeded']:
            result['alerts'].append(
                f"CPU使用率超过阈值: {metrics.cpu_usage:.1f}% > {self.resource_limits['cpu_threshold']}%"
            )
        if limit_checks['memory_exceeded']:
            result['alerts'].append(
                f"内存使用率超过阈值: {metrics.memory_usage:.1f}% > {self.resource_limits['memory_threshold']}%"
            )
        if limit_checks['threads_exceeded']:
            result['alerts'].append(
                f"线程数超过限制: {metrics.thread_count} > {self.resource_limits['max_threads']}"
            )
            
        return result

    def get_history_statistics(self) -> Dict[str, Any]:
        """获取历史统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.metrics_history:
            return {}
            
        stats = {
            'time_range': {
                'start': self.metrics_history[0].timestamp.isoformat(),
                'end': self.metrics_history[-1].timestamp.isoformat()
            },
            'samples': len(self.metrics_history),
            'cpu_stats': {},
            'memory_stats': {},
            'thread_stats': {},
            'io_stats': {},
            'network_stats': {}
        }
        
        # 提取数据序列
        cpu_data = [m.cpu_usage for m in self.metrics_history]
        memory_data = [m.memory_usage for m in self.metrics_history]
        thread_data = [m.thread_count for m in self.metrics_history]
        io_read_data = [m.io_read_bytes for m in self.metrics_history]
        io_write_data = [m.io_write_bytes for m in self.metrics_history]
        net_sent_data = [m.network_sent_bytes for m in self.metrics_history]
        net_recv_data = [m.network_recv_bytes for m in self.metrics_history]
        
        # 计算统计值
        stats['cpu_stats'] = {
            'min': np.min(cpu_data),
            'max': np.max(cpu_data),
            'avg': np.mean(cpu_data),
            'std': np.std(cpu_data)
        }
        
        stats['memory_stats'] = {
            'min': np.min(memory_data),
            'max': np.max(memory_data),
            'avg': np.mean(memory_data),
            'std': np.std(memory_data)
        }
        
        stats['thread_stats'] = {
            'min': np.min(thread_data),
            'max': np.max(thread_data),
            'avg': np.mean(thread_data),
            'std': np.std(thread_data)
        }
        
        stats['io_stats'] = {
            'read': {
                'min': np.min(io_read_data),
                'max': np.max(io_read_data),
                'avg': np.mean(io_read_data)
            },
            'write': {
                'min': np.min(io_write_data),
                'max': np.max(io_write_data),
                'avg': np.mean(io_write_data)
            }
        }
        
        stats['network_stats'] = {
            'sent': {
                'min': np.min(net_sent_data),
                'max': np.max(net_sent_data),
                'avg': np.mean(net_sent_data)
            },
            'recv': {
                'min': np.min(net_recv_data),
                'max': np.max(net_recv_data),
                'avg': np.mean(net_recv_data)
            }
        }
        
        return stats

    def add_custom_metric(self, name: str, value: float) -> None:
        """添加自定义指标
        
        Args:
            name: 指标名称
            value: 指标值
        """
        if self.metrics_history:
            self.metrics_history[-1].custom_metrics[name] = value

    def clear_history(self) -> None:
        """清空历史记录"""
        self.metrics_history.clear()
        self.last_io_stats = None
        self.last_network_stats = None
        self.last_collection_time = None
