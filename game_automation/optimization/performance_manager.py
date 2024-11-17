from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

@dataclass
class PerformanceMetrics:
    """性能指标数据"""
    cpu_usage: float
    memory_usage: float
    thread_count: int
    operation_count: int
    average_response_time: float
    error_count: int
    timestamp: datetime

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, sampling_interval: float = 1.0):
        """初始化性能监控器
        
        Args:
            sampling_interval: 采样间隔(秒)
        """
        self.sampling_interval = sampling_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 性能统计
        self.operation_count = 0
        self.error_count = 0
        self.response_times: List[float] = []
        
        # 资源限制
        self.cpu_threshold = 80.0  # CPU使用率阈值
        self.memory_threshold = 80.0  # 内存使用率阈值
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=psutil.cpu_count(),
            thread_name_prefix="perf_monitor"
        )

    def start_monitoring(self) -> None:
        """启动性能监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        detailed_logger.info("启动性能监控")

    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        detailed_logger.info("停止性能监控")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查资源使用
                self._check_resource_usage(metrics)
                
                # 清理旧数据
                self._cleanup_old_metrics()
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                detailed_logger.error(f"性能监控异常: {str(e)}")

    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标
        
        Returns:
            PerformanceMetrics: 性能指标数据
        """
        process = psutil.Process()
        
        # 收集CPU和内存使用率
        cpu_usage = process.cpu_percent()
        memory_usage = process.memory_percent()
        
        # 收集线程信息
        thread_count = threading.active_count()
        
        # 计算平均响应时间
        avg_response_time = (
            np.mean(self.response_times)
            if self.response_times else 0.0
        )
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            thread_count=thread_count,
            operation_count=self.operation_count,
            average_response_time=avg_response_time,
            error_count=self.error_count,
            timestamp=datetime.now()
        )

    def _check_resource_usage(self, metrics: PerformanceMetrics) -> None:
        """检查资源使用情况
        
        Args:
            metrics: 性能指标数据
        """
        # 检查CPU使用率
        if metrics.cpu_usage > self.cpu_threshold:
            detailed_logger.warning(
                f"CPU使用率过高: {metrics.cpu_usage:.1f}%"
            )
            
        # 检查内存使用率
        if metrics.memory_usage > self.memory_threshold:
            detailed_logger.warning(
                f"内存使用率过高: {metrics.memory_usage:.1f}%"
            )

    def _cleanup_old_metrics(self, max_age: int = 3600) -> None:
        """清理旧的指标数据
        
        Args:
            max_age: 最大保留时间(秒)
        """
        cutoff_time = datetime.now() - timedelta(seconds=max_age)
        self.metrics_history = [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]

    def record_operation(self, response_time: float) -> None:
        """记录操作执行
        
        Args:
            response_time: 响应时间(秒)
        """
        self.operation_count += 1
        self.response_times.append(response_time)
        
        # 限制响应时间列表大小
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

    def record_error(self) -> None:
        """记录错误"""
        self.error_count += 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标
        
        Returns:
            PerformanceMetrics: 性能指标数据
        """
        return self._collect_metrics()

    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.metrics_history:
            return {}
            
        # 计算各项指标的统计值
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        thread_count = [m.thread_count for m in self.metrics_history]
        response_times = self.response_times
        
        return {
            'cpu_usage': {
                'min': min(cpu_usage),
                'max': max(cpu_usage),
                'avg': np.mean(cpu_usage),
                'std': np.std(cpu_usage)
            },
            'memory_usage': {
                'min': min(memory_usage),
                'max': max(memory_usage),
                'avg': np.mean(memory_usage),
                'std': np.std(memory_usage)
            },
            'thread_count': {
                'min': min(thread_count),
                'max': max(thread_count),
                'avg': np.mean(thread_count)
            },
            'response_time': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'avg': np.mean(response_times) if response_times else 0,
                'p95': np.percentile(response_times, 95) if response_times else 0,
                'p99': np.percentile(response_times, 99) if response_times else 0
            },
            'operation_count': self.operation_count,
            'error_count': self.error_count,
            'error_rate': (
                self.error_count / self.operation_count
                if self.operation_count > 0 else 0
            )
        }

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        """初始化资源管理器"""
        self.resources: Dict[str, Any] = {}
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        self.resource_usage: Dict[str, int] = {}
        self.resource_limits: Dict[str, int] = {}

    async def acquire_resource(self, resource_id: str,
                             timeout: Optional[float] = None) -> bool:
        """获取资源
        
        Args:
            resource_id: 资源ID
            timeout: 超时时间(秒)
            
        Returns:
            bool: 是否成功获取资源
        """
        if resource_id not in self.resource_locks:
            self.resource_locks[resource_id] = asyncio.Lock()
            self.resource_usage[resource_id] = 0
            
        try:
            # 获取资源锁
            if timeout:
                lock_acquired = await asyncio.wait_for(
                    self.resource_locks[resource_id].acquire(),
                    timeout
                )
            else:
                await self.resource_locks[resource_id].acquire()
                lock_acquired = True
                
            if lock_acquired:
                # 检查资源限制
                if (resource_id in self.resource_limits and
                    self.resource_usage[resource_id] >= self.resource_limits[resource_id]):
                    self.resource_locks[resource_id].release()
                    return False
                    
                self.resource_usage[resource_id] += 1
                return True
                
            return False
            
        except asyncio.TimeoutError:
            return False
        except Exception as e:
            detailed_logger.error(f"获取资源失败: {str(e)}")
            return False

    def release_resource(self, resource_id: str) -> None:
        """释放资源
        
        Args:
            resource_id: 资源ID
        """
        if resource_id in self.resource_locks:
            try:
                self.resource_usage[resource_id] -= 1
                self.resource_locks[resource_id].release()
            except Exception as e:
                detailed_logger.error(f"释放资源失败: {str(e)}")

    def set_resource_limit(self, resource_id: str, limit: int) -> None:
        """设置资源限制
        
        Args:
            resource_id: 资源ID
            limit: 限制值
        """
        self.resource_limits[resource_id] = limit

    def get_resource_usage(self, resource_id: str) -> int:
        """获取资源使用量
        
        Args:
            resource_id: 资源ID
            
        Returns:
            int: 资源使用量
        """
        return self.resource_usage.get(resource_id, 0)

class ConcurrencyManager:
    """并发管理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """初始化并发管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers or psutil.cpu_count(),
            thread_name_prefix="concurrency"
        )
        self.task_queues: Dict[str, asyncio.Queue] = {}
        self.workers: Dict[str, List[asyncio.Task]] = {}
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.start_monitoring()

    async def submit_task(self, queue_id: str,
                         task_func: Callable,
                         *args,
                         **kwargs) -> Any:
        """提交任务
        
        Args:
            queue_id: 队列ID
            task_func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 任务结果
        """
        # 确保队列存在
        if queue_id not in self.task_queues:
            self.task_queues[queue_id] = asyncio.Queue()
            self.workers[queue_id] = []
            
        # 创建任务
        task = {
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'result_future': asyncio.Future()
        }
        
        # 提交任务
        start_time = time.time()
        try:
            await self.task_queues[queue_id].put(task)
            
            # 确保有工作线程处理任务
            self._ensure_workers(queue_id)
            
            # 等待结果
            result = await task['result_future']
            
            # 记录性能指标
            response_time = time.time() - start_time
            self.performance_monitor.record_operation(response_time)
            
            return result
            
        except Exception as e:
            self.performance_monitor.record_error()
            raise GameAutomationError(f"任务执行失败: {str(e)}")

    def _ensure_workers(self, queue_id: str,
                       min_workers: int = 1,
                       max_workers: int = 5) -> None:
        """确保有足够的工作线程
        
        Args:
            queue_id: 队列ID
            min_workers: 最小工作线程数
            max_workers: 最大工作线程数
        """
        # 清理已完成的工作线程
        self.workers[queue_id] = [
            worker for worker in self.workers[queue_id]
            if not worker.done()
        ]
        
        # 创建新的工作线程
        current_workers = len(self.workers[queue_id])
        if current_workers < min_workers:
            for _ in range(min_workers - current_workers):
                worker = asyncio.create_task(
                    self._worker_loop(queue_id)
                )
                self.workers[queue_id].append(worker)

    async def _worker_loop(self, queue_id: str) -> None:
        """工作线程循环
        
        Args:
            queue_id: 队列ID
        """
        queue = self.task_queues[queue_id]
        
        while True:
            try:
                # 获取任务
                task = await queue.get()
                
                # 执行任务
                try:
                    result = await task['func'](
                        *task['args'],
                        **task['kwargs']
                    )
                    task['result_future'].set_result(result)
                except Exception as e:
                    task['result_future'].set_exception(e)
                    
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                detailed_logger.error(f"工作线程异常: {str(e)}")

    def shutdown(self) -> None:
        """关闭并发管理器"""
        # 停止性能监控
        self.performance_monitor.stop_monitoring()
        
        # 取消所有工作线程
        for workers in self.workers.values():
            for worker in workers:
                worker.cancel()
                
        # 关闭线程池
        self.thread_pool.shutdown()
