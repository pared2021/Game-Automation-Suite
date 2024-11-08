import unittest
import time
import asyncio
import psutil
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from game_automation.game_engine import GameEngine
from game_automation.optimization.resource_manager import ResourceManager
from game_automation.optimization.multi_threading import thread_pool

class TestPerformanceSuite(unittest.TestCase):
    def setUp(self):
        self.game_engine = GameEngine()
        self.resource_manager = ResourceManager()
        self.process = psutil.Process(os.getpid())
        gc.collect()  # 清理垃圾收集器

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        return self.process.memory_info().rss / 1024 / 1024  # 转换为MB

    def get_cpu_usage(self):
        """获取CPU使用情况"""
        return self.process.cpu_percent(interval=1)

    async def test_long_running_stability(self):
        """测试长时间运行的稳定性"""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # 运行30分钟的自动化测试
        await self.game_engine.initialize()
        asyncio.create_task(self.game_engine.run_game_loop())
        
        memory_samples = []
        cpu_samples = []
        
        for _ in range(30):  # 30分钟，每分钟采样一次
            await asyncio.sleep(60)
            memory_samples.append(self.get_memory_usage())
            cpu_samples.append(self.get_cpu_usage())
        
        end_memory = self.get_memory_usage()
        total_time = time.time() - start_time
        
        # 验证内存使用是否稳定（不应该持续增长）
        memory_growth = end_memory - start_memory
        self.assertLess(memory_growth, 100)  # 内存增长不应超过100MB
        
        # 验证CPU使用是否稳定
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        self.assertLess(avg_cpu, 80)  # 平均CPU使用率不应超过80%
        
        # 验证运行时间
        self.assertGreaterEqual(total_time, 1800)  # 至少运行30分钟

    async def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        # 执行密集型操作
        for _ in range(1000):
            await self.game_engine.get_game_state()
            if _ % 100 == 0:
                gc.collect()
                current_memory = self.get_memory_usage()
                # 确保内存增长不超过初始内存的20%
                self.assertLess(current_memory, initial_memory * 1.2)

    async def test_concurrent_performance(self):
        """测试并发性能"""
        async def concurrent_task():
            await self.game_engine.get_game_state()
            await asyncio.sleep(0.1)
        
        # 测试不同并发级别
        concurrency_levels = [5, 10, 20, 50]
        for concurrency in concurrency_levels:
            start_time = time.time()
            tasks = [concurrent_task() for _ in range(concurrency)]
            await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            # 验证执行时间是否在可接受范围内
            self.assertLess(execution_time, concurrency * 0.2)

    def test_resource_usage_monitoring(self):
        """测试资源使用监控"""
        # 监控CPU使用率
        cpu_usage = self.resource_manager.monitor_cpu_usage(duration=5)
        self.assertIsInstance(cpu_usage, float)
        self.assertLess(cpu_usage, 90)  # CPU使用率不应超过90%
        
        # 监控内存使用率
        memory_usage = self.resource_manager.monitor_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertLess(memory_usage, 80)  # 内存使用率不应超过80%
        
        # 监控磁盘I/O
        io_stats = self.resource_manager.monitor_io_usage()
        self.assertIsInstance(io_stats, dict)
        self.assertIn('read_bytes', io_stats)
        self.assertIn('write_bytes', io_stats)

    async def test_response_time(self):
        """测试响应时间"""
        response_times = []
        
        for _ in range(100):
            start_time = time.time()
            await self.game_engine.get_game_state()
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # 验证平均响应时间和最大响应时间
        self.assertLess(avg_response_time, 0.1)  # 平均响应时间应小于100ms
        self.assertLess(max_response_time, 0.5)  # 最大响应时间应小于500ms

    def test_thread_pool_efficiency(self):
        """测试线程池效率"""
        def cpu_bound_task():
            result = 0
            for i in range(1000000):
                result += i
            return result
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.time()
            results = list(executor.map(cpu_bound_task, range(10)))
            execution_time = time.time() - start_time
            
            # 验证执行时间和结果
            self.assertLess(execution_time, 5)  # 应在5秒内完成
            self.assertEqual(len(results), 10)

    def run_async_test(self, test_method):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_method())

    def test_all(self):
        """运行所有性能测试"""
        self.run_async_test(self.test_long_running_stability)
        self.run_async_test(self.test_memory_leak_detection)
        self.run_async_test(self.test_concurrent_performance)
        self.test_resource_usage_monitoring()
        self.run_async_test(self.test_response_time)
        self.test_thread_pool_efficiency()

if __name__ == '__main__':
    unittest.main()
