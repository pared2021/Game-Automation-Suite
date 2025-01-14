"""测试缓存监控器"""

import os
import time
import shutil
import unittest
from unittest.mock import Mock
from datetime import datetime, timedelta

from game_automation.core.resource.cache.monitor import (
    CacheMonitor,
    CacheMetrics,
    CacheMetricType,
    CacheMetricValue
)


class TestCacheMonitor(unittest.TestCase):
    """测试缓存监控器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.test_dir = "data/test/monitor"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.monitor = CacheMonitor(
            save_dir=self.test_dir,
            save_interval=1,
            cleanup_interval=1,
            cleanup_threshold=1
        )
        
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_record_operation(self):
        """测试记录操作"""
        # 记录一些操作
        self.monitor.record_operation(
            "get",
            "key1",
            True,
            0.1,
            1024,
            "memory",
            {"hit": True}
        )
        self.monitor.record_operation(
            "put",
            "key2",
            True,
            0.2,
            2048,
            "disk",
            {"compressed": True}
        )
        
        # 获取指标
        metrics = self.monitor.get_metrics()
        
        # 验证操作计数
        self.assertEqual(
            len(metrics["cache_operation_total"]),
            2
        )
        
        # 验证操作延迟
        self.assertEqual(
            len(metrics["cache_operation_duration_seconds"]),
            2
        )
        
        # 验证数据大小
        self.assertEqual(
            len(metrics["cache_data_size_bytes"]),
            2
        )
        
    def test_metrics_filtering(self):
        """测试指标过滤"""
        # 记录一些操作
        self.monitor.record_operation(
            "get",
            "key1",
            True,
            0.1,
            1024,
            "memory"
        )
        time.sleep(1)
        self.monitor.record_operation(
            "put",
            "key2",
            True,
            0.2,
            2048,
            "disk"
        )
        
        # 获取最近的指标
        now = datetime.now()
        metrics = self.monitor.get_metrics(
            start_time=now - timedelta(seconds=1)
        )
        
        # 验证只返回最近的指标
        self.assertEqual(
            len(metrics["cache_operation_total"]),
            1
        )
        
    def test_metrics_persistence(self):
        """测试指标持久化"""
        # 记录一些操作
        self.monitor.record_operation(
            "get",
            "key1",
            True,
            0.1,
            1024,
            "memory"
        )
        
        # 等待自动保存
        time.sleep(1.1)
        
        # 验证指标文件
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith("metrics_") for f in files))
        
        # 创建新的监控器
        new_monitor = CacheMonitor(
            save_dir=self.test_dir,
            save_interval=1,
            cleanup_interval=1,
            cleanup_threshold=1
        )
        
        # 验证指标加载
        metrics = new_monitor.get_metrics()
        self.assertGreater(len(metrics["cache_operation_total"]), 0)
        
    def test_metrics_cleanup(self):
        """测试指标清理"""
        # 记录一些操作
        self.monitor.record_operation(
            "get",
            "key1",
            True,
            0.1,
            1024,
            "memory"
        )
        
        # 等待自动保存
        time.sleep(1.1)
        
        # 修改文件时间
        for file in os.listdir(self.test_dir):
            if file.startswith("metrics_"):
                filepath = os.path.join(self.test_dir, file)
                old_time = datetime.now() - timedelta(days=2)
                os.utime(filepath, (old_time.timestamp(), old_time.timestamp()))
                
        # 记录新操作触发清理
        self.monitor.record_operation(
            "put",
            "key2",
            True,
            0.2,
            2048,
            "disk"
        )
        
        # 等待清理
        time.sleep(1.1)
        
        # 验证旧文件被清理
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 1)
        
    def test_cleanup(self):
        """测试完全清理"""
        # 记录一些操作
        self.monitor.record_operation(
            "get",
            "key1",
            True,
            0.1,
            1024,
            "memory"
        )
        
        # 等待自动保存
        time.sleep(1.1)
        
        # 执行完全清理
        self.monitor.cleanup()
        
        # 验证内存指标被清理
        metrics = self.monitor.get_metrics()
        self.assertEqual(len(metrics), 0)
        
        # 验证文件被清理
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 0)
        

if __name__ == "__main__":
    unittest.main()
