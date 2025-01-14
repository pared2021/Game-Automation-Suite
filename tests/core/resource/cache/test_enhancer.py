"""测试缓存增强器"""

import os
import time
import shutil
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from game_automation.core.resource.cache.enhancer import (
    CacheEnhancer,
    PreloadStrategy,
    PreloadStats
)


class TestCacheEnhancer(unittest.TestCase):
    """测试缓存增强器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.test_dir = "data/test/cache"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.preload_callback = Mock()
        self.monitor_callback = Mock()
        
        self.enhancer = CacheEnhancer(
            compression_level=6,
            compression_threshold=1024,
            verify_data=True,
            preload_keys=["key1", "key2", "key3"],
            preload_callback=self.preload_callback,
            monitor_callback=self.monitor_callback,
            preload_strategy=PreloadStrategy.ALL,
            preload_batch_size=2,
            preload_interval=0.1,
            monitor_save_dir=self.test_dir
        )
        
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_compression(self):
        """测试数据压缩"""
        # 测试小数据不压缩
        data = b"hello"
        compressed, is_compressed = self.enhancer.compress(data)
        self.assertFalse(is_compressed)
        self.assertEqual(data, compressed)
        
        # 测试大数据压缩
        data = b"x" * 2048
        compressed, is_compressed = self.enhancer.compress(data)
        self.assertTrue(is_compressed)
        self.assertLess(len(compressed), len(data))
        
        # 测试解压缩
        decompressed = self.enhancer.decompress(compressed, is_compressed)
        self.assertEqual(data, decompressed)
        
    def test_verify(self):
        """测试数据验证"""
        # 测试新数据
        data = {"key": "value"}
        self.assertTrue(self.enhancer.verify("key1", data))
        
        # 测试相同数据
        self.assertTrue(self.enhancer.verify("key1", data))
        
        # 测试不同数据
        data2 = {"key": "value2"}
        self.assertFalse(self.enhancer.verify("key1", data2))
        
    def test_preload(self):
        """测试缓存预热"""
        # 模拟预加载数据
        self.preload_callback.side_effect = lambda key: {"data": key}
        
        # 执行预热
        self.enhancer.preload()
        
        # 验证回调调用
        self.assertEqual(self.preload_callback.call_count, 3)
        self.preload_callback.assert_any_call("key1")
        self.preload_callback.assert_any_call("key2")
        self.preload_callback.assert_any_call("key3")
        
        # 验证预热统计
        stats = self.enhancer.get_preload_stats()
        self.assertEqual(stats.status, "completed")
        self.assertEqual(stats.total_keys, 3)
        self.assertEqual(stats.loaded_keys, 3)
        self.assertEqual(stats.failed_keys, 0)
        self.assertEqual(stats.progress, 100)
        
    def test_preload_priority(self):
        """测试优先级预热"""
        # 设置优先级
        self.enhancer.set_key_priority("key1", 1)
        self.enhancer.set_key_priority("key2", 3)
        self.enhancer.set_key_priority("key3", 2)
        
        # 设置预热策略
        self.enhancer._preload_strategy = PreloadStrategy.PRIORITY
        
        # 记录预热顺序
        preloaded_keys = []
        self.preload_callback.side_effect = lambda key: preloaded_keys.append(key)
        
        # 执行预热
        self.enhancer.preload()
        
        # 验证预热顺序
        self.assertEqual(preloaded_keys, ["key2", "key3", "key1"])
        
    def test_preload_recent(self):
        """测试最近访问预热"""
        # 设置访问时间
        self.enhancer.update_access_time("key1")
        time.sleep(0.1)
        self.enhancer.update_access_time("key2")
        time.sleep(0.1)
        self.enhancer.update_access_time("key3")
        
        # 设置预热策略
        self.enhancer._preload_strategy = PreloadStrategy.RECENT
        
        # 记录预热顺序
        preloaded_keys = []
        self.preload_callback.side_effect = lambda key: preloaded_keys.append(key)
        
        # 执行预热
        self.enhancer.preload()
        
        # 验证预热顺序
        self.assertEqual(preloaded_keys, ["key3", "key2", "key1"])
        
    def test_monitor(self):
        """测试监控回调"""
        # 执行一些操作
        data = b"x" * 2048
        compressed, is_compressed = self.enhancer.compress(data)
        self.enhancer.decompress(compressed, is_compressed)
        
        # 验证监控回调
        self.assertTrue(self.monitor_callback.called)
        
        # 验证监控指标
        metrics = self.enhancer._monitor.get_metrics()
        self.assertIn("cache_operation_total", metrics)
        self.assertIn("cache_operation_duration_seconds", metrics)
        self.assertIn("cache_data_size_bytes", metrics)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试压缩错误
        with patch("pickle.dumps", side_effect=Exception("Compression error")):
            with self.assertRaises(Exception):
                self.enhancer.compress({"key": "value"})
                
        # 测试解压缩错误
        with patch("pickle.loads", side_effect=Exception("Decompression error")):
            with self.assertRaises(Exception):
                self.enhancer.decompress(b"invalid", True)
                
        # 测试预热错误
        self.preload_callback.side_effect = Exception("Preload error")
        self.enhancer.preload()
        stats = self.enhancer.get_preload_stats()
        self.assertEqual(stats.failed_keys, 3)
        

if __name__ == "__main__":
    unittest.main()
