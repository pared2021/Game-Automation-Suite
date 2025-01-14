"""测试缓存序列化器"""

import os
import shutil
import unittest
from datetime import datetime, timedelta

from game_automation.core.resource.cache.serializer import CacheSerializer


class TestCacheSerializer(unittest.TestCase):
    """测试缓存序列化器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.test_dir = "data/test/serializer"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.serializer = CacheSerializer(self.test_dir)
        
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_json_serialization(self):
        """测试 JSON 序列化"""
        # 基本数据类型
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
        
        # 保存数据
        self.assertTrue(
            self.serializer.save(data, "test.json", format="json")
        )
        
        # 加载数据
        loaded = self.serializer.load("test.json", format="json")
        self.assertEqual(loaded, data)
        
    def test_pickle_serialization(self):
        """测试 Pickle 序列化"""
        # 使用可序列化的数据类型
        data = {
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "datetime": datetime.now()
        }
        
        # 保存数据
        self.assertTrue(
            self.serializer.save(data, "test.pkl", format="pickle")
        )
        
        # 加载数据
        loaded = self.serializer.load("test.pkl", format="pickle")
        self.assertEqual(loaded, data)
        
    def test_datetime_handling(self):
        """测试日期时间处理"""
        # 包含日期时间的数据
        now = datetime.now()
        data = {
            "datetime": now,
            "list": [now - timedelta(days=1), now],
            "dict": {"time": now + timedelta(days=1)}
        }
        
        # 保存数据
        self.assertTrue(
            self.serializer.save(data, "test.json", format="json")
        )
        
        # 加载数据
        loaded = self.serializer.load("test.json", format="json")
        
        # 验证日期时间
        self.assertEqual(loaded["datetime"], now)
        self.assertEqual(loaded["list"][0], now - timedelta(days=1))
        self.assertEqual(loaded["list"][1], now)
        self.assertEqual(loaded["dict"]["time"], now + timedelta(days=1))
        
    def test_file_operations(self):
        """测试文件操作"""
        # 保存数据
        data = {"key": "value"}
        self.assertTrue(
            self.serializer.save(data, "test/nested/file.json")
        )
        
        # 验证目录创建
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "test/nested"))
        )
        
        # 列出文件
        files = self.serializer.list_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], "test/nested/file.json")
        
        # 删除文件
        self.assertTrue(self.serializer.delete("test/nested/file.json"))
        self.assertFalse(
            os.path.exists(os.path.join(self.test_dir, "test/nested/file.json"))
        )
        
    def test_error_handling(self):
        """测试错误处理"""
        # 加载不存在的文件
        self.assertIsNone(self.serializer.load("not_exists.json"))
        self.assertEqual(
            self.serializer.load("not_exists.json", default=42),
            42
        )
        
        # 保存到无效路径
        self.assertFalse(
            self.serializer.save({"key": "value"}, "/invalid/path.json")
        )
        
        # 加载无效 JSON
        with open(os.path.join(self.test_dir, "invalid.json"), "w") as f:
            f.write("invalid json")
        self.assertIsNone(self.serializer.load("invalid.json"))
        
    def test_file_listing(self):
        """测试文件列表"""
        # 创建测试文件
        self.serializer.save({"key": "value"}, "test1.json")
        self.serializer.save({"key": "value"}, "test2.json")
        self.serializer.save({"key": "value"}, "test/test3.json")
        self.serializer.save({"key": "value"}, "test.txt")
        
        # 列出所有文件
        files = self.serializer.list_files()
        self.assertEqual(len(files), 4)
        
        # 按模式列出文件
        json_files = self.serializer.list_files(pattern=".json")
        self.assertEqual(len(json_files), 3)
        
        # 非递归列出文件
        root_files = self.serializer.list_files(recursive=False)
        self.assertEqual(len(root_files), 3)
        

if __name__ == "__main__":
    unittest.main()
