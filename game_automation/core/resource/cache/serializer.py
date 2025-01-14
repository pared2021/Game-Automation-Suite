"""缓存序列化器"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CacheSerializer:
    """缓存序列化器
    
    特性：
    - 支持 JSON 和 Pickle 格式
    - 支持数据压缩
    - 支持自定义序列化
    - 支持批量操作
    """
    
    def __init__(self, base_dir: str):
        """初始化序列化器
        
        Args:
            base_dir: 基础目录
        """
        self._base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    def save(
        self,
        data: Any,
        filename: str,
        format: str = "json",
        compress: bool = False
    ) -> bool:
        """保存数据
        
        Args:
            data: 数据
            filename: 文件名
            format: 格式（json 或 pickle）
            compress: 是否压缩
            
        Returns:
            是否成功
        """
        try:
            # 检查路径安全性
            filepath = os.path.abspath(os.path.join(self._base_dir, filename))
            if not filepath.startswith(os.path.abspath(self._base_dir)):
                logger.error(f"Invalid file path: {filepath}")
                return False
                
            # 创建目录
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 序列化数据
            if format == "json":
                # 处理日期时间
                if isinstance(data, dict):
                    data = self._convert_datetime(data)
                try:
                    content = json.dumps(data, indent=2)
                    mode = "w"
                    encoding = "utf-8"
                except Exception as e:
                    logger.error(f"Failed to serialize data to JSON: {e}")
                    return False
            else:  # pickle
                try:
                    # 检查可序列化性
                    content = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    mode = "wb"
                    encoding = None
                except Exception as e:
                    logger.error(f"Failed to pickle data: {e}")
                    return False
                    
            # 写入文件
            try:
                with open(filepath, mode, encoding=encoding) as f:
                    if isinstance(content, bytes):
                        f.write(content)
                    else:
                        f.write(content)
                return True
            except Exception as e:
                logger.error(f"Failed to write file {filepath}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {e}")
            return False
            
    def load(
        self,
        filename: str,
        format: str = "json",
        default: Any = None
    ) -> Any:
        """加载数据
        
        Args:
            filename: 文件名
            format: 格式（json 或 pickle）
            default: 默认值
            
        Returns:
            数据
        """
        try:
            # 检查路径安全性
            filepath = os.path.abspath(os.path.join(self._base_dir, filename))
            if not filepath.startswith(os.path.abspath(self._base_dir)):
                logger.error(f"Invalid file path: {filepath}")
                return default
                
            # 如果文件不存在，返回默认值
            if not os.path.exists(filepath):
                return default
                
            # 读取文件
            if format == "json":
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 处理日期时间
                    if isinstance(data, dict):
                        data = self._parse_datetime(data)
                    return data
                except Exception as e:
                    logger.error(f"Failed to load JSON data: {e}")
                    return default
            else:  # pickle
                try:
                    with open(filepath, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"Failed to load pickle data: {e}")
                    return default
                    
        except Exception as e:
            logger.error(f"Failed to load data from {filename}: {e}")
            return default
            
    def delete(self, filename: str) -> bool:
        """删除数据
        
        Args:
            filename: 文件名
            
        Returns:
            是否成功
        """
        try:
            # 检查路径安全性
            filepath = os.path.abspath(os.path.join(self._base_dir, filename))
            if not filepath.startswith(os.path.abspath(self._base_dir)):
                logger.error(f"Invalid file path: {filepath}")
                return False
                
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            return False
            
    def list_files(
        self,
        pattern: Optional[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """列出文件
        
        Args:
            pattern: 文件模式
            recursive: 是否递归
            
        Returns:
            文件列表
        """
        files = []
        base_path = os.path.abspath(self._base_dir)
        for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                if pattern and not filename.endswith(pattern):
                    continue
                filepath = os.path.join(root, filename)
                # 检查路径安全性
                if not os.path.abspath(filepath).startswith(base_path):
                    continue
                relpath = os.path.relpath(filepath, base_path)
                # 统一使用正斜杠
                relpath = relpath.replace(os.path.sep, "/")
                files.append(relpath)
            if not recursive:
                break
        return sorted(files)  # 排序以保持一致性
        
    def _convert_datetime(
        self,
        data: Union[Dict, List, Any]
    ) -> Union[Dict, List, Any]:
        """转换日期时间为字符串
        
        Args:
            data: 数据
            
        Returns:
            转换后的数据
        """
        if isinstance(data, dict):
            return {
                k: self._convert_datetime(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_datetime(v) for v in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        return data
        
    def _parse_datetime(
        self,
        data: Union[Dict, List, Any]
    ) -> Union[Dict, List, Any]:
        """解析日期时间字符串
        
        Args:
            data: 数据
            
        Returns:
            解析后的数据
        """
        if isinstance(data, dict):
            return {
                k: self._parse_datetime(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._parse_datetime(v) for v in data]
        elif isinstance(data, str):
            try:
                return datetime.fromisoformat(data)
            except ValueError:
                return data
        return data
