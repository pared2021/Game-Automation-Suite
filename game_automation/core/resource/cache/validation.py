"""缓存验证模块"""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Callable


class CacheValidator:
    """缓存验证器
    
    特性：
    - 支持数据完整性验证
    - 支持自定义验证规则
    - 支持验证统计
    - 支持验证缓存
    """
    
    def __init__(
        self,
        verify_data: bool = True,
        verify_interval: Optional[int] = None,
        custom_validator: Optional[Callable[[str, Any], bool]] = None
    ):
        """初始化验证器
        
        Args:
            verify_data: 是否验证数据
            verify_interval: 验证间隔（秒）
            custom_validator: 自定义验证函数
        """
        self._verify_data = verify_data
        self._verify_interval = verify_interval
        self._custom_validator = custom_validator
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            'validations': 0,
            'failures': 0,
            'cache_hits': 0,
            'custom_validations': 0
        }
        
    def _compute_hash(self, data: Any) -> str:
        """计算数据哈希值
        
        Args:
            data: 数据
            
        Returns:
            哈希值
        """
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
    def _should_validate(self, key: str) -> bool:
        """判断是否需要验证
        
        Args:
            key: 缓存键
            
        Returns:
            是否需要验证
        """
        if not self._verify_data:
            return False
            
        if key not in self._validation_cache:
            return True
            
        if not self._verify_interval:
            self._stats['cache_hits'] += 1
            return False
            
        last_validation = self._validation_cache[key]['timestamp']
        return time.time() - last_validation >= self._verify_interval
        
    def validate(self, key: str, data: Any) -> bool:
        """验证数据
        
        Args:
            key: 缓存键
            data: 缓存数据
            
        Returns:
            验证是否通过
        """
        self._stats['validations'] += 1
        
        # 检查是否需要验证
        if not self._should_validate(key):
            return True
            
        # 计算数据哈希值
        current_hash = self._compute_hash(data)
        
        # 检查缓存中的哈希值
        if key in self._validation_cache:
            if current_hash != self._validation_cache[key]['hash']:
                self._stats['failures'] += 1
                return False
                
        # 执行自定义验证
        if self._custom_validator:
            self._stats['custom_validations'] += 1
            if not self._custom_validator(key, data):
                self._stats['failures'] += 1
                return False
                
        # 更新验证缓存
        self._validation_cache[key] = {
            'hash': current_hash,
            'timestamp': time.time()
        }
        
        return True
        
    def invalidate(self, key: str) -> None:
        """使缓存键失效
        
        Args:
            key: 缓存键
        """
        if key in self._validation_cache:
            del self._validation_cache[key]
            
    def clear_cache(self) -> None:
        """清除验证缓存"""
        self._validation_cache.clear()
        
    def get_stats(self) -> dict:
        """获取验证统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        total = stats['validations']
        if total > 0:
            stats['failure_rate'] = stats['failures'] / total
            stats['cache_hit_rate'] = stats['cache_hits'] / total
            stats['custom_validation_rate'] = stats['custom_validations'] / total
        else:
            stats['failure_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
            stats['custom_validation_rate'] = 0.0
        return stats
        
    def clear_stats(self) -> None:
        """清除统计信息"""
        self._stats = {
            'validations': 0,
            'failures': 0,
            'cache_hits': 0,
            'custom_validations': 0
        }
