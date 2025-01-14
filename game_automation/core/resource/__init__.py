"""资源管理系统

该模块提供了一个完整的资源管理系统，用于管理游戏自动化过程中的各类资源：
- 图像资源（截图、模板图像）
- 配置资源（JSON、YAML）
- 模型资源（AI模型、权重文件）
- 缓存资源（临时数据、中间结果）

主要功能：
- 资源加载和释放
- 资源缓存管理
- 资源监控和统计
- 错误处理和恢复
"""

from .base import ResourceBase, ResourceState, ResourceType
from .manager import ResourceManager
from .loader import ResourceLoader
from .cache import CacheManager
from .monitor import MonitorManager
from .errors import (
    ResourceError,
    ResourceNotFoundError,
    ResourceLoadError,
    ResourceStateError,
    CacheError,
    MonitorError
)

__all__ = [
    'ResourceBase',
    'ResourceState',
    'ResourceType',
    'ResourceManager',
    'ResourceLoader',
    'CacheManager',
    'MonitorManager',
    'ResourceError',
    'ResourceNotFoundError',
    'ResourceLoadError',
    'ResourceStateError',
    'CacheError',
    'MonitorError'
]
