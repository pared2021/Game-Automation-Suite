"""图像资源实现"""

import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from ..base import ResourceBase, ResourceType
from ..errors import ResourceLoadError


class ImageResource(ResourceBase):
    """图像资源
    
    支持的图像格式：
    - PNG
    - JPEG
    - BMP
    
    特性：
    - 自动缓存
    - 图像预处理
    - 图像变换
    """
    
    def __init__(
        self,
        key: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        preprocess: bool = True
    ):
        """初始化图像资源
        
        Args:
            key: 资源标识符
            path: 图像文件路径
            metadata: 资源元数据
            preprocess: 是否进行预处理
        """
        super().__init__(key, ResourceType.IMAGE, metadata)
        self._path = Path(path)
        self._preprocess = preprocess
        self._image: Optional[np.ndarray] = None
        self._gray: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None
        
    @property
    def path(self) -> Path:
        """获取图像路径"""
        return self._path
        
    @property
    def image(self) -> Optional[np.ndarray]:
        """获取原始图像"""
        return self._image
        
    @property
    def gray(self) -> Optional[np.ndarray]:
        """获取灰度图像"""
        return self._gray
        
    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """获取图像尺寸 (width, height)"""
        return self._size
        
    async def _do_load(self) -> None:
        """加载图像
        
        Raises:
            ResourceLoadError: 图像加载失败
        """
        try:
            # 检查文件是否存在
            if not self._path.exists():
                raise ResourceLoadError(
                    self.key,
                    f"Image file not found: {self._path}"
                )
                
            # 读取图像
            self._image = cv2.imread(str(self._path))
            if self._image is None:
                raise ResourceLoadError(
                    self.key,
                    f"Failed to load image: {self._path}"
                )
                
            # 获取图像尺寸
            height, width = self._image.shape[:2]
            self._size = (width, height)
            
            # 预处理
            if self._preprocess:
                await self._preprocess_image()
                
        except Exception as e:
            raise ResourceLoadError(self.key, cause=e)
            
    async def _do_unload(self) -> None:
        """释放图像"""
        self._image = None
        self._gray = None
        self._size = None
        
    async def _preprocess_image(self) -> None:
        """预处理图像
        
        - 转换为灰度图
        - 其他预处理步骤
        """
        if self._image is not None:
            self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            
    def resize(self, width: int, height: int) -> np.ndarray:
        """调整图像大小
        
        Args:
            width: 目标宽度
            height: 目标高度
            
        Returns:
            调整后的图像
            
        Raises:
            ValueError: 图像未加载
        """
        if self._image is None:
            raise ValueError("Image not loaded")
            
        return cv2.resize(self._image, (width, height))
        
    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """裁剪图像
        
        Args:
            x: 左上角 x 坐标
            y: 左上角 y 坐标
            width: 裁剪宽度
            height: 裁剪高度
            
        Returns:
            裁剪后的图像
            
        Raises:
            ValueError: 图像未加载或参数无效
        """
        if self._image is None:
            raise ValueError("Image not loaded")
            
        if (
            x < 0 or y < 0 or
            width <= 0 or height <= 0 or
            x + width > self._size[0] or
            y + height > self._size[1]
        ):
            raise ValueError("Invalid crop parameters")
            
        return self._image[y:y+height, x:x+width]
        
    def match_template(
        self,
        template: 'ImageResource',
        threshold: float = 0.8,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> Optional[Tuple[int, int, float]]:
        """模板匹配
        
        Args:
            template: 模板图像
            threshold: 匹配阈值
            method: 匹配方法
            
        Returns:
            匹配结果 (x, y, score)，如果未匹配则返回 None
            
        Raises:
            ValueError: 图像未加载或参数无效
        """
        if self._gray is None or template.gray is None:
            raise ValueError("Image or template not loaded")
            
        # 执行模板匹配
        result = cv2.matchTemplate(self._gray, template.gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 根据匹配方法选择结果
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            score = 1 - min_val
            loc = min_loc
        else:
            score = max_val
            loc = max_loc
            
        # 检查阈值
        if score < threshold:
            return None
            
        return (loc[0], loc[1], score)
