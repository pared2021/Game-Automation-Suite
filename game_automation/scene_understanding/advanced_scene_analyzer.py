from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
import json
import logging
from datetime import datetime

from PIL import Image
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class SceneAnalyzerError(Exception):
    """场景分析器错误"""
    pass

class SceneElement:
    """场景元素"""
    def __init__(
        self,
        element_id: str,
        element_type: str,
        bounds: Dict[str, int],
        confidence: float,
        properties: Dict = None
    ):
        self.element_id = element_id
        self.element_type = element_type
        self.bounds = bounds
        self.confidence = confidence
        self.properties = properties or {}
        
    def to_dict(self) -> Dict:
        """转换为字典
        
        Returns:
            Dict: 元素信息字典
        """
        return {
            'id': self.element_id,
            'type': self.element_type,
            'bounds': self.bounds,
            'confidence': self.confidence,
            'properties': self.properties
        }

class AdvancedSceneAnalyzer:
    """高级场景分析器"""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._templates: Dict[str, Image.Image] = {}
        self._features: Dict[str, np.ndarray] = {}
        self._scene_cache: Dict[str, Dict] = {}
        self._feature_extractor = None
        self._classifier = None
        
    async def initialize(self):
        """初始化分析器"""
        if not self._initialized:
            try:
                # 加载配置
                config_path = Path("config/config.json")
                if not config_path.exists():
                    raise SceneAnalyzerError("配置文件不存在")
                    
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                    
                # 加载模板
                await self._load_templates()
                
                # 初始化特征提取器
                self._feature_extractor = models.resnet18(pretrained=True)
                self._feature_extractor.fc = nn.Identity()  # 移除最后的全连接层
                self._feature_extractor.eval()
                
                # 初始化分类器
                self._classifier = models.resnet18(pretrained=True)
                num_classes = len(self._templates)
                self._classifier.fc = nn.Linear(512, num_classes)
                self._classifier.eval()
                
                # 提取特征
                await self._extract_features()
                
                self._initialized = True
                
            except Exception as e:
                raise SceneAnalyzerError(f"初始化失败: {str(e)}")
                
    async def cleanup(self):
        """清理资源"""
        if self._initialized:
            self._templates.clear()
            self._features.clear()
            self._scene_cache.clear()
            self._initialized = False
            
    async def _load_templates(self):
        """加载模板图片"""
        try:
            template_dir = Path(self._config["paths"]["templates"])
            if not template_dir.exists():
                raise SceneAnalyzerError("模板目录不存在")
                
            # 加载所有PNG文件
            for template_path in template_dir.glob("**/*.png"):
                template_id = template_path.stem
                template = Image.open(template_path)
                self._templates[template_id] = template
                
        except Exception as e:
            raise SceneAnalyzerError(f"加载模板失败: {str(e)}")
            
    async def _extract_features(self):
        """提取特征"""
        try:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            for template_id, template in self._templates.items():
                # 转换图像
                tensor = transform(template).unsqueeze(0)
                
                # 提取特征
                with torch.no_grad():
                    features = self._feature_extractor(tensor)
                    
                self._features[template_id] = features.numpy()
                
        except Exception as e:
            raise SceneAnalyzerError(f"提取特征失败: {str(e)}")
            
    async def analyze_scene(
        self,
        screenshot: Image.Image
    ) -> Dict[str, Any]:
        """分析场景
        
        Args:
            screenshot: 屏幕截图
            
        Returns:
            Dict[str, Any]: 场景信息
        """
        if not self._initialized:
            raise SceneAnalyzerError("未初始化")
            
        try:
            # 转换图像
            scene_array = np.array(screenshot)
            
            # 场景分类
            scene_type = await self._classify_scene(scene_array)
            
            # 元素检测
            elements = await self._detect_elements(scene_array)
            
            # 计算置信度
            confidence = await self._calculate_confidence(scene_type, elements)
            
            # 缓存结果
            scene_info = {
                'type': scene_type,
                'elements': {
                    e.element_id: e.to_dict()
                    for e in elements
                },
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            self._scene_cache[scene_type] = scene_info
            
            return scene_info
            
        except Exception as e:
            logging.error(f"场景分析失败: {str(e)}")
            return {
                'type': 'unknown',
                'elements': {},
                'confidence': 0.0,
                'error': str(e)
            }
            
    async def _classify_scene(
        self,
        scene_array: np.ndarray
    ) -> str:
        """场景分类
        
        Args:
            scene_array: 场景图像数组
            
        Returns:
            str: 场景类型
        """
        try:
            # 转换图像
            scene = Image.fromarray(scene_array)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensor = transform(scene).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = self._feature_extractor(tensor)
                
            # 计算相似度
            max_similarity = -1
            best_match = 'unknown'
            
            for template_id, template_features in self._features.items():
                similarity = cosine_similarity(
                    features.numpy().reshape(1, -1),
                    template_features.reshape(1, -1)
                )[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = template_id
                    
            # 检查置信度
            threshold = self._config["scene"]["match_threshold"]
            if max_similarity < threshold:
                return 'unknown'
                
            return best_match
            
        except Exception as e:
            logging.error(f"场景分类失败: {str(e)}")
            return 'unknown'
            
    async def _detect_elements(
        self,
        scene_array: np.ndarray
    ) -> List[SceneElement]:
        """检测元素
        
        Args:
            scene_array: 场景图像数组
            
        Returns:
            List[SceneElement]: 元素列表
        """
        try:
            elements = []
            
            # 转换为灰度图
            gray = cv2.cvtColor(scene_array, cv2.COLOR_RGB2GRAY)
            
            # 遍历模板
            for template_id, template in self._templates.items():
                # 转换模板
                template_array = np.array(template)
                template_gray = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)
                
                # 模板匹配
                result = cv2.matchTemplate(
                    gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED
                )
                
                # 查找匹配位置
                threshold = self._config["scene"]["match_threshold"]
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    # 计算边界
                    bounds = {
                        'left': int(pt[0]),
                        'top': int(pt[1]),
                        'right': int(pt[0] + template_array.shape[1]),
                        'bottom': int(pt[1] + template_array.shape[0])
                    }
                    
                    # 创建元素
                    element = SceneElement(
                        element_id=f"{template_id}_{len(elements)}",
                        element_type=template_id,
                        bounds=bounds,
                        confidence=float(result[pt[1], pt[0]])
                    )
                    
                    elements.append(element)
                    
            return elements
            
        except Exception as e:
            logging.error(f"元素检测失败: {str(e)}")
            return []
            
    async def _calculate_confidence(
        self,
        scene_type: str,
        elements: List[SceneElement]
    ) -> float:
        """计算置信度
        
        Args:
            scene_type: 场景类型
            elements: 元素列表
            
        Returns:
            float: 置信度
        """
        try:
            # 如果场景未知
            if scene_type == 'unknown':
                return 0.0
                
            # 如果没有元素
            if not elements:
                return 0.0
                
            # 计算平均置信度
            confidences = [
                element.confidence
                for element in elements
                if element.element_type == scene_type
            ]
            
            if not confidences:
                return 0.0
                
            return sum(confidences) / len(confidences)
            
        except Exception as e:
            logging.error(f"计算置信度失败: {str(e)}")
            return 0.0
            
    async def get_element(
        self,
        element_id: str,
        scene_type: str = None
    ) -> Optional[Dict]:
        """获取元素信息
        
        Args:
            element_id: 元素ID
            scene_type: 场景类型
            
        Returns:
            Optional[Dict]: 元素信息
        """
        if scene_type and scene_type in self._scene_cache:
            scene = self._scene_cache[scene_type]
            return scene['elements'].get(element_id)
            
        for scene in self._scene_cache.values():
            if element_id in scene['elements']:
                return scene['elements'][element_id]
                
        return None
        
    async def find_elements(
        self,
        element_type: str = None,
        min_confidence: float = 0.0,
        **filters
    ) -> List[Dict]:
        """查找元素
        
        Args:
            element_type: 元素类型
            min_confidence: 最小置信度
            **filters: 过滤条件
            
        Returns:
            List[Dict]: 元素列表
        """
        elements = []
        
        for scene in self._scene_cache.values():
            for element in scene['elements'].values():
                if element_type and element['type'] != element_type:
                    continue
                    
                if element['confidence'] < min_confidence:
                    continue
                    
                match = True
                for key, value in filters.items():
                    if key not in element or element[key] != value:
                        match = False
                        break
                        
                if match:
                    elements.append(element)
                    
        return elements
