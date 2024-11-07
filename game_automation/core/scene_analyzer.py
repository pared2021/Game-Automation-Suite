import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from ..nlp.advanced_language_processor import advanced_language_processor
from utils.logger import detailed_logger
from utils.error_handler import log_exception, GameAutomationError

@dataclass
class SceneObject:
    """场景对象数据类"""
    id: str
    type: str
    name: str
    position: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    attributes: Dict[str, Any]

@dataclass
class SceneContext:
    """场景上下文数据类"""
    scene_type: str
    objects: List[SceneObject]
    text_elements: List[Dict[str, Any]]
    ui_elements: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

class SceneAnalyzer:
    """
    场景分析器核心类
    负责游戏场景的分析、对象检测和场景理解
    """
    def __init__(self):
        self.logger = detailed_logger
        self.language_processor = advanced_language_processor
        self._scene_templates = {}
        self._object_templates = {}
        self._last_scene: Optional[SceneContext] = None

    @log_exception
    async def initialize(self) -> None:
        """初始化场景分析器"""
        try:
            # 加载场景模板
            await self._load_scene_templates()
            # 加载对象模板
            await self._load_object_templates()
            self.logger.info("Scene Analyzer initialized successfully")
        except Exception as e:
            raise GameAutomationError(f"Failed to initialize scene analyzer: {str(e)}")

    @log_exception
    async def analyze_game_scene(self, 
                               screen: np.ndarray, 
                               text: str) -> SceneContext:
        """
        分析游戏场景
        :param screen: 屏幕截图
        :param text: OCR识别的文本
        :return: 场景上下文对象
        """
        try:
            # 场景类型识别
            scene_type = await self._determine_scene_type(screen, text)
            
            # 对象检测
            objects = await self._detect_objects(screen)
            
            # 文本元素分析
            text_elements = await self._analyze_text_elements(text)
            
            # UI元素检测
            ui_elements = await self._detect_ui_elements(screen)
            
            # 分析对象间关系
            relationships = await self._analyze_relationships(objects, text_elements)
            
            # 创建场景上下文
            scene_context = SceneContext(
                scene_type=scene_type,
                objects=objects,
                text_elements=text_elements,
                ui_elements=ui_elements,
                relationships=relationships
            )
            
            self._last_scene = scene_context
            return scene_context
            
        except Exception as e:
            raise GameAutomationError(f"Failed to analyze game scene: {str(e)}")

    @log_exception
    async def _determine_scene_type(self, 
                                  screen: np.ndarray, 
                                  text: str) -> str:
        """
        确定场景类型
        :param screen: 屏幕截图
        :param text: OCR识别的文本
        :return: 场景类型
        """
        try:
            # 基于模板匹配
            for template_name, template in self._scene_templates.items():
                similarity = await self._calculate_scene_similarity(screen, template)
                if similarity > 0.8:  # 相似度阈值
                    return template_name
            
            # 基于文本分析
            scene_type = await self.language_processor.classify_scene(text)
            if scene_type:
                return scene_type
                
            return "unknown"
        except Exception as e:
            self.logger.error(f"Error determining scene type: {str(e)}")
            return "unknown"

    @log_exception
    async def _detect_objects(self, screen: np.ndarray) -> List[SceneObject]:
        """
        检测场景中的对象
        :param screen: 屏幕截图
        :return: 检测到的对象列表
        """
        objects = []
        try:
            # 对每个对象模板进行匹配
            for obj_id, template in self._object_templates.items():
                matches = await self._template_matching(screen, template['image'])
                for match in matches:
                    x, y, w, h = match
                    confidence = template['confidence']
                    
                    obj = SceneObject(
                        id=obj_id,
                        type=template['type'],
                        name=template['name'],
                        position=(x, y, w, h),
                        confidence=confidence,
                        attributes=template.get('attributes', {})
                    )
                    objects.append(obj)
                    
            return objects
        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}")
            return []

    @log_exception
    async def _analyze_text_elements(self, text: str) -> List[Dict[str, Any]]:
        """
        分析文本元素
        :param text: OCR识别的文本
        :return: 文本元素列表
        """
        try:
            return await self.language_processor.extract_text_elements(text)
        except Exception as e:
            self.logger.error(f"Error analyzing text elements: {str(e)}")
            return []

    @log_exception
    async def _detect_ui_elements(self, screen: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测UI元素
        :param screen: 屏幕截图
        :return: UI元素列表
        """
        try:
            # 边缘检测
            edges = cv2.Canny(screen, 100, 200)
            
            # 轮廓检测
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ui_elements = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 100:  # 过滤小区域
                    ui_elements.append({
                        'type': 'ui_element',
                        'position': (x, y, w, h)
                    })
                    
            return ui_elements
        except Exception as e:
            self.logger.error(f"Error detecting UI elements: {str(e)}")
            return []

    @log_exception
    async def _analyze_relationships(self,
                                   objects: List[SceneObject],
                                   text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分析对象间的关系
        :param objects: 检测到的对象列表
        :param text_elements: 文本元素列表
        :return: 关系列表
        """
        relationships = []
        try:
            # 分析对象间的空间关系
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    rel = self._analyze_spatial_relationship(obj1, obj2)
                    if rel:
                        relationships.append(rel)
            
            # 分析对象与文本的关系
            for obj in objects:
                for text_elem in text_elements:
                    rel = self._analyze_text_object_relationship(obj, text_elem)
                    if rel:
                        relationships.append(rel)
                        
            return relationships
        except Exception as e:
            self.logger.error(f"Error analyzing relationships: {str(e)}")
            return []

    @staticmethod
    def _analyze_spatial_relationship(obj1: SceneObject, 
                                    obj2: SceneObject) -> Optional[Dict[str, Any]]:
        """分析两个对象间的空间关系"""
        x1, y1, w1, h1 = obj1.position
        x2, y2, w2, h2 = obj2.position
        
        # 计算中心点
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        # 确定相对位置
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        if abs(dx) > abs(dy):
            relation = "right_of" if dx > 0 else "left_of"
        else:
            relation = "below" if dy > 0 else "above"
            
        return {
            'type': 'spatial',
            'object1_id': obj1.id,
            'object2_id': obj2.id,
            'relation': relation
        }

    @staticmethod
    def _analyze_text_object_relationship(obj: SceneObject,
                                        text_elem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析对象与文本间的关系"""
        # TODO: 实现文本与对象的关系分析
        return None

    async def _template_matching(self,
                               screen: np.ndarray,
                               template: np.ndarray,
                               threshold: float = 0.8) -> List[Tuple[int, int, int, int]]:
        """
        模板匹配
        :param screen: 屏幕截图
        :param template: 模板图像
        :param threshold: 匹配阈值
        :return: 匹配位置列表
        """
        try:
            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            matches = []
            
            for pt in zip(*locations[::-1]):
                matches.append((
                    int(pt[0]),
                    int(pt[1]),
                    template.shape[1],
                    template.shape[0]
                ))
                
            return matches
        except Exception as e:
            self.logger.error(f"Error in template matching: {str(e)}")
            return []

    async def _calculate_scene_similarity(self,
                                        screen: np.ndarray,
                                        template: np.ndarray) -> float:
        """
        计算场景相似度
        :param screen: 屏幕截图
        :param template: 模板图像
        :return: 相似度分数
        """
        try:
            # 调整大小以匹配
            if screen.shape != template.shape:
                template = cv2.resize(template, (screen.shape[1], screen.shape[0]))
            
            # 计算直方图相似度
            hist1 = cv2.calcHist([screen], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([template], [0], None, [256], [0, 256])
            
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0, similarity)  # 确保返回值非负
            
        except Exception as e:
            self.logger.error(f"Error calculating scene similarity: {str(e)}")
            return 0.0

    async def _load_scene_templates(self) -> None:
        """加载场景模板"""
        # TODO: 实现场景模板加载逻辑
        pass

    async def _load_object_templates(self) -> None:
        """加载对象模板"""
        # TODO: 实现对象模板加载逻辑
        pass

    @property
    def last_scene(self) -> Optional[SceneContext]:
        """获取最后分析的场景"""
        return self._last_scene

# 创建全局实例
scene_analyzer = SceneAnalyzer()
