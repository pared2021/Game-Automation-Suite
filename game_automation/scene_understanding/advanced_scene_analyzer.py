import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
from collections import deque

from .scene_analyzer import SceneAnalyzer, SceneAnalysisError
from utils.logger import detailed_logger

class AdvancedSceneAnalyzer(SceneAnalyzer):
    """高级场景分析器，提供更复杂的场景分析功能"""

    def __init__(self, template_dir: str = "resources/templates", history_size: int = 10):
        """初始化高级场景分析器
        
        Args:
            template_dir: 模板图片目录
            history_size: 保存的场景历史记录数量
        """
        super().__init__(template_dir)
        self.scene_history = deque(maxlen=history_size)
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def analyze_screenshot(self, screenshot: np.ndarray) -> Dict[str, any]:
        """增强的场景分析
        
        Args:
            screenshot: OpenCV格式的截图

        Returns:
            Dict: 详细的分析结果
        """
        # 获取基础分析结果
        base_result = super().analyze_screenshot(screenshot)
        
        # 添加高级分析结果
        advanced_result = {
            **base_result,
            'features': self._extract_features(screenshot),
            'motion': self._detect_motion(),
            'objects': self._detect_objects(screenshot),
            'text_regions': self._detect_text_regions(screenshot)
        }
        
        # 更新场景历史
        self.scene_history.append({
            'timestamp': datetime.now(),
            'scene_type': advanced_result['scene_type'],
            'features': advanced_result['features']
        })
        
        return advanced_result

    def _extract_features(self, image: np.ndarray) -> Dict[str, any]:
        """提取图像特征
        
        Args:
            image: 图像数据

        Returns:
            Dict: 特征信息
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 提取SIFT特征
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            # 计算颜色直方图
            color_hist = self._calculate_color_histogram(image)
            
            # 计算边缘密度
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges > 0)
            
            return {
                'keypoint_count': len(keypoints) if keypoints is not None else 0,
                'color_distribution': color_hist,
                'edge_density': float(edge_density)
            }
            
        except Exception as e:
            detailed_logger.error(f"特征提取失败: {str(e)}")
            return {
                'keypoint_count': 0,
                'color_distribution': [],
                'edge_density': 0.0
            }

    def _detect_motion(self) -> Dict[str, float]:
        """检测场景变化中的运动
        
        Returns:
            Dict: 运动信息
        """
        if len(self.scene_history) < 2:
            return {'motion_level': 0.0, 'direction': None}
            
        try:
            # 获取最近两个场景的特征
            current = self.scene_history[-1]['features']
            previous = self.scene_history[-2]['features']
            
            # 计算特征变化
            feature_change = abs(current['keypoint_count'] - previous['keypoint_count'])
            edge_change = abs(current['edge_density'] - previous['edge_density'])
            
            # 综合评估运动水平
            motion_level = (feature_change / max(current['keypoint_count'], 1) + edge_change) / 2
            
            return {
                'motion_level': float(motion_level),
                'direction': self._estimate_motion_direction()
            }
            
        except Exception as e:
            detailed_logger.error(f"运动检测失败: {str(e)}")
            return {'motion_level': 0.0, 'direction': None}

    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, any]]:
        """检测图像中的对象
        
        Args:
            image: 图像数据

        Returns:
            List[Dict]: 检测到的对象列表
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用级联分类器检测物体
            objects = []
            
            # 阈值化
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # 计算轮廓特征
                area = cv2.contourArea(contour)
                if area < 100:  # 过滤小物体
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                objects.append({
                    'position': (x, y),
                    'size': (w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
            
            return objects
            
        except Exception as e:
            detailed_logger.error(f"物体检测失败: {str(e)}")
            return []

    def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, any]]:
        """检测可能包含文本的区域
        
        Args:
            image: 图像数据

        Returns:
            List[Dict]: 文本区域列表
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值化
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # 查找文本区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            contours, _ = cv2.findContours(
                dilated,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # 使用启发式规则过滤可能的文本区域
                if 2.0 < aspect_ratio < 10.0 and h > 8:
                    text_regions.append({
                        'position': (x, y),
                        'size': (w, h),
                        'confidence': self._calculate_text_confidence(gray[y:y+h, x:x+w])
                    })
            
            return text_regions
            
        except Exception as e:
            detailed_logger.error(f"文本区域检测失败: {str(e)}")
            return []

    def _calculate_color_histogram(self, image: np.ndarray) -> List[float]:
        """计算颜色直方图
        
        Args:
            image: 图像数据

        Returns:
            List[float]: 归一化的直方图数据
        """
        try:
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().tolist()
            return hist
        except Exception as e:
            detailed_logger.error(f"颜色直方图计算失败: {str(e)}")
            return []

    def _estimate_motion_direction(self) -> Optional[str]:
        """估计运动方向
        
        Returns:
            Optional[str]: 运动方向描述
        """
        if len(self.scene_history) < 2:
            return None
            
        try:
            current = self.scene_history[-1]
            previous = self.scene_history[-2]
            
            # 比较特征变化来推测方向
            current_features = current['features']
            previous_features = previous['features']
            
            # 简单的启发式方向判断
            if current_features['edge_density'] > previous_features['edge_density']:
                return 'approaching'
            elif current_features['edge_density'] < previous_features['edge_density']:
                return 'receding'
            else:
                return 'static'
                
        except Exception as e:
            detailed_logger.error(f"运动方向估计失败: {str(e)}")
            return None

    def _calculate_text_confidence(self, region: np.ndarray) -> float:
        """计算区域包含文本的置信度
        
        Args:
            region: 图像区域数据

        Returns:
            float: 置信度分数 (0-1)
        """
        try:
            # 计算区域的一些特征来评估是否包含文本
            
            # 计算垂直和水平梯度
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            
            # 文本通常有强烈的垂直边缘
            gradient_ratio = np.mean(np.abs(sobelx)) / (np.mean(np.abs(sobely)) + 1e-6)
            
            # 计算局部方差（文本区域通常方差较大）
            local_var = np.var(region)
            
            # 综合评分
            score = (gradient_ratio * 0.7 + min(local_var / 1000.0, 1.0) * 0.3)
            return float(min(max(score, 0.0), 1.0))
            
        except Exception as e:
            detailed_logger.error(f"文本置信度计算失败: {str(e)}")
            return 0.0

    def get_scene_history(self) -> List[Dict]:
        """获取场景历史记录
        
        Returns:
            List[Dict]: 场景历史记录列表
        """
        return list(self.scene_history)

    def clear_history(self) -> None:
        """清除场景历史记录"""
        self.scene_history.clear()
