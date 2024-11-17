from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from game_automation.core.decision_maker import Condition
from utils.logger import detailed_logger

@dataclass
class SceneState:
    """场景状态数据类"""
    scene_type: str
    brightness: float
    complexity: float
    elements: Dict[str, Any]
    timestamp: datetime

class SceneCondition:
    """场景状态到条件的转换处理器"""

    def __init__(self):
        """初始化场景条件转换器"""
        self._state_cache: Dict[str, SceneState] = {}
        self._cache_duration = timedelta(seconds=5)  # 缓存有效期

    def create_condition(self, state: Dict[str, Any]) -> Condition:
        """从场景状态创建条件
        
        Args:
            state: 场景状态数据
            
        Returns:
            Condition: 对应的条件对象
        """
        scene_type = state.get('scene_type', 'unknown')
        
        # 创建条件参数
        condition_params = {
            'scene_type': scene_type,
            'state': {
                'brightness': state.get('scene_state', {}).get('brightness', 0),
                'complexity': state.get('scene_state', {}).get('complexity', 0)
            }
        }
        
        # 添加场景特定的条件参数
        if scene_type != 'unknown':
            condition_params.update(self._extract_scene_specific_params(state))
        
        return Condition(
            condition_type='scene_state',
            parameters=condition_params
        )

    def _extract_scene_specific_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取场景特定的参数
        
        Args:
            state: 场景状态数据
            
        Returns:
            Dict[str, Any]: 场景特定的参数
        """
        scene_type = state.get('scene_type')
        scene_state = state.get('scene_state', {})
        
        params = {}
        
        # 根据场景类型提取特定参数
        if scene_type == 'battle':
            params['in_combat'] = scene_state.get('in_combat', False)
            params['enemy_count'] = scene_state.get('enemy_count', 0)
            params['health_percentage'] = scene_state.get('health_percentage', 100)
            
        elif scene_type == 'menu':
            params['menu_level'] = scene_state.get('menu_level', 0)
            params['available_options'] = scene_state.get('available_options', [])
            
        elif scene_type == 'dialog':
            params['dialog_type'] = scene_state.get('dialog_type', 'normal')
            params['has_choices'] = scene_state.get('has_choices', False)
            
        return params

    def update_state_cache(self, scene_type: str, state: Dict[str, Any]) -> None:
        """更新状态缓存
        
        Args:
            scene_type: 场景类型
            state: 场景状态数据
        """
        self._state_cache[scene_type] = SceneState(
            scene_type=scene_type,
            brightness=state.get('scene_state', {}).get('brightness', 0),
            complexity=state.get('scene_state', {}).get('complexity', 0),
            elements=state.get('scene_state', {}).get('elements', {}),
            timestamp=datetime.now()
        )

    def get_cached_state(self, scene_type: str) -> Optional[SceneState]:
        """获取缓存的状态
        
        Args:
            scene_type: 场景类型
            
        Returns:
            Optional[SceneState]: 缓存的场景状态，如果未找到或已过期则返回None
        """
        if scene_type not in self._state_cache:
            return None
            
        state = self._state_cache[scene_type]
        if datetime.now() - state.timestamp > self._cache_duration:
            del self._state_cache[scene_type]
            return None
            
        return state

    def compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """比较两个场景状态的相似度
        
        Args:
            state1: 第一个场景状态
            state2: 第二个场景状态
            
        Returns:
            float: 状态相似度(0-1)
        """
        if state1.get('scene_type') != state2.get('scene_type'):
            return 0.0
            
        scene_state1 = state1.get('scene_state', {})
        scene_state2 = state2.get('scene_state', {})
        
        # 计算基础属性的相似度
        brightness_diff = abs(scene_state1.get('brightness', 0) - scene_state2.get('brightness', 0))
        complexity_diff = abs(scene_state1.get('complexity', 0) - scene_state2.get('complexity', 0))
        
        # 归一化差异值
        brightness_similarity = max(0, 1 - brightness_diff / 255)
        complexity_similarity = max(0, 1 - complexity_diff / 100)
        
        # 计算场景特定属性的相似度
        specific_similarity = self._compare_specific_attributes(scene_state1, scene_state2)
        
        # 加权平均
        weights = {
            'brightness': 0.3,
            'complexity': 0.3,
            'specific': 0.4
        }
        
        similarity = (
            weights['brightness'] * brightness_similarity +
            weights['complexity'] * complexity_similarity +
            weights['specific'] * specific_similarity
        )
        
        return similarity

    def _compare_specific_attributes(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """比较场景特定属性的相似度
        
        Args:
            state1: 第一个场景状态
            state2: 第二个场景状态
            
        Returns:
            float: 特定属性相似度(0-1)
        """
        # 获取所有属性键
        all_keys = set(state1.keys()) | set(state2.keys())
        if not all_keys:
            return 1.0  # 如果没有特定属性，认为完全相似
            
        matches = 0
        for key in all_keys:
            if key in ['brightness', 'complexity']:
                continue  # 跳过基础属性
                
            val1 = state1.get(key)
            val2 = state2.get(key)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值类型，计算相对差异
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    matches += 1
                else:
                    diff = abs(val1 - val2) / max_val
                    matches += max(0, 1 - diff)
                    
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                # 列表类型，计算交集比例
                union = set(val1) | set(val2)
                intersection = set(val1) & set(val2)
                if union:
                    matches += len(intersection) / len(union)
                else:
                    matches += 1
                    
            else:
                # 其他类型，直接比较相等性
                matches += 1 if val1 == val2 else 0
                
        return matches / len(all_keys)

    def clear_expired_cache(self) -> None:
        """清理过期的缓存"""
        current_time = datetime.now()
        expired_keys = [
            scene_type for scene_type, state in self._state_cache.items()
            if current_time - state.timestamp > self._cache_duration
        ]
        
        for key in expired_keys:
            del self._state_cache[key]
