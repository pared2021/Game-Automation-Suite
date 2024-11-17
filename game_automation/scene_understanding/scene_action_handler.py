from typing import Dict, Optional
import numpy as np
from datetime import datetime

from game_automation.core.decision_maker import Action, Condition
from game_automation.scene_understanding.scene_analyzer import SceneAnalyzer
from utils.logger import detailed_logger

class SceneActionHandler:
    """处理场景相关的Action和Condition"""

    def __init__(self, scene_analyzer: SceneAnalyzer):
        """初始化场景动作处理器
        
        Args:
            scene_analyzer: 场景分析器实例
        """
        self.scene_analyzer = scene_analyzer
        self.last_analysis: Optional[Dict] = None
        self.last_analysis_time: Optional[datetime] = None

    def evaluate_scene_type(self, condition: Condition, context: Dict) -> bool:
        """评估场景类型条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - scene_type: 期望的场景类型
            context: 上下文数据，需包含:
                    - screenshot: 当前截图数据
        
        Returns:
            bool: 条件是否满足
        """
        if 'scene_type' not in condition.parameters:
            detailed_logger.error("场景类型条件缺少必需参数: scene_type")
            return False

        if 'screenshot' not in context:
            detailed_logger.error("上下文缺少必需数据: screenshot")
            return False

        try:
            # 分析场景
            analysis = self.scene_analyzer.analyze_screenshot(context['screenshot'])
            self.last_analysis = analysis
            self.last_analysis_time = datetime.now()

            expected_type = condition.parameters['scene_type']
            actual_type = analysis['scene_type']

            return expected_type == actual_type

        except Exception as e:
            detailed_logger.error(f"场景类型评估失败: {str(e)}")
            return False

    def evaluate_scene_changed(self, condition: Condition, context: Dict) -> bool:
        """评估场景切换条件
        
        Args:
            condition: 条件对象
            context: 上下文数据，需包含:
                    - screenshot: 当前截图数据
        
        Returns:
            bool: 条件是否满足
        """
        if 'screenshot' not in context:
            detailed_logger.error("上下文缺少必需数据: screenshot")
            return False

        try:
            # 分析场景
            analysis = self.scene_analyzer.analyze_screenshot(context['screenshot'])
            self.last_analysis = analysis
            self.last_analysis_time = datetime.now()

            return analysis['scene_changed']

        except Exception as e:
            detailed_logger.error(f"场景切换评估失败: {str(e)}")
            return False

    def evaluate_scene_brightness(self, condition: Condition, context: Dict) -> bool:
        """评估场景亮度条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - threshold: 亮度阈值
                      - operator: 比较运算符 (greater_than/less_than)
            context: 上下文数据，需包含:
                    - screenshot: 当前截图数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'threshold' not in params:
            detailed_logger.error("场景亮度条件缺少必需参数: threshold")
            return False

        if 'screenshot' not in context:
            detailed_logger.error("上下文缺少必需数据: screenshot")
            return False

        try:
            # 分析场景
            analysis = self.scene_analyzer.analyze_screenshot(context['screenshot'])
            self.last_analysis = analysis
            self.last_analysis_time = datetime.now()

            brightness = analysis['scene_state']['brightness']
            threshold = params['threshold']
            operator = params.get('operator', 'greater_than')

            if operator == 'greater_than':
                return brightness > threshold
            elif operator == 'less_than':
                return brightness < threshold
            else:
                detailed_logger.error(f"不支持的运算符: {operator}")
                return False

        except Exception as e:
            detailed_logger.error(f"场景亮度评估失败: {str(e)}")
            return False

    def evaluate_scene_complexity(self, condition: Condition, context: Dict) -> bool:
        """评估场景复杂度条件
        
        Args:
            condition: 条件对象，参数需包含:
                      - threshold: 复杂度阈值
                      - operator: 比较运算符 (greater_than/less_than)
            context: 上下文数据，需包含:
                    - screenshot: 当前截图数据
        
        Returns:
            bool: 条件是否满足
        """
        params = condition.parameters
        if 'threshold' not in params:
            detailed_logger.error("场景复杂度条件缺少必需参数: threshold")
            return False

        if 'screenshot' not in context:
            detailed_logger.error("上下文缺少必需数据: screenshot")
            return False

        try:
            # 分析场景
            analysis = self.scene_analyzer.analyze_screenshot(context['screenshot'])
            self.last_analysis = analysis
            self.last_analysis_time = datetime.now()

            complexity = analysis['scene_state']['complexity']
            threshold = params['threshold']
            operator = params.get('operator', 'greater_than')

            if operator == 'greater_than':
                return complexity > threshold
            elif operator == 'less_than':
                return complexity < threshold
            else:
                detailed_logger.error(f"不支持的运算符: {operator}")
                return False

        except Exception as e:
            detailed_logger.error(f"场景复杂度评估失败: {str(e)}")
            return False

    async def handle_save_template(self, action: Action) -> bool:
        """处理保存模板动作
        
        Args:
            action: 动作对象，参数需包含:
                   - template_name: 模板名称
                   - screenshot: 截图数据
        
        Returns:
            bool: 是否成功
        """
        params = action.parameters
        if 'template_name' not in params or 'screenshot' not in params:
            detailed_logger.error("保存模板动作缺少必需参数: template_name, screenshot")
            return False

        try:
            return self.scene_analyzer.save_template(
                params['screenshot'],
                params['template_name']
            )
        except Exception as e:
            detailed_logger.error(f"保存模板失败: {str(e)}")
            return False

    def register_handlers(self, decision_maker) -> None:
        """注册场景相关的动作和条件处理器
        
        Args:
            decision_maker: DecisionMaker实例
        """
        # 注册条件处理器
        decision_maker.register_condition_handler("scene_type", self.evaluate_scene_type)
        decision_maker.register_condition_handler("scene_changed", self.evaluate_scene_changed)
        decision_maker.register_condition_handler("scene_brightness", self.evaluate_scene_brightness)
        decision_maker.register_condition_handler("scene_complexity", self.evaluate_scene_complexity)

        # 注册动作处理器
        decision_maker.register_action_handler("save_template", self.handle_save_template)
