from utils.logger import setup_logger

class SimpleAIDecisionMaker:
    def __init__(self, game_engine):
        self.logger = setup_logger('simple_ai_decision_maker')
        self.game_engine = game_engine
        self.actions = ["use_health_potion", "attack", "defend", "explore"]

    def make_decision(self, game_state):
        """根据游戏状态做出简单决策"""
        if game_state['health'] < 20:
            action = "use_health_potion"
        elif game_state['enemy_nearby']:
            action = "attack"
        else:
            action = "explore"
        
        self.logger.info(f"Decision made: {action}")
        return action

# 实例化简单 AI 决策制定器
simple_ai_decision_maker = SimpleAIDecisionMaker(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用
