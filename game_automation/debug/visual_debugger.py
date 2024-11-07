import pygame
import asyncio
from utils.logger import setup_logger
from utils.config_manager import config_manager

class VisualDebugger:
    def __init__(self, game_engine):
        self.logger = setup_logger('visual_debugger')
        self.config = config_manager.get('debug', {})
        self.game_engine = game_engine
        self.screen = None
        self.font = None
        self.running = False

    async def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Game Automation Debug Tool")
        self.font = pygame.font.Font(None, 24)
        self.logger.info("Visual debugger initialized")

    async def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill((0, 0, 0))
            await self.draw_game_state()
            await self.draw_ai_decision()
            pygame.display.flip()
            await asyncio.sleep(0.1)

        pygame.quit()

    async def draw_game_state(self):
        game_state = await self.game_engine.get_game_state()
        y = 10
        for key, value in game_state.items():
            text = self.font.render(f"{key}: {value}", True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 30

    async def draw_ai_decision(self):
        decision = await self.game_engine.ai_decision_maker.get_last_decision()
        text = self.font.render(f"AI Decision: {decision}", True, (255, 255, 0))
        self.screen.blit(text, (10, 500))

    def stop(self):
        self.running = False

visual_debugger = VisualDebugger(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用