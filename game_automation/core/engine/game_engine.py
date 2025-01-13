from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from ..base.base_manager import BaseManager
from ..events.event_manager import EventManager, Event, EventType
from ..recognition.async_manager import AsyncManager
from ..recognition.image_capture import ImageCapture
from ..recognition.image_processor import ImageProcessor
from ..recognition.state_analyzer import StateAnalyzer, GameState
from ..task.task_executor import TaskExecutor
from ...utils.logger import get_logger

logger = get_logger(__name__)

class GameEngine(BaseManager):
    """Game automation engine that coordinates all components"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        event_manager: EventManager,
        async_manager: AsyncManager,
        device_manager: DeviceManager,
        scene_analyzer: AdvancedSceneAnalyzer,
        context_manager: ContextManager,
        task_executor: TaskExecutor,
        error_manager: Optional[ErrorManager] = None
    ):
        """Initialize game engine
        
        Args:
            config: Configuration dictionary
            event_manager: Event manager instance
            async_manager: Async manager instance
            device_manager: Device manager instance
            scene_analyzer: Scene analyzer instance
            context_manager: Context manager instance
            task_executor: Task executor instance
            error_manager: Error manager instance (optional)
        """
        super().__init__()
        self.config = config
        self.event_manager = event_manager
        self.async_manager = async_manager
        self.device_manager = device_manager
        self.scene_analyzer = scene_analyzer
        self.context_manager = context_manager
        self.task_executor = task_executor
        self.error_manager = error_manager or ErrorManager()
        
        # Initialize components
        self.image_capture = ImageCapture(event_manager, async_manager)
        self.image_processor = ImageProcessor(event_manager, async_manager, self.image_capture)
        self.state_analyzer = StateAnalyzer(event_manager, async_manager)
        
        # Engine state
        self._initialized = False
        self._running = False
        
        # Register event handlers
        self.event_manager.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self.event_manager.subscribe(EventType.STATE_TIMEOUT, self._on_state_timeout)
        self.event_manager.subscribe(EventType.STATE_ACTION, self._on_state_action)
        
    async def initialize(self):
        """Initialize engine"""
        if not self._initialized:
            try:
                # Initialize components
                await self.image_capture.initialize()
                await self.image_processor.initialize()
                await self.state_analyzer.initialize()
                await self.task_executor.initialize()
                await self.device_manager.initialize()
                await self.scene_analyzer.initialize()
                await self.context_manager.initialize()
                await self.error_manager.initialize()
                
                # Load game states from config
                await self._load_game_states()
                
                self._initialized = True
                logger.info("Game engine initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize game engine: {e}")
                raise
                
    async def cleanup(self):
        """Clean up resources"""
        await self.stop()
        
        # Cleanup components
        await self.image_capture.cleanup()
        await self.image_processor.cleanup()
        await self.state_analyzer.cleanup()
        await self.task_executor.cleanup()
        await self.device_manager.cleanup()
        await self.scene_analyzer.cleanup()
        await self.context_manager.cleanup()
        await self.error_manager.cleanup()
        
        await super().cleanup()
        
    async def start(self):
        """Start game engine"""
        if not self._initialized:
            await self.initialize()
            
        if not self._running:
            # Start components
            await self.image_capture.start_capture()
            await self.image_processor.start_processing()
            await self.task_executor.start()
            await self.device_manager.connect()
            
            self._running = True
            logger.info("Game engine started")
            
    async def stop(self):
        """Stop game engine"""
        if self._running:
            # Stop components
            await self.image_capture.stop_capture()
            await self.image_processor.stop_processing()
            await self.task_executor.stop()
            await self.device_manager.disconnect()
            
            self._running = False
            logger.info("Game engine stopped")
            
    async def _load_game_states(self):
        """Load game states from config"""
        states_config = self.config.get('states', {})
        for name, state_config in states_config.items():
            try:
                state = GameState.from_dict({
                    'name': name,
                    **state_config
                })
                self.state_analyzer.add_state(state)
                logger.debug(f"Loaded game state: {name}")
            except Exception as e:
                logger.error(f"Failed to load game state {name}: {e}")
                
    async def _on_state_changed(self, event: Event):
        """Handle state changed event
        
        Args:
            event: Event instance
        """
        state_name = event.data['state']
        confidence = event.data['confidence']
        logger.info(f"Game state changed to {state_name} with confidence {confidence:.2f}")
        
        # Get state actions
        state = self.state_analyzer._states.get(state_name)
        if state and state.actions:
            # Execute state actions
            for action in state.actions:
                await self.event_manager.emit(Event(
                    EventType.STATE_ACTION,
                    {
                        'state': state_name,
                        'action': action
                    }
                ))
                
    async def _on_state_timeout(self, event: Event):
        """Handle state timeout event
        
        Args:
            event: Event instance
        """
        state_name = event.data['state']
        logger.warning(f"Game state {state_name} timed out")
        
    async def _on_state_action(self, event: Event):
        """Handle state action event
        
        Args:
            event: Event instance
        """
        state_name = event.data['state']
        action = event.data['action']
        logger.info(f"Executing action {action} for state {state_name}")
        
        # Add action to task queue
        await self.task_executor.add_task({
            'type': 'state_action',
            'state': state_name,
            'action': action
        })
