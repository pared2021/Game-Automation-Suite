import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger
from ..device.device_manager import DeviceManager
from ..ai.ai_decision_maker import AIDecisionMaker
from .task_manager import TaskManager, Task, TaskStatus, TaskPriority

class GameEngine:
    """Game automation engine that coordinates all components"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the game engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device_manager = DeviceManager()
        self.task_manager = TaskManager()
        self.ai_decision_maker = AIDecisionMaker(self)
        
        self.running = False
        self.last_state_update = None
        self.current_game_state = {}
        
        self.state_update_callbacks = []

    @log_exception
    async def initialize(self) -> None:
        """Initialize the engine"""
        detailed_logger.info("Initializing game engine...")
        
        # Initialize device connection
        await self.device_manager.connect()
        
        # Initialize AI decision maker
        await self.ai_decision_maker.initialize()
        
        # Load task state
        try:
            self.task_manager.load_state("data/task_state/latest.json")
        except Exception as e:
            detailed_logger.warning(f"Failed to load task state: {str(e)}")
        
        detailed_logger.info("Game engine initialization complete")

    @log_exception
    async def start(self) -> None:
        """Start the engine"""
        if self.running:
            return
            
        self.running = True
        detailed_logger.info("Game engine started")
        
        # Start main loop
        asyncio.create_task(self._main_loop())

    @log_exception
    async def stop(self) -> None:
        """Stop the engine"""
        if not self.running:
            return
            
        self.running = False
        detailed_logger.info("Game engine stopped")
        
        # Save current task state
        self.task_manager.save_state("data/task_state/latest.json")

    async def _main_loop(self) -> None:
        """Main engine loop"""
        while self.running:
            try:
                # Update game state
                await self.update_game_state()
                
                # Get AI decision
                decision = await self.ai_decision_maker.make_decision(self.current_game_state)
                
                # Create task from decision
                if decision:
                    task = self._create_task_from_decision(decision)
                    self.task_manager.add_task(task)
                
                # Execute next task
                self.task_manager.execute_next_task()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                detailed_logger.error(f"Main loop error: {str(e)}")
                await asyncio.sleep(1)

    @log_exception
    async def update_game_state(self) -> Dict[str, Any]:
        """Update the game state
        
        Returns:
            Dict[str, Any]: Current game state
        """
        try:
            # Get UI state
            ui_state = await self.device_manager.queue_operation(
                self.device_manager.get_ui_automator().get_ui_state
            )
            
            # Update state
            self.current_game_state.update({
                'ui_state': ui_state,
                'tasks': {
                    'total': len(self.task_manager.tasks),
                    'pending': len(self.task_manager.task_queue),
                    'running': len(self.task_manager.running_tasks),
                    'completed': len(self.task_manager.completed_tasks),
                    'failed': len(self.task_manager.failed_tasks)
                },
                'timestamp': datetime.now().isoformat()
            })
            
            # Trigger callbacks
            self._trigger_state_update()
            
            self.last_state_update = datetime.now()
            return self.current_game_state
            
        except Exception as e:
            detailed_logger.error(f"Failed to update game state: {str(e)}")
            return self.current_game_state

    def _create_task_from_decision(self, decision: str) -> Task:
        """Create a task from AI decision
        
        Args:
            decision: AI decision result
            
        Returns:
            Task: Created task instance
        """
        task_id = f"task_{len(self.task_manager.tasks) + 1}"
        return Task(
            task_id=task_id,
            name=f"Execute {decision}",
            priority=TaskPriority.NORMAL
        )

    def register_state_update_callback(self, callback) -> None:
        """Register state update callback
        
        Args:
            callback: Callback function
        """
        self.state_update_callbacks.append(callback)

    def _trigger_state_update(self) -> None:
        """Trigger all state update callbacks"""
        for callback in self.state_update_callbacks:
            try:
                callback(self.current_game_state)
            except Exception as e:
                detailed_logger.error(f"State update callback failed: {str(e)}")

    @property
    def is_running(self) -> bool:
        """Get engine running status
        
        Returns:
            bool: Whether engine is running
        """
        return self.running

    def get_current_state(self) -> Dict[str, Any]:
        """Get current game state
        
        Returns:
            Dict[str, Any]: Copy of current game state
        """
        return self.current_game_state.copy()

    def get_task_manager(self) -> TaskManager:
        """Get task manager
        
        Returns:
            TaskManager: Task manager instance
        """
        return self.task_manager

    def get_device_manager(self) -> DeviceManager:
        """Get device manager
        
        Returns:
            DeviceManager: Device manager instance
        """
        return self.device_manager

    def get_ai_decision_maker(self) -> AIDecisionMaker:
        """Get AI decision maker
        
        Returns:
            AIDecisionMaker: AI decision maker instance
        """
        return self.ai_decision_maker
