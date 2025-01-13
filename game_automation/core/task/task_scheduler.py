"""Task scheduler for game automation."""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
from croniter import croniter

from .task_types import Task, TaskStatus
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ScheduledTask:
    """Scheduled task"""
    
    def __init__(
        self,
        task: Task,
        schedule: str,
        max_retries: int = 3
    ):
        """Initialize scheduled task
        
        Args:
            task: Task to schedule
            schedule: Cron-style schedule
            max_retries: Maximum retry attempts
        """
        self.task = task
        self.schedule = schedule
        self.max_retries = max_retries
        self.retries = 0
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.running = False
        self._cron = croniter(schedule)
        self._update_next_run()
        
    def _update_next_run(self):
        """Update next run time"""
        self.next_run = self._cron.get_next(datetime)
        
    def should_run(self) -> bool:
        """Check if task should run
        
        Returns:
            bool: Whether task should run
        """
        if self.running:
            return False
            
        now = datetime.now()
        return self.next_run and now >= self.next_run

class TaskScheduler:
    """Task scheduler"""
    
    def __init__(self):
        """Initialize task scheduler"""
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    def schedule_task(
        self,
        task: Task,
        schedule: str,
        max_retries: int = 3
    ) -> None:
        """Schedule task
        
        Args:
            task: Task to schedule
            schedule: Cron-style schedule
            max_retries: Maximum retry attempts
        """
        scheduled = ScheduledTask(task, schedule, max_retries)
        self.scheduled_tasks[task.task_id] = scheduled
        logger.info(
            f"Scheduled task: {task.name} ({task.task_id}) "
            f"with schedule: {schedule}"
        )
        
    def cancel_schedule(self, task_id: str) -> None:
        """Cancel task schedule
        
        Args:
            task_id: Task ID
        """
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            logger.info(f"Cancelled schedule for task: {task_id}")
            
    async def start(self):
        """Start scheduler"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Task scheduler started")
        
    async def stop(self):
        """Stop scheduler"""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            
        logger.info("Task scheduler stopped")
        
    async def _run(self):
        """Run scheduler"""
        while self._running:
            try:
                await self._check_schedules()
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                
            # Check every minute
            await asyncio.sleep(60)
            
    async def _check_schedules(self):
        """Check scheduled tasks"""
        now = datetime.now()
        
        for task_id, scheduled in list(self.scheduled_tasks.items()):
            if scheduled.should_run():
                # Execute task
                scheduled.running = True
                try:
                    success = await scheduled.task.execute()
                    
                    if success:
                        # Reset retries on success
                        scheduled.retries = 0
                        scheduled.last_run = now
                        scheduled._update_next_run()
                        
                    else:
                        # Handle failure
                        scheduled.retries += 1
                        if scheduled.retries >= scheduled.max_retries:
                            logger.error(
                                f"Task failed after {scheduled.retries} retries: "
                                f"{scheduled.task.name}"
                            )
                            self.cancel_schedule(task_id)
                        else:
                            # Retry after delay
                            delay = min(300, 60 * (2 ** scheduled.retries))
                            scheduled.next_run = now + timedelta(seconds=delay)
                            
                except Exception as e:
                    logger.error(
                        f"Task execution failed: {scheduled.task.name} - {str(e)}"
                    )
                    scheduled.retries += 1
                    
                finally:
                    scheduled.running = False
                    
    def get_next_run(self, task_id: str) -> Optional[datetime]:
        """Get next run time for task
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[datetime]: Next run time or None if not scheduled
        """
        scheduled = self.scheduled_tasks.get(task_id)
        return scheduled.next_run if scheduled else None
        
    def get_schedule(self, task_id: str) -> Optional[str]:
        """Get task schedule
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[str]: Task schedule or None if not scheduled
        """
        scheduled = self.scheduled_tasks.get(task_id)
        return scheduled.schedule if scheduled else None
