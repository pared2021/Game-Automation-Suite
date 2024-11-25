import os
import asyncio
import subprocess
import time
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime
import platform

from utils.error_handler import log_exception, DeviceConnectionError
from utils.logger import detailed_logger
from .ui_automator import UIAutomator

class MockUIAutomator:
    """Mock UI Automator for simulation mode"""
    def __init__(self, device_id: str = "mock_device"):
        self.device_id = device_id
        self.connected = True
        detailed_logger.info(f"Initialized Mock UI Automator for {device_id}")

    def get_device_info(self) -> Dict:
        """Get mock device info"""
        return {
            'device_id': self.device_id,
            'status': 'connected',
            'mode': 'simulation'
        }

    def get_ui_state(self) -> Dict:
        """Get mock UI state"""
        return {
            'screen': 'mock_screen',
            'elements': [],
            'timestamp': datetime.now().isoformat()
        }

class DeviceManager:
    """Device manager responsible for device connection, monitoring and reconnection"""
    
    def __init__(self, simulation_mode: bool = False):
        """Initialize device manager
        
        Args:
            simulation_mode: Whether to run in simulation mode
        """
        self.device_id: Optional[str] = None
        self.ui_automator: Optional[UIAutomator] = None
        self.connected: bool = False
        self.monitor_task = None
        self.simulation_mode = simulation_mode
        
        # Reconnection control
        self._max_reconnect_attempts = 3
        self._reconnect_delay = 2
        self._last_reconnect_time = 0
        self._min_reconnect_interval = 5
        
        # Concurrency control
        self._operation_lock = asyncio.Lock()
        self._resource_lock = asyncio.Lock()
        self._operation_queue = deque()
        self._active_operations: Dict[str, datetime] = {}
        self._max_concurrent_operations = 3
        self._operation_timeout = 30
        
        # Resource management
        self._resources: Dict[str, any] = {}
        self._cleanup_scheduled = False

        if not simulation_mode:
            try:
                self._adb_path = self._get_adb_path()
            except DeviceConnectionError as e:
                detailed_logger.warning(f"ADB not found: {str(e)}. Falling back to simulation mode.")
                self.simulation_mode = True

    @log_exception
    def _get_adb_path(self) -> str:
        """Get ADB executable path
        
        Returns:
            str: ADB executable path
        
        Raises:
            DeviceConnectionError: If ADB not found
        """
        # Check environment variable
        android_home = os.environ.get('ANDROID_HOME')
        if android_home:
            if platform.system() == 'Windows':
                adb_path = os.path.join(android_home, 'platform-tools', 'adb.exe')
            else:
                adb_path = os.path.join(android_home, 'platform-tools', 'adb')
            if os.path.exists(adb_path):
                return adb_path

        # Check common paths
        common_paths = []
        if platform.system() == 'Windows':
            common_paths = [
                r'C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe',
                r'C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe'
            ]
        else:
            common_paths = [
                '/usr/local/bin/adb',
                '/usr/bin/adb',
                os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
                os.path.expanduser('~/Android/Sdk/platform-tools/adb')
            ]
        
        for path in common_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path

        raise DeviceConnectionError("ADB executable not found")

    @log_exception
    async def connect(self, device_id: Optional[str] = None) -> None:
        """Connect to device
        
        Args:
            device_id: Optional device ID. If not provided, connect to first available device
        """
        if self.simulation_mode:
            self.device_id = device_id or "mock_device"
            self.ui_automator = MockUIAutomator(self.device_id)
            self.connected = True
            detailed_logger.info(f"Connected to mock device: {self.device_id}")
            
            # Start device monitoring
            if self.monitor_task is None:
                self.monitor_task = asyncio.create_task(self._monitor_device())
            return

        try:
            if device_id is None:
                # Get connected devices list
                result = subprocess.run(
                    [self._adb_path, 'devices'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse devices list
                lines = result.stdout.strip().split('\n')[1:]
                devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
                
                if not devices:
                    raise DeviceConnectionError("No connected devices found")
                
                device_id = devices[0]
            
            # Clean up existing resources
            if self.connected:
                await self._cleanup_resources()
            
            # Try to connect device
            self.ui_automator = UIAutomator(device_id)
            self.device_id = device_id
            self.connected = True
            
            detailed_logger.info(f"Connected to device: {device_id}")
            
            # Start device monitoring
            if self.monitor_task is None:
                self.monitor_task = asyncio.create_task(self._monitor_device())
        
        except Exception as e:
            self.connected = False
            raise DeviceConnectionError(f"Failed to connect device: {str(e)}")

    @log_exception
    async def disconnect(self) -> None:
        """Disconnect device"""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
        
        await self._cleanup_resources()
        
        self.connected = False
        self.device_id = None
        self.ui_automator = None
        detailed_logger.info("Device disconnected")

    @log_exception
    async def check_connection(self) -> bool:
        """Check device connection status
        
        Returns:
            bool: Whether device is properly connected
        """
        if not self.device_id or not self.ui_automator:
            return False
            
        try:
            if self.simulation_mode:
                return True

            # Try to get device info to verify connection
            device_info = self.ui_automator.get_device_info()
            return bool(device_info)
        except Exception as e:
            detailed_logger.warning(f"Device connection check failed: {str(e)}")
            return False

    async def _monitor_device(self) -> None:
        """Monitor device status and handle disconnection/reconnection"""
        while True:
            try:
                if not await self.check_connection():
                    current_time = time.time()
                    if current_time - self._last_reconnect_time < self._min_reconnect_interval:
                        await asyncio.sleep(1)
                        continue
                        
                    self._last_reconnect_time = current_time
                    await self._handle_disconnection()
                
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                detailed_logger.error(f"Device monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _handle_disconnection(self) -> None:
        """Handle device disconnection"""
        detailed_logger.warning("Device disconnection detected")
        self.connected = False
        await self._cleanup_resources()
        
        if self.simulation_mode:
            try:
                await self.connect(self.device_id)
                return
            except Exception as e:
                detailed_logger.error(f"Mock device reconnection failed: {str(e)}")
                return

        for attempt in range(self._max_reconnect_attempts):
            try:
                detailed_logger.info(f"Attempting to reconnect device (attempt {attempt + 1}/{self._max_reconnect_attempts})")
                await self.connect(self.device_id)
                if self.connected:
                    return
            except DeviceConnectionError as e:
                detailed_logger.error(f"Reconnection failed: {str(e)}")
                if attempt < self._max_reconnect_attempts - 1:
                    await asyncio.sleep(self._reconnect_delay)
        
        detailed_logger.error("Device reconnection failed, stopping monitoring")
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

    async def queue_operation(self, operation_func, *args, **kwargs) -> any:
        """Queue operation for execution
        
        Args:
            operation_func: Operation function to execute
            *args: Position arguments
            **kwargs: Keyword arguments
            
        Returns:
            any: Operation result
        """
        # Create operation task
        async def execute_operation():
            while True:
                if await self._acquire_operation_slot():
                    op_id = f"op_{len(self._active_operations)}"
                    try:
                        result = await operation_func(*args, **kwargs)
                        return result
                    finally:
                        await self._release_operation_slot(op_id)
                await asyncio.sleep(0.1)
        
        # Queue operation
        task = asyncio.create_task(execute_operation())
        self._operation_queue.append(task)
        
        try:
            return await task
        finally:
            self._operation_queue.remove(task)

    @property
    def is_connected(self) -> bool:
        """Get device connection status
        
        Returns:
            bool: Whether device is connected
        """
        return self.connected

    def get_ui_automator(self) -> Optional[UIAutomator]:
        """Get UI automator
        
        Returns:
            Optional[UIAutomator]: UI automator instance or None if not connected
        """
        return self.ui_automator if self.connected else None

    @property
    def active_operations_count(self) -> int:
        """Get current active operations count
        
        Returns:
            int: Active operations count
        """
        return len(self._active_operations)

    @property
    def queued_operations_count(self) -> int:
        """Get queued operations count
        
        Returns:
            int: Queued operations count
        """
        return len(self._operation_queue)
