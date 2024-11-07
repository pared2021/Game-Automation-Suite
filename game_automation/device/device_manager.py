import subprocess
import time
import asyncio
from utils.logger import setup_logger
from utils.error_handler import log_exception, DeviceError
from utils.config_manager import config_manager

class DeviceManager:
    def __init__(self):
        self.logger = setup_logger('device_manager')
        self.adb_path = self._get_adb_path()
        self.connected = False
        self.device_id = None
        self.max_retries = config_manager.get('performance.max_retries', 3)
        self.retry_delay = config_manager.get('performance.error_retry_delay', 5)
        self.fps_history = []
        self.response_time_history = []

    @log_exception
    def _get_adb_path(self):
        adb_path = config_manager.get('device.adb_path', 'auto')
        if adb_path == 'auto':
            return 'adb'  # Assume ADB is in system PATH
        return adb_path

    @log_exception
    async def connect(self, device_id=None):
        if device_id is None:
            devices = await self.get_connected_devices()
            if not devices:
                raise DeviceError("No devices found")
            device_id = devices[0]
        
        self.device_id = device_id
        try:
            await self.execute_adb_command(['connect', device_id])
            self.connected = True
            self.logger.info(f"Connected to device: {device_id}")
        except Exception as e:
            self.logger.error(f"Failed to connect to device {device_id}: {str(e)}")
            self.connected = False
            raise DeviceError(f"Failed to connect to device {device_id}")

    @log_exception
    async def check_connection(self):
        if not self.connected:
            return False
        try:
            await self.execute_adb_command(['devices'])
            return True
        except Exception:
            self.connected = False
            return False

    @log_exception
    async def auto_reconnect(self):
        for attempt in range(self.max_retries):
            try:
                await self.connect(self.device_id)
                return
            except DeviceError:
                self.logger.warning(f"Reconnection attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
        raise DeviceError(f"Failed to reconnect after {self.max_retries} attempts")

    @log_exception
    async def get_connected_devices(self):
        result = await self.execute_adb_command(['devices'])
        lines = result.strip().split('\n')[1:]
        return [line.split('\t')[0] for line in lines if line.endswith('device')]

    @log_exception
    async def execute_adb_command(self, command):
        full_command = [self.adb_path, '-s', self.device_id] + command if self.device_id else [self.adb_path] + command
        start_time = time.time()
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        end_time = time.time()
        
        if process.returncode != 0:
            error_message = stderr.decode().strip()
            self.logger.error(f"ADB command failed: {error_message}")
            raise DeviceError(f"ADB command failed: {error_message}")
        
        response_time = end_time - start_time
        self.response_time_history.append(response_time)
        self.logger.debug(f"ADB command executed: {' '.join(full_command)}")
        self.logger.debug(f"Response time: {response_time:.4f} seconds")
        
        return stdout.decode().strip()

    @log_exception
    async def get_device_info(self):
        model = await self.execute_adb_command(['shell', 'getprop', 'ro.product.model'])
        android_version = await self.execute_adb_command(['shell', 'getprop', 'ro.build.version.release'])
        return {
            "model": model,
            "android_version": android_version
        }

    @log_exception
    async def monitor_performance(self):
        fps = await self.get_fps()
        self.fps_history.append(fps)
        avg_response_time = sum(self.response_time_history) / len(self.response_time_history) if self.response_time_history else 0
        self.logger.info(f"Current FPS: {fps}, Average Response Time: {avg_response_time:.4f} seconds")

    @log_exception
    async def get_fps(self):
        result = await self.execute_adb_command(['shell', 'dumpsys', 'gfxinfo'])
        fps_line = [line for line in result.split('\n') if 'fps' in line]
        if fps_line:
            return float(fps_line[0].split(':')[1].strip().split(' ')[0])
        return 0

    @log_exception
    async def capture_screen(self):
        await self.execute_adb_command(['shell', 'screencap', '-p', '/sdcard/screen.png'])
        await self.execute_adb_command(['pull', '/sdcard/screen.png', 'screen.png'])
        self.logger.debug("Screen captured and saved as screen.png")

    @log_exception
    async def tap(self, x, y):
        await self.execute_adb_command(['shell', 'input', 'tap', str(x), str(y)])
        self.logger.debug(f"Tap performed at ({x}, {y})")

    @log_exception
    async def swipe(self, start_x, start_y, end_x, end_y, duration):
        await self.execute_adb_command(['shell', 'input', 'swipe', str(start_x), str(start_y), str(end_x), str(end_y), str(duration)])
        self.logger.debug(f"Swipe performed from ({start_x}, {start_y}) to ({end_x}, {end_y}) with duration {duration}ms")

device_manager = DeviceManager()