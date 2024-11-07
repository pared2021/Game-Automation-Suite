import asyncio
import subprocess
from utils.config_manager import config_manager
from utils.logger import setup_logger
from utils.error_handler import log_exception, DeviceError

class EmulatorManager:
    def __init__(self):
        self.logger = setup_logger('emulator_manager')
        self.config = config_manager.get('emulator', {})
        self.connected_emulator = None

    @log_exception
    async def connect(self):
        if self.config.get('auto_detect', True):
            await self.auto_detect_and_connect()
        else:
            await self.connect_default()

    async def auto_detect_and_connect(self):
        emulator_types = sorted(self.config.get('types', []), key=lambda x: x['priority'])
        for emulator in emulator_types:
            try:
                self.logger.info(f"Attempting to connect to {emulator['name']} emulator")
                connection_method = getattr(self, emulator['connection_method'], None)
                if connection_method:
                    await connection_method(emulator)
                    self.connected_emulator = emulator
                    self.logger.info(f"Successfully connected to {emulator['name']} emulator")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to connect to {emulator['name']} emulator: {str(e)}")

        raise DeviceError("Failed to connect to any emulator")

    async def connect_default(self):
        default_type = self.config.get('default_type', 'adb')
        emulator = next((e for e in self.config.get('types', []) if e['name'] == default_type), None)
        if not emulator:
            raise DeviceError(f"Default emulator type '{default_type}' not found in configuration")

        connection_method = getattr(self, emulator['connection_method'], None)
        if connection_method:
            await connection_method(emulator)
            self.connected_emulator = emulator
        else:
            raise DeviceError(f"Connection method '{emulator['connection_method']}' not implemented")

    async def adb_connect(self, emulator):
        devices = await self.get_adb_devices()
        if not devices:
            raise DeviceError("No ADB devices found")
        # Connect to the first available device
        await self.execute_adb_command(['connect', devices[0]])

    async def nox_connect(self, emulator):
        # Implement Nox-specific connection logic
        nox_path = emulator.get('path', 'C:/Program Files/Nox/bin/Nox.exe')
        # Start Nox emulator if not running
        subprocess.Popen([nox_path])
        # Wait for Nox to start and connect via ADB
        await asyncio.sleep(10)
        await self.adb_connect(emulator)

    async def bluestacks_connect(self, emulator):
        # Implement BlueStacks-specific connection logic
        bluestacks_path = emulator.get('path', 'C:/Program Files/BlueStacks/bluestacks.exe')
        # Start BlueStacks emulator if not running
        subprocess.Popen([bluestacks_path])
        # Wait for BlueStacks to start and connect via ADB
        await asyncio.sleep(10)
        await self.adb_connect(emulator)

    async def get_adb_devices(self):
        result = await self.execute_adb_command(['devices'])
        lines = result.strip().split('\n')[1:]
        return [line.split('\t')[0] for line in lines if line.endswith('device')]

    async def execute_adb_command(self, command):
        process = await asyncio.create_subprocess_exec(
            'adb', *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise DeviceError(f"ADB command failed: {stderr.decode().strip()}")
        return stdout.decode().strip()

    async def disconnect(self):
        if self.connected_emulator:
            self.logger.info(f"Disconnecting from {self.connected_emulator['name']} emulator")
            # Implement disconnection logic here
            self.connected_emulator = None

    async def restart_emulator(self):
        if self.connected_emulator:
            await self.disconnect()
            await self.connect()
        else:
            raise DeviceError("No emulator is currently connected")

    async def execute_custom_script(self, script_name):
        script = self.config.get('custom_scripts', {}).get(script_name)
        if script:
            # Execute the custom script
            # This is a placeholder and should be implemented based on your needs
            self.logger.info(f"Executing custom script: {script_name}")
            # Example: subprocess.run(script, shell=True)
        else:
            raise DeviceError(f"Custom script '{script_name}' not found in configuration")

emulator_manager = EmulatorManager()