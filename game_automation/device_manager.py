import subprocess
import platform
import os
from utils.error_handler import log_exception, DeviceError

class DeviceManager:
    def __init__(self):
        self.platform = platform.system()
        self.adb_path = self._get_adb_path()

    @log_exception
    def _get_adb_path(self):
        if self.platform == "Windows":
            return os.path.join(os.environ.get("LOCALAPPDATA"), "Android", "Sdk", "platform-tools", "adb.exe")
        elif self.platform in ["Linux", "Darwin"]:
            return "adb"
        else:
            raise DeviceError(f"Unsupported platform: {self.platform}")

    @log_exception
    def get_connected_devices(self):
        result = subprocess.run([self.adb_path, "devices"], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        return [line.split('\t')[0] for line in lines if line.endswith('device')]

    @log_exception
    def select_device(self, device_id):
        if device_id in self.get_connected_devices():
            return device_id
        else:
            raise DeviceError(f"Device {device_id} not found or not connected.")

    @log_exception
    def execute_adb_command(self, device_id, command):
        full_command = [self.adb_path, "-s", device_id] + command
        result = subprocess.run(full_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise DeviceError(f"ADB command failed: {result.stderr}")
        return result.stdout.strip()

    @log_exception
    def get_device_info(self, device_id):
        model = self.execute_adb_command(device_id, ["shell", "getprop", "ro.product.model"])
        android_version = self.execute_adb_command(device_id, ["shell", "getprop", "ro.build.version.release"])
        return {
            "model": model,
            "android_version": android_version,
            "platform": self.platform
        }

    @log_exception
    def install_app(self, device_id, apk_path):
        return self.execute_adb_command(device_id, ["install", "-r", apk_path])

    @log_exception
    def uninstall_app(self, device_id, package_name):
        return self.execute_adb_command(device_id, ["uninstall", package_name])

    @log_exception
    def start_app(self, device_id, package_name, activity_name):
        return self.execute_adb_command(device_id, ["shell", "am", "start", "-n", f"{package_name}/{activity_name}"])

    @log_exception
    def stop_app(self, device_id, package_name):
        return self.execute_adb_command(device_id, ["shell", "am", "force-stop", package_name])

device_manager = DeviceManager()