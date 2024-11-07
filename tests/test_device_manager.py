import unittest
from unittest.mock import patch, MagicMock
from game_automation.device.device_manager import DeviceManager

class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        self.device_manager = DeviceManager()

    @patch('subprocess.run')
    def test_get_connected_devices(self, mock_run):
        mock_run.return_value.stdout = "List of devices attached\ndevice1\tdevice\ndevice2\tdevice\n"
        devices = self.device_manager.get_connected_devices()
        self.assertEqual(devices, ["device1", "device2"])

    def test_select_device(self):
        self.device_manager.get_connected_devices = MagicMock(return_value=["device1", "device2"])
        selected_device = self.device_manager.select_device("device1")
        self.assertEqual(selected_device, "device1")

    def test_select_device_not_found(self):
        self.device_manager.get_connected_devices = MagicMock(return_value=["device1", "device2"])
        with self.assertRaises(ValueError):
            self.device_manager.select_device("device3")

    @patch('subprocess.run')
    def test_execute_adb_command(self, mock_run):
        mock_run.return_value.stdout = "command output"
        output = self.device_manager.execute_adb_command("device1", ["shell", "ls"])
        self.assertEqual(output, "command output")

    @patch('subprocess.run')
    def test_get_device_info(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="Model X"),
            MagicMock(stdout="10")
        ]
        info = self.device_manager.get_device_info("device1")
        self.assertEqual(info, {"model": "Model X", "android_version": "10"})

if __name__ == '__main__':
    unittest.main()