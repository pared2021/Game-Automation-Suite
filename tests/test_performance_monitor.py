import unittest
from unittest.mock import patch, Mock
from utils.performance_monitor import PerformanceMonitor

class TestPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = PerformanceMonitor(interval=0.1)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_monitor(self, mock_net_io, mock_disk, mock_memory, mock_cpu):
        mock_cpu.return_value = 50
        mock_memory.return_value.percent = 60
        mock_disk.return_value.percent = 70
        mock_net_io.side_effect = [
            Mock(bytes_sent=1000, bytes_recv=2000),
            Mock(bytes_sent=1500, bytes_recv=3000)
        ]

        self.monitor.start()
        import time
        time.sleep(0.3)  # Allow for a few monitoring cycles
        self.monitor.stop()

        stats = self.monitor.get_stats()
        self.assertIn('cpu_percent', stats)
        self.assertIn('memory_percent', stats)
        self.assertIn('disk_usage', stats)
        self.assertIn('network_io', stats)

    def test_record_response_time(self):
        self.monitor.record_response_time(0.5)
        self.monitor.record_response_time(0.7)
        stats = self.monitor.get_stats()
        self.assertIn('response_time', stats)
        self.assertEqual(stats['response_time']['avg'], 0.6)

    @patch('psutil.cpu_percent')
    def test_check_alerts(self, mock_cpu):
        mock_cpu.return_value = 95
        self.monitor._check_alerts()
        alerts = self.monitor.get_alerts()
        self.assertIn('High CPU usage: 95%', alerts)

    def test_auto_adjust(self):
        original_interval = self.monitor.interval
        self.monitor.stats['cpu_percent'] = [95] * 11
        self.monitor._auto_adjust()
        self.assertGreater(self.monitor.interval, original_interval)

    @patch('psutil.cpu_percent', side_effect=psutil.NoSuchProcess(1))
    def test_monitor_error_handling(self, mock_cpu):
        self.monitor._monitor()
        # Assert that the method doesn't raise an exception

if __name__ == '__main__':
    unittest.main()