import time
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self, interval=1, history_size=100):
        self.interval = interval
        self.running = False
        self.history_size = history_size
        self.stats = {
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'disk_usage': deque(maxlen=history_size),
        }

    def start(self):
        self.running = True
        threading.Thread(target=self._monitor, daemon=True).start()

    def stop(self):
        self.running = False

    def _monitor(self):
        while self.running:
            self.stats['cpu_percent'].append(psutil.cpu_percent())
            self.stats['memory_percent'].append(psutil.virtual_memory().percent)
            self.stats['disk_usage'].append(psutil.disk_usage('/').percent)
            time.sleep(self.interval)

    def get_stats(self):
        return {
            'cpu_percent': self._calculate_average('cpu_percent'),
            'memory_percent': self._calculate_average('memory_percent'),
            'disk_usage': self._calculate_average('disk_usage'),
        }

    def _calculate_average(self, stat_name):
        values = self.stats[stat_name]
        return sum(values) / len(values) if values else 0

    def get_detailed_stats(self):
        return {key: list(value) for key, value in self.stats.items()}