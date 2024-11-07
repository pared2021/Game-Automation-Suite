import time
import psutil
import threading
from collections import deque
import matplotlib.pyplot as plt

class PerformanceMonitor:
    def __init__(self, interval=1, history_size=100):
        self.interval = interval
        self.running = False
        self.history_size = history_size
        self.stats = {
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'disk_usage': deque(maxlen=history_size),
            'network_sent': deque(maxlen=history_size),
            'network_recv': deque(maxlen=history_size),
            'fps': deque(maxlen=history_size),
            'response_time': deque(maxlen=history_size)
        }
        self.last_network_io = psutil.net_io_counters()
        self.last_time = time.time()

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

            network_io = psutil.net_io_counters()
            current_time = time.time()
            time_diff = current_time - self.last_time
            self.stats['network_sent'].append((network_io.bytes_sent - self.last_network_io.bytes_sent) / time_diff)
            self.stats['network_recv'].append((network_io.bytes_recv - self.last_network_io.bytes_recv) / time_diff)
            self.last_network_io = network_io
            self.last_time = current_time

            time.sleep(self.interval)

    def get_stats(self):
        return {
            'cpu_percent': self._calculate_average('cpu_percent'),
            'memory_percent': self._calculate_average('memory_percent'),
            'disk_usage': self._calculate_average('disk_usage'),
            'network_sent': self._calculate_average('network_sent'),
            'network_recv': self._calculate_average('network_recv'),
            'fps': self._calculate_average('fps'),
            'response_time': self._calculate_average('response_time')
        }

    def _calculate_average(self, stat_name):
        values = self.stats[stat_name]
        return sum(values) / len(values) if values else 0

    def get_detailed_stats(self):
        return {key: list(value) for key, value in self.stats.items()}

    def record_fps(self, fps):
        self.stats['fps'].append(fps)

    def record_response_time(self, response_time):
        self.stats['response_time'].append(response_time)

    def plot_performance(self, filename='performance_plot.png'):
        fig, axs = plt.subplots(4, 1, figsize=(12, 20))
        
        axs[0].plot(self.stats['cpu_percent'])
        axs[0].set_title('CPU Usage (%)')
        
        axs[1].plot(self.stats['memory_percent'])
        axs[1].set_title('Memory Usage (%)')
        
        axs[2].plot(self.stats['fps'])
        axs[2].set_title('FPS')
        
        axs[3].plot(self.stats['response_time'])
        axs[3].set_title('Response Time (ms)')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

performance_monitor = PerformanceMonitor()