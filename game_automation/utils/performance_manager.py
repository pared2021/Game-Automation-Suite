import json
import os
import time
import psutil
import threading
from collections import deque
import matplotlib.pyplot as plt
from utils.error_handler import log_exception
from utils.logger import detailed_logger
from utils.config_manager import config_manager

class PerformanceManager:
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
        self.logger = detailed_logger
        self.config = config_manager.get('performance', {})
        self.image_processing_frequency = self.config.get('image_processing_frequency', 1)
        self.image_processing_quality = self.config.get('image_processing_quality', 'medium')
        self.ai_update_frequency = self.config.get('ai_update_frequency', 1)
        self.background_tasks_limit = self.config.get('background_tasks_limit', 5)

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

    def adjust_settings(self, run_mode):
        if run_mode == 'power_saving':
            self.image_processing_frequency = 2
            self.image_processing_quality = 'low'
            self.ai_update_frequency = 2
            self.background_tasks_limit = 3
        elif run_mode == 'balanced':
            self.image_processing_frequency = 1
            self.image_processing_quality = 'medium'
            self.ai_update_frequency = 1
            self.background_tasks_limit = 5
        elif run_mode == 'high_performance':
            self.image_processing_frequency = 1
            self.image_processing_quality = 'high'
            self.ai_update_frequency = 1
            self.background_tasks_limit = 10
        self.logger.info(f"Performance settings adjusted for {run_mode} mode")

    def get_image_processing_params(self):
        quality_params = {
            'low': {'interpolation': cv2.INTER_NEAREST, 'scale': 0.5},
            'medium': {'interpolation': cv2.INTER_LINEAR, 'scale': 0.75},
            'high': {'interpolation': cv2.INTER_CUBIC, 'scale': 1.0}
        }
        return quality_params[self.image_processing_quality]

    def should_process_image(self, frame_count):
        return frame_count % self.image_processing_frequency == 0

    def should_update_ai(self, update_count):
        return update_count % self.ai_update_frequency == 0

    def get_background_tasks_limit(self):
        return self.background_tasks_limit

    async def optimize_image(self, image):
        params = self.get_image_processing_params()
        if params['scale'] != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * params['scale']), int(height * params['scale']))
            image = cv2.resize(image, new_size, interpolation=params['interpolation'])
        return image

    async def reduce_memory_usage(self):
        import gc
        gc.collect()
        self.logger.info("Performed garbage collection to reduce memory usage")

    async def optimize_database_queries(self, query):
        if 'SELECT *' in query:
            self.logger.warning("Consider specifying columns instead of using SELECT *")
        if 'ORDER BY' in query and 'LIMIT' not in query:
            self.logger.warning("Consider adding LIMIT to ORDER BY queries")
        return query

    def get_rendering_settings(self):
        if self.image_processing_quality == 'low':
            return {'antialiasing': False, 'shadows': False, 'texture_quality': 'low'}
        elif self.image_processing_quality == 'medium':
            return {'antialiasing': True, 'shadows': True, 'texture_quality': 'medium'}
        else:
            return {'antialiasing': True, 'shadows': True, 'texture_quality': 'high'}

    async def optimize_network_requests(self, request_data):
        optimized_data = request_data.copy()
        if 'cache_control' not in optimized_data.get('headers', {}):
            optimized_data.setdefault('headers', {})['cache_control'] = 'max-age=3600'
        return optimized_data

    def get_multithreading_settings(self):
        if self.image_processing_quality == 'low':
            return {'max_threads': 2, 'thread_priority': 'low'}
        elif self.image_processing_quality == 'medium':
            return {'max_threads': 4, 'thread_priority': 'normal'}
        else:
            return {'max_threads': 8, 'thread_priority': 'high'}

performance_manager = PerformanceManager()
