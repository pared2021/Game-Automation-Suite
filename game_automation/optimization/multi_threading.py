import threading
import queue
from utils.logger import detailed_logger

class ThreadPool:
    def __init__(self, num_threads):
        self.logger = detailed_logger
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.threads = []
        self._create_threads()

    def _create_threads(self):
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while True:
            task, args, kwargs = self.task_queue.get()
            try:
                task(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in thread: {str(e)}")
            finally:
                self.task_queue.task_done()

    def add_task(self, task, *args, **kwargs):
        self.task_queue.put((task, args, kwargs))

    def wait_completion(self):
        self.task_queue.join()

thread_pool = ThreadPool(4)  # 创建一个有4个线程的线程池
