import threading
import queue
from utils.logger import detailed_logger

class ThreadPool:
    def __init__(self, num_threads):
        self.logger = detailed_logger
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.threads = []
        self.lock = threading.Lock()  # 添加锁机制
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
            if task is None:  # 结束信号
                break
            try:
                with self.lock:  # 确保任务执行时的并发控制
                    task(*args, **kwargs)
            except (ValueError, TypeError) as e:
                self.logger.error(f"Specific error in thread: {str(e)}")
                # 可以在这里添加重试机制
            except Exception as e:
                self.logger.error(f"Unexpected error in thread: {str(e)}")
            finally:
                self.task_queue.task_done()

    def add_task(self, task, *args, **kwargs):
        self.task_queue.put((task, args, kwargs))

    def wait_completion(self):
        self.task_queue.join()

    def adjust_pool_size(self, new_size):
        current_size = len(self.threads)
        if new_size > current_size:
            for _ in range(new_size - current_size):
                thread = threading.Thread(target=self._worker)
                thread.daemon = True
                thread.start()
                self.threads.append(thread)
        elif new_size < current_size:
            for _ in range(current_size - new_size):
                self.task_queue.put((None, None, None))  # 发送结束信号
                self.threads.pop()  # 这里需要实现线程的安全退出

# 创建全局实例
thread_pool = ThreadPool(4)  # 创建一个有4个线程的线程池
