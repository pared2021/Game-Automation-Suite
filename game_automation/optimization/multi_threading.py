from game_automation.optimization.thread_manager import thread_pool

# 使用示例
async def main():
    # 示例任务
    def example_task(data):
        print(f"Processing {data}")

    # 添加任务到线程池
    for i in range(10):
        thread_pool.add_task(example_task, f"Task {i}")

    # 等待所有任务完成
    thread_pool.wait_completion()

if __name__ == "__main__":
    main()
