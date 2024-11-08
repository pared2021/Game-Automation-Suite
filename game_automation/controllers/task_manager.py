from game_automation.controllers.task_manager_module import TaskManager

# 创建全局实例
task_manager = TaskManager(None)  # 初始化时传入 None，后续在 GameEngine 中设置正确的引用
