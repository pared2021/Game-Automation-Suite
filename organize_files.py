import os
import shutil

# 定义源目录和目标目录的映射
directory_mapping = {
    "actions": "game_automation/actions",
    "ai": "game_automation/ai",
    "analysis": "game_automation/analysis",
    "blessing": "game_automation/blessing",
    "config": "game_automation/config",
    "controllers": "game_automation/controllers",
    "core": "game_automation/core",
    "debug": "game_automation/debug",
    "device": "game_automation/device",
    "difficulty": "game_automation/difficulty",
    "game_types": "game_automation/game_types",
    "gui": "game_automation/gui",
    "i18n": "game_automation/i18n",
    "multimodal": "game_automation/multimodal",
    "nlp": "game_automation/nlp",
    "ocr_prediction": "game_automation/ocr_prediction",
    "optimization": "game_automation/optimization",
    "plugins": "game_automation/plugins",
    "reasoning": "game_automation/reasoning",
    "resources": "game_automation/resources",
    "rogue": "game_automation/rogue",
    "scene_understanding": "game_automation/scene_understanding",
    "security": "game_automation/security",
    "testing": "game_automation/testing",
    "utils": "game_automation/utils",
    "visualization": "game_automation/visualization",
    "web/templates": "game_automation/web/templates"
}

# 移动文件
for dir_name, target_dir in directory_mapping.items():
    source_dir = os.path.join("game_automation", dir_name)
    if os.path.exists(source_dir):
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(target_dir, file_name)
            if not os.path.exists(target_file):  # 检查目标文件是否存在
                shutil.move(source_file, target_dir)

print("文件移动操作已完成。")
