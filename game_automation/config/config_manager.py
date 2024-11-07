import json
import os
import yaml

class ConfigManager:
    def __init__(self):
        self.config = {}
        self.load_all_configs()

    def load_all_configs(self):
        config_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(config_dir):
            if filename.endswith('.json') or filename.endswith('.yaml'):
                file_path = os.path.join(config_dir, filename)
                self.load_config(file_path)

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config = json.load(f)
            elif file_path.endswith('.yaml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            self.config.update(config)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

config_manager = ConfigManager()