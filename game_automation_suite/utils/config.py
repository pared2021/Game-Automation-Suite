import json

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        """
        从配置文件中加载配置数据。
        """
        try:
            with open(self.config_file, 'r') as file:
                config_data = json.load(file)
            print(f"Config loaded from {self.config_file}")
            return config_data
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.config_file}.")
            return {}

    def get(self, key, default=None):
        """
        获取配置项的值。
        """
        return self.config_data.get(key, default)