import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self, filename: str = 'config.yaml'):
        config_path = os.path.join(self.config_dir, filename)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def save_config(self, filename: str = 'config.yaml'):
        config_path = os.path.join(self.config_dir, filename)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def load_preset(self, preset_name: str):
        preset_path = os.path.join(self.config_dir, 'presets', f'{preset_name}.yaml')
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as f:
                preset_config = yaml.safe_load(f)
            self.config.update(preset_config)
        else:
            raise FileNotFoundError(f"Preset {preset_name} not found")

    def save_preset(self, preset_name: str):
        preset_dir = os.path.join(self.config_dir, 'presets')
        os.makedirs(preset_dir, exist_ok=True)
        preset_path = os.path.join(preset_dir, f'{preset_name}.yaml')
        with open(preset_path, 'w') as f:
            yaml.dump(self.config, f)

config_manager = ConfigManager()