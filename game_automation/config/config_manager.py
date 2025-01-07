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

    def get(self, key):
        """Get configuration value by key"""
        value = self.config.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found and no default provided")
        return value

    def get_with_default(self, key, default):
        """Get configuration value with fallback to default"""
        return self.config.get(key, default)

    def set(self, key, value, save_to_file=False):
        """Set configuration value and optionally save to file"""
        self.config[key] = value
        if save_to_file:
            self.save_config(key)

    def save_config(self, key):
        """Save configuration to appropriate file"""
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine which file contains this key
        if key in self.config.get('game_settings', {}):
            config_file = 'game_settings.yaml'
        elif key in self.config.get('resource_paths', {}):
            config_file = 'resource_paths.yaml'
        else:
            config_file = 'deploy.template.yaml'
            
        file_path = os.path.join(config_dir, config_file)
        
        # Load existing config
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml'):
                config = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                config = json.load(f)
                
        # Update config
        config[key] = self.config[key]
        
        # Save back to file
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml'):
                yaml.safe_dump(config, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                json.dump(config, f, indent=4)

    def validate_configs(self):
        """Validate all loaded configurations"""
        required_configs = [
            'game_settings.yaml',
            'resource_paths.yaml',
            'deploy.template.yaml'
        ]
        
        errors = []
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if required config files exist
        for config_file in required_configs:
            if not os.path.exists(os.path.join(config_dir, config_file)):
                errors.append(f"Missing required config file: {config_file}")
                continue
                
            # Validate file content
            try:
                with open(os.path.join(config_dir, config_file), 'r') as f:
                    if config_file.endswith('.yaml'):
                        config = yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        config = json.load(f)
                    
                    # Validate specific config values
                    if config_file == 'game_settings.yaml':
                        self._validate_game_settings(config)
                    elif config_file == 'resource_paths.yaml':
                        self._validate_resource_paths(config)
                        
            except Exception as e:
                errors.append(f"Invalid config format in {config_file}: {str(e)}")
        
        # Validate specific required fields
        if not self.config.get('game_settings'):
            errors.append("Missing required game_settings configuration")
            
        if not self.config.get('resource_paths'):
            errors.append("Missing required resource_paths configuration")
            
        return len(errors) == 0, errors

    def _validate_game_settings(self, config):
        """Validate game settings configuration"""
        errors = []
        
        # Validate battle settings
        battle_settings = config.get('battle', {})
        if not isinstance(battle_settings, dict):
            errors.append("Invalid battle settings format")
        else:
            auto_battle = battle_settings.get('auto_battle', {})
            if not isinstance(auto_battle, dict):
                errors.append("Invalid auto_battle settings format")
            else:
                strategy = auto_battle.get('strategy')
                if strategy not in ['balanced', 'aggressive', 'defensive']:
                    errors.append("Invalid auto_battle strategy")
                    
        # Validate collection settings
        collection_settings = config.get('collection', {})
        if not isinstance(collection_settings, dict):
            errors.append("Invalid collection settings format")
        else:
            radius = collection_settings.get('collection_radius')
            if not isinstance(radius, (int, float)) or radius < 0 or radius > 100:
                errors.append("Invalid collection_radius value")
                
        return errors

    def _validate_resource_paths(self, config):
        """Validate resource paths configuration"""
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Invalid resource_paths format")
        else:
            for path_type, path_value in config.items():
                if not isinstance(path_value, str):
                    errors.append(f"Invalid path value for {path_type}")
                elif not os.path.exists(path_value):
                    errors.append(f"Path does not exist: {path_value}")
                    
        return errors

config_manager = ConfigManager()
