import importlib
import os
from utils.logger import detailed_logger

class PluginManager:
    def __init__(self):
        self.logger = detailed_logger
        self.plugins = {}

    def load_plugins(self):
        plugin_dir = 'game_automation/plugins'
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = f'game_automation.plugins.{filename[:-3]}'
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, 'register_plugin'):
                        plugin = module.register_plugin()
                        self.plugins[plugin.name] = plugin
                        self.logger.info(f"Loaded plugin: {plugin.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load plugin {filename}: {str(e)}")

    def get_plugin(self, name):
        return self.plugins.get(name)

    async def execute_plugin(self, name, game_engine):
        plugin = self.get_plugin(name)
        if plugin:
            try:
                result = await plugin.execute(game_engine)
                self.logger.info(f"Executed plugin {name}: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Error executing plugin {name}: {str(e)}")
        else:
            self.logger.warning(f"Plugin not found: {name}")

plugin_manager = PluginManager()
