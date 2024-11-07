import importlib
import os

class PluginManager:
    def __init__(self, plugin_dir='plugins'):
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                module = importlib.import_module(f'{self.plugin_dir}.{module_name}')
                if hasattr(module, 'register_plugin'):
                    plugin = module.register_plugin()
                    self.plugins[module_name] = plugin

    def get_plugin(self, name):
        return self.plugins.get(name)

    def execute_plugin(self, name, *args, **kwargs):
        plugin = self.get_plugin(name)
        if plugin:
            return plugin.execute(*args, **kwargs)
        else:
            raise ValueError(f"Plugin {name} not found")

plugin_manager = PluginManager()