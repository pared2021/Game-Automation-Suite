class ExamplePlugin:
    def __init__(self):
        self.name = "example_plugin"

    def execute(self, game_engine):
        game_engine.logger.info("Example plugin executed")
        # 在这里添加插件的具体功能

def register_plugin():
    return ExamplePlugin()