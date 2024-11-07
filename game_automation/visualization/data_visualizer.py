import matplotlib.pyplot as plt
from utils.logger import setup_logger

class DataVisualizer:
    def __init__(self):
        self.logger = setup_logger('data_visualizer')

    def plot_performance_over_time(self, performance_data):
        plt.figure(figsize=(10, 6))
        plt.plot(performance_data)
        plt.title('Player Performance Over Time')
        plt.xlabel('Game Sessions')
        plt.ylabel('Performance Score')
        plt.savefig('performance_plot.png')
        self.logger.info("Performance plot saved as performance_plot.png")

    def plot_difficulty_changes(self, difficulty_data):
        plt.figure(figsize=(10, 6))
        plt.plot(difficulty_data)
        plt.title('Game Difficulty Changes')
        plt.xlabel('Adjustments')
        plt.ylabel('Difficulty Level')
        plt.savefig('difficulty_plot.png')
        self.logger.info("Difficulty plot saved as difficulty_plot.png")

    def create_heatmap(self, action_frequencies):
        plt.figure(figsize=(12, 8))
        plt.imshow(action_frequencies, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Action Frequency Heatmap')
        plt.xlabel('Actions')
        plt.ylabel('Game States')
        plt.savefig('action_heatmap.png')
        self.logger.info("Action frequency heatmap saved as action_heatmap.png")

data_visualizer = DataVisualizer()