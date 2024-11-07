import networkx as nx
from pyDatalog import pyDatalog
from utils.logger import detailed_logger
from utils.config_manager import config_manager

class InferenceEngine:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('reasoning', {})
        self.knowledge_graph = nx.DiGraph()
        pyDatalog.create_terms('X, Y, Z, relation, infer')

    async def add_fact(self, subject, predicate, object):
        self.knowledge_graph.add_edge(subject, object, relation=predicate)
        pyDatalog.assert_fact(f'relation', subject, predicate, object)
        self.logger.info(f"Added fact: {subject} {predicate} {object}")

    async def query(self, query):
        results = pyDatalog.ask(query)
        self.logger.info(f"Query results: {results}")
        return results

    async def infer(self, start_node, end_node):
        paths = list(nx.all_simple_paths(self.knowledge_graph, start_node, end_node))
        inferences = []
        for path in paths:
            inference = " -> ".join([f"{path[i]} {self.knowledge_graph[path[i]][path[i+1]]['relation']} {path[i+1]}" for i in range(len(path)-1)])
            inferences.append(inference)
        self.logger.info(f"Inferences: {inferences}")
        return inferences

    async def add_rule(self, rule):
        pyDatalog.load(rule)
        self.logger.info(f"Added rule: {rule}")

    async def visualize_knowledge_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.knowledge_graph)
        nx.draw(self.knowledge_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.knowledge_graph, 'relation')
        nx.draw_networkx_edge_labels(self.knowledge_graph, pos, edge_labels=edge_labels)
        plt.title("Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('knowledge_graph.png')
        self.logger.info("Knowledge graph visualization saved as 'knowledge_graph.png'")

inference_engine = InferenceEngine()
