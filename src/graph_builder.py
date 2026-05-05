import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
    
    def add_triple(self, subject: str, relation: str, obj: str):
        """Thêm một triple (subject, relation, obj) vào đồ thị"""
        self.graph.add_node(subject, type='entity')
        self.graph.add_node(obj, type='entity')
        self.graph.add_edge(subject, obj, relation=relation)
    
    def add_triples(self, triples: List[Tuple[str, str, str]]):
        """Thêm nhiều triples từ list"""
        for subj, rel, obj in triples:
            self.add_triple(subj, rel, obj)
    
    def bfs_traversal(self, start_node: str, max_depth: int = 2) -> Dict:
        """Duyệt đồ thị BFS trong phạm vi max_depth hops"""
        result = {0: [start_node]}
        visited = {start_node}
        current_level = [start_node]
        
        for depth in range(1, max_depth + 1):
            next_level = []
            for node in current_level:
                # Lấy neighbors (cả incoming và outgoing)
                neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            result[depth] = next_level
            current_level = next_level
        
        return result
    
    def get_context(self, entity: str, hops: int = 2) -> str:
        """Lấy context text cho một entity"""
        if entity not in self.graph.nodes:
            return f"Entity '{entity}' not found"
        
        context_lines = [f"Information about {entity}:"]
        
        # Lấy tất cả edges trong phạm vi hops
        for node in self.graph.nodes:
            if node == entity:
                continue
            # Kiểm tra edge từ entity -> node
            if self.graph.has_edge(entity, node):
                rel = self.graph[entity][node]['relation']
                context_lines.append(f"- {entity} {rel} {node}")
            # Kiểm tra edge từ node -> entity
            if self.graph.has_edge(node, entity):
                rel = self.graph[node][entity]['relation']
                context_lines.append(f"- {node} {rel} {entity}")
        
        return "\n".join(context_lines)
    
    def visualize(self, output_path: str = "outputs/graphs/knowledge_graph.png"):
        """Vẽ đồ thị và lưu thành file"""
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        
        # Vẽ nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=2500, 
                               node_color='lightblue', alpha=0.9)
        
        # Vẽ edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                               arrows=True, arrowsize=20, width=1.5)
        
        # Vẽ labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # Vẽ edge labels (quan hệ)
        edge_labels = {(u, v): d['relation'] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
        
        plt.title("Knowledge Graph - Tech Companies (NetworkX)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Graph saved to {output_path}")
    
    def get_stats(self) -> Dict:
        """Thống kê đồ thị"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'nodes': list(self.graph.nodes()),
            'is_directed': self.graph.is_directed()
        }