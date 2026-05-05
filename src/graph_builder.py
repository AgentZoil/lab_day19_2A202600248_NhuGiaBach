import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json
import os
from collections import defaultdict
import textwrap

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_triple(self, subject: str, relation: str, obj: str):
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(subject, obj, relation=relation)
    
    def add_triples(self, triples: List[Tuple[str, str, str]]):
        for subj, rel, obj in triples:
            self.add_triple(subj, rel, obj)
    
    def get_neighbors(self, entity: str, hops: int = 2) -> Dict[int, List[str]]:
        if entity not in self.graph.nodes:
            return {}
        result = {0: [entity]}
        visited = {entity}
        current = [entity]
        for depth in range(1, hops + 1):
            next_level = []
            for node in current:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            result[depth] = next_level
            current = next_level
        return result
    
    def bfs_traversal(self, start: str, max_depth: int = 2) -> List[str]:
        if start not in self.graph.nodes:
            return []
        visited = set()
        queue = [(start, 0)]
        result = []
        while queue:
            node, depth = queue.pop(0)
            if node not in visited and depth <= max_depth:
                visited.add(node)
                result.append(node)
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        return result
    
    def get_context(self, entity: str, hops: int = 2) -> str:
        if entity not in self.graph.nodes:
            return f"Entity '{entity}' not found"
        lines = [f"Information about {entity}:"]
        visited = {entity}
        current = [entity]
        for depth in range(hops):
            next_nodes = []
            for node in current:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        rel = self.graph[node][neighbor]['relation']
                        lines.append(f"- {node} {rel} {neighbor}")
                        visited.add(neighbor)
                        next_nodes.append(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        rel = self.graph[neighbor][node]['relation']
                        lines.append(f"- {neighbor} {rel} {node}")
                        visited.add(neighbor)
                        next_nodes.append(neighbor)
            current = next_nodes
        return '\n'.join(lines) if len(lines) > 1 else f"No connected information for {entity}"
    
    def _node_category(self, node: str) -> str:
        if node.isdigit():
            return "year"
        if node in {"California", "Washington", "Texas", "San Francisco", "Redmond", "Menlo Park", "Cupertino", "Austin", "Bellevue", "Mountain View", "Seattle", "Santa Clara"}:
            return "location"
        if node in {"OpenAI", "Google", "Microsoft", "Meta", "Facebook", "Amazon", "Apple", "Tesla", "NVIDIA", "Alphabet Inc."}:
            return "company"
        return "person"

    def _short_label(self, label: str, width: int = 10) -> str:
        return "\n".join(textwrap.wrap(label, width=width)) if len(label) > width else label

    def _triangle_layout(self, graph: nx.DiGraph):
        """Arrange nodes in a layered triangle-like layout."""
        layers = {
            "company": [],
            "person": [],
            "location": [],
            "year": [],
        }
        for node in graph.nodes():
            layers[self._node_category(node)].append(node)

        pos = {}
        layer_y = {
            "company": 2.8,
            "person": 1.3,
            "location": 0.0,
            "year": -0.7,
        }
        x_span = {
            "company": 2.2,
            "person": 4.8,
            "location": 4.2,
            "year": 4.2,
        }

        for category, nodes in layers.items():
            if not nodes:
                continue
            count = len(nodes)
            span = x_span[category]
            xs = [0.0] if count == 1 else [i * (2 * span) / max(count - 1, 1) - span for i in range(count)]

            for i, node in enumerate(nodes):
                y_offset = 0.0
                if count > 1:
                    y_offset = 0.08 * ((i % 3) - 1)
                pos[node] = (xs[i], layer_y[category] + y_offset)

        return pos

    def visualize(
        self,
        output_path: str = "outputs/graphs/knowledge_graph.png",
        focus_nodes: Optional[List[str]] = None,
        hops: int = 1,
        show_edge_labels: bool = True,
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph = self.graph
        if focus_nodes:
            nodes = set()
            for node in focus_nodes:
                if node in graph:
                    nodes.add(node)
                    frontier = {node}
                    for _ in range(hops):
                        next_frontier = set()
                        for current in frontier:
                            next_frontier.update(graph.successors(current))
                            next_frontier.update(graph.predecessors(current))
                        nodes.update(next_frontier)
                        frontier = next_frontier
            if nodes:
                graph = graph.subgraph(nodes).copy()

        plt.figure(figsize=(16, 11))
        pos = nx.spring_layout(graph, seed=42, k=0.42, iterations=120)

        categories = defaultdict(list)
        for node in graph.nodes():
            categories[self._node_category(node)].append(node)

        color_map = {
            "company": "#4E79A7",
            "person": "#F28E2B",
            "location": "#59A14F",
            "year": "#E15759",
        }

        for category, nodes in categories.items():
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_size=1700 if category == "company" else 950,
                node_color=color_map.get(category, "#9C755F"),
                alpha=0.92,
                linewidths=1.2,
                edgecolors="white",
            )

        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color="#888888",
            arrows=True,
            arrowsize=11,
            width=0.95,
            connectionstyle="arc3,rad=0.06",
        )

        labels = {node: self._short_label(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6.5, font_weight='bold')

        edge_labels = {(u, v): d['relation'] for u, v, d in graph.edges(data=True)}
        if show_edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels,
                font_size=6.2,
                label_pos=0.55,
                rotate=False,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
            )

        title = "Knowledge Graph - Tech Companies"
        if focus_nodes:
            title = f"Knowledge Graph Focus: {', '.join(focus_nodes)}"
        plt.title(title, fontsize=15, pad=18)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_path}")
    
    def get_stats(self) -> Dict:
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'nodes': list(self.graph.nodes()),
            'edges': [(u, v, d['relation']) for u, v, d in self.graph.edges(data=True)]
        }
    
    def save_graph(self, path: str):
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_graph(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def export_interactive_html(
        self,
        output_path: str = "outputs/graphs/knowledge_graph.html",
        focus_nodes: Optional[List[str]] = None,
        hops: int = 1,
    ):
        """Export a force-directed interactive HTML graph using vis-network."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        graph = self.graph
        if focus_nodes:
            nodes = set()
            for node in focus_nodes:
                if node in graph:
                    nodes.add(node)
                    frontier = {node}
                    for _ in range(hops):
                        next_frontier = set()
                        for current in frontier:
                            next_frontier.update(graph.successors(current))
                            next_frontier.update(graph.predecessors(current))
                        nodes.update(next_frontier)
                        frontier = next_frontier
            if nodes:
                graph = graph.subgraph(nodes).copy()

        node_colors = {
            "company": "#4E79A7",
            "person": "#F28E2B",
            "location": "#59A14F",
            "year": "#E15759",
        }

        nodes_data = []
        for node in graph.nodes():
            category = self._node_category(node)
            nodes_data.append({
                "id": node,
                "label": self._short_label(node, width=12),
                "title": f"<b>{node}</b><br>Type: {category}",
                "group": category,
                "color": node_colors.get(category, "#9C755F"),
                "shape": "box" if category == "company" else "ellipse",
            })

        edges_data = []
        for u, v, data in graph.edges(data=True):
            relation = data.get("relation", "")
            edges_data.append({
                "from": u,
                "to": v,
                "label": relation,
                "arrows": "to",
                "font": {
                    "size": 13,
                    "align": "middle",
                    "strokeWidth": 3,
                    "strokeColor": "#ffffff",
                    "vadjust": -4,
                },
                "title": relation,
                "color": {"color": "#888888", "highlight": "#333333"},
            })

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Graph Interactive View</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f6f8fb;
    }}
    .wrap {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 12px;
      height: 100vh;
      padding: 12px;
      box-sizing: border-box;
    }}
    #network {{
      background: white;
      border-radius: 16px;
      border: 1px solid #e5e7eb;
      min-height: 0;
    }}
    #info {{
      background: white;
      border-radius: 16px;
      border: 1px solid #e5e7eb;
      padding: 16px;
      overflow: auto;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 8px 0;
    }}
    .muted {{
      color: #6b7280;
      font-size: 13px;
      line-height: 1.5;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      margin: 4px 4px 0 0;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
    }}
    code {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div id="network"></div>
    <div id="info">
      <h1>Knowledge Graph</h1>
      <div class="muted">
        Click any node to inspect its details. Use mouse wheel to zoom and drag to pan.
      </div>
      <div style="margin-top: 12px;">
        <span class="pill">Company</span>
        <span class="pill">Person</span>
        <span class="pill">Location</span>
        <span class="pill">Year</span>
      </div>
      <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
      <div id="details" class="muted">No node selected.</div>
    </div>
  </div>

  <script>
    const nodes = new vis.DataSet({json.dumps(nodes_data)});
    const edges = new vis.DataSet({json.dumps(edges_data)});
    const container = document.getElementById('network');
    const details = document.getElementById('details');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      layout: {{
        improvedLayout: true
      }},
      nodes: {{
        borderWidth: 1,
        shadow: true,
        font: {{
          size: 14,
          face: 'Arial'
        }}
      }},
      edges: {{
        smooth: {{
          type: 'dynamic'
        }},
        shadow: false,
        width: 1.2,
        font: {{
          size: 13,
          align: 'middle',
          strokeWidth: 3,
          strokeColor: '#ffffff'
        }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 120,
        navigationButtons: true,
        keyboard: true
      }},
      physics: {{
        enabled: true,
        barnesHut: {{
          gravitationalConstant: -4200,
          springLength: 160,
          springConstant: 0.03,
          damping: 0.09,
          avoidOverlap: 0.6
        }}
      }}
    }};

    const network = new vis.Network(container, data, options);

    network.on('click', function (params) {{
      if (params.nodes.length === 0) {{
        details.innerHTML = 'No node selected.';
        return;
      }}
      const id = params.nodes[0];
      const node = nodes.get(id);
      const neighbors = network.getConnectedNodes(id);
      const connectedEdges = network.getConnectedEdges(id).map(eid => edges.get(eid));
      const relList = connectedEdges.map(e => `• ${{e.from}} -- ${{e.label || ''}} --> ${{e.to}}`).join('<br>');
      details.innerHTML = `
        <div><b>Node:</b> ${{node.id}}</div>
        <div><b>Type:</b> ${{node.group}}</div>
        <div><b>Connected nodes:</b> ${{neighbors.length}}</div>
        <div style="margin-top:8px;"><b>Relations</b></div>
        <code>${{relList || 'No relations'}}</code>
      `;
    }});
  </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Interactive graph saved to {output_path}")
