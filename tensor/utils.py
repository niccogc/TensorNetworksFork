import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import re

def visualize_tensornetwork(tensornetwork, layout='grid'):
    """
    Visualize the tensornetwork as a graph.

    Parameters:
        tensornetwork: The tensor network object containing nodes and edges.
        layout: The layout for visualization. Options are 'grid', 'horizontal', or 'vertical'.
    """
    G = nx.DiGraph()

    # Add nodes with their shapes and names
    for node in tensornetwork.nodes:
        G.add_node(node.name, shape=node.shape)

    # Add edges with sizes
    for node in tensornetwork.nodes:
        for label, connected_node in node.connections.items():
            size = node.dim_size(label)  # Use the dim_size method to get the size of the dimension
            G.add_edge(node.name, connected_node.name, size=size)

    pos = {}
    visited = set()

    # Try to detect grid structure by node names (e.g., "A{layer}_{col}")
    grid_nodes = []
    grid_pattern = re.compile(r"^A(\d+)_(\d+)$")
    for node in tensornetwork.nodes:
        m = grid_pattern.match(node.name)
        if m:
            grid_nodes.append((int(m.group(1)), int(m.group(2)), node.name))
    if grid_nodes:
        # Find grid dimensions
        max_layer = max(layer for layer, col, name in grid_nodes)
        min_layer = min(layer for layer, col, name in grid_nodes)
        min_col = min(col for layer, col, name in grid_nodes)
        # Arrange grid nodes in a 2D grid: x=col, y increases with layer (bottom layer at y=2)
        for layer, col, name in grid_nodes:
            pos[name] = (col, 2 + (layer - min_layer))
            visited.add(name)
        # Place C and X nodes below grid
        for node in tensornetwork.nodes:
            if node.name not in pos:
                col = None
                if node.name.startswith("C"):
                    try:
                        col = int(node.name[1:])
                    except Exception:
                        pass
                    if col is not None:
                        pos[node.name] = (col, 0)  # C at y=0
                        visited.add(node.name)
                elif node.name.startswith("X"):
                    try:
                        col = int(node.name[1:])
                    except Exception:
                        pass
                    if col is not None:
                        pos[node.name] = (col, 1)  # X at y=1
                        visited.add(node.name)
        # Fallback for any remaining nodes
        y_offset = 2 + (max_layer - min_layer) + 1
        for node in tensornetwork.nodes:
            if node.name not in pos:
                pos[node.name] = (len(pos), y_offset)
                y_offset += 1
    else:
        # Fallback: old logic
        def traverse_and_position(main_nodes):
            for i, node in enumerate(main_nodes):
                pos[node.name] = (i * 2, 0)
                visited.add(node.name)
            queue = deque(main_nodes)
            while queue:
                node = queue.popleft()
                x, y = pos[node.name]
                for label, connected_node in node.connections.items():
                    if connected_node.name not in visited and not node.is_horizontal_bond(label):
                        visited.add(connected_node.name)
                        pos[connected_node.name] = (x, y - 1)
                        queue.append(connected_node)
        traverse_and_position(tensornetwork.main_nodes)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')

    labels = {node: f"{node}\n{tuple(G.nodes[node]['shape'])}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    edge_labels = {(u, v): f"{d['size']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Tensor Network Visualization")
    plt.show()