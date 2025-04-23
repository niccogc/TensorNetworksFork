import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

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

    # Traverse the network to determine positions
    pos = {}
    visited = set()

    def traverse_and_position(main_nodes):
        # Assign x-positions based on main nodes
        for i, node in enumerate(main_nodes):
            pos[node.name] = (i * 2, 0)  # Main nodes are spaced horizontally
            visited.add(node.name)

        # Traverse non-horizontal connections to determine y-positions
        queue = deque(main_nodes)
        while queue:
            node = queue.popleft()
            x, y = pos[node.name]

            for label, connected_node in node.connections.items():
                if connected_node.name not in visited and not node.is_horizontal_bond(label):
                    visited.add(connected_node.name)
                    pos[connected_node.name] = (x, y - 1)  # Shift vertically for non-horizontal connections
                    queue.append(connected_node)

    # Start traversal from main nodes
    traverse_and_position(tensornetwork.main_nodes)

    # Draw the graph
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')

    # Add node labels for shapes and names
    labels = {node: f"{node}\n{tuple(G.nodes[node]['shape'])}" for node in G.nodes}  # Display both name and shape as a tuple
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)  # Increase font size for better visibility

    # Add edge labels for sizes
    edge_labels = {(u, v): f"{d['size']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)  # Increased font size for edge labels

    plt.title("Tensor Network Visualization")
    plt.show()