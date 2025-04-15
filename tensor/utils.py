import networkx as nx
import matplotlib.pyplot as plt

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

    def traverse_and_position(node, x, y):
        if node.name in visited:
            return
        visited.add(node.name)
        pos[node.name] = (x, y)

        # Traverse left connections
        left_x = x - 1
        for left_label in node.left_labels:
            if left_label in node.connections:
                traverse_and_position(node.connections[left_label], left_x, y)
                left_x -= 1

        # Traverse right connections
        right_x = x + 1
        for right_label in node.right_labels:
            if right_label in node.connections:
                traverse_and_position(node.connections[right_label], right_x, y)
                right_x += 1

        # Traverse other connections vertically
        up_y = y + 1
        for label, connected_node in node.connections.items():
            if label not in node.left_labels + node.right_labels:
                traverse_and_position(connected_node, x, up_y)
                up_y += 1

    # Start traversal from main nodes
    x_offset = 0
    for main_node in tensornetwork.main_nodes:
        traverse_and_position(main_node, x_offset, 0)
        x_offset += 2

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