#%%
from tensor.layers import MainNodeLayer, InputNodeLayer
from +

bond_dim = 3
input_features = 5
output_shape = 3

input_nodes = []
main_nodes = []

def zip_connect(nodes1, nodes2, label='z{0}', priority=1):
"""Connects two lists of nodes with a zip connection."""
if len(nodes1) != len(nodes2):
    raise ValueError("The number of nodes in both lists must be the same.")
for i, (n1, n2) in enumerate(zip(nodes1, nodes2), 1):
    n1.connect(n2, label.format(i), priority=priority)

def horizontal_connect(self, nodes):
"""Connects nodes horizontally."""
if len(nodes) < 2:
    return
for n1, n2 in zip(nodes[:-1], nodes[1:]):
    if n1.right_labels and n2.left_labels and n1.right_labels[0] != n2.left_labels[0]:
        raise ValueError(f"Right label of the first node does not match left label of the second node. Nodes: {n1.name}, {n2.name}")
    n1.connect(n2, n1.right_labels[0], priority=1)

for n in [1, 2, 3]:
    main_node_layer = MainNodeLayer(
        num_carriages, bond_dim, input_features, output_shape=output_shape,
        down_label='p{0}', constrict_bond=False, perturb=True
    )
    horizontal_connect(main_node_layer.nodes)
    input_node_layer = InputNodeLayer(num_carriages, input_features, label='p{0}', dtype=dtype)
    zip_connect(input_node_layer.nodes, main_node_layer.nodes, label='p{0}')

    if squeeze:
        for node in main_node_layer.nodes:
            node.squeeze(main_node_layer.labels)

    input_nodes.append(input_node_layer)
    main_nodes.append(main_node_layer)

# Create a TensorNetwork
tensor_network = TensorNetwork(input_node_layer.nodes, main_node_layer.nodes, output_labels=main_node_layer.labels)