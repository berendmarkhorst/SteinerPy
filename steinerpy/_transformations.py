"""Graph transformations with explicit original-node and edge provenance."""

from dataclasses import dataclass

import networkx as nx


@dataclass
class SplitNodes:
    """Paid node traversals and original-edge arcs, ready for terminal attachment."""

    graph: nx.DiGraph
    split: dict
    node_arcs: dict
    edge_arcs: dict


def split_node_costs(
    original, node_costs, weight="weight", *, default_edge_cost=1, split=None
):
    """Split an undirected graph without modifying its nodes or cost mappings.

    Every node has an input/output pair; its traversal arc charges the node cost.
    Cross-arcs retain original edge identities in both directions. Integer labels
    are collision-free by default; a legacy adapter may supply explicit pairs.
    Cost/connectivity validation belongs to the caller before transformation.
    """
    if original.is_directed():
        raise ValueError("Node-cost splitting requires an undirected graph.")
    split = (
        {node: (2 * i, 2 * i + 1) for i, node in enumerate(original)}
        if split is None
        else {node: tuple(pair) for node, pair in split.items()}
    )
    if set(split) != set(original) or any(len(pair) != 2 for pair in split.values()):
        raise ValueError("Split labels must provide one input/output pair per node.")
    labels = [label for pair in split.values() for label in pair]
    if len(set(labels)) != len(labels):
        raise ValueError("Split labels must be unique.")

    graph = nx.DiGraph()
    node_costs = node_costs or {}
    node_arcs = {}
    edge_arcs = {}
    for node, (enter, leave) in split.items():
        graph.add_edge(enter, leave, **{weight: node_costs.get(node, 0)})
        node_arcs[enter, leave] = node
    for u, v, data in original.edges(data=True):
        for a, b in ((u, v), (v, u)):
            arc = (split[a][1], split[b][0])
            graph.add_edge(*arc, **{weight: data.get(weight, default_edge_cost)})
            edge_arcs[arc] = (u, v)
    return SplitNodes(graph, split, node_arcs, edge_arcs)
