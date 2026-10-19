"""Graph transformations with explicit original-node and edge provenance."""

from dataclasses import dataclass

import networkx as nx


def _fresh_label(base, existing):
    if base not in existing:
        return base
    index = 0
    while "{}{}".format(base, index) in existing:
        index += 1
    return "{}{}".format(base, index)


@dataclass
class GroupTerminals:
    """Artificial terminals and connectors mapped to (group index, member).

    Directed connector keys are arc tuples; undirected keys are frozensets.
    """

    graph: nx.Graph
    terminals: list
    connector_edges: dict


def attach_group_terminals(original, groups, weight="weight", *, ports=None):
    """Attach zero-cost group terminals to a copy of a working graph.

    ``ports`` maps original group members to their working output nodes after
    splitting; without splitting, members are their own ports. Directed
    connectors enter sink-only terminals, so a group cannot create shortcuts.
    """
    groups = tuple(tuple(group) for group in groups)
    if not groups or any(not group for group in groups):
        raise ValueError("Each group must contain at least one vertex.")
    ports = {node: node for node in original} if ports is None else dict(ports)
    for group in groups:
        for member in group:
            if member not in ports:
                raise ValueError(
                    "group vertex {!r} is not in the graph.".format(member)
                )
            if ports[member] not in original:
                raise ValueError(
                    "Group port {!r} is not in the graph.".format(ports[member])
                )

    graph = original.copy()
    existing = set(graph)
    terminals = []
    connectors = {}
    for index, group in enumerate(groups):
        terminal = _fresh_label("__group_{}__".format(index), existing)
        existing.add(terminal)
        terminals.append(terminal)
        graph.add_node(terminal)
        for member in group:
            port = ports[member]
            edge = (port, terminal) if graph.is_directed() else (terminal, port)
            graph.add_edge(*edge, **{weight: 0})
            key = edge if graph.is_directed() else frozenset(edge)
            connectors[key] = (index, member)
    return GroupTerminals(graph, terminals, connectors)


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
