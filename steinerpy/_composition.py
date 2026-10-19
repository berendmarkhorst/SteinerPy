"""Compile original-space node costs and group choices into directed cuts."""

from dataclasses import dataclass
import math

import networkx as nx

from . import mathematical_model as mm
from .objects import DirectedSteinerProblem, NodeWeightedSolution


def split_node_costs(spec):
    """Move node costs onto arcs, retaining original node and edge identities."""
    graph = nx.DiGraph()
    split = {node: (2 * i, 2 * i + 1) for i, node in enumerate(spec.graph)}
    node_arcs = {}
    edge_arcs = {}
    for node, (enter, leave) in split.items():
        graph.add_edge(
            enter, leave, **{spec.weight: (spec.node_costs or {}).get(node, 0)}
        )
        node_arcs[enter, leave] = node
    for u, v, data in spec.graph.edges(data=True):
        for a, b in ((u, v), (v, u)):
            arc = (split[a][1], split[b][0])
            graph.add_edge(*arc, **{spec.weight: data.get(spec.weight, 1)})
            edge_arcs[arc] = (u, v)
    return graph, split, node_arcs, edge_arcs


@dataclass
class CompiledGroups:
    graph: nx.DiGraph
    root: int
    terminals: list
    source_arcs: list
    node_arcs: dict
    edge_arcs: dict


def compile_groups(spec):
    """Compose paid-node traversal with terminal sinks and one chosen root.

    Non-negative costs admit an optimal tree, rooted at a selected member of
    any group. Exactly one source arc chooses that root; sink-only connectors
    cannot create original-space shortcuts. Every path to a group sink leaves
    a member's output node, so its node-cost arc must have been traversed.
    """
    graph, split, node_arcs, edge_arcs = split_node_costs(spec)
    groups = list(spec.groups) + [(t,) for group in spec.terminal_groups for t in group]
    feasible_nodes = set()
    for component in nx.connected_components(spec.graph):
        if all(component.intersection(group) for group in groups):
            feasible_nodes.update(component)
    if not feasible_nodes:
        raise RuntimeError(
            "Group/node-cost instance is infeasible: groups cannot be connected."
        )
    root = len(graph)
    graph.add_node(root)
    source_arcs = [
        (root, split[node][0])
        for node in min(groups, key=len)
        if node in feasible_nodes
    ]
    for arc in source_arcs:
        graph.add_edge(*arc, **{spec.weight: 0})
    terminals = []
    for group in groups:
        sink = len(graph)
        terminals.append(sink)
        graph.add_node(sink)
        for node in group:
            graph.add_edge(split[node][1], sink, **{spec.weight: 0})
    return CompiledGroups(graph, root, terminals, source_arcs, node_arcs, edge_arcs)


def decode_groups(spec, compiled, selected_arcs, objective):
    """Map and independently validate a candidate in the original graph."""
    selected = nx.DiGraph()
    selected.add_node(compiled.root)
    selected.add_edges_from(selected_arcs)
    reachable = nx.descendants(selected, compiled.root) | {compiled.root}
    used = {(u, v) for u, v in selected.edges if u in reachable}
    if sum(arc in used for arc in compiled.source_arcs) != 1:
        raise RuntimeError(
            "Composed solution does not select exactly one original root."
        )
    nodes = [node for arc, node in compiled.node_arcs.items() if arc in used]
    edge_set = {compiled.edge_arcs[arc] for arc in used if arc in compiled.edge_arcs}
    edges = [edge for edge in spec.graph.edges if edge in edge_set]
    original = nx.Graph()
    original.add_nodes_from(nodes)
    original.add_edges_from(edges)
    if not nodes or set(original) != set(nodes) or not nx.is_connected(original):
        raise RuntimeError(
            "Composed solution does not form a connected original subgraph."
        )
    if any(not set(group).intersection(nodes) for group in spec.groups):
        raise RuntimeError("Composed solution does not cover every group.")
    if any(t not in original for group in spec.terminal_groups for t in group):
        raise RuntimeError("Composed solution omits mandatory terminals.")
    cost = sum(spec.graph.edges[e].get(spec.weight, 1) for e in edges)
    cost += sum((spec.node_costs or {}).get(node, 0) for node in nodes)
    if not math.isclose(cost, objective, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError("Composed solution objective does not match original costs.")
    return edges, nodes, cost


class GroupComposition:
    """Private solve strategy for connected group choices and node costs."""

    def __init__(self, spec):
        self.spec = spec
        self.cut_stats = {}

    def get_solution(
        self, time_limit=300, log_file="", solver="highs", threads=None, exact=True
    ):
        """Solve exactly and expose only validated original-space results."""
        if not exact:
            raise NotImplementedError(
                "Group/node-cost composition does not support heuristic-only solving."
            )
        solver = solver.lower()
        if solver not in ("highs", "gurobi"):
            raise ValueError(
                "Unknown solver {!r}; choose 'highs' or 'gurobi'.".format(solver)
            )
        compiled = compile_groups(self.spec)
        view = DirectedSteinerProblem(
            compiled.graph, compiled.root, compiled.terminals, weight=self.spec.weight
        )
        builder = mm.build_model if solver == "highs" else mm.build_model_gurobi
        runner = mm.run_model if solver == "highs" else mm.run_model_gurobi
        model, x, y1, y2, z = builder(
            view, time_limit=time_limit, logfile=log_file, threads=threads
        )
        try:
            model.addConstr(sum(y1[a] for a in compiled.source_arcs) == 1)
            gap, runtime, objective, _, status = runner(
                model, view, x, y2, z, return_status=True
            )
            self.cut_stats = view.cut_stats
            if status == "infeasible":
                raise RuntimeError(
                    "Group/node-cost instance is infeasible: groups cannot be connected."
                )
            if not math.isfinite(objective):
                raise RuntimeError(
                    "Group/node-cost solve stopped without a connectivity-valid feasible incumbent."
                )
            arcs = [
                a
                for a in view.arcs
                if (model.variableValue(y1[a]) if solver == "highs" else y1[a].X) > 0.5
            ]
            edges, nodes, cost = decode_groups(self.spec, compiled, arcs, objective)
            return NodeWeightedSolution(
                gap=gap if status == "optimal" else math.inf,
                runtime=runtime,
                objective=cost,
                selected_edges=edges,
                original_selected_edges=edges,
                selected_nodes=nodes,
            )
        finally:
            if solver == "gurobi":
                model.dispose()

    def get_optimal_solutions(self, *args, **kwargs):
        """Enumeration needs no-good cuts on decoded original selections."""
        raise NotImplementedError(
            "Group/node-cost composition does not support optimum enumeration."
        )
