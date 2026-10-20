"""Compile original-space node costs and group choices into directed cuts."""

from dataclasses import dataclass
import math

import networkx as nx

from . import mathematical_model as mm
from ._transformations import split_node_costs, attach_group_terminals
from .objects import DirectedSteinerProblem, NodeWeightedSolution


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
    split_nodes = split_node_costs(spec.graph, spec.node_costs, spec.weight)
    graph = split_nodes.graph
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
        (root, split_nodes.split[node][0])
        for node in min(groups, key=len)
        if node in feasible_nodes
    ]
    for arc in source_arcs:
        graph.add_edge(*arc, **{spec.weight: 0})
    group_terminals = attach_group_terminals(
        graph,
        groups,
        spec.weight,
        ports={node: pair[1] for node, pair in split_nodes.split.items()},
    )
    return CompiledGroups(
        group_terminals.graph,
        root,
        group_terminals.terminals,
        source_arcs,
        split_nodes.node_arcs,
        split_nodes.edge_arcs,
    )


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
    if spec.hop_limit is not None and len(edges) > spec.hop_limit:
        raise RuntimeError("Composed solution exceeds the original-edge hop limit.")
    original = nx.Graph()
    original.add_nodes_from(nodes)
    original.add_edges_from(edges)
    if not nodes or set(original) != set(nodes) or not nx.is_connected(original):
        raise RuntimeError(
            "Composed solution does not form a connected original subgraph."
        )
    if spec.max_degree is not None and any(
        d > spec.max_degree for _, d in original.degree()
    ):
        raise RuntimeError("Composed solution exceeds the original-node degree limit.")
    if any(not set(group).intersection(nodes) for group in spec.groups):
        raise RuntimeError("Composed solution does not cover every group.")
    if any(t not in original for group in spec.terminal_groups for t in group):
        raise RuntimeError("Composed solution omits mandatory terminals.")
    cost = spec.objective(None).value(edges, nodes)
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
            compiled.graph,
            compiled.root,
            compiled.terminals,
            weight=self.spec.weight,
            hop_limit=self.spec.hop_limit,
            max_degree=self.spec.max_degree,
        )
        # Only cross-arcs represent original edges; paid and artificial arcs do not.
        view._hop_arcs = tuple(compiled.edge_arcs)
        view._degree_arc_groups = {node: [] for node in self.spec.graph}
        for arc, (u, v) in compiled.edge_arcs.items():
            view._degree_arc_groups[u].append(arc)
            view._degree_arc_groups[v].append(arc)
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
                    "Group/node-cost instance is infeasible: no solution meets all requirements."
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
