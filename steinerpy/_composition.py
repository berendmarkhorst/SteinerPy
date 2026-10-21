"""Compile original-space node costs and group choices into directed cuts."""

from dataclasses import dataclass, replace
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

    Non-negative costs admit an optimal tree, rooted at the directed root or
    a selected group member. Exactly one source arc chooses that root; sink-only
    connectors cannot create original-space shortcuts. Every path to a group
    sink leaves a member's output node, so its node-cost arc must be traversed.
    """
    split_nodes = split_node_costs(spec.graph, spec.node_costs, spec.weight)
    graph = split_nodes.graph
    mandatory = [t for group in spec.terminal_groups for t in group]
    if spec.graph.is_directed() and spec.root not in mandatory:
        mandatory.append(spec.root)
    groups = list(spec.groups) + [(t,) for t in mandatory]
    required = groups if spec.budget is None else [(t,) for t in mandatory]
    if spec.graph.is_directed():
        feasible_nodes = nx.descendants(spec.graph, spec.root) | {spec.root}
        if any(feasible_nodes.isdisjoint(group) for group in required):
            feasible_nodes = set()
    else:
        feasible_nodes = set()
        for component in nx.connected_components(spec.graph):
            if all(component.intersection(group) for group in required):
                feasible_nodes.update(component)
    if not feasible_nodes:
        raise RuntimeError(
            "Group/node-cost instance is infeasible: groups cannot be connected."
        )
    root = len(graph)
    graph.add_node(root)
    root_choices = (
        min(groups, key=len)
        if spec.budget is None
        else (
            required[0]
            if required
            else {node for group in spec.groups for node in group}
        )
    )
    if spec.graph.is_directed():
        root_choices = {spec.root}
    source_arcs = [
        (root, split_nodes.split[node][0])
        for node in spec.graph
        if node in root_choices and node in feasible_nodes
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


def make_view(spec, compiled):
    view = DirectedSteinerProblem(
        compiled.graph,
        compiled.root,
        compiled.terminals,
        weight=spec.weight,
        hop_limit=spec.hop_limit,
        max_degree=spec.max_degree,
    )
    view._hop_arcs = tuple(compiled.edge_arcs)
    view._degree_arc_groups = {node: [] for node in spec.graph}
    for arc, (u, v) in compiled.edge_arcs.items():
        view._degree_arc_groups[u].append(arc)
        view._degree_arc_groups[v].append(arc)
    return view


def decode_groups(spec, compiled, selected_arcs, objective, *, require_all_groups=True):
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
    original = nx.DiGraph() if spec.graph.is_directed() else nx.Graph()
    original.add_nodes_from(nodes)
    original.add_edges_from(edges)
    if spec.graph.is_directed():
        connected = spec.root in original and (
            nx.descendants(original, spec.root) | {spec.root}
        ) == set(nodes)
    else:
        connected = bool(nodes) and nx.is_connected(original)
    if not connected or set(original) != set(nodes):
        raise RuntimeError(
            "Composed solution does not form a connected original subgraph."
        )
    if spec.max_degree is not None and any(
        d > spec.max_degree for _, d in original.degree()
    ):
        raise RuntimeError("Composed solution exceeds the original-node degree limit.")
    if require_all_groups and any(
        not set(group).intersection(nodes) for group in spec.groups
    ):
        raise RuntimeError("Composed solution does not cover every group.")
    if any(t not in original for group in spec.terminal_groups for t in group):
        raise RuntimeError("Composed solution omits mandatory terminals.")
    cost = replace(spec, budget=None).objective(None).value(edges, nodes)
    if objective is not None and not math.isclose(
        cost, objective, rel_tol=1e-7, abs_tol=1e-7
    ):
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
        if self.spec.budget is not None:
            from ._composition_budget import solve_budget

            return solve_budget(self, time_limit, log_file, solver, threads)
        compiled = compile_groups(self.spec)
        view = make_view(self.spec, compiled)
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
