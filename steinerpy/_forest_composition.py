"""Forest requirements share original edge and node selections and costs."""

import networkx as nx

from ._selection_model import SelectionModel, solve_selection
from .objects import BudgetSolution, NodeWeightedSolution


class ForestComposition:
    def __init__(self, spec):
        self.spec = spec
        self.cut_stats = {}
        self.roots = [
            spec.root if spec.graph.is_directed() and spec.root is not None else g[0]
            for g in spec.terminal_groups
        ]
        self.demands = [
            tuple(dict.fromkeys((root,) + group))
            for root, group in zip(self.roots, spec.terminal_groups)
        ]

    def decode(self, edges, nodes):
        spec = self.spec
        chosen = set(nodes)
        selected = nx.DiGraph() if spec.graph.is_directed() else nx.Graph()
        selected.add_nodes_from(nodes)
        selected.add_edges_from(edges)
        if (
            not chosen <= set(spec.graph)
            or set(selected) != chosen
            or any(not spec.graph.has_edge(*e) for e in edges)
        ):
            raise RuntimeError("Forest solution contains invalid original selections.")
        reachable = []
        for root in self.roots:
            if root not in chosen:
                raise RuntimeError("Forest solution omits a required root.")
            reachable.append(
                nx.descendants(selected, root) | {root}
                if spec.graph.is_directed()
                else nx.node_connected_component(selected, root)
            )
        attached = set().union(*reachable)
        group_reachable = set()
        if spec.groups:
            candidates = (
                [nx.descendants(selected, spec.root) | {spec.root}]
                if spec.graph.is_directed()
                else list(nx.connected_components(selected))
            )
            candidates = [c for c in candidates if attached | c == chosen]
            if not candidates:
                raise RuntimeError(
                    "Forest solution contains disconnected group components."
                )
            group_reachable = max(
                candidates,
                key=lambda c: sum(bool(c.intersection(g)) for g in spec.groups),
            )
            attached |= group_reachable
        if attached != chosen:
            raise RuntimeError(
                "Forest solution contains nodes disconnected from all roots."
            )
        missed = {
            f"group_{i}_terminal_{j}": 1
            for i, group in enumerate(self.demands)
            for j, t in enumerate(group)
            if t not in reachable[i]
        }
        if (spec.budget is None or spec.groups) and missed:
            raise RuntimeError("Forest solution omits mandatory terminals.")
        if spec.groups:
            missed = {
                f"group_{i}": 1
                for i, group in enumerate(spec.groups)
                if group_reachable.isdisjoint(group)
            }
            if spec.budget is None and missed:
                raise RuntimeError("Forest solution omits connected group choices.")
        if spec.max_degree is not None and any(
            d > spec.max_degree for _, d in selected.degree()
        ):
            raise RuntimeError(
                "Forest solution exceeds the original-node degree limit."
            )
        if spec.hop_limit is not None and len(edges) > spec.hop_limit:
            raise RuntimeError("Forest solution exceeds the original-edge hop limit.")
        edge_cost = sum(spec.graph.edges[e].get(spec.weight, 1) for e in edges)
        if spec.budget is not None and edge_cost > spec.budget + 1e-7:
            raise RuntimeError("Forest solution exceeds the original-edge budget.")
        cost = edge_cost + sum((spec.node_costs or {}).get(n, 0) for n in chosen)
        covered = self.total_requirements - len(missed)
        return edges, nodes, cost, covered, missed

    @property
    def total_requirements(self):
        return (
            len(self.spec.groups) if self.spec.groups else sum(map(len, self.demands))
        )

    def get_solution(
        self, time_limit=300, log_file="", solver="highs", threads=None, exact=True
    ):
        if not exact:
            raise NotImplementedError(
                "Forest composition does not support heuristic-only solving."
            )
        spec = self.spec
        model = SelectionModel(spec, solver, time_limit, log_file, threads)
        try:
            coverage = []
            for root, group in zip(self.roots, self.demands):
                model.add(model.nodes[root] == 1)
                for terminal in group:
                    connected = model.variable(binary=True)
                    coverage.append(connected)
                    model.add(connected <= model.nodes[terminal])
                    if spec.budget is None or spec.groups or terminal == root:
                        model.add(connected == 1)
                    if terminal != root:
                        model.flow({root: connected}, {terminal: connected})
            roots = dict.fromkeys(self.roots, 1)
            if spec.groups:
                if spec.graph.is_directed():
                    group_roots = {spec.root: 1}
                else:
                    group_roots = {n: model.variable(binary=True) for n in model.nodes}
                    model.add(model.sum(group_roots.values()) == 1)
                    for n, root in group_roots.items():
                        model.add(root <= model.nodes[n])
                coverage = model.choices(
                    spec.groups, group_roots, spec.budget is not None
                )
                for n, root in group_roots.items():
                    roots.setdefault(n, root)
            model.attach_selected_nodes(roots)
            result, gap, runtime = solve_selection(
                model,
                coverage if spec.budget is not None else [],
                self.decode,
                time_limit,
            )
            edges, nodes, cost, covered, missed = result
            common = dict(
                gap=gap,
                runtime=runtime,
                selected_edges=edges,
                original_selected_edges=edges,
            )
            if spec.budget is None:
                return NodeWeightedSolution(
                    objective=cost, selected_nodes=nodes, **common
                )
            solution = BudgetSolution(
                objective=covered,
                connected_terminals=covered,
                total_terminals=self.total_requirements,
                penalties=missed,
                **common,
            )
            solution.selected_nodes = nodes
            solution.total_cost = cost
            return solution
        finally:
            model.close()

    def get_optimal_solutions(self, *args, **kwargs):
        raise NotImplementedError(
            "Forest composition does not support optimum enumeration."
        )
