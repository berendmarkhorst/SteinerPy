"""Compose connectivity, objectives and limits in original graph coordinates."""

import networkx as nx

from ._selection_model import SelectionModel, solve_selection
from .objects import BudgetSolution, NodeWeightedSolution


class FeatureComposition:
    def __init__(self, spec):
        self.spec = spec
        self.cut_stats = {}
        self.coverage_mode = (
            spec.budget is not None
            and spec.node_prizes is None
            and spec.node_weights is None
            and spec.penalty_cost is None
        )
        self.optional = self.coverage_mode or spec.penalty_cost is not None
        self.objective = spec.objective(None)
        self.roots = [
            spec.root if spec.root is not None else g[0] for g in spec.terminal_groups
        ]
        self.demands = [
            tuple(dict.fromkeys((root,) + group))
            for root, group in zip(self.roots, spec.terminal_groups)
        ]
        if not self.roots and spec.root is not None:
            self.roots = [spec.root]
            self.demands = [(spec.root,)]

    @property
    def choice_root(self):
        if self.spec.root is not None:
            return self.spec.root
        if len(self.roots) == 1:
            return self.roots[0]
        return None

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
        if spec.groups or not self.roots:
            if self.choice_root is not None:
                candidates = [self.reached(selected, self.choice_root)]
            elif not spec.graph.is_directed():
                candidates = list(nx.connected_components(selected))
            else:
                condensed = nx.condensation(selected)
                sources = [n for n, d in condensed.in_degree() if d == 0]
                candidates = [chosen] if len(sources) == 1 else []
            if not chosen and not spec.groups:
                candidates = [set()]
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
        if (not self.optional or spec.groups) and missed:
            raise RuntimeError("Forest solution omits mandatory terminals.")
        if spec.groups:
            missed = {
                f"group_{i}": 1
                for i, group in enumerate(spec.groups)
                if group_reachable.isdisjoint(group)
            }
            if not self.optional and missed:
                raise RuntimeError("Forest solution omits connected group choices.")
        if spec.max_degree is not None and any(
            d > spec.max_degree for _, d in selected.degree()
        ):
            raise RuntimeError(
                "Forest solution exceeds the original-node degree limit."
            )
        if spec.hop_limit is not None and len(edges) > spec.hop_limit:
            raise RuntimeError("Forest solution exceeds the original-edge hop limit.")
        if any(
            selected.degree(n) > 1
            for n in (spec.partial_terminals or ())
            if n in chosen
        ):
            raise RuntimeError(
                "Feature solution violates an original-node leaf constraint."
            )
        edge_cost = sum(spec.graph.edges[e].get(spec.weight, 1) for e in edges)
        if spec.budget is not None and edge_cost > spec.budget + 1e-7:
            raise RuntimeError("Forest solution exceeds the original-edge budget.")
        node_cost = sum((spec.node_costs or {}).get(n, 0) for n in chosen)
        if spec.node_budget is not None and node_cost > spec.node_budget + 1e-7:
            raise RuntimeError("Feature solution exceeds the original-node budget.")
        if spec.penalty_budget is not None and len(missed) > spec.penalty_budget:
            raise RuntimeError(
                "Feature solution exceeds the missed-requirement budget."
            )
        cost = (
            self.objective.canonical_value(edges, chosen, penalty_count=len(missed))
            if not self.coverage_mode
            else edge_cost + node_cost
        )
        covered = self.total_requirements - len(missed)
        return edges, nodes, cost, covered, missed

    @staticmethod
    def reached(graph, root):
        return (
            nx.descendants(graph, root) | {root}
            if graph.is_directed()
            else nx.node_connected_component(graph, root)
        )

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
                    if not self.optional or spec.groups or terminal == root:
                        model.add(connected == 1)
                    if terminal != root:
                        model.flow({root: connected}, {terminal: connected})
            roots = dict.fromkeys(self.roots, 1)
            if spec.groups or not self.roots:
                if self.choice_root is not None:
                    group_roots = {self.choice_root: 1}
                else:
                    group_roots = {n: model.variable(binary=True) for n in model.nodes}
                    if spec.groups:
                        model.add(model.sum(group_roots.values()) == 1)
                    else:
                        model.add(model.sum(group_roots.values()) <= 1)
                    for n, root in group_roots.items():
                        model.add(root <= model.nodes[n])
                coverage = model.choices(spec.groups, group_roots, self.optional)
                for n, root in group_roots.items():
                    roots.setdefault(n, root)
            model.attach_selected_nodes(roots)
            if spec.penalty_budget is not None:
                model.add(len(coverage) - model.sum(coverage) <= spec.penalty_budget)
            if not self.coverage_mode:
                model.cost = (
                    model.sum(
                        self.objective.edge_cost(e) * x for e, x in model.edges.items()
                    )
                    + model.sum(
                        self.objective.node_costs.get(n, 0) * x
                        for n, x in model.nodes.items()
                    )
                    + self.objective.penalty_cost
                    * (len(coverage) - model.sum(coverage))
                )
            result, gap, runtime = solve_selection(
                model,
                coverage if self.coverage_mode else [],
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
            if not self.coverage_mode:
                solution = NodeWeightedSolution(
                    objective=self.objective.value(
                        edges, nodes, penalty_count=len(missed)
                    ),
                    selected_nodes=nodes,
                    **common,
                )
                solution.penalties = missed
            else:
                solution = BudgetSolution(
                    objective=covered,
                    connected_terminals=covered,
                    total_terminals=self.total_requirements,
                    penalties=missed,
                    **common,
                )
            solution.selected_nodes = nodes
            solution.total_cost = sum(
                spec.graph.edges[e].get(spec.weight, 1) for e in edges
            ) + sum((spec.node_costs or {}).get(n, 0) for n in nodes)
            return solution
        finally:
            model.close()

    def get_optimal_solutions(self, *args, **kwargs):
        raise NotImplementedError(
            "Forest composition does not support optimum enumeration."
        )
