"""Original-space objective coefficients and legacy reporting conventions.

The common minimization is sum(edge costs) + sum(node costs). Legacy penalty
formulations additionally need a cost per unconnected terminal requirement.
The reported value is ``report_sign * canonical_value + report_offset``.
This describes objective values, not connectivity, constraints, or solver gaps.
"""

from dataclasses import dataclass, field, replace
from typing import Dict


@dataclass(frozen=True)
class ObjectiveSpec:
    """Coefficients for binary selections in the original graph.

    Undirected edges use frozenset keys; directed arcs use ordered tuple keys.
    ``penalty_count`` counts terminal requirements, including separate forest
    requirements for the same node, rather than unique omitted vertices.
    """

    edge_costs: Dict
    node_costs: Dict = field(default_factory=dict)
    directed: bool = False
    penalty_cost: float = 0
    report_sign: int = 1
    report_offset: float = 0

    @classmethod
    def from_graph(cls, graph, weight="weight", *, default_edge_cost=1):
        """Describe edge costs on the supplied original or working graph."""
        return cls(
            edge_costs={
                (u, v) if graph.is_directed() else frozenset((u, v)): data.get(
                    weight, default_edge_cost
                )
                for u, v, data in graph.edges(data=True)
            },
            directed=graph.is_directed(),
        )

    def edge_cost(self, edge):
        """Look up one edge, preserving arc direction for directed graphs."""
        return self.edge_costs[tuple(edge) if self.directed else frozenset(edge)]

    @classmethod
    def from_prizes(cls, graph, prizes, weight="weight", *, penalty_cost=1000):
        """Describe the native penalty objective; only positive prizes earn rewards."""
        return replace(
            cls.from_graph(graph, weight),
            node_costs={
                node: -prizes[node] for node in graph if prizes.get(node, 0) > 0
            },
            penalty_cost=penalty_cost,
        )

    @classmethod
    def from_node_weights(cls, graph, weights):
        """Minimize negative signed node weights; edges supply only topology."""
        return cls(
            edge_costs={
                (u, v) if graph.is_directed() else frozenset((u, v)): 0
                for u, v in graph.edges
            },
            node_costs={node: -weights.get(node, 0.0) for node in graph},
            directed=graph.is_directed(),
            report_sign=-1,
        )

    def _selection_costs(self, edges, nodes):
        keys = dict.fromkeys(
            tuple(edge) if self.directed else frozenset(edge) for edge in edges
        )
        return (
            sum(self.edge_costs[key] for key in keys),
            sum(self.node_costs.get(node, 0) for node in dict.fromkeys(nodes)),
        )

    def canonical_value(self, edges=(), nodes=(), *, penalty_count=0):
        """Evaluate binary selections once, before reporting sign and offset."""
        edge_cost, node_cost = self._selection_costs(edges, nodes)
        return edge_cost + node_cost + self.penalty_cost * penalty_count

    def reported_value(self, canonical_value):
        """Convert a canonical minimum to the existing variant's reported value."""
        return self.report_sign * canonical_value + self.report_offset

    def value(self, edges=(), nodes=(), *, penalty_count=0):
        """Evaluate an original-space selection in the variant's reporting units."""
        edge_cost, node_cost = self._selection_costs(edges, nodes)
        # Cancel prize offsets before adding small edge costs, as PC backmapping does.
        return (
            self.report_sign * edge_cost
            + (self.report_sign * node_cost + self.report_offset)
            + self.report_sign * self.penalty_cost * penalty_count
        )


def objective_from_spec(spec, variant, *, pc_transform=None):
    """Describe a validated strategy's objective without constructing a solver.

    ``variant`` is the resolved strategy name, or None for group composition.
    ``pc_transform`` may specify the solve-time override for prize/MWCS paths;
    otherwise the configured default is used. Path eligibility remains the
    validation/solver layer's responsibility. Geometry needs its Hanan grid
    before graph coefficients can be described.
    """
    if spec.graph is None:
        raise NotImplementedError(
            "Rectilinear objective coefficients require the Hanan grid."
        )

    graph = spec.graph
    node_costs = {}
    penalty_cost = 0
    sign = 1
    offset = 0
    topology_only = False
    default_edge_cost = 1
    transform = (
        spec.options.get("pc_transform", variant == "directed_prize_collecting")
        if pc_transform is None
        else pc_transform
    )

    if variant is None:
        node_costs = dict(spec.node_costs or {})
        if spec.budget is not None:
            node_costs = {}
            topology_only = True
            penalty_cost, sign, offset = 1, -1, len(spec.groups)
    elif variant == "node_weighted":
        node_costs = dict(spec.node_weights)
        # The legacy node-splitting helper defaults missing original costs to 0.
        default_edge_cost = 0
    elif variant == "budgeted_max_weight_connected" or (
        variant == "max_weight_connected" and transform
    ):
        return ObjectiveSpec.from_node_weights(graph, spec.node_weights)
    elif variant in {"prize_collecting", "directed_prize_collecting"}:
        if transform:
            node_costs = {node: -spec.node_prizes.get(node, 0) for node in graph}
            offset = sum(spec.node_prizes.get(node, 0) for node in graph)
        else:
            return ObjectiveSpec.from_prizes(
                graph,
                spec.node_prizes,
                spec.weight,
                penalty_cost=(
                    spec.penalty_cost
                    if "penalty_cost" in spec.supplied_features
                    else 1000
                ),
            )
    elif variant == "max_weight_connected":
        # Its legacy default is the penalty ILP, not the signed MWCS objective.
        prizes = {node: max(0, spec.node_weights.get(node, 0)) for node in graph}
        return ObjectiveSpec.from_prizes(
            graph, prizes, spec.weight, penalty_cost=max(prizes.values(), default=0) + 1
        )
    elif (
        variant in {"steiner", "directed", "hop_constrained", "directed_group"}
        and spec.budget is not None
    ):
        # The edge budget is a constraint; the objective counts connected demands.
        topology_only = True
        penalty_cost = 1
        sign = -1
        if variant == "directed_group":
            offset = len(spec.groups)
        elif variant in {"directed", "hop_constrained"}:
            offset = len({spec.root}.union(spec.terminal_groups[0]))
        else:
            offset = sum(map(len, spec.terminal_groups))
    elif variant not in {
        "steiner",
        "directed",
        "hop_constrained",
        "group",
        "directed_group",
        "partial_terminal",
        "full_terminal",
    }:
        raise NotImplementedError(
            "No original-space objective for {!r}.".format(variant)
        )

    objective = ObjectiveSpec.from_graph(
        graph, spec.weight, default_edge_cost=default_edge_cost
    )
    return replace(
        objective,
        edge_costs=(
            dict.fromkeys(objective.edge_costs, 0)
            if topology_only
            else objective.edge_costs
        ),
        node_costs=node_costs,
        penalty_cost=penalty_cost,
        report_sign=sign,
        report_offset=offset,
    )
