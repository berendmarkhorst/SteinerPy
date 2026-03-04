import networkx as nx
import highspy as hp
import warnings
from typing import List, Tuple, Dict, Optional
import logging
from .mathematical_model import (
    build_model, run_model,
    build_prize_collecting_model, run_prize_collecting_model,
    build_budget_model, run_budget_model,
)
from .graph_reducer import preprocess_graph, reduction_stats, map_solution_to_original, ReductionTracker

logger = logging.getLogger(__name__)


def node_split_graph(
    graph: nx.Graph,
    terminal_groups: List[List],
    node_weights: Dict,
    weight: str = "weight",
) -> Tuple[nx.DiGraph, List[List], Dict]:
    """
    Transform a node-weighted graph into an edge-weighted *directed* graph by
    splitting each node v into v_in and v_out connected by a directed edge whose
    weight equals node_weights[v].  Original edges become directed cross-edges
    between the split nodes in both directions (to preserve undirected semantics).

    :param graph: original undirected graph.
    :param terminal_groups: nested list of terminals.
    :param node_weights: dict mapping node -> node cost/weight.
    :param weight: edge-attribute name for weights.
    :return: (new_graph, new_terminal_groups, node_map)
             where node_map maps every split-node name back to its original node.
    """
    new_graph = nx.DiGraph()
    node_map: Dict[str, object] = {}

    for v in graph.nodes():
        v_in, v_out = f"{v}_in", f"{v}_out"
        node_map[v_in] = v
        node_map[v_out] = v
        new_graph.add_edge(v_in, v_out, **{weight: node_weights.get(v, 0)})

    for u, v, data in graph.edges(data=True):
        edge_cost = data.get(weight, 0)
        # Both directions to preserve undirected semantics
        new_graph.add_edge(f"{u}_out", f"{v}_in", **{weight: edge_cost})
        new_graph.add_edge(f"{v}_out", f"{u}_in", **{weight: edge_cost})

    new_terminal_groups = [[f"{t}_in" for t in group] for group in terminal_groups]
    return new_graph, new_terminal_groups, node_map


class BaseSteinerProblem:
    def __init__(self, graph: nx.Graph, terminal_groups: List[List], weight="weight", preprocess=True, **kwargs):
        """
        Initialize the SteinerProblem (can be tree or forest).
        :param graph: networkx graph (Graph or DiGraph).
        :param terminal_groups: nested list of terminals.
        :param weight: edge attribute specified by this string as the edge weight.
        """
        self.original_graph = graph
        self.preprocess = preprocess

        if preprocess:
            if isinstance(graph, nx.DiGraph):
                raise ValueError("Graph preprocessing is not supported for directed graphs. Use preprocess=False.")
            self.graph, self.reduction_tracker = preprocess_graph(graph, terminal_groups, weight)
            stats = reduction_stats(self.original_graph, self.graph)
            logger.info(f"Graph reduced: {stats['nodes_removed']} nodes ({stats['node_reduction_percent']:.1f}%), "
                  f"{stats['edges_removed']} edges ({stats['edge_reduction_percent']:.1f}%) removed")
        else:
            self.graph = graph
            self.reduction_tracker = ReductionTracker()

        self.terminal_groups = terminal_groups
        self.weight = weight
        self.edges = list(self.graph.edges())
        # For directed graphs arcs are one-directional; for undirected both directions
        if isinstance(self.graph, nx.DiGraph):
            self.arcs = self.edges
        else:
            self.arcs = self.edges + [(v, u) for (u, v) in self.edges]
        self.nodes = list(self.graph.nodes())
        self.steiner_points = set(self.nodes) - set([t for group in terminal_groups for t in group])
        self.roots = [group[0] for group in self.terminal_groups]

        # Extract global modifiers like max_degree or budget from kwargs
        self.max_degree = kwargs.get('max_degree', None)
        self.budget = kwargs.get('budget', None)
        

    def __repr__(self):
        return f"Problem with a graph of {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges and {self.terminal_groups} as terminal groups."

    def get_solution(self, time_limit: float = 300, log_file: str = "") -> 'Solution':
        """
        Get the solution of the Steiner Problem using HighsPy.

        When ``budget`` was supplied at construction time the solver maximises
        the number of connected terminals subject to the budget constraint and
        returns a :class:`BudgetSolution`.  Otherwise the solver minimises the
        total edge cost and returns a plain :class:`Solution`.

        Optional modifiers set at construction time (``max_degree``, ``budget``)
        are applied automatically regardless of which problem class is used.

        :param time_limit: time limit in seconds.
        :param log_file: path to the log file.
        :return: :class:`Solution` (or :class:`BudgetSolution` when a budget is set).
        """
        if self.budget is not None:
            # Budget-constrained: maximise connected terminals
            model, x, y1, y2, z, f, penalty_vars = build_budget_model(
                self, time_limit=time_limit, logfile=log_file
            )
            gap, runtime, connected_count, selected_edges, penalties = run_budget_model(
                model, self, x, penalty_vars
            )

            if self.preprocess:
                original_selected_edges = map_solution_to_original(
                    selected_edges, self.reduction_tracker, self.graph
                )
            else:
                original_selected_edges = selected_edges

            total_terminals = sum(len(g) for g in self.terminal_groups)

            return BudgetSolution(
                gap=gap,
                runtime=runtime,
                objective=connected_count,
                selected_edges=selected_edges,
                original_selected_edges=original_selected_edges,
                was_preprocessed=self.preprocess,
                connected_terminals=connected_count,
                total_terminals=total_terminals,
                penalties=penalties,
            )

        model, x, y1, y2, z = build_model(self, time_limit=time_limit, logfile=log_file)
        gap, runtime, objective, selected_edges = run_model(model, self, x, y2, z)

        # Map solution back to original graph if preprocessing was used
        if self.preprocess:
            original_selected_edges = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original_selected_edges = selected_edges

        solution = Solution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            was_preprocessed=self.preprocess,
        )

        return solution

class Solution:
    def __init__(self, gap: float, runtime: float, objective: float, 
                 selected_edges: List[Tuple], original_selected_edges: List[Tuple] = None,
                 was_preprocessed: bool = False):
        self.gap = gap
        self.runtime = runtime
        self.objective = objective
        self.selected_edges = selected_edges  # Edges in the (possibly reduced) graph
        self.original_selected_edges = original_selected_edges or selected_edges  # Edges in the original graph
        self.was_preprocessed = was_preprocessed
    
    @property
    def edges(self):
        """Return the edges in the original graph."""
        return self.original_selected_edges


class SteinerProblem(BaseSteinerProblem):
    pass

class PrizeCollectingProblem(SteinerProblem):  # Inherit from SteinerProblem instead of BaseSteinerProblem
    def __init__(self, graph, terminal_groups, node_prizes, penalty_cost=1000, penalty_budget=None, **kwargs):
        """
        Prize Collecting Steiner Problem - extends regular Steiner problem.
        :param node_prizes: dict mapping node -> prize value
        :param penalty_cost: cost per unconnected terminal
        :param penalty_budget: maximum total penalty allowed (optional)
        """
        # Initialize base Steiner problem first
        super().__init__(graph, terminal_groups, **kwargs)
        
        # Add prize collecting specific attributes
        self.node_prizes = node_prizes
        self.penalty_cost = penalty_cost
        self.penalty_budget = penalty_budget
        
    def get_solution(self, time_limit: float = 300, log_file: str = "") -> 'PrizeCollectingSolution':
        """Override to use prize collecting model."""
        model, x, y1, y2, z, f, node_vars, penalty_vars = build_prize_collecting_model(
            self, time_limit=time_limit, logfile=log_file
        )
        
        gap, runtime, objective, selected_edges, selected_nodes, penalties = run_prize_collecting_model(
            model, self, x, node_vars, penalty_vars
        )

        # Map solution back to original graph if preprocessing was used
        if self.preprocess:
            original_selected_edges = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original_selected_edges = selected_edges

        edge_cost = sum(self.graph.edges[e][self.weight] for e in selected_edges)

        solution = PrizeCollectingSolution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
            penalties=penalties,
            total_prize=sum(self.node_prizes.get(node, 0) for node in selected_nodes),
            edge_cost=edge_cost,
            was_preprocessed=self.preprocess
        )

        return solution
    
class PrizeCollectingSolution(Solution):
    def __init__(self, selected_nodes: List[str] = None, penalties: Dict = None, 
                 total_prize: float = 0, edge_cost: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = selected_nodes or []
        self.penalties = penalties or {}
        self.total_prize = total_prize
        self.edge_cost = edge_cost
        
    @property
    def net_value(self):
        """Total prize collected minus edge costs and penalties."""
        return self.total_prize - self.edge_cost - sum(self.penalties.values())
    
    def __repr__(self):
        return f"PrizeCollectingSolution(objective={self.objective:.2f}, prizes={self.total_prize:.2f}, nodes={len(self.selected_nodes)}, edges={len(self.edges)})"


# ---------------------------------------------------------------------------
# Node-Weighted Steiner Tree (NWST) and Maximum-Weight Connected Subgraph (MWCS)
# ---------------------------------------------------------------------------

class NodeWeightedSteinerProblem(BaseSteinerProblem):
    """
    Node-Weighted Steiner Tree Problem (NWST).

    Converts the node-weighted graph to an edge-weighted graph by splitting each
    node v into v_in and v_out connected by an edge of cost = node_weights[v].
    Then solves the standard Steiner Tree problem on the transformed graph.

    Terminal node costs are always incurred (they are part of every solution) and
    are added as a constant to the reported objective value.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], node_weights: Dict,
                 weight: str = "weight", **kwargs):
        """
        :param graph: networkx undirected graph.
        :param terminal_groups: nested list of terminals.
        :param node_weights: dict mapping node -> node cost.
        :param weight: edge attribute name for weights.

        Note: graph preprocessing is not supported for node-weighted problems because
        the node-splitting transformation produces a directed graph internally.
        """
        self.node_weights = node_weights
        self.original_terminal_groups_nw = terminal_groups

        transformed_graph, new_terminal_groups, self._nw_node_map = node_split_graph(
            graph, terminal_groups, node_weights, weight=weight
        )

        # The transformed graph is a DiGraph; preprocessing is not applicable
        super().__init__(transformed_graph, new_terminal_groups, weight=weight,
                         preprocess=False, **kwargs)

    def get_solution(self, time_limit: float = 300, log_file: str = "") -> 'NodeWeightedSolution':
        """Solve and map solution back to the original node-weighted graph."""
        from .mathematical_model import build_model, run_model

        model, x, y1, y2, z = build_model(self, time_limit=time_limit, logfile=log_file)
        gap, runtime, objective, _ = run_model(model, self, x, y2, z)

        # Use arc (y1) variables for the actual directed tree structure instead of edge
        # (x) variables, to avoid degenerate zero-weight cross-edges being included.
        # Also exclude arcs pointing INTO root nodes (roots have no parents in an arborescence).
        root_nodes_split = set(self.roots)
        used_arcs = [
            (u, v) for (u, v) in self.arcs
            if model.variableValue(y1[(u, v)]) > 0.5 and v not in root_nodes_split
        ]

        # Terminal node costs are always incurred (not captured in the solver objective
        # because the path ends at t_in without traversing t_in -> t_out for non-root terminals)
        non_root_terminals = [
            t
            for group in self.original_terminal_groups_nw
            for t in group[1:]  # exclude root (first element) — its cost is already counted
        ]
        terminal_cost = sum(self.node_weights.get(t, 0) for t in non_root_terminals)
        adjusted_objective = objective + terminal_cost

        original_selected_edges = self._map_split_edges_to_original(used_arcs)
        selected_nodes = self._extract_original_nodes(used_arcs)

        return NodeWeightedSolution(
            gap=gap,
            runtime=runtime,
            objective=adjusted_objective,
            selected_edges=original_selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
        )

    def _is_node_cost_edge(self, u, v) -> bool:
        """Return True if the split-graph edge (u, v) represents a node cost."""
        orig_u = self._nw_node_map.get(u)
        orig_v = self._nw_node_map.get(v)
        return orig_u is not None and orig_u == orig_v

    def _map_split_edges_to_original(self, split_edges: List[Tuple]) -> List[Tuple]:
        """Map edges from the split graph back to edges in the original graph."""
        seen = set()
        original_edges = []
        for u, v in split_edges:
            if self._is_node_cost_edge(u, v):
                continue
            orig_u = self._nw_node_map.get(u)
            orig_v = self._nw_node_map.get(v)
            if orig_u is not None and orig_v is not None and orig_u != orig_v:
                key = tuple(sorted([str(orig_u), str(orig_v)]))
                if key not in seen:
                    seen.add(key)
                    original_edges.append((orig_u, orig_v))
        return original_edges

    def _extract_original_nodes(self, split_edges: List[Tuple]) -> List:
        """Extract original nodes from node-cost edges in the split graph, plus terminals."""
        nodes: set = set()
        for u, v in split_edges:
            if self._is_node_cost_edge(u, v):
                nodes.add(self._nw_node_map[u])
        # Terminal nodes are always part of the solution
        for group in self.original_terminal_groups_nw:
            for t in group:
                nodes.add(t)
        return sorted(nodes, key=lambda n: str(n))


class NodeWeightedSolution(Solution):
    """Solution for the Node-Weighted Steiner Tree Problem."""

    def __init__(self, selected_nodes: Optional[List] = None, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = selected_nodes or []

    def __repr__(self):
        return (f"NodeWeightedSolution(objective={self.objective:.2f}, "
                f"nodes={len(self.selected_nodes)}, edges={len(self.edges)})")


class MaxWeightConnectedSubgraph(PrizeCollectingProblem):
    """
    Maximum-Weight Connected Subgraph (MWCS).

    Finds a connected subgraph maximising the sum of node weights.  Nodes with
    positive weights are modelled as optional terminals with prizes; nodes with
    negative weights are treated as Steiner points that are included only when
    they are necessary connectors.  A user-supplied (or automatically chosen)
    root node anchors the solution.
    """

    def __init__(self, graph: nx.Graph, node_weights: Dict, root=None,
                 weight: str = "weight", **kwargs):
        """
        :param graph: networkx undirected graph.
        :param node_weights: dict mapping node -> weight (positive or negative).
        :param root: optional root node; defaults to the highest-weight node.
        :param weight: edge attribute name for edge weights.
        """
        if root is None:
            root = max(graph.nodes(), key=lambda v: node_weights.get(v, 0))
        self.mwcs_root = root

        # Nodes with positive weight become optional terminals with prizes.
        # Negative-weight nodes are left as Steiner points (no prize).
        positive_nodes = [v for v in graph.nodes() if node_weights.get(v, 0) > 0]
        all_terminals = [root] + [v for v in positive_nodes if v != root]
        terminal_groups = [all_terminals]

        node_prizes = {v: max(0.0, node_weights.get(v, 0.0)) for v in graph.nodes()}

        # Use a penalty equal to the maximum positive prize so that connecting
        # a terminal is always preferred over paying the penalty.
        max_prize = max(node_prizes.values()) if node_prizes else 1.0
        penalty_cost = max_prize + 1.0

        super().__init__(
            graph=graph,
            terminal_groups=terminal_groups,
            node_prizes=node_prizes,
            penalty_cost=penalty_cost,
            weight=weight,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Degree-Constrained Steiner Tree Problem
# ---------------------------------------------------------------------------

class DegreeConstrainedSteinerProblem(SteinerProblem):
    """
    Degree-Constrained Steiner Tree Problem.

    .. deprecated:: 0.2.0
        Pass ``max_degree`` directly to :class:`SteinerProblem` (or any other
        problem class) instead::

            SteinerProblem(graph, terminal_groups, max_degree=2)

        This class is kept for backward compatibility and will be removed in a
        future version.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], max_degree: int, **kwargs):
        """
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param max_degree: maximum allowed degree for any node in the solution.
        """
        warnings.warn(
            "DegreeConstrainedSteinerProblem is deprecated. "
            "Pass max_degree directly to SteinerProblem (or any other problem class) instead: "
            "SteinerProblem(graph, terminal_groups, max_degree=max_degree).",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs['max_degree'] = max_degree
        super().__init__(graph, terminal_groups, **kwargs)


# ---------------------------------------------------------------------------
# Budget-Constrained Steiner Tree Problem
# ---------------------------------------------------------------------------

class BudgetConstrainedSteinerProblem(SteinerProblem):
    """
    Budget-Constrained Steiner Tree Problem.

    .. deprecated:: 0.2.0
        Pass ``budget`` directly to :class:`SteinerProblem` (or any other
        problem class) instead::

            SteinerProblem(graph, terminal_groups, budget=10.0)

        This class is kept for backward compatibility and will be removed in a
        future version.
    """

    def __init__(self, graph: nx.Graph, terminal_groups: List[List], budget: float, **kwargs):
        """
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param budget: maximum total edge cost allowed.
        """
        warnings.warn(
            "BudgetConstrainedSteinerProblem is deprecated. "
            "Pass budget directly to SteinerProblem (or any other problem class) instead: "
            "SteinerProblem(graph, terminal_groups, budget=budget).",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs['budget'] = budget
        super().__init__(graph, terminal_groups, **kwargs)


class BudgetSolution(Solution):
    """Solution for the Budget-Constrained Steiner Tree Problem."""

    def __init__(self, connected_terminals: int = 0, total_terminals: int = 0,
                 penalties: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.connected_terminals = connected_terminals
        self.total_terminals = total_terminals
        self.penalties = penalties or {}

    def __repr__(self):
        return (f"BudgetSolution(connected={self.connected_terminals}/{self.total_terminals}, "
                f"edges={len(self.edges)})")


# ---------------------------------------------------------------------------
# Directed Steiner Tree Problem (Steiner Arborescence)
# ---------------------------------------------------------------------------

class DirectedSteinerProblem(BaseSteinerProblem):
    """
    Directed Steiner Tree Problem (Steiner Arborescence Problem).

    The graph is a directed graph (nx.DiGraph).  A designated root node must
    have a directed path to every terminal node.  Edges only allow flow in the
    direction they are defined; reverse arcs are not added.
    """

    def __init__(self, graph: nx.DiGraph, root, terminals: List, weight: str = "weight", **kwargs):
        """
        :param graph: networkx DiGraph.
        :param root: root node — the source of the arborescence.
        :param terminals: list of terminal nodes that must be reachable from root.
        :param weight: edge attribute for edge weights.
        """
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("DirectedSteinerProblem requires a directed graph (nx.DiGraph).")

        # Build a single terminal group with root as the first element
        terminal_group = [root] + [t for t in terminals if t != root]

        # Preprocessing is not supported for directed graphs
        kwargs['preprocess'] = False

        super().__init__(graph, [terminal_group], weight=weight, **kwargs)