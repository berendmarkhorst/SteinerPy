import networkx as nx
import highspy as hp
from typing import List, Tuple, Dict
import logging
from .mathematical_model import build_model, run_model, build_prize_collecting_model, run_prize_collecting_model
from .graph_reducer import preprocess_graph, reduction_stats, map_solution_to_original, ReductionTracker

logger = logging.getLogger(__name__)

class BaseSteinerProblem:
    def __init__(self, graph: nx.Graph, terminal_groups: List[List], weight="weight", preprocess=True, **kwargs):
        """
        Initialize the SteinerProblem (can be tree or forest).
        :param graph: networkx graph.
        :param terminal_groups: nested list of terminals.
        :param weight: edge attribute specified by this string as the edge weight.
        """
        self.original_graph = graph
        self.preprocess = preprocess
        
        if preprocess:
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
        :param time_limit: time limit in seconds.
        :param log_file: path to the log file.
        :return: Solution object.
        """

        model, x, y1, y2, z, f = build_model(self, time_limit=time_limit, logfile=log_file)
        gap, runtime, objective, selected_edges = run_model(model, self, x)

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

class PrizeCollectingProblem(BaseSteinerProblem):
    def __init__(self, graph, terminal_groups, node_prizes, penalty_budget=None, **kwargs):
        """
        Prize Collecting Steiner Problem.
        :param node_prizes: dict mapping node -> prize value
        :param penalty_budget: maximum penalty allowed (optional)
        """
        self.node_prizes = node_prizes
        self.penalty_budget = penalty_budget
        super().__init__(graph, terminal_groups, **kwargs)
        
    def get_solution(self, time_limit: float = 300, log_file: str = "") -> 'PrizeCollectingSolution':
        """Override to use prize collecting model."""
        model, x, y, penalty_vars = build_prize_collecting_model(self, time_limit=time_limit, logfile=log_file)
        gap, runtime, objective, selected_edges, selected_nodes, penalties = run_prize_collecting_model(model, self, x, y, penalty_vars)

        if self.preprocess:
            original_selected_edges = map_solution_to_original(selected_edges, self.reduction_tracker, self.graph)
        else:
            original_selected_edges = selected_edges

        solution = PrizeCollectingSolution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
            penalties=penalties,
            total_prize=sum(self.node_prizes.get(node, 0) for node in selected_nodes),
            was_preprocessed=self.preprocess,
        )

        return solution

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

        solution = PrizeCollectingSolution(
            gap=gap,
            runtime=runtime,
            objective=objective,
            selected_edges=selected_edges,
            original_selected_edges=original_selected_edges,
            selected_nodes=selected_nodes,
            penalties=penalties,
            total_prize=sum(self.node_prizes.get(node, 0) for node in selected_nodes),
            was_preprocessed=self.preprocess
        )

        return solution
    
class PrizeCollectingSolution(Solution):
    def __init__(self, selected_nodes: List[str] = None, penalties: Dict = None, 
                 total_prize: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = selected_nodes or []
        self.penalties = penalties or {}
        self.total_prize = total_prize
        
    @property
    def net_value(self):
        """Total prize collected minus edge costs and penalties."""
        return self.total_prize - sum(self.penalties.values())
    
    def __repr__(self):
        return f"PrizeCollectingSolution(objective={self.objective:.2f}, prizes={self.total_prize:.2f}, nodes={len(self.selected_nodes)}, edges={len(self.edges)})"