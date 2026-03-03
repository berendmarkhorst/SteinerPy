import networkx as nx
import highspy as hp
from typing import List, Tuple
import logging
from .mathematical_model import build_model, run_model
from .graph_reducer import preprocess_graph, reduction_stats, map_solution_to_original, ReductionTracker

logger = logging.getLogger(__name__)

class SteinerProblem:
    def __init__(self, graph: nx.Graph, terminal_groups: List[List], weight="weight", preprocess=True):
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


